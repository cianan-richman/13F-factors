"""SEC-wide Form 13F `SUMMARYPAGE` quantitative filer screener (`2015–2026`).

Auto-discovers quarterly Form 13F Data Set `.zip` archives from SEC's official page
(`https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets`) and extracts
ONLY the tiny metadata tables (`SUBMISSION.tsv`, `COVERPAGE.tsv`, `SUMMARYPAGE.tsv`)
via HTTP Range reads (fetching ~300 KB per quarter instead of ~80 MB `INFOTABLE.tsv`!)
with automatic full-download fallback if Range requests are unsupported.

Computes exact entity-level summary statistics (`n_quarters`, `median_positions`,
`median_aum_usd`, `max_aum_usd`, `first_period`, `last_period`) across ALL SEC 13F-HR
CIKs, producing an objective quantitative filter funnel before final active-fundamental
curation.

Outputs:
  - `data/processed/edgar_13f_filer_summary.parquet`
  - `outputs/filer_screening_candidates.csv`

Usage:
    python -m src.ingest.filer_screener
    python -m src.ingest.filer_screener --parse-only
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import struct
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.edgar_client import EdgarBlockedError, EdgarClient
from src.ingest.master_index import DEFAULT_OUTPUT_PARQUET as DEFAULT_INDEX_PARQUET

logger = logging.getLogger(__name__)

SEC_13F_DATASETS_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
RAW_SUMMARY_DIR = RAW_DATA_DIR / "edgar_13f_summary"
DEFAULT_FILER_SUMMARY_PARQUET = PROCESSED_DATA_DIR / "edgar_13f_filer_summary.parquet"
DEFAULT_CANDIDATES_CSV = OUTPUTS_DIR / "filer_screening_candidates.csv"

# Target metadata members inside each SEC Form 13F Data Set ZIP
TARGET_TSVS = ("SUBMISSION.tsv", "COVERPAGE.tsv", "SUMMARYPAGE.tsv")

# Regex for finding quarterly 13F dataset .zip links on SEC landing page
_ZIP_HREF_RE = re.compile(r'href=["\']([^"\']+form13f\.zip)["\']', re.IGNORECASE)
_YEAR_RE = re.compile(r"(20\d{2})")


class _HttpRangeFile(io.RawIOBase):
    """Minimal seekable file-like object backed by HTTP Range requests for `zipfile.ZipFile`.

    Reads only the ZIP end-of-central-directory tail (~64 KB) plus the exact byte ranges
    of requested members (`SUBMISSION.tsv`, `COVERPAGE.tsv`, `SUMMARYPAGE.tsv`), avoiding
    downloading the multi-gigabyte `INFOTABLE.tsv` payloads.
    """

    def __init__(self, client: EdgarClient, url: str, size: int) -> None:
        self._client = client
        self._url = url
        self._size = size
        self._pos = 0
        # Prefetch tail 64 KB (covers EOCD + central directory for small-member ZIPs)
        tail_start = max(0, size - 65536)
        self._tail_start = tail_start
        self._tail_buf = self._fetch_range(tail_start, size - 1)

    def _fetch_range(self, start: int, end: int) -> bytes:
        resp = self._client.get(self._url, headers={"Range": f"bytes={start}-{end}"})
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"Range GET failed ({resp.status_code}) for {self._url}")
        return resp.content

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"Unsupported whence={whence}")
        self._pos = max(0, min(self._size, self._pos))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            end = self._size - 1
        else:
            end = min(self._size - 1, self._pos + size - 1)
        if end < self._pos:
            return b""
        if self._pos >= self._tail_start:
            rel_s = self._pos - self._tail_start
            rel_e = end - self._tail_start + 1
            data = self._tail_buf[rel_s:rel_e]
        else:
            data = self._fetch_range(self._pos, end)
        self._pos += len(data)
        return data


def discover_dataset_zip_urls(
    client: EdgarClient,
    *,
    min_year: int = 2015,
) -> list[tuple[str, str]]:
    """Return sorted `[(dataset_key, full_url)]` for SEC Form 13F ZIPs from `min_year` onward."""
    resp = client.get(SEC_13F_DATASETS_PAGE)
    html = resp.text
    seen: dict[str, str] = {}
    for m in _ZIP_HREF_RE.finditer(html):
        href = m.group(1).strip()
        full_url = urljoin(SEC_13F_DATASETS_PAGE, href)
        fname = Path(full_url.split("?")[0]).name
        years = [int(y) for y in _YEAR_RE.findall(fname)]
        if not years or max(years) < min_year:
            continue
        key = fname.replace(".zip", "").lower()
        seen[key] = full_url
    return sorted(seen.items())


def _extract_summary_from_zipfile(zf: zipfile.ZipFile, dataset_key: str) -> pd.DataFrame:
    """Join `SUBMISSION.tsv`, `COVERPAGE.tsv`, and `SUMMARYPAGE.tsv` inside one SEC 13F ZIP."""
    member_by_upper = {Path(n).name.upper(): n for n in zf.namelist()}
    dfs: dict[str, pd.DataFrame] = {}
    for target in TARGET_TSVS:
        real_name = member_by_upper.get(target.upper())
        if real_name is None:
            return pd.DataFrame()
        raw_bytes = zf.read(real_name)
        dfs[target] = pd.read_csv(
            io.BytesIO(raw_bytes),
            sep="\t",
            dtype=str,
            on_bad_lines="skip",
            low_memory=False,
        )

    sub = dfs["SUBMISSION.tsv"]
    cov = dfs["COVERPAGE.tsv"]
    smr = dfs["SUMMARYPAGE.tsv"]

    # Normalize column names to uppercase
    sub.columns = [c.strip().upper() for c in sub.columns]
    cov.columns = [c.strip().upper() for c in cov.columns]
    smr.columns = [c.strip().upper() for c in smr.columns]

    # Keep only 13F-HR (initial holdings reports; exclude 13F-NT notices & /A amendments)
    if "SUBMISSIONTYPE" in sub.columns:
        sub = sub[sub["SUBMISSIONTYPE"].str.strip().str.upper() == "13F-HR"].copy()
    if sub.empty:
        return pd.DataFrame()

    sub_cols = [
        c for c in ("ACCESSION_NUMBER", "CIK", "FILING_DATE", "PERIODOFREPORT") if c in sub.columns
    ]
    cov_cols = [
        c
        for c in ("ACCESSION_NUMBER", "FILINGMANAGER_NAME", "REPORTTYPE", "ISCOMBINATION")
        if c in cov.columns
    ]
    smr_cols = [
        c
        for c in (
            "ACCESSION_NUMBER",
            "TABLEENTRYTOTAL",
            "TABLEVALUETOTAL",
            "OTHERINCLUDEDMANAGERSCOUNT",
        )
        if c in smr.columns
    ]

    merged = (
        sub[sub_cols]
        .merge(cov[cov_cols].drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER", how="left")
        .merge(
            smr[smr_cols].drop_duplicates("ACCESSION_NUMBER"), on="ACCESSION_NUMBER", how="inner"
        )
    )
    if merged.empty:
        return pd.DataFrame()

    merged["DATASET_KEY"] = dataset_key
    return merged


def fetch_or_load_dataset_summary(
    client: EdgarClient | None,
    dataset_key: str,
    url: str,
    *,
    cache_dir: Path = RAW_SUMMARY_DIR,
    parse_only: bool = False,
) -> pd.DataFrame:
    """Load cached compact summary Parquet for one SEC 13F dataset, fetching via Range if absent."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_key}.parquet"
    if cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            cache_path.unlink(missing_ok=True)

    if parse_only or client is None:
        return pd.DataFrame()

    # Probe file size via HEAD (or 1-byte Range GET) to use fast partial ZIP extraction
    try:
        probe = client.get(url, headers={"Range": "bytes=0-0"})
        cr = probe.headers.get("Content-Range", "")
        if probe.status_code == 206 and "/" in cr:
            total_size = int(cr.split("/")[-1])
            rf = _HttpRangeFile(client, url, total_size)
            with zipfile.ZipFile(rf, "r") as zf:
                df = _extract_summary_from_zipfile(zf, dataset_key)
        else:
            # Fallback to full download if Range not honored
            full = client.get(url)
            with zipfile.ZipFile(io.BytesIO(full.content), "r") as zf:
                df = _extract_summary_from_zipfile(zf, dataset_key)
    except (zipfile.BadZipFile, struct.error, RuntimeError) as exc:
        logger.warning(
            "Range extraction failed for %s (%s); falling back to full GET", dataset_key, exc
        )
        full = client.get(url)
        with zipfile.ZipFile(io.BytesIO(full.content), "r") as zf:
            df = _extract_summary_from_zipfile(zf, dataset_key)

    if not df.empty:
        tmp = cache_path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
    return df


def normalize_summary_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw SEC `SUMMARYPAGE` records and normalize Jan 3, 2023 `<value>` unit switch."""
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()
    df["cik"] = df["CIK"].astype(str).str.strip().str.lstrip("0").str.zfill(10)
    df["accession_number"] = df["ACCESSION_NUMBER"].astype(str).str.strip()
    df["filing_date"] = pd.to_datetime(df["FILING_DATE"], format="mixed", errors="coerce")
    df["period_of_report"] = pd.to_datetime(df["PERIODOFREPORT"], format="mixed", errors="coerce")
    df["manager_name"] = df.get("FILINGMANAGER_NAME", "").fillna("").astype(str).str.strip()
    df["table_entry_total"] = pd.to_numeric(df["TABLEENTRYTOTAL"], errors="coerce")
    df["table_value_raw"] = pd.to_numeric(df["TABLEVALUETOTAL"], errors="coerce")
    df["other_managers_count"] = pd.to_numeric(
        df.get("OTHERINCLUDEDMANAGERSCOUNT", 0), errors="coerce"
    ).fillna(0)

    df = df.dropna(subset=["filing_date", "table_entry_total", "table_value_raw"]).copy()
    df = df.drop_duplicates(subset=["accession_number"], keep="last").copy()

    # Statutory SEC unit rule: filings before 2023-01-03 report TABLEVALUETOTAL in $1,000s;
    # filings on/after 2023-01-03 report in exact $1.
    is_pre_2023 = df["filing_date"] < pd.Timestamp("2023-01-03")
    df["aum_usd"] = np.where(is_pre_2023, df["table_value_raw"] * 1000.0, df["table_value_raw"])

    # Self-healing guard for 1,000x SEC unit-reporting mistakes in SUMMARYPAGE:
    # 1) Post-2023 filings still reported in $1,000s (e.g., Baupost: avg position < $250k on >=5 pos
    #    while pre-2023 raw scale matches post-2023 raw scale)
    ent_safe = np.where(df["table_entry_total"] > 0, df["table_entry_total"], 1.0)
    avg_pos = np.where(df["table_entry_total"] > 0, df["aum_usd"] / ent_safe, np.nan)
    post_under = (
        (~is_pre_2023)
        & (df["table_entry_total"] >= 5)
        & (avg_pos < 250_000.0)
        & (df["aum_usd"] > 0)
    )
    df["aum_usd"] = np.where(post_under, df["aum_usd"] * 1000.0, df["aum_usd"])

    # 2) Pre-2023 filings already reported in exact $1 (e.g., Kahn Brothers, Hahn, WestEnd,
    #    Chartist, Select Equity): if multiplying by 1,000 yields avg position > $1.5B across
    #    >=20 positions (or >$5B across >=5 positions), raw value was already in $1.
    #    (Excludes Berkshire Hathaway which is excluded from active fundamental roster anyway.)
    pre_over = is_pre_2023 & (
        ((df["table_entry_total"] >= 20) & (avg_pos > 1_500_000_000.0))
        | ((df["table_entry_total"] >= 5) & (avg_pos > 5_000_000_000.0))
    )
    df["aum_usd"] = np.where(pre_over, df["aum_usd"] / 1000.0, df["aum_usd"])

    return df


# Transparent, auditable name-pattern exclusion for non-fundamental institutional categories
_NON_FUNDAMENTAL_RE = re.compile(
    r"\b(?:"
    r"BANK|BANCORP|BANCSHARES|TRUST CO|TRUST COMPANY|TRUST CORP|NORTHERN TRUST|STATE STREET|"
    r"WEALTH|PRIVATE WEALTH|FINANCIAL ADVIS|FINANCIAL GROUP|FINANCIAL PLANNING|ADVISORY SERVICES|"
    r"INSURANCE|ASSURANCE|MUTUAL LIFE|LIFE INS|REINSURANCE|BERKSHIRE HATHAWAY|LOEWS CORP|"
    r"RETIREMENT|PENSION|TJANSTEPENSION|TEACHERS|EMPLOYEES|PUBLIC EMPLOYEES|ENDOWMENT|"
    r"FOUNDATION|UNIVERSITY|SOVEREIGN|NORGES BANK|CENTRAL BANK|SWISS NATIONAL|"
    r"PUBLIC INVESTMENT FUND|MUBADALA|TEMASEK|QATAR INVESTMENT|ABU DHABI INVESTMENT|"
    r"VANGUARD|BLACKROCK|DIMENSIONAL|GEODE|INDEX|ETF|ISHARES|INVESCO|FIDELITY|FMR LLC|"
    r"SCHWAB|MERRILL|MORGAN STANLEY|GOLDMAN SACHS|JPMORGAN|JP MORGAN|CITIGROUP|WELLS FARGO|"
    r"UBS |CREDIT SUISSE|BARCLAYS|DEUTSCHE BANK|HSBC|BNP PARIBAS|ROYAL BANK|TORONTO DOMINION|"
    r"BANK OF AMERICA|BANK OF NEW YORK|BNY MELLON|PNC |US BANCORP|TRUIST|AMERIPRISE|"
    r"EDWARD JONES|RAYMOND JAMES|LPL FINANCIAL|STIFEL|OPPENHEIMER|JEFFERIES|NOMURA|MACQUARIE|"
    r"SUSQUEHANNA|JANE STREET|CITADEL|TWO SIGMA|RENAISSANCE TECH|MILLENNIUM|POINT72|BALYASNY|"
    r"WOLVERINE|JUMP TRADING|VIRTU|HUDSON RIVER|TOWER RESEARCH|DESHAW|D\. E\. SHAW|AQR |"
    r"OPTIVER|AKUNA|IMC |DRW |FLOW TRADERS|BRIDGEWATER|MAN GROUP|WORLDQUANT|ARROWSTREET|"
    r"ACADIAN|PANAGORA|NUMERIC|QMA |APG ASSET|"
    r"PRIVATE EQUITY|VENTURE|VENTURES|BUYOUT|KKR |BLACKSTONE|APOLLO|CARLYLE|TPG |WARBURG|"
    r"SILVER LAKE|THOMA BRAVO|VISTA EQUITY|BAIN CAPITAL|ADVENT INT|GENERAL ATLANTIC|"
    r"SB INVESTMENT|SOFTBANK|ICAHN|ELLIOTT|ALPHABET|MICROSOFT|AMAZON|META PLATFORMS|INTEL CORP"
    r")\b",
    re.IGNORECASE,
)


def compute_entity_funnel(
    summary_df: pd.DataFrame,
    master_index_path: Path = DEFAULT_INDEX_PARQUET,
    *,
    min_quarters: int = 8,
    min_median_aum_usd: float = 2_000_000_000.0,
    min_median_positions: int = 10,
    max_median_positions: int = 100,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    """Compute exact SEC-wide entity filter funnel and candidate table."""
    idx_df = pd.read_parquet(master_index_path)
    total_ciks_all = int(idx_df["cik"].nunique())
    hr_idx = idx_df[idx_df["form_type"] == "13F-HR"]
    hr_counts = hr_idx.groupby("cik")["accession_number"].nunique()
    ciks_ge_min_q = set(hr_counts[hr_counts >= min_quarters].index.astype(str))

    # Aggregate SUMMARYPAGE per CIK (deduplicating any overlapping quarters by period_of_report)
    per_q = summary_df.sort_values(["cik", "period_of_report", "filing_date"]).drop_duplicates(
        subset=["cik", "period_of_report"], keep="last"
    )
    agg = per_q.groupby("cik", as_index=False).agg(
        manager_name=("manager_name", "last"),
        n_summary_quarters=("period_of_report", "nunique"),
        first_period=("period_of_report", "min"),
        last_period=("period_of_report", "max"),
        median_positions=("table_entry_total", "median"),
        min_positions=("table_entry_total", "min"),
        max_positions=("table_entry_total", "max"),
        median_aum_usd=("aum_usd", "median"),
        max_aum_usd=("aum_usd", "max"),
        latest_aum_usd=("aum_usd", "last"),
        median_other_mgrs=("other_managers_count", "median"),
    )
    agg["n_index_hr_quarters"] = (
        agg["cik"].map(hr_counts).fillna(agg["n_summary_quarters"]).astype(int)
    )
    agg["passes_continuity"] = agg["cik"].isin(ciks_ge_min_q) | (
        agg["n_summary_quarters"] >= min_quarters
    )
    agg["passes_scale"] = agg["passes_continuity"] & (agg["median_aum_usd"] >= min_median_aum_usd)
    agg["passes_min_pos"] = agg["passes_scale"] & (agg["median_positions"] >= min_median_positions)
    agg["passes_max_pos"] = agg["passes_min_pos"] & (
        agg["median_positions"] <= max_median_positions
    )
    agg["is_non_fundamental_name"] = agg["manager_name"].str.contains(_NON_FUNDAMENTAL_RE, na=False)
    agg["passes_active_fundamental_screen"] = agg["passes_max_pos"] & (
        ~agg["is_non_fundamental_name"]
    )
    agg["is_active_2025_plus"] = agg["last_period"] >= pd.Timestamp("2025-01-01")

    funnel = {
        "step0_all_edgar_ciks": total_ciks_all,
        "step1_continuity_ge_8q": len(ciks_ge_min_q),
        "step2_scale_ge_2B": int(agg["passes_scale"].sum()),
        "step3_min_pos_ge_10": int(agg["passes_min_pos"].sum()),
        "step4_max_pos_le_100": int(agg["passes_max_pos"].sum()),
        "step5_exclude_non_fundamental_names": int(agg["passes_active_fundamental_screen"].sum()),
        "step5_active_2025_plus": int(
            (agg["passes_active_fundamental_screen"] & agg["is_active_2025_plus"]).sum()
        ),
        "step5_closed_historical": int(
            (agg["passes_active_fundamental_screen"] & (~agg["is_active_2025_plus"])).sum()
        ),
    }

    candidates = (
        agg[agg["passes_active_fundamental_screen"]]
        .sort_values("median_aum_usd", ascending=False)
        .reset_index(drop=True)
    )
    return funnel, agg, candidates


def run_filer_screener(
    *,
    parse_only: bool = False,
    max_workers: int = 4,
    min_year: int = 2015,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    """Fetch/load SEC 13F SUMMARYPAGE tables across `2015–2026` and compute exact entity funnel."""
    cfg = load_config()
    frames: list[pd.DataFrame] = []

    if parse_only:
        for p in sorted(RAW_SUMMARY_DIR.glob("*.parquet")):
            frames.append(pd.read_parquet(p))
    else:
        with EdgarClient(
            user_agent=cfg.sec_user_agent,
            max_requests_per_sec=float(cfg.settings["edgar"].get("max_requests_per_sec", 8)),
        ) as client:
            datasets = discover_dataset_zip_urls(client, min_year=min_year)
            logger.info(
                "Discovered %d SEC Form 13F quarterly data sets (%d+).", len(datasets), min_year
            )
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {
                    pool.submit(fetch_or_load_dataset_summary, client, k, u): k for k, u in datasets
                }
                for fut in as_completed(futs):
                    key = futs[fut]
                    try:
                        df = fut.result()
                        if not df.empty:
                            frames.append(df)
                    except EdgarBlockedError:
                        raise
                    except Exception as exc:
                        logger.warning("Failed to fetch dataset %s: %s", key, exc)

    if not frames:
        raise RuntimeError("No SEC 13F SUMMARYPAGE data loaded.")

    raw_all = pd.concat(frames, ignore_index=True)
    clean_all = normalize_summary_table(raw_all)
    funnel, entity_summary, candidates = compute_entity_funnel(clean_all)

    DEFAULT_FILER_SUMMARY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    entity_summary.to_parquet(DEFAULT_FILER_SUMMARY_PARQUET, index=False)
    DEFAULT_CANDIDATES_CSV.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(DEFAULT_CANDIDATES_CSV, index=False)
    return funnel, entity_summary, candidates


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen all SEC 13F-HR entities via SUMMARYPAGE.")
    parser.add_argument(
        "--parse-only", action="store_true", help="Use local cached summary parquets."
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args(argv)
    funnel, entity_summary, candidates = run_filer_screener(
        parse_only=args.parse_only,
        max_workers=args.workers,
    )
    print("\n=== SEC-WIDE 13F ENTITY FILTER FUNNEL (2015-2026) ===")
    for k, v in funnel.items():
        print(f"  {k:40s}: {v:7,d}")
    print(f"\nCandidate pool written -> {DEFAULT_CANDIDATES_CSV} ({len(candidates):,} entities)")
    print("\nTop 40 screened candidates by median 13F AUM:")
    show = candidates.head(40).assign(
        med_aum_B=lambda d: (d["median_aum_usd"] / 1e9).round(2),
        med_pos=lambda d: d["median_positions"].round(1),
        first_q=lambda d: d["first_period"].dt.strftime("%Y-%m"),
        last_q=lambda d: d["last_period"].dt.strftime("%Y-%m"),
    )[
        [
            "cik",
            "manager_name",
            "n_summary_quarters",
            "med_pos",
            "med_aum_B",
            "first_q",
            "last_q",
            "is_active_2025_plus",
        ]
    ]
    print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
