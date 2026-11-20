"""SEC EDGAR `companyfacts` XBRL loader for strictly point-in-time unadjusted shares outstanding.

Maps our US Common Stock `yahoo_ticker` universe to SEC issuer CIKs via SEC's official
`company_tickers.json`, fetches historical XBRL share-count facts concurrently via our
rate-limited `EdgarClient` (`<= 8 req/s`), caches compact extracted share-count records
(`data/raw/sec_companyfacts/{cik}.json.gz`, ~3 KB/issuer), and writes:

  - `data/processed/shares_outstanding_pit.parquet`
  - `outputs/shares_outstanding_coverage.csv`

Point-in-time discipline:
  - Every fact retains BOTH `filed_date` (when the 10-Q/10-K/20-F became public on EDGAR) and
    `end_date` (as-of balance-sheet / cover-page date).
  - Downstream PIT joins must ONLY read facts with `filed_date <= t - 1` and apply strictly
    forward splits occurring in `(end_date, t]` to compute exact unadjusted shares on day `t`.

Usage:
    python -m src.ingest.shares_outstanding
    python -m src.ingest.shares_outstanding --limit 50
    python -m src.ingest.shares_outstanding --parse-only
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.cusip_mapper import DEFAULT_CUSIP_MAP_PARQUET
from src.ingest.edgar_client import EdgarBlockedError, EdgarClient
from src.ingest.market_data import filter_queryable_tickers

logger = logging.getLogger(__name__)

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL_TMPL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

RAW_CF_DIR = RAW_DATA_DIR / "sec_companyfacts"
DEFAULT_TICKERS_CACHE = RAW_CF_DIR / "company_tickers.json"
DEFAULT_SHARES_PARQUET = PROCESSED_DATA_DIR / "shares_outstanding_pit.parquet"
DEFAULT_SHARES_COVERAGE_CSV = OUTPUTS_DIR / "shares_outstanding_coverage.csv"

# Ordered by preference: cover-page PIT shares -> balance-sheet -> weighted avg -> IFRS
SHARE_FACT_TAGS: tuple[tuple[str, str, int], ...] = (
    ("dei", "EntityCommonStockSharesOutstanding", 1),
    ("us-gaap", "CommonStockSharesOutstanding", 2),
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic", 3),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", 4),
    ("us-gaap", "CommonStockSharesIssued", 5),
    ("ifrs-full", "NumberOfSharesIssued", 6),
    ("ifrs-full", "WeightedAverageShares", 7),
)
TAG_SCHEMA_VERSION = 2
ALLOWED_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}
MIN_END_DATE = "2013-01-01"


def fetch_sec_company_tickers(
    client: EdgarClient,
    *,
    cache_path: Path = DEFAULT_TICKERS_CACHE,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Return `{TICKER_UPPER: 10-digit-zero-padded-CIK}` from SEC `company_tickers.json`."""
    if not cache_path.exists() or force_refresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = client.get(SEC_COMPANY_TICKERS_URL)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(cache_path)

    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for entry in raw.values():
        tkr = str(entry.get("ticker") or "").strip().upper()
        cik_raw = entry.get("cik_str")
        if tkr and cik_raw is not None:
            cik_padded = f"{int(cik_raw):010d}"
            mapping[tkr] = cik_padded
            # Also index slash/dot normalized Yahoo syntax (e.g., BRK-B)
            mapping[tkr.replace(".", "-")] = cik_padded
    return mapping


def extract_share_facts_payload(raw_companyfacts: dict[str, Any], cik: str) -> dict[str, Any]:
    """Extract only the compact share-count XBRL arrays from a full `companyfacts` JSON."""
    facts_root = raw_companyfacts.get("facts") or {}
    extracted: dict[str, list[dict[str, Any]]] = {}

    for taxonomy, tag, _ in SHARE_FACT_TAGS:
        tag_obj = (facts_root.get(taxonomy) or {}).get(tag) or {}
        shares_list = (tag_obj.get("units") or {}).get("shares") or []
        if shares_list:
            extracted[f"{taxonomy}:{tag}"] = [
                {
                    "end": item.get("end"),
                    "filed": item.get("filed"),
                    "val": item.get("val"),
                    "form": item.get("form"),
                    "accn": item.get("accn"),
                    "fp": item.get("fp"),
                    "fy": item.get("fy"),
                }
                for item in shares_list
                if item.get("val") is not None and item.get("filed") and item.get("end")
            ]

    return {
        "cik": cik,
        "entityName": str(raw_companyfacts.get("entityName") or ""),
        "status": "ok",
        "tag_schema_version": TAG_SCHEMA_VERSION,
        "tags": extracted,
    }


def raw_companyfacts_path(cik: str, raw_dir: Path = RAW_CF_DIR) -> Path:
    return raw_dir / f"CIK{cik}.json.gz"


def fetch_issuer_share_facts(
    client: EdgarClient,
    cik: str,
    *,
    raw_dir: Path = RAW_CF_DIR,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Fetch and cache compact XBRL share-count facts for one issuer CIK."""
    dest = raw_companyfacts_path(cik, raw_dir)
    if dest.exists() and not force_refresh:
        try:
            cached = json.loads(gzip.decompress(dest.read_bytes()).decode("utf-8"))
            if cached.get("tag_schema_version") == TAG_SCHEMA_VERSION:
                return cached
        except (gzip.BadGzipFile, EOFError, OSError, json.JSONDecodeError):
            logger.warning("Corrupt cached companyfacts at %s; re-fetching.", dest)
            dest.unlink(missing_ok=True)

    url = SEC_COMPANYFACTS_URL_TMPL.format(cik=cik)
    resp = client.get(url, allow_not_found=True)
    if resp.status_code == 404:
        payload: dict[str, Any] = {
            "cik": cik,
            "entityName": "",
            "status": "not_found",
            "tag_schema_version": TAG_SCHEMA_VERSION,
            "tags": {},
        }
    else:
        payload = extract_share_facts_payload(resp.json(), cik)

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8")))
    tmp.replace(dest)
    return payload


def parse_share_facts_rows(
    payload: dict[str, Any],
    *,
    min_end_date: str = MIN_END_DATE,
) -> list[dict[str, Any]]:
    """Convert extracted XBRL share-count facts for one CIK into clean PIT rows.

    Handles dual-class issuers carefully:
    - When `dei:EntityCommonStockSharesOutstanding` reports multiple distinct share counts on
      the exact same `(filed, end, accn)` (e.g., Alphabet Class A + B + C), they represent
      separate share classes on the cover page -> sum distinct values to obtain total common
      shares outstanding, while storing `n_classes_reported` for auditability.
    - Deduplicates identical `(filed, end, accn, val)` repeats cleanly.
    """
    cik = str(payload.get("cik") or "")
    entity_name = str(payload.get("entityName") or "")
    tags_map = payload.get("tags") or {}

    rows: list[dict[str, Any]] = []
    for taxonomy, tag, priority in SHARE_FACT_TAGS:
        tag_key = f"{taxonomy}:{tag}"
        items = tags_map.get(tag_key) or []
        if not items:
            continue

        # Group by (filed, end, accn, form) to cleanly handle dual-class cover-page splits
        grouped: dict[tuple[str, str, str, str], set[float]] = {}
        for it in items:
            form = str(it.get("form") or "").strip().upper()
            if form not in ALLOWED_FORMS:
                continue
            end_s = str(it.get("end") or "").strip()
            filed_s = str(it.get("filed") or "").strip()
            if not end_s or not filed_s or end_s < min_end_date:
                continue
            try:
                val = float(it["val"])
            except (TypeError, ValueError):
                continue
            if val <= 0:
                continue
            accn = str(it.get("accn") or "").strip()
            grouped.setdefault((filed_s, end_s, accn, form), set()).add(val)

        for (filed_s, end_s, accn, form), distinct_vals in grouped.items():
            # For cover-page DEI tag, multiple distinct values in the same filing & date
            # reflect multi-class common share counts (e.g., Class A + Class B + Class C).
            # For balance-sheet / weighted-avg tags, use max to avoid double-counting sub-totals.
            if tag == "EntityCommonStockSharesOutstanding":
                shares_val = float(sum(distinct_vals))
            else:
                shares_val = float(max(distinct_vals))

            rows.append(
                {
                    "issuer_cik": cik,
                    "entity_name": entity_name,
                    "filed_date": filed_s,
                    "end_date": end_s,
                    "shares_outstanding": shares_val,
                    "form": form,
                    "accession_number": accn,
                    "tag_source": tag_key,
                    "tag_priority": priority,
                    "n_classes_reported": len(distinct_vals),
                }
            )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    # Within each (issuer_cik, filed_date, end_date), keep highest-priority tag (priority 1 > 2 > 3)
    df = (
        df.sort_values(["filed_date", "end_date", "tag_priority", "shares_outstanding"])
        .drop_duplicates(subset=["issuer_cik", "filed_date", "end_date"], keep="first")
        .reset_index(drop=True)
    )
    return df.to_dict(orient="records")


def ingest_shares_outstanding(
    cusip_map_df: pd.DataFrame,
    client: EdgarClient | None = None,
    *,
    raw_dir: Path = RAW_CF_DIR,
    tickers_cache_path: Path = DEFAULT_TICKERS_CACHE,
    output_parquet: Path = DEFAULT_SHARES_PARQUET,
    coverage_csv: Path = DEFAULT_SHARES_COVERAGE_CSV,
    parse_only: bool = False,
    force_refresh: bool = False,
    max_workers: int = 5,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and parse strictly PIT unadjusted shares outstanding for our US Common Stocks."""
    if not parse_only and client is None:
        raise ValueError("EdgarClient is required unless parse_only=True.")

    tickers = filter_queryable_tickers(cusip_map_df)
    if limit is not None:
        tickers = tickers[:limit]

    if parse_only:
        if not tickers_cache_path.exists():
            raise FileNotFoundError(
                f"SEC company_tickers cache missing at {tickers_cache_path}. Run live first."
            )
        raw_t = json.loads(tickers_cache_path.read_text(encoding="utf-8"))
        tkr_to_cik = {}
        for entry in raw_t.values():
            t = str(entry.get("ticker") or "").strip().upper()
            c = entry.get("cik_str")
            if t and c is not None:
                cik_p = f"{int(c):010d}"
                tkr_to_cik[t] = cik_p
                tkr_to_cik[t.replace(".", "-")] = cik_p
    else:
        assert client is not None
        tkr_to_cik = fetch_sec_company_tickers(
            client,
            cache_path=tickers_cache_path,
            force_refresh=force_refresh,
        )

    ticker_cik_pairs = [(t, tkr_to_cik.get(t, "")) for t in tickers]
    unique_ciks = sorted({c for _, c in ticker_cik_pairs if c})
    workers = 1 if parse_only else max(1, min(max_workers, max(1, len(unique_ciks))))

    def _process_cik(cik: str) -> tuple[str, str, list[dict[str, Any]]]:
        try:
            if parse_only:
                p = raw_companyfacts_path(cik, raw_dir)
                if not p.exists():
                    return cik, "not_cached", []
                payload = json.loads(gzip.decompress(p.read_bytes()).decode("utf-8"))
            else:
                assert client is not None
                payload = fetch_issuer_share_facts(
                    client,
                    cik,
                    raw_dir=raw_dir,
                    force_refresh=force_refresh,
                )
            if payload.get("status") != "ok":
                return cik, str(payload.get("status") or "not_found"), []
            parsed = parse_share_facts_rows(payload)
            return cik, ("ok" if parsed else "no_share_tags"), parsed
        except EdgarBlockedError:
            raise
        except Exception as exc:
            logger.warning("Failed companyfacts for CIK %s: %s", cik, exc)
            return cik, "fetch_error", []

    all_rows: list[dict[str, Any]] = []
    cik_status: dict[str, str] = {}

    if workers == 1:
        iterator = (
            tqdm(unique_ciks, desc="SEC companyfacts", unit="cik") if show_progress else unique_ciks
        )
        for cik in iterator:
            c_id, st, rows = _process_cik(cik)
            cik_status[c_id] = st
            all_rows.extend(rows)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_cik, c): c for c in unique_ciks}
            pbar = (
                tqdm(total=len(unique_ciks), desc="SEC companyfacts", unit="cik")
                if show_progress
                else None
            )
            try:
                for fut in as_completed(futures):
                    c_id, st, rows = fut.result()
                    cik_status[c_id] = st
                    all_rows.extend(rows)
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

    # Map issuer_cik back onto yahoo_ticker (preserving dual-class tickers sharing one CIK,
    # e.g. GOOG & GOOGL both map to Alphabet CIK 0001652044)
    cik_to_tickers: dict[str, list[str]] = {}
    for tkr, cik in ticker_cik_pairs:
        if cik:
            cik_to_tickers.setdefault(cik, []).append(tkr)

    expanded_rows: list[dict[str, Any]] = []
    for r in all_rows:
        for tkr in cik_to_tickers.get(r["issuer_cik"], []):
            expanded_rows.append({"yahoo_ticker": tkr, **r})

    shares_df = pd.DataFrame(expanded_rows)
    if not shares_df.empty:
        shares_df["filed_date"] = pd.to_datetime(shares_df["filed_date"])
        shares_df["end_date"] = pd.to_datetime(shares_df["end_date"])
        shares_df = (
            shares_df.sort_values(["yahoo_ticker", "filed_date", "end_date"])
            .reset_index(drop=True)
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    shares_df.to_parquet(output_parquet, index=False)

    # Coverage report weighted by tracked US Common Stock 13F dollars
    cm = cusip_map_df[cusip_map_df["is_us_common_stock"]].copy()
    cm_tkr = (
        cm.groupby("yahoo_ticker", as_index=False)["total_value_usd"]
        .sum()
    )
    tkr_usd = dict(zip(cm_tkr["yahoo_ticker"], cm_tkr["total_value_usd"], strict=False))
    cov_records = []
    for tkr, cik in ticker_cik_pairs:
        st = "no_sec_cik" if not cik else cik_status.get(cik, "unknown")
        cov_records.append(
            {
                "yahoo_ticker": tkr,
                "issuer_cik": cik,
                "status": st,
                "total_value_usd": tkr_usd.get(tkr, 0.0),
            }
        )
    cov_detail = pd.DataFrame(cov_records)
    tot_usd = float(cov_detail["total_value_usd"].sum()) or 1.0
    cov_summary = (
        cov_detail.groupby("status", as_index=False)
        .agg(
            ticker_count=("yahoo_ticker", "count"),
            total_value_usd=("total_value_usd", "sum"),
        )
        .sort_values("total_value_usd", ascending=False)
        .reset_index(drop=True)
    )
    cov_summary["pct_queryable_dollars"] = cov_summary["total_value_usd"] / tot_usd * 100.0
    coverage_csv.parent.mkdir(parents=True, exist_ok=True)
    cov_summary.to_csv(coverage_csv, index=False)

    return shares_df, cov_summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download & parse strictly PIT SEC companyfacts shares outstanding."
    )
    parser.add_argument(
        "--cusip-map",
        type=Path,
        default=DEFAULT_CUSIP_MAP_PARQUET,
        help=f"Input CUSIP-FIGI map Parquet (default: {DEFAULT_CUSIP_MAP_PARQUET}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on top-dollar tickers to process (for smoke testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent EDGAR workers (default: config edgar.max_workers).",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip network requests and rebuild Parquet from local cache only.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-fetch companyfacts even if cached locally.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config()
    edgar_cfg = cfg.settings["edgar"]

    if not args.cusip_map.exists():
        raise FileNotFoundError(
            f"CUSIP map not found at {args.cusip_map}. "
            "Run `python -m src.ingest.cusip_mapper` first."
        )
    cusip_map = pd.read_parquet(args.cusip_map)
    workers = args.workers if args.workers is not None else int(edgar_cfg.get("max_workers", 5))

    if args.parse_only:
        shares_df, cov_summary = ingest_shares_outstanding(
            cusip_map,
            client=None,
            parse_only=True,
            limit=args.limit,
        )
    else:
        with EdgarClient(
            user_agent=cfg.sec_user_agent,
            max_requests_per_sec=float(edgar_cfg.get("max_requests_per_sec", 8.0)),
            max_retries=int(edgar_cfg.get("max_retries", 5)),
            backoff_base_sec=float(edgar_cfg.get("backoff_base_sec", 1.0)),
            pool_maxsize=max(workers + 2, 10),
        ) as client:
            shares_df, cov_summary = ingest_shares_outstanding(
                cusip_map,
                client=client,
                force_refresh=args.force_refresh,
                max_workers=workers,
                limit=args.limit,
            )

    n_tkrs = shares_df["yahoo_ticker"].nunique() if not shares_df.empty else 0
    print(
        f"PIT shares outstanding written -> {DEFAULT_SHARES_PARQUET}\n"
        f"  Tickers covered: {n_tkrs:,} | PIT fact rows: {len(shares_df):,}\n"
        f"  Coverage CSV -> {DEFAULT_SHARES_COVERAGE_CSV}"
    )
    print("\n=== SEC CompanyFacts Shares Outstanding Coverage ===")
    print(cov_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
