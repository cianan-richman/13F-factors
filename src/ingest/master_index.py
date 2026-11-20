"""Download, cache, and parse SEC EDGAR quarterly `master.idx` / `master.gz` archives.

Builds a consolidated local Parquet index of 13F filings (default: 2015 Q1 -> current quarter)
so curated fund CIKs, affiliated/subsidiary filers, amendments (13F-HR/A), and accession numbers
can be audited and filtered offline before fetching individual filing XMLs.

Usage:
    python -m src.ingest.master_index
    python -m src.ingest.master_index --start-year 2015 --refresh-current
    python -m src.ingest.master_index --year 2024 --qtr QTR1
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import logging
import re
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.edgar_client import EdgarClient

logger = logging.getLogger(__name__)

FULL_INDEX_BASE_URL = "https://www.sec.gov/Archives/edgar/full-index"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives"

RAW_INDEX_DIR = RAW_DATA_DIR / "edgar_index"
DEFAULT_OUTPUT_PARQUET = PROCESSED_DATA_DIR / "edgar_13f_index.parquet"

_ACCESSION_RE = re.compile(r"edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})\.txt$", re.IGNORECASE)


def current_year_quarter(today: dt.date | None = None) -> tuple[int, int]:
    """Return `(year, quarter)` for today (or a supplied UTC date)."""
    ref = today or dt.datetime.now(dt.UTC).date()
    return ref.year, (ref.month - 1) // 3 + 1


def iter_quarters(
    start_year: int,
    start_qtr: int,
    end_year: int,
    end_qtr: int,
) -> Iterator[tuple[int, int]]:
    """Yield inclusive `(year, quarter)` pairs in chronological order."""
    if not (1 <= start_qtr <= 4 and 1 <= end_qtr <= 4):
        raise ValueError("Quarters must be integers between 1 and 4.")
    if (start_year, start_qtr) > (end_year, end_qtr):
        raise ValueError("Start (year, quarter) must not be after end (year, quarter).")

    year, qtr = start_year, start_qtr
    while (year, qtr) <= (end_year, end_qtr):
        yield year, qtr
        if qtr == 4:
            year += 1
            qtr = 1
        else:
            qtr += 1


def raw_quarter_path(year: int, qtr: int, raw_dir: Path = RAW_INDEX_DIR) -> Path:
    """Return local gzip cache path for a quarterly master index."""
    return raw_dir / f"{year}_QTR{qtr}_master.idx.gz"


def fetch_quarter_master_index(
    client: EdgarClient,
    year: int,
    qtr: int,
    raw_dir: Path = RAW_INDEX_DIR,
    *,
    force_refresh: bool = False,
) -> Path | None:
    """Fetch and cache a quarterly SEC EDGAR master index as gzipped text.

    Returns the cached file path, or `None` if the quarter has not been published yet (HTTP 404).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_quarter_path(year, qtr, raw_dir)

    if dest.exists() and not force_refresh:
        try:
            gzip.decompress(dest.read_bytes())
            return dest
        except (gzip.BadGzipFile, EOFError, OSError):
            logger.warning("Corrupt cached index at %s; deleting and re-fetching.", dest)
            dest.unlink(missing_ok=True)

    gz_url = f"{FULL_INDEX_BASE_URL}/{year}/QTR{qtr}/master.gz"
    resp = client.get(gz_url, allow_not_found=True)

    if resp.status_code == 200:
        # Validate gzip payload before writing atomically
        gzip.decompress(resp.content)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(dest)
        return dest

    # Fallback to uncompressed master.idx if master.gz is absent
    idx_url = f"{FULL_INDEX_BASE_URL}/{year}/QTR{qtr}/master.idx"
    resp_idx = client.get(idx_url, allow_not_found=True)
    if resp_idx.status_code == 200:
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(resp_idx.content))
        tmp.replace(dest)
        return dest

    if resp.status_code == 404 and resp_idx.status_code == 404:
        logger.info("SEC master index not yet available for %d QTR%d (HTTP 404).", year, qtr)
        return None

    resp_idx.raise_for_status()
    return None


def parse_master_index_text(
    text: str,
    *,
    form_types: Iterable[str] | None = ("13F-HR", "13F-HR/A"),
    index_year: int | None = None,
    index_qtr: int | None = None,
) -> pd.DataFrame:
    """Parse raw SEC `master.idx` text into a structured DataFrame.

    Parameters
    ----------
    text:
        Decoded contents of an SEC EDGAR `master.idx` file.
    form_types:
        Allowed SEC form types (case-insensitive). Pass `None` to keep all forms.
    index_year, index_qtr:
        Optional metadata columns recording which quarterly index archive the row came from.
    """
    allowed_forms = {f.strip().upper() for f in form_types} if form_types is not None else None
    lines = text.splitlines()

    # Locate the dashed divider line directly beneath `CIK|Company Name|Form Type|...`
    start_idx = 0
    for i, line in enumerate(lines[:50]):
        if line.startswith("-----"):
            start_idx = i + 1
            break
    else:
        for i, line in enumerate(lines[:50]):
            if line.strip().upper().startswith("CIK|COMPANY NAME|FORM TYPE|"):
                start_idx = i + 1
                break

    records: list[dict[str, object]] = []
    for raw_line in lines[start_idx:]:
        if not raw_line or "|" not in raw_line:
            continue
        parts = raw_line.split("|")
        if len(parts) != 5:
            continue

        cik_raw, company_name, form_type, date_filed, filename = (p.strip() for p in parts)
        form_upper = form_type.upper()
        if allowed_forms is not None and form_upper not in allowed_forms:
            continue
        if not cik_raw.isdigit():
            continue

        cik_int = int(cik_raw)
        cik_padded = f"{cik_int:010d}"

        accession_number = ""
        accession_nodash = ""
        filing_dir_url = ""
        match = _ACCESSION_RE.search(filename)
        if match:
            path_cik, accession_number = match.group(1), match.group(2)
            accession_nodash = accession_number.replace("-", "")
            filing_dir_url = f"{ARCHIVES_BASE_URL}/edgar/data/{path_cik}/{accession_nodash}/"

        records.append(
            {
                "cik": cik_padded,
                "cik_int": cik_int,
                "company_name": company_name,
                "form_type": form_upper,
                "date_filed": date_filed,
                "filename": filename,
                "accession_number": accession_number,
                "accession_nodash": accession_nodash,
                "filing_dir_url": filing_dir_url,
                "index_year": index_year,
                "index_qtr": index_qtr,
            }
        )

    columns = [
        "cik",
        "cik_int",
        "company_name",
        "form_type",
        "date_filed",
        "filename",
        "accession_number",
        "accession_nodash",
        "filing_dir_url",
        "index_year",
        "index_qtr",
    ]
    df = pd.DataFrame.from_records(records, columns=columns)
    if not df.empty:
        df["date_filed"] = pd.to_datetime(df["date_filed"], format="%Y-%m-%d", errors="coerce")
        df = df.dropna(subset=["date_filed"]).reset_index(drop=True)
    return df


def read_cached_quarter_index(
    path: Path,
    *,
    form_types: Iterable[str] | None = ("13F-HR", "13F-HR/A"),
    index_year: int | None = None,
    index_qtr: int | None = None,
) -> pd.DataFrame:
    """Read and parse a locally cached `.idx.gz` (or `.idx`) master index file."""
    if path.suffix == ".gz":
        raw_bytes = gzip.decompress(path.read_bytes())
    else:
        raw_bytes = path.read_bytes()
    # SEC historical index files occasionally contain ISO-8859-1 bytes in company names.
    text = raw_bytes.decode("latin-1")
    return parse_master_index_text(
        text,
        form_types=form_types,
        index_year=index_year,
        index_qtr=index_qtr,
    )


def build_master_index(
    client: EdgarClient,
    *,
    start_year: int = 2015,
    start_qtr: int = 1,
    end_year: int | None = None,
    end_qtr: int | None = None,
    form_types: Iterable[str] | None = ("13F-HR", "13F-HR/A"),
    raw_dir: Path = RAW_INDEX_DIR,
    output_path: Path = DEFAULT_OUTPUT_PARQUET,
    refresh_current: bool = True,
    force_refresh_all: bool = False,
    max_workers: int = 5,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Download/cache quarterly master indexes concurrently and build a consolidated Parquet."""
    cur_year, cur_qtr = current_year_quarter()
    eff_end_year = end_year if end_year is not None else cur_year
    eff_end_qtr = end_qtr if end_qtr is not None else cur_qtr

    quarters = list(iter_quarters(start_year, start_qtr, eff_end_year, eff_end_qtr))
    workers = max(1, min(max_workers, len(quarters)))

    def _process_quarter(yq: tuple[int, int]) -> pd.DataFrame | None:
        year, qtr = yq
        is_current = (year, qtr) == (cur_year, cur_qtr)
        force = force_refresh_all or (refresh_current and is_current)
        cached_path = fetch_quarter_master_index(
            client,
            year,
            qtr,
            raw_dir=raw_dir,
            force_refresh=force,
        )
        if cached_path is None:
            return None
        qdf = read_cached_quarter_index(
            cached_path,
            form_types=form_types,
            index_year=year,
            index_qtr=qtr,
        )
        return qdf if not qdf.empty else None

    frames: list[pd.DataFrame] = []
    if workers == 1:
        iterator: Iterable[tuple[int, int]] = (
            tqdm(quarters, desc="EDGAR master.idx", unit="qtr") if show_progress else quarters
        )
        for yq in iterator:
            qdf = _process_quarter(yq)
            if qdf is not None:
                frames.append(qdf)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_quarter, yq): yq for yq in quarters}
            pbar = (
                tqdm(total=len(quarters), desc="EDGAR master.idx", unit="qtr")
                if show_progress
                else None
            )
            try:
                for fut in as_completed(futures):
                    qdf = fut.result()
                    if qdf is not None:
                        frames.append(qdf)
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = (
            combined.drop_duplicates(subset=["cik", "accession_number", "form_type"])
            .sort_values(["date_filed", "cik", "accession_number"])
            .reset_index(drop=True)
        )
    else:
        combined = parse_master_index_text("", form_types=form_types)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    logger.info(
        "Wrote %d indexed filings (%d unique CIKs) to %s",
        len(combined),
        combined["cik"].nunique() if not combined.empty else 0,
        output_path,
    )
    return combined


def load_13f_index(path: Path = DEFAULT_OUTPUT_PARQUET) -> pd.DataFrame:
    """Load the consolidated local Parquet 13F master index."""
    if not path.exists():
        raise FileNotFoundError(
            f"Master index not found at {path}. Run `python -m src.ingest.master_index` first."
        )
    return pd.read_parquet(path)


def search_filers(index_df: pd.DataFrame, query: str) -> pd.DataFrame:
    """Search unique CIKs and company names matching a case-insensitive substring/regex."""
    mask = index_df["company_name"].str.contains(query, case=False, na=False)
    subset = index_df.loc[mask]
    if subset.empty:
        return pd.DataFrame(
            columns=["cik", "company_name", "filing_count", "first_filed", "last_filed"]
        )
    summary = (
        subset.groupby("cik", as_index=False)
        .agg(
            company_name=("company_name", "last"),
            filing_count=("accession_number", "nunique"),
            first_filed=("date_filed", "min"),
            last_filed=("date_filed", "max"),
        )
        .sort_values(["filing_count", "last_filed"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill and parse SEC EDGAR quarterly master.idx files into Parquet."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First calendar year to download (default: 2015, which captures Q4 2014 13Fs).",
    )
    parser.add_argument(
        "--start-qtr",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="First quarter of --start-year (default: 1).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Last calendar year to download (default: current year).",
    )
    parser.add_argument(
        "--end-qtr",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Last quarter of --end-year (default: current quarter).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Single-quarter backfill year (must be paired with --qtr).",
    )
    parser.add_argument(
        "--qtr",
        choices=["QTR1", "QTR2", "QTR3", "QTR4"],
        default=None,
        help="Single-quarter backfill quarter (must be paired with --year).",
    )
    parser.add_argument(
        "--include-nt",
        action="store_true",
        help="Also include 13F-NT and 13F-NT/A notice filings (useful for dedup auditing).",
    )
    parser.add_argument(
        "--refresh-current",
        action="store_true",
        default=True,
        help="Re-download the current in-progress quarter even if cached (default: True).",
    )
    parser.add_argument(
        "--no-refresh-current",
        action="store_false",
        dest="refresh_current",
        help="Use cached current quarter if present.",
    )
    parser.add_argument(
        "--force-refresh-all",
        action="store_true",
        help="Re-download every quarter even if already cached locally.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent download workers (default: config edgar.max_workers).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    if (args.year is None) != (args.qtr is None):
        raise SystemExit("Error: --year and --qtr must both be provided or both be omitted.")

    cfg = load_config()
    edgar_cfg = cfg.settings["edgar"]
    form_types = list(edgar_cfg.get("form_types", ["13F-HR", "13F-HR/A"]))
    if args.include_nt:
        form_types.extend(["13F-NT", "13F-NT/A"])

    if args.year is not None:
        qtr_num = int(args.qtr.replace("QTR", ""))
        start_year = end_year = args.year
        start_qtr = end_qtr = qtr_num
        output_path = PROCESSED_DATA_DIR / f"edgar_13f_index_{args.year}_{args.qtr}.parquet"
    else:
        start_year = args.start_year
        start_qtr = args.start_qtr
        end_year = args.end_year
        end_qtr = args.end_qtr
        output_path = DEFAULT_OUTPUT_PARQUET

    workers = args.workers if args.workers is not None else int(edgar_cfg.get("max_workers", 5))
    with EdgarClient(
        user_agent=cfg.sec_user_agent,
        max_requests_per_sec=float(edgar_cfg.get("max_requests_per_sec", 8.0)),
        max_retries=int(edgar_cfg.get("max_retries", 5)),
        backoff_base_sec=float(edgar_cfg.get("backoff_base_sec", 1.0)),
        pool_maxsize=max(workers + 2, 10),
    ) as client:
        df = build_master_index(
            client,
            start_year=start_year,
            start_qtr=start_qtr,
            end_year=end_year,
            end_qtr=end_qtr,
            form_types=form_types,
            output_path=output_path,
            refresh_current=args.refresh_current,
            force_refresh_all=args.force_refresh_all,
            max_workers=workers,
        )

    print(
        f"Indexed {len(df):,} filings across {df['cik'].nunique():,} unique CIKs -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
