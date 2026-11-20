"""Map SEC 13F CUSIPs to persistent `shareClassFIGI`, tickers, and security types via OpenFIGI.

Reads all unique long-equity (`SH`, non-option) CUSIPs from `edgar_13f_holdings_raw.parquet`,
queries the OpenFIGI `v3/mapping` API with `includeUnlistedEquities=True` (incrementally
cached to `data/raw/openfigi/cusip_cache.jsonl` so interrupted runs resume instantly),
applies deterministic US-exchange candidate selection plus manual overrides from
`config/cusip_overrides.yaml`, and writes:

  - `data/processed/cusip_figi_map.parquet`
  - `outputs/cusip_mapping_summary.csv`

Usage:
    python -m src.ingest.cusip_mapper
    python -m src.ingest.cusip_mapper --limit 50
    python -m src.ingest.cusip_mapper --resolve-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.filings import DEFAULT_HOLDINGS_RAW_PARQUET

logger = logging.getLogger(__name__)

OPENFIGI_MAPPING_URL = "https://api.openfigi.com/v3/mapping"
RAW_OPENFIGI_DIR = RAW_DATA_DIR / "openfigi"
DEFAULT_CACHE_JSONL = RAW_OPENFIGI_DIR / "cusip_cache.jsonl"
DEFAULT_CUSIP_MAP_PARQUET = PROCESSED_DATA_DIR / "cusip_figi_map.parquet"
DEFAULT_SUMMARY_CSV = OUTPUTS_DIR / "cusip_mapping_summary.csv"

# Bloomberg exchange codes for US composite and US primary/OTC equity venues (ordered by preference)
US_EXCHANGE_PRIORITY: dict[str, int] = {
    "US": 100,  # US Composite (canonical US ticker)
    "UN": 90,   # NYSE
    "UW": 90,   # Nasdaq Global Select
    "UQ": 85,   # Nasdaq Global Market
    "UR": 85,   # Nasdaq Capital Market
    "UA": 80,   # NYSE American
    "UP": 80,   # NYSE Arca
    "UF": 75,   # Cboe BZX
    "UV": 70,   # IEX
    "UD": 60,   # FINRA / OTC
    "UO": 55,   # OTC Bulletin Board / Pink
    "PQ": 50,   # OTC Pink
}

_NON_COMMON_KEYWORDS = (
    "WARRANT",
    "RIGHT",
    "UNIT",
    "PREF",
    "ETF",
    "ETP",
    "ETN",
    "REIT",
    "ADR",
    "GDR",
    "DEPOSITARY",
    "CLOSED-END",
    "MLP",
    "ROYALTY",
    "TRUST",
    "TRACKING",
)

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_CUSIP_CLEAN_RE = re.compile(r"^[0-9A-Z]{9}$")


def normalize_yahoo_ticker(ticker: str) -> str:
    """Convert OpenFIGI/Bloomberg equity ticker syntax into Yahoo Finance format.

    Examples:
        `BRK/B` -> `BRK-B`, `BF/B` -> `BF-B`, `BRK.B` -> `BRK-B`, `GOOG` -> `GOOG`.
    """
    cleaned = (ticker or "").strip().upper()
    if not cleaned:
        return ""
    # Normalize class separators (e.g., `BRK/B` or `BRK B` -> `BRK-B`)
    cleaned = re.sub(r"[/.\s]+", "-", cleaned)
    return cleaned.strip("-")


def is_common_stock_candidate(rec: dict[str, Any]) -> bool:
    """Return True if an OpenFIGI record is a US Common Stock (excludes REITs/ETFs/ADRs)."""
    exch = str(rec.get("exchCode") or "").upper()
    if exch not in US_EXCHANGE_PRIORITY:
        return False

    sec_type2 = str(rec.get("securityType2") or "").strip().lower()
    sec_type = str(rec.get("securityType") or "").strip().upper()
    market_sec = str(rec.get("marketSector") or "").strip().lower()

    if market_sec and market_sec != "equity":
        return False
    if sec_type2 != "common stock":
        return False
    if any(kw in sec_type for kw in _NON_COMMON_KEYWORDS):
        return False
    return True


def _score_candidate(rec: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Rank OpenFIGI candidates for a single CUSIP (higher tuple = better)."""
    exch = str(rec.get("exchCode") or "").upper()
    exch_score = US_EXCHANGE_PRIORITY.get(exch, 0)
    is_equity = 1 if str(rec.get("marketSector") or "").lower() == "equity" else 0
    is_common = 1 if is_common_stock_candidate(rec) else 0
    has_sc_figi = 1 if rec.get("shareClassFIGI") else 0
    ticker = str(rec.get("ticker") or "").strip()
    clean_ticker = 1 if (ticker and not ticker.endswith(" WI") and len(ticker) <= 8) else 0
    return (is_common, exch_score, is_equity, has_sc_figi, clean_ticker)


def select_best_figi_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Deterministically select the canonical US equity record from OpenFIGI candidates."""
    if not candidates:
        return None
    return max(candidates, key=_score_candidate)


def openfigi_id_type(cusip: str) -> str:
    """Return `ID_CINS` for letter-prefixed international CUSIPs (CINS), else `ID_CUSIP`."""
    return "ID_CINS" if cusip and cusip[0].isalpha() else "ID_CUSIP"


class OpenFigiClient:
    """Rate-limited OpenFIGI v3/mapping client supporting keyed and unkeyed tiers."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_retries: int = 5,
        timeout: tuple[float, float] = (10.0, 30.0),
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.max_retries = max_retries
        self.timeout = timeout

        # Official OpenFIGI v3/mapping limits:
        #   No key:   10 jobs/req, 25 req/min  (~2.6s spacing for safe margin)
        #   With key: 100 jobs/req, 25 req/6s  (~0.28s spacing for safe margin)
        if self.api_key:
            self.batch_size = 100
            self.min_interval_sec = 0.30  # ~3.3 req/s, safely under 25 req / 6s
            self.default_workers = 4
            logger.info(
                "OpenFIGI client using authenticated key tier (100 CUSIPs/batch, 4 workers)."
            )
        else:
            self.batch_size = 10
            self.min_interval_sec = 2.6
            self.default_workers = 1
            logger.info(
                "OPENFIGI_API_KEY not set; falling back to keyless OpenFIGI tier "
                "(10 CUSIPs/batch, ~23 req/min — slower but checkpointed per batch)."
            )

        self._lock = threading.Lock()
        self._next_allowed_ts = 0.0

        self._session = session or requests.Session()
        if session is None:
            adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8, max_retries=0)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        self._session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            self._session.headers["X-OPENFIGI-APIKEY"] = self.api_key

    def __enter__(self) -> OpenFigiClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def _acquire_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            scheduled = max(now, self._next_allowed_ts)
            self._next_allowed_ts = scheduled + self.min_interval_sec
        wait_sec = scheduled - now
        if wait_sec > 0:
            time.sleep(wait_sec)

    def map_cusip_batch(self, cusips: list[str]) -> list[dict[str, Any]]:
        """Send a single OpenFIGI v3/mapping POST batch for up to `self.batch_size` CUSIPs."""
        if not cusips:
            return []
        if len(cusips) > self.batch_size:
            raise ValueError(f"Batch size {len(cusips)} exceeds limit {self.batch_size}")

        payload = [
            {
                "idType": openfigi_id_type(c),
                "idValue": c,
                "marketSecDes": "Equity",
                "includeUnlistedEquities": True,
            }
            for c in cusips
        ]

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._acquire_slot()
            try:
                resp = self._session.post(
                    OPENFIGI_MAPPING_URL,
                    json=payload,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise
                sleep_sec = 2.0 * (2**attempt)
                logger.warning("OpenFIGI request error (%s); retrying in %.1fs", exc, sleep_sec)
                time.sleep(sleep_sec)
                continue

            if resp.status_code == 200:
                results = resp.json()
                if not isinstance(results, list) or len(results) != len(cusips):
                    raise RuntimeError(
                        f"Unexpected OpenFIGI response length {len(results)} vs {len(cusips)}"
                    )
                return results

            if resp.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                reset_hdr = resp.headers.get("ratelimit-reset") or resp.headers.get("Retry-After")
                if reset_hdr and str(reset_hdr).isdigit():
                    sleep_sec = max(float(reset_hdr) + 1.0, 2.0)
                else:
                    sleep_sec = min(65.0, 5.0 * (2**attempt))
                logger.warning(
                    "OpenFIGI HTTP %d; backing off %.1fs (attempt %d/%d)",
                    resp.status_code,
                    sleep_sec,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(sleep_sec)
                continue

            resp.raise_for_status()

        assert last_exc is not None
        raise last_exc


def load_cusip_cache(cache_path: Path = DEFAULT_CACHE_JSONL) -> dict[str, dict[str, Any]]:
    """Load incrementally cached OpenFIGI per-CUSIP responses from JSONL."""
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cusip = str(obj.get("cusip") or "").upper()
            if cusip:
                cache[cusip] = obj
    return cache


def append_cusip_cache(
    entries: list[dict[str, Any]],
    cache_path: Path = DEFAULT_CACHE_JSONL,
) -> None:
    """Append newly fetched OpenFIGI results atomically-per-line to JSONL cache."""
    if not entries:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, separators=(",", ":")) + "\n")


def extract_long_equity_cusips(holdings_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique long `SH`-only CUSIPs ordered by total tracked dollar-quarters."""
    sh = holdings_df[
        (holdings_df["ssh_prnamt_type"] == "SH") & (holdings_df["put_call"] == "")
    ].copy()
    sh["cusip"] = sh["cusip"].astype(str).str.strip().str.upper()
    sh = sh[sh["cusip"].str.match(_CUSIP_CLEAN_RE)]

    summary = (
        sh.groupby("cusip", as_index=False)
        .agg(
            sample_issuer=("name_of_issuer", "last"),
            sample_title=("title_of_class", "last"),
            total_value_usd=("value_usd", "sum"),
            row_count=("accession_number", "count"),
            n_filers=("filer_id", "nunique"),
        )
        .sort_values("total_value_usd", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def resolve_cusip_table(
    cusip_stats_df: pd.DataFrame,
    cache: dict[str, dict[str, Any]],
    overrides: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Combine cached OpenFIGI responses + manual overrides into the clean CUSIP map DataFrame."""
    overrides_norm = {k.strip().upper(): v for k, v in (overrides or {}).items()}
    rows: list[dict[str, Any]] = []

    for rec in cusip_stats_df.to_dict(orient="records"):
        cusip = str(rec["cusip"]).upper()
        if cusip in overrides_norm:
            ov = overrides_norm[cusip]
            raw_tkr = str(ov.get("ticker") or "").strip().upper()
            sc_figi = str(ov.get("share_class_figi") or "").strip()
            sec_type2 = str(ov.get("security_type2") or "Common Stock").strip()
            is_common = bool(ov.get("is_us_common_stock", sec_type2.lower() == "common stock"))
            rows.append(
                {
                    "cusip": cusip,
                    "security_id": sc_figi or f"CUSIP:{cusip}",
                    "share_class_figi": sc_figi,
                    "composite_figi": str(ov.get("composite_figi") or ""),
                    "figi": str(ov.get("figi") or ""),
                    "figi_ticker": raw_tkr,
                    "yahoo_ticker": normalize_yahoo_ticker(raw_tkr),
                    "figi_name": str(ov.get("name") or rec["sample_issuer"]),
                    "exch_code": str(ov.get("exch_code") or "US").upper(),
                    "market_sector": "Equity",
                    "security_type": str(ov.get("security_type") or sec_type2),
                    "security_type2": sec_type2,
                    "is_us_common_stock": is_common,
                    "mapping_source": "manual_override",
                    "mapping_error": "",
                    "sample_issuer": rec["sample_issuer"],
                    "sample_title": rec["sample_title"],
                    "total_value_usd": rec["total_value_usd"],
                    "n_filers": rec["n_filers"],
                }
            )
            continue

        cached = cache.get(cusip)
        if not cached:
            rows.append(
                {
                    "cusip": cusip,
                    "security_id": f"CUSIP:{cusip}",
                    "share_class_figi": "",
                    "composite_figi": "",
                    "figi": "",
                    "figi_ticker": "",
                    "yahoo_ticker": "",
                    "figi_name": "",
                    "exch_code": "",
                    "market_sector": "",
                    "security_type": "",
                    "security_type2": "",
                    "is_us_common_stock": False,
                    "mapping_source": "not_queried",
                    "mapping_error": "",
                    "sample_issuer": rec["sample_issuer"],
                    "sample_title": rec["sample_title"],
                    "total_value_usd": rec["total_value_usd"],
                    "n_filers": rec["n_filers"],
                }
            )
            continue

        candidates = cached.get("data") or []
        best = select_best_figi_candidate(candidates)
        if best is None:
            rows.append(
                {
                    "cusip": cusip,
                    "security_id": f"CUSIP:{cusip}",
                    "share_class_figi": "",
                    "composite_figi": "",
                    "figi": "",
                    "figi_ticker": "",
                    "yahoo_ticker": "",
                    "figi_name": "",
                    "exch_code": "",
                    "market_sector": "",
                    "security_type": "",
                    "security_type2": "",
                    "is_us_common_stock": False,
                    "mapping_source": "unmapped",
                    "mapping_error": str(
                        cached.get("error") or cached.get("warning") or "no_match"
                    ),
                    "sample_issuer": rec["sample_issuer"],
                    "sample_title": rec["sample_title"],
                    "total_value_usd": rec["total_value_usd"],
                    "n_filers": rec["n_filers"],
                }
            )
            continue

        sc_figi = str(best.get("shareClassFIGI") or "").strip()
        comp_figi = str(best.get("compositeFIGI") or "").strip()
        figi = str(best.get("figi") or "").strip()
        raw_tkr = str(best.get("ticker") or "").strip().upper()
        rows.append(
            {
                "cusip": cusip,
                "security_id": sc_figi or comp_figi or figi or f"CUSIP:{cusip}",
                "share_class_figi": sc_figi,
                "composite_figi": comp_figi,
                "figi": figi,
                "figi_ticker": raw_tkr,
                "yahoo_ticker": normalize_yahoo_ticker(raw_tkr),
                "figi_name": str(best.get("name") or ""),
                "exch_code": str(best.get("exchCode") or "").upper(),
                "market_sector": str(best.get("marketSector") or ""),
                "security_type": str(best.get("securityType") or ""),
                "security_type2": str(best.get("securityType2") or ""),
                "is_us_common_stock": is_common_stock_candidate(best),
                "mapping_source": "openfigi",
                "mapping_error": "",
                "sample_issuer": rec["sample_issuer"],
                "sample_title": rec["sample_title"],
                "total_value_usd": rec["total_value_usd"],
                "n_filers": rec["n_filers"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Second pass: stitch retired CUSIPs whose US listing dropped off OpenFIGI (leaving only a
    # foreign-exchange listing) when they share an identical `share_class_figi` with an already
    # verified US Common Stock (e.g., Priceline -> BKNG, United Technologies -> RTX, old GE -> GE).
    us_by_sc = (
        df[df["is_us_common_stock"] & (df["share_class_figi"] != "")]
        .sort_values("total_value_usd", ascending=False)
        .drop_duplicates("share_class_figi")
        .set_index("share_class_figi")
    )
    stitch_mask = (
        (~df["is_us_common_stock"])
        & (df["security_type2"].str.lower() == "common stock")
        & (df["share_class_figi"].isin(us_by_sc.index))
    )
    for idx in df.index[stitch_mask]:
        sc = df.at[idx, "share_class_figi"]
        ref = us_by_sc.loc[sc]
        df.at[idx, "figi_ticker"] = ref["figi_ticker"]
        df.at[idx, "yahoo_ticker"] = ref["yahoo_ticker"]
        df.at[idx, "exch_code"] = ref["exch_code"]
        df.at[idx, "is_us_common_stock"] = True
        df.at[idx, "mapping_source"] = "openfigi_figi_stitch"

    return df


def build_cusip_map(
    holdings_df: pd.DataFrame,
    client: OpenFigiClient | None = None,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    cache_path: Path = DEFAULT_CACHE_JSONL,
    output_path: Path = DEFAULT_CUSIP_MAP_PARQUET,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV,
    resolve_only: bool = False,
    limit: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Fetch missing CUSIPs from OpenFIGI into JSONL cache and write the resolved mapping table."""
    cusip_stats = extract_long_equity_cusips(holdings_df)
    if limit is not None:
        cusip_stats = cusip_stats.head(limit).reset_index(drop=True)

    overrides_norm = {k.strip().upper() for k in (overrides or {})}
    cache = load_cusip_cache(cache_path)

    missing = [
        c
        for c in cusip_stats["cusip"].tolist()
        if c not in overrides_norm
        and (
            c not in cache
            or (
                openfigi_id_type(c) == "ID_CINS"
                and cache[c].get("query_id_type") != "ID_CINS"
                and not cache[c].get("data")
            )
        )
    ]

    if missing and not resolve_only:
        if client is None:
            raise ValueError("OpenFigiClient is required unless resolve_only=True.")
        batches = [
            missing[i : i + client.batch_size]
            for i in range(0, len(missing), client.batch_size)
        ]
        workers = max(1, min(getattr(client, "default_workers", 1), len(batches)))
        cache_lock = threading.Lock()

        def _fetch_batch(batch: list[str]) -> int:
            results = client.map_cusip_batch(batch)
            new_entries: list[dict[str, Any]] = []
            for cusip, res in zip(batch, results, strict=True):
                new_entries.append(
                    {
                        "cusip": cusip,
                        "query_id_type": openfigi_id_type(cusip),
                        "data": res.get("data", []),
                        "error": res.get("error", ""),
                        "warning": res.get("warning", ""),
                    }
                )
            with cache_lock:
                for entry in new_entries:
                    cache[entry["cusip"]] = entry
                append_cusip_cache(new_entries, cache_path)
            return len(batch)

        pbar = (
            tqdm(total=len(missing), desc="OpenFIGI CUSIP map", unit="cusip")
            if show_progress
            else None
        )
        try:
            if workers == 1:
                for batch in batches:
                    n_done = _fetch_batch(batch)
                    if pbar is not None:
                        pbar.update(n_done)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_fetch_batch, b): b for b in batches}
                    for fut in as_completed(futures):
                        n_done = fut.result()
                        if pbar is not None:
                            pbar.update(n_done)
        finally:
            if pbar is not None:
                pbar.close()

    resolved = resolve_cusip_table(cusip_stats, cache, overrides=overrides)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved.to_parquet(output_path, index=False)

    # Write human-readable summary by securityType2 & mapping_source
    tot_usd = float(resolved["total_value_usd"].sum()) or 1.0
    summary = (
        resolved.groupby(["mapping_source", "security_type2", "is_us_common_stock"], as_index=False)
        .agg(
            cusip_count=("cusip", "count"),
            unique_security_ids=("security_id", "nunique"),
            total_value_usd=("total_value_usd", "sum"),
        )
        .sort_values("total_value_usd", ascending=False)
        .reset_index(drop=True)
    )
    summary["pct_tracked_dollars"] = summary["total_value_usd"] / tot_usd * 100.0
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv_path, index=False)

    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map 13F long-equity CUSIPs to shareClassFIGI & tickers via OpenFIGI."
    )
    parser.add_argument(
        "--holdings",
        type=Path,
        default=DEFAULT_HOLDINGS_RAW_PARQUET,
        help=f"Input raw holdings Parquet (default: {DEFAULT_HOLDINGS_RAW_PARQUET}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on top-dollar CUSIPs to map (useful for smoke testing).",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Do not hit OpenFIGI; resolve solely from local JSONL cache + cusip_overrides.yaml.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config()

    if not args.holdings.exists():
        raise FileNotFoundError(
            f"Raw holdings file not found at {args.holdings}. "
            "Run `python -m src.ingest.filings` first."
        )
    holdings_df = pd.read_parquet(args.holdings)

    if args.resolve_only:
        resolved = build_cusip_map(
            holdings_df,
            client=None,
            overrides=cfg.cusip_overrides,
            resolve_only=True,
            limit=args.limit,
        )
    else:
        with OpenFigiClient(api_key=cfg.openfigi_api_key) as client:
            resolved = build_cusip_map(
                holdings_df,
                client=client,
                overrides=cfg.cusip_overrides,
                limit=args.limit,
            )

    tot_usd = float(resolved["total_value_usd"].sum()) or 1.0
    mapped = resolved[
        resolved["mapping_source"].isin(["openfigi", "openfigi_figi_stitch", "manual_override"])
    ]
    common = resolved[resolved["is_us_common_stock"]]

    multi_cusip = (
        common[common["share_class_figi"] != ""]
        .groupby("share_class_figi")["cusip"]
        .nunique()
    )
    stitched_count = int((multi_cusip > 1).sum())

    mapped_pct = len(mapped) / max(1, len(resolved)) * 100.0
    mapped_usd_pct = mapped["total_value_usd"].sum() / tot_usd * 100.0
    common_usd_pct = common["total_value_usd"].sum() / tot_usd * 100.0

    print(
        f"CUSIP map written -> {DEFAULT_CUSIP_MAP_PARQUET}\n"
        f"  Total CUSIPs:       {len(resolved):,}\n"
        f"  Mapped (any type):  {len(mapped):,} ({mapped_pct:.1f}% CUSIPs, "
        f"{mapped_usd_pct:.2f}% tracked $)\n"
        f"  US Common Stocks:   {len(common):,} CUSIPs -> "
        f"{common['security_id'].nunique():,} unique security_ids "
        f"({common_usd_pct:.2f}% tracked $)\n"
        f"  CUSIP-change stitches (multi-CUSIP -> 1 shareClassFIGI): {stitched_count}\n"
        f"  Summary CSV -> {DEFAULT_SUMMARY_CSV}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
