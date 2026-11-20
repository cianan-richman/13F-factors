"""Batched Yahoo Finance (`yfinance`) loader with strict PIT un-splitting and rolling beta.

Downloads daily OHLCV, `Stock Splits`, and `Dividends` (`auto_adjust=False, actions=True`) for
`SPY` and all clean US Common Stock `yahoo_ticker`s in `data/processed/cusip_figi_map.parquet`
starting `2014-01-01` (providing a full 252-trading-day 2014 burn-in for PIT rolling beta on
`2015-01-02`).

Key point-in-time guarantees:
  1. Mathematically un-splits Yahoo's split-adjusted `Close`, `Volume`, and `Dividends` using the
     explicit `Stock Splits` column to reconstruct raw historical prices (`raw_close`), raw
     dividends (`raw_dividend`), and `cum_split_to_latest` (enabling exact forward-only split
     normalization across mid-gap 13F splits without lookahead).
  2. Reconstructs forward-only daily total returns `ret_total`.
  3. Computes split-invariant 20-day Average Daily Dollar Volume (`adv_20d_usd`) and strictly
     point-in-time 252-day trailing rolling beta (`beta_252d`) vs. `SPY` — never using Yahoo's
     static lookahead `info['beta']`.
  4. Quantifies the exact survivorship/delisted coverage gap against our 13F holdings.

Usage:
    python -m src.ingest.market_data
    python -m src.ingest.market_data --limit 50
    python -m src.ingest.market_data --process-only
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.cusip_mapper import DEFAULT_CUSIP_MAP_PARQUET

logger = logging.getLogger(__name__)

RAW_YF_DIR = RAW_DATA_DIR / "yfinance"
RAW_YF_BATCH_DIR = RAW_YF_DIR / "batches"
DEFAULT_MARKET_PRICES_PARQUET = PROCESSED_DATA_DIR / "market_prices.parquet"
DEFAULT_SPY_PARQUET = PROCESSED_DATA_DIR / "spy_prices.parquet"
DEFAULT_SURVIVORSHIP_CSV = OUTPUTS_DIR / "market_data_survivorship_report.csv"

# 1-year burn-in before 2015-01-01 so 252-day rolling beta is populated on day 1 of IS window
DEFAULT_MARKET_START = "2014-01-01"
_BLOOMBERG_DEAD_TICKER_RE = re.compile(r"^\d+[A-Z]$")


def filter_queryable_tickers(cusip_map_df: pd.DataFrame) -> list[str]:
    """Return sorted unique Yahoo-queryable US Common Stock tickers from `cusip_figi_map`."""
    cm = cusip_map_df[cusip_map_df["is_us_common_stock"]].copy()
    tickers = cm["yahoo_ticker"].fillna("").astype(str).str.strip().str.upper()
    valid = tickers[(tickers != "") & (~tickers.str.match(_BLOOMBERG_DEAD_TICKER_RE))]
    # Order by total tracked dollar weight descending for clean smoke-testing with --limit
    cm_valid = cm.loc[valid.index].assign(yahoo_ticker=valid)
    ordered = (
        cm_valid.groupby("yahoo_ticker", as_index=False)["total_value_usd"]
        .sum()
        .sort_values("total_value_usd", ascending=False)["yahoo_ticker"]
        .tolist()
    )
    return ordered


def compute_pit_price_series(
    raw_ticker_df: pd.DataFrame,
    *,
    ticker: str,
) -> pd.DataFrame:
    """Reconstruct raw unadjusted prices, forward-only total returns, and ADV for one ticker.

    Parameters
    ----------
    raw_ticker_df:
        DataFrame indexed by DatetimeIndex (trading dates) with columns `Close`, `Volume`,
        and optionally `Dividends` and `Stock Splits` (from `yfinance` with
        `auto_adjust=False, actions=True`).
    """
    if raw_ticker_df.empty or "Close" not in raw_ticker_df.columns:
        return pd.DataFrame()

    df = raw_ticker_df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    close_sadj = pd.to_numeric(df["Close"], errors="coerce")
    valid_mask = close_sadj.notna() & (close_sadj > 0)
    df = df.loc[valid_mask].copy()
    if df.empty:
        return pd.DataFrame()

    close_sadj = pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=np.float64)
    vol_sadj = (
        pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "Volume" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    div_sadj = (
        pd.to_numeric(df["Dividends"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "Dividends" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )
    splits_raw = (
        pd.to_numeric(df["Stock Splits"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if "Stock Splits" in df.columns
        else np.zeros(len(df), dtype=np.float64)
    )

    # Yahoo encodes non-split days as 0.0; map to multiplicative split factor s_t (1.0 normal)
    s_t = np.where(splits_raw > 0, splits_raw, 1.0)

    # Future cumulative split factor strictly AFTER day t: F_fut[t] = prod_{tau > t} s_tau
    # Reverse cumprod shifted by 1 gives exact future product without including day t itself.
    rev_cumprod = np.cumprod(s_t[::-1])[::-1]
    cum_split_to_latest = np.ones_like(s_t)
    if len(s_t) > 1:
        cum_split_to_latest[:-1] = rev_cumprod[1:]

    # Un-split Yahoo's split-adjusted Close, Volume, and Dividends back to historical raw basis
    raw_close = close_sadj * cum_split_to_latest
    raw_volume = np.where(cum_split_to_latest > 0, vol_sadj / cum_split_to_latest, vol_sadj)
    raw_dividend = div_sadj * cum_split_to_latest

    # Dollar volume is invariant to stock splits: raw_close * raw_volume == close_sadj * vol_sadj
    dollar_volume = raw_close * raw_volume

    # Strictly forward-only daily total return: (P_raw[t] * s[t] + D_raw[t]) / P_raw[t-1] - 1
    ret_total = np.full(len(df), np.nan, dtype=np.float64)
    if len(df) > 1:
        ret_total[1:] = (raw_close[1:] * s_t[1:] + raw_dividend[1:]) / raw_close[:-1] - 1.0

    adv_20d = pd.Series(dollar_volume, index=df.index).rolling(window=20, min_periods=10).mean()

    return pd.DataFrame(
        {
            "date": df.index,
            "yahoo_ticker": ticker,
            "raw_close": raw_close,
            "split_adj_close": close_sadj,
            "raw_volume": raw_volume,
            "dollar_volume": dollar_volume,
            "split_factor": s_t,
            "cum_split_to_latest": cum_split_to_latest,
            "raw_dividend": raw_dividend,
            "ret_total": ret_total,
            "adv_20d_usd": adv_20d.to_numpy(dtype=np.float64),
        }
    )


def compute_rolling_beta_panel(
    prices_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    *,
    window: int = 252,
    min_periods: int = 126,
) -> pd.DataFrame:
    """Attach strictly PIT trailing rolling OLS equity beta (`beta_252d`) vs. `SPY`."""
    if prices_df.empty or spy_df.empty:
        out = prices_df.copy()
        out["beta_252d"] = np.nan
        return out

    spy_ret = (
        spy_df.set_index("date")["ret_total"]
        .sort_index()
        .rename("spy_ret")
    )

    ret_wide = prices_df.pivot(index="date", columns="yahoo_ticker", values="ret_total")
    aligned_spy = spy_ret.reindex(ret_wide.index)

    # Vectorized rolling OLS beta: Cov(r_i, r_spy) / Var(r_spy) using strictly past [t-W+1, t]
    cov_wide = ret_wide.rolling(window=window, min_periods=min_periods).cov(aligned_spy)
    var_spy = aligned_spy.rolling(window=window, min_periods=min_periods).var()
    beta_wide = cov_wide.div(var_spy, axis=0)

    # Stack back onto long panel cleanly
    beta_long = (
        beta_wide.stack(future_stack=True)
        .rename("beta_252d")
        .reset_index()
    )
    merged = prices_df.merge(beta_long, on=["date", "yahoo_ticker"], how="left")
    return merged


def _extract_ticker_frame_from_batch(
    batch_raw: pd.DataFrame,
    ticker: str,
    batch_len: int,
) -> pd.DataFrame:
    if batch_raw.empty:
        return pd.DataFrame()
    if isinstance(batch_raw.columns, pd.MultiIndex):
        if ticker in batch_raw.columns.get_level_values(0):
            return batch_raw[ticker].copy()
        if ticker in batch_raw.columns.get_level_values(1):
            return batch_raw.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()
    if batch_len == 1:
        return batch_raw.copy()
    return pd.DataFrame()


def download_yfinance_batch(
    tickers: list[str],
    *,
    start_date: str = DEFAULT_MARKET_START,
    end_date: str | None = None,
    max_retries: int = 4,
) -> pd.DataFrame:
    """Download one batch of tickers via `yfinance` with YFRateLimitError detection & retry."""
    if not tickers:
        return pd.DataFrame()

    remaining = list(tickers)
    collected: list[pd.DataFrame] = []

    for attempt in range(max_retries):
        if not remaining:
            break
        try:
            # Clear yfinance's shared error dict before each batch so we detect rate limits cleanly
            if hasattr(yf, "shared") and hasattr(yf.shared, "_ERRORS"):
                yf.shared._ERRORS.clear()
            raw = yf.download(
                tickers=remaining,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=True,
                group_by="ticker",
                threads=4,
                progress=False,
            )
        except Exception as exc:
            if attempt >= max_retries - 1:
                raise
            sleep_sec = 5.0 * (2**attempt)
            logger.warning(
                "yfinance batch exception (%s); retrying in %.1fs (attempt %d/%d)",
                exc,
                sleep_sec,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_sec)
            continue

        batch_errors = dict(getattr(getattr(yf, "shared", None), "_ERRORS", {}))
        rate_limited: list[str] = []
        batch_collected = 0

        for tkr in remaining:
            err_msg = repr(batch_errors.get(tkr, ""))
            if any(
                tok in err_msg for tok in ("RateLimit", "Rate limited", "Too Many Requests")
            ):
                rate_limited.append(tkr)
                continue
            t_df = _extract_ticker_frame_from_batch(raw, tkr, len(remaining))
            if t_df.empty:
                continue
            cleaned = compute_pit_price_series(t_df, ticker=tkr)
            if not cleaned.empty:
                collected.append(cleaned)
                batch_collected += 1

        # Guard against yfinance version changes where 429 errors aren't exposed in _ERRORS:
        # if a large batch returns 0 tickers at all, treat it as a rate-limit cooldown event.
        if batch_collected == 0 and not rate_limited and len(remaining) >= 10:
            rate_limited = list(remaining)

        remaining = rate_limited
        if remaining:
            if attempt >= max_retries - 1:
                logger.error(
                    "yfinance still rate-limited on %d tickers after %d attempts: %s",
                    len(remaining),
                    max_retries,
                    remaining[:10],
                )
                break
            sleep_sec = min(90.0, 20.0 * (attempt + 1))
            logger.warning(
                "yfinance rate-limited on %d tickers; cooling down %.0fs (attempt %d/%d)",
                len(remaining),
                sleep_sec,
                attempt + 1,
                max_retries,
            )
            time.sleep(sleep_sec)

    if not collected:
        return pd.DataFrame()
    return pd.concat(collected, ignore_index=True)


def build_survivorship_report(
    cusip_map_df: pd.DataFrame,
    priced_tickers: set[str],
) -> pd.DataFrame:
    """Quantify Yahoo Finance survivorship/delisted coverage across US Common Stock 13F dollars."""
    cm = cusip_map_df[cusip_map_df["is_us_common_stock"]].copy()
    cm["yahoo_ticker_clean"] = cm["yahoo_ticker"].fillna("").astype(str).str.strip().str.upper()

    def _status(row: pd.Series) -> str:
        tkr = row["yahoo_ticker_clean"]
        if not tkr or _BLOOMBERG_DEAD_TICKER_RE.match(tkr):
            return "dead_bloomberg_id"
        if tkr in priced_tickers:
            return "priced_in_yfinance"
        return "missing_or_delisted_in_yfinance"

    cm["price_coverage_status"] = cm.apply(_status, axis=1)
    tot_usd = float(cm["total_value_usd"].sum()) or 1.0

    summary = (
        cm.groupby("price_coverage_status", as_index=False)
        .agg(
            cusip_count=("cusip", "count"),
            unique_security_ids=("security_id", "nunique"),
            total_value_usd=("total_value_usd", "sum"),
        )
        .sort_values("total_value_usd", ascending=False)
        .reset_index(drop=True)
    )
    summary["pct_us_common_dollars"] = summary["total_value_usd"] / tot_usd * 100.0
    return summary


def ingest_market_data(
    cusip_map_df: pd.DataFrame,
    *,
    benchmark_ticker: str = "SPY",
    start_date: str = DEFAULT_MARKET_START,
    end_date: str | None = None,
    beta_window: int = 252,
    batch_size: int = 50,
    batch_dir: Path = RAW_YF_BATCH_DIR,
    prices_output_path: Path = DEFAULT_MARKET_PRICES_PARQUET,
    spy_output_path: Path = DEFAULT_SPY_PARQUET,
    survivorship_csv_path: Path = DEFAULT_SURVIVORSHIP_CSV,
    process_only: bool = False,
    force_refresh: bool = False,
    limit: int | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download/cache batched `yfinance` data, compute PIT rolling beta, and audit survivorship."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    tickers = filter_queryable_tickers(cusip_map_df)
    if limit is not None:
        tickers = tickers[:limit]

    # 1. Benchmark (SPY)
    spy_cache = batch_dir / f"benchmark_{benchmark_ticker}.parquet"
    if not process_only and (force_refresh or not spy_cache.exists()):
        spy_df = download_yfinance_batch(
            [benchmark_ticker],
            start_date=start_date,
            end_date=end_date,
        )
        if spy_df.empty:
            raise RuntimeError(
                f"Failed to download benchmark ticker {benchmark_ticker} from yfinance."
            )
        spy_df.to_parquet(spy_cache, index=False)
    elif spy_cache.exists():
        spy_df = pd.read_parquet(spy_cache)
    else:
        raise FileNotFoundError(
            f"Benchmark cache missing at {spy_cache}. Run without --process-only first."
        )

    # 2. Equity batches (deterministic batch files keyed by batch index + size)
    batches = [tickers[i : i + batch_size] for i in range(0, len(tickers), batch_size)]
    pbar = (
        tqdm(total=len(batches), desc="yfinance batches", unit="batch") if show_progress else None
    )
    batch_frames: list[pd.DataFrame] = []

    try:
        for b_idx, batch in enumerate(batches):
            b_hash = hashlib.md5(",".join(batch).encode("utf-8")).hexdigest()[:8]
            b_path = batch_dir / f"batch_{b_idx:04d}_{b_hash}.parquet"
            if not process_only and (force_refresh or not b_path.exists()):
                bdf = download_yfinance_batch(
                    batch,
                    start_date=start_date,
                    end_date=end_date,
                )
                # Never cache an empty batch (likely rate-limited) so re-runs retry cleanly
                if not bdf.empty:
                    bdf.to_parquet(b_path, index=False)
                # Polite pause between batches to stay below Yahoo Finance's per-minute ceiling
                time.sleep(2.0)
            elif b_path.exists():
                bdf = pd.read_parquet(b_path)
            else:
                bdf = pd.DataFrame()

            if not bdf.empty:
                batch_frames.append(bdf)
            if pbar is not None:
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    if batch_frames:
        prices_df = pd.concat(batch_frames, ignore_index=True)
        prices_df["date"] = pd.to_datetime(prices_df["date"])
        prices_df = (
            prices_df.drop_duplicates(subset=["date", "yahoo_ticker"])
            .sort_values(["yahoo_ticker", "date"])
            .reset_index(drop=True)
        )
    else:
        prices_df = pd.DataFrame()

    spy_df["date"] = pd.to_datetime(spy_df["date"])
    spy_df = spy_df.sort_values("date").reset_index(drop=True)

    # 3. Compute strictly PIT 252-day trailing rolling beta vs. SPY
    prices_df = compute_rolling_beta_panel(
        prices_df,
        spy_df,
        window=beta_window,
        min_periods=min(126, beta_window),
    )

    prices_output_path.parent.mkdir(parents=True, exist_ok=True)
    spy_output_path.parent.mkdir(parents=True, exist_ok=True)
    prices_df.to_parquet(prices_output_path, index=False)
    spy_df.to_parquet(spy_output_path, index=False)

    priced_set = set(prices_df["yahoo_ticker"].unique()) if not prices_df.empty else set()
    surv_df = build_survivorship_report(cusip_map_df, priced_set)
    survivorship_csv_path.parent.mkdir(parents=True, exist_ok=True)
    surv_df.to_csv(survivorship_csv_path, index=False)

    return prices_df, spy_df, surv_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download & cache batched yfinance OHLCV/splits/dividends + PIT rolling beta."
    )
    parser.add_argument(
        "--cusip-map",
        type=Path,
        default=DEFAULT_CUSIP_MAP_PARQUET,
        help=f"Input CUSIP-FIGI map Parquet (default: {DEFAULT_CUSIP_MAP_PARQUET}).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_MARKET_START,
        help=f"First date to download (default: {DEFAULT_MARKET_START} for 1-yr beta burn-in).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Tickers per yfinance batch call (default: 50).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on top-dollar tickers to download (smoke test).",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip network calls and rebuild Parquet + rolling beta from cached batches.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download all batches even if cached locally.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config()
    hedging_cfg = cfg.settings["hedging"]

    if not args.cusip_map.exists():
        raise FileNotFoundError(
            f"CUSIP map not found at {args.cusip_map}. "
            "Run `python -m src.ingest.cusip_mapper` first."
        )
    cusip_map = pd.read_parquet(args.cusip_map)

    prices_df, spy_df, surv_df = ingest_market_data(
        cusip_map,
        benchmark_ticker=str(hedging_cfg.get("benchmark_ticker", "SPY")),
        start_date=args.start_date,
        end_date=cfg.settings["window"].get("end_date"),
        beta_window=int(hedging_cfg.get("rolling_beta_window_days", 252)),
        batch_size=args.batch_size,
        process_only=args.process_only,
        force_refresh=args.force_refresh,
        limit=args.limit,
    )

    n_tickers = prices_df["yahoo_ticker"].nunique() if not prices_df.empty else 0
    date_min = prices_df["date"].min().date() if not prices_df.empty else "N/A"
    date_max = prices_df["date"].max().date() if not prices_df.empty else "N/A"
    print(
        f"Market prices written -> {DEFAULT_MARKET_PRICES_PARQUET}\n"
        f"  Tickers priced: {n_tickers:,} | Rows: {len(prices_df):,} | "
        f"Dates: {date_min}..{date_max}\n"
        f"  SPY rows:       {len(spy_df):,} -> {DEFAULT_SPY_PARQUET}\n"
        f"  Survivorship report -> {DEFAULT_SURVIVORSHIP_CSV}"
    )
    print("\n=== US Common Stock Survivorship / Price Coverage ===")
    print(surv_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
