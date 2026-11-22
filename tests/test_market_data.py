"""Unit tests for yfinance un-splitting, forward total returns, and PIT rolling beta."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.ingest.market_data import (
    _inclusive_to_yf_end_date,
    build_survivorship_report,
    compute_pit_price_series,
    compute_rolling_beta_panel,
    download_yfinance_batch,
    ingest_market_data,
)


def test_compute_pit_price_series_unsplits_correctly_and_preserves_total_return() -> None:
    # Stock trades at $400 on day 0, splits 4-for-1 on day 1 (trades at $100),
    # then rises 5% to $105 and pays a $1.00 raw cash dividend on day 2.
    # Yahoo Finance returns split-adjusted Close, Volume, Dividends:
    #   day 0: Close=100, Vol=4_000_000, Div=0, Split=0
    #   day 1: Close=100, Vol=4_000_000, Div=0, Split=4.0
    #   day 2: Close=105, Vol=4_000_000, Div=1.0, Split=0
    dates = pd.bdate_range("2020-08-28", periods=3)
    raw_yf = pd.DataFrame(
        {
            "Close": [100.0, 100.0, 105.0],
            "Volume": [4_000_000.0, 4_000_000.0, 4_000_000.0],
            "Dividends": [0.0, 0.0, 1.0],
            "Stock Splits": [0.0, 4.0, 0.0],
        },
        index=dates,
    )

    out = compute_pit_price_series(raw_yf, ticker="AAPL")
    assert len(out) == 3

    # Day 0 (pre-split): raw_close must be $400, raw_volume 1M, cum_split_to_latest = 4.0
    assert np.isclose(out.loc[0, "raw_close"], 400.0)
    assert np.isclose(out.loc[0, "raw_volume"], 1_000_000.0)
    assert np.isclose(out.loc[0, "cum_split_to_latest"], 4.0)
    assert np.isclose(out.loc[0, "dollar_volume"], 400_000_000.0)

    # Day 1 (split ex-date): raw_close = $100, cum_split_to_latest = 1.0, return = 0%
    assert np.isclose(out.loc[1, "raw_close"], 100.0)
    assert np.isclose(out.loc[1, "cum_split_to_latest"], 1.0)
    assert np.isclose(out.loc[1, "ret_total"], 0.0)

    # Day 2: (105 * 1 + 1) / 100 - 1 = +6.0%
    assert np.isclose(out.loc[2, "raw_close"], 105.0)
    assert np.isclose(out.loc[2, "raw_dividend"], 1.0)
    assert np.isclose(out.loc[2, "ret_total"], 0.06)


def test_compute_rolling_beta_panel_is_strictly_pit_no_future_leakage() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=300)
    spy_ret = rng.normal(0.0004, 0.01, size=len(dates))
    # True beta = 1.5 for first 260 days, then jumps to -2.0 on days 260..299
    stock_ret = np.where(np.arange(len(dates)) < 260, 1.5 * spy_ret, -2.0 * spy_ret)

    spy_df = pd.DataFrame({"date": dates, "ret_total": spy_ret})
    prices_df = pd.DataFrame(
        {"date": dates, "yahoo_ticker": "TEST", "ret_total": stock_ret}
    )

    out = compute_rolling_beta_panel(prices_df, spy_df, window=252, min_periods=126)

    # Prior to min_periods (index 124), beta must be NaN
    assert np.isnan(out.loc[100, "beta_252d"])
    # On day 259 (strictly before regime flip at day 260), beta must be exact 1.50 (zero lookahead!)
    assert np.isclose(out.loc[259, "beta_252d"], 1.5, atol=1e-10)
    # After regime flip (day 299), rolling beta must have moved downward
    assert out.loc[299, "beta_252d"] < 1.2


def test_build_survivorship_report_categorizes_dead_and_priced_dollars() -> None:
    cmap = pd.DataFrame(
        [
            {
                "cusip": "037833100",
                "security_id": "S1",
                "yahoo_ticker": "AAPL",
                "is_us_common_stock": True,
                "total_value_usd": 800.0,
            },
            {
                "cusip": "999999999",
                "security_id": "S2",
                "yahoo_ticker": "TWTR",
                "is_us_common_stock": True,
                "total_value_usd": 150.0,
            },
            {
                "cusip": "888888888",
                "security_id": "S3",
                "yahoo_ticker": "1812212D",
                "is_us_common_stock": True,
                "total_value_usd": 50.0,
            },
        ]
    )
    rep = build_survivorship_report(cmap, priced_tickers={"AAPL"})
    by_status = rep.set_index("price_coverage_status")["pct_us_common_dollars"].to_dict()
    assert np.isclose(by_status["priced_in_yfinance"], 80.0)
    assert np.isclose(by_status["missing_or_delisted_in_yfinance"], 15.0)
    assert np.isclose(by_status["dead_bloomberg_id"], 5.0)


def test_yfinance_end_date_is_inclusive_on_download_and_cache(tmp_path: Path) -> None:
    """Pinned end_date='2026-11-20' passes exclusive end='2026-11-21' to yfinance and clips."""
    assert _inclusive_to_yf_end_date("2026-11-20") == "2026-11-21"
    assert _inclusive_to_yf_end_date(None) is None

    dates = pd.DatetimeIndex(["2026-11-19", "2026-11-20", "2026-11-23"])
    fake_raw = pd.DataFrame(
        {
            "Close": [100.0, 102.0, 105.0],
            "Volume": [1_000_000.0, 1_000_000.0, 1_000_000.0],
            "Dividends": [0.0, 0.0, 0.0],
            "Stock Splits": [0.0, 0.0, 0.0],
        },
        index=dates,
    )

    captured: dict[str, object] = {}

    def _fake_download(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return fake_raw

    with patch("src.ingest.market_data.yf.download", side_effect=_fake_download):
        df = download_yfinance_batch(["SPY"], start_date="2026-11-19", end_date="2026-11-20")

    assert captured["end"] == "2026-11-21"
    assert list(df["date"]) == [pd.Timestamp("2026-11-19"), pd.Timestamp("2026-11-20")]

    # Cached parquets containing bars past end_date must also be clipped strictly <= end_date
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    spy_full = compute_pit_price_series(fake_raw, ticker="SPY")
    aapl_full = compute_pit_price_series(fake_raw, ticker="AAPL")
    spy_full.to_parquet(batch_dir / "benchmark_SPY.parquet", index=False)
    aapl_full.to_parquet(batch_dir / "batch_0000_cached.parquet", index=False)

    cmap = pd.DataFrame(
        [
            {
                "cusip": "037833100",
                "security_id": "S1",
                "yahoo_ticker": "AAPL",
                "is_us_common_stock": True,
                "total_value_usd": 1_000.0,
            }
        ]
    )
    prices_df, spy_df, _ = ingest_market_data(
        cmap,
        end_date="2026-11-20",
        batch_dir=batch_dir,
        prices_output_path=tmp_path / "market_prices.parquet",
        spy_output_path=tmp_path / "spy_prices.parquet",
        survivorship_csv_path=tmp_path / "surv.csv",
        process_only=True,
        show_progress=False,
    )
    assert prices_df["date"].max() == pd.Timestamp("2026-11-20")
    assert spy_df["date"].max() == pd.Timestamp("2026-11-20")
