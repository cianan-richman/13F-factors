"""Unit and adversarial PIT QA tests for Stage 3 backwards/forward compute (`src.pit.compute`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pit.compute import (
    Stage3MarketMetrics,
    build_pit_stage3_snapshot,
    compute_masked_rolling_beta_lag1,
)
from src.pit.ledger import PitHoldingsLedger
from src.pit.panel import PitMarketStore


def _make_cal(start: str = "2022-01-03", periods: int = 320) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


def _make_price_series(
    ticker: str,
    dates: pd.DatetimeIndex,
    daily_rets: np.ndarray,
    *,
    start_price: float = 100.0,
    split_idx: int | None = None,
    split_factor: float = 1.0,
    adv_usd: float = 50_000_000.0,
) -> pd.DataFrame:
    """Build synthetic `market_prices` rows from explicit daily returns + optional stock split."""
    n = len(dates)
    r = np.asarray(daily_rets, dtype=np.float64).copy()
    r[0] = 0.0
    tr = start_price * np.cumprod(1.0 + r)

    s_t = np.ones(n, dtype=np.float64)
    if split_idx is not None:
        s_t[split_idx] = split_factor
    rev = np.cumprod(s_t[::-1])[::-1]
    cum_fut = np.ones(n, dtype=np.float64)
    cum_fut[:-1] = rev[1:]

    # Economic price without dividends: split_adj_close * cum_fut = raw_close
    split_adj = tr / rev[0]
    raw_close = split_adj * cum_fut
    ret_total = np.full(n, np.nan, dtype=np.float64)
    ret_total[1:] = (raw_close[1:] * s_t[1:]) / raw_close[:-1] - 1.0

    raw_vol = adv_usd / raw_close
    return pd.DataFrame(
        {
            "date": dates,
            "yahoo_ticker": ticker,
            "raw_close": raw_close,
            "split_adj_close": split_adj,
            "raw_volume": raw_vol,
            "dollar_volume": np.full(n, adv_usd),
            "split_factor": s_t,
            "cum_split_to_latest": cum_fut,
            "raw_dividend": np.zeros(n),
            "ret_total": ret_total,
            "adv_20d_usd": np.full(n, adv_usd),
        }
    )


def test_beta_lag1_matches_exact_ols_and_excludes_query_date() -> None:
    """`beta_252d_lag1` matches exact masked OLS on `[t-252, t-1]` and ignores day `t` & `t+1`."""
    rng = np.random.default_rng(42)
    cal = _make_cal("2022-01-03", periods=280)
    r_spy = rng.normal(0.0005, 0.01, size=len(cal))
    r_spy[0] = np.nan
    r_stk = 1.35 * r_spy + rng.normal(0.0, 0.005, size=len(cal))
    # Inject a NaN trading halt inside the 252d window to verify masked SPY variance!
    r_stk[50] = np.nan

    ret_wide = pd.DataFrame({"AAA": r_stk}, index=cal)
    spy_s = pd.Series(r_spy, index=cal)

    beta_base, n_obs = compute_masked_rolling_beta_lag1(
        ret_wide, spy_s, window=252, min_periods=250
    )

    # Inject huge outliers ON query_date cal[265] & AFTER -> MUST NOT change cal[265]!
    ret_mut = ret_wide.copy()
    ret_mut.iloc[265, 0] = 5.0
    ret_mut.iloc[266, 0] = -0.95
    beta_mut, _ = compute_masked_rolling_beta_lag1(ret_mut, spy_s, window=252, min_periods=250)

    qdt = cal[265]
    # Exact manual OLS on strictly [cal[13]..cal[264]] (252 calendar slots, 1 NaN -> 251 valid obs)
    win_x = r_stk[265 - 252 : 265]
    win_y = r_spy[265 - 252 : 265]
    ok = np.isfinite(win_x) & np.isfinite(win_y)
    expected_beta = float(np.polyfit(win_y[ok], win_x[ok], 1)[0])

    assert int(n_obs.at[qdt, "AAA"]) == int(ok.sum())
    assert np.isclose(beta_base.at[qdt, "AAA"], expected_beta, atol=1e-11)
    assert np.isclose(beta_mut.at[qdt, "AAA"], beta_base.at[qdt, "AAA"], atol=1e-12)


def test_beta_lag1_masks_bankruptcy_reorg_return_discontinuity() -> None:
    """Single-day >500% Chapter-11 reorg discontinuity (e.g. CHRD +25,733%) must be masked."""
    rng = np.random.default_rng(123)
    cal = _make_cal("2022-01-03", periods=280)
    r_spy = rng.normal(0.0004, 0.01, size=len(cal))
    r_spy[0] = np.nan
    r_stk = 1.40 * r_spy + rng.normal(0.0, 0.004, size=len(cal))
    r_stk[100] = 257.33  # Simulated Yahoo Finance pre/post-Ch.11 splice (+25,733%)

    ret_wide = pd.DataFrame({"CHRD": r_stk}, index=cal)
    spy_s = pd.Series(r_spy, index=cal)
    beta_lag1, n_obs = compute_masked_rolling_beta_lag1(
        ret_wide, spy_s, window=252, min_periods=250
    )

    qdt = cal[265]
    # Glitch day is excluded -> 251 valid observations and clean beta near 1.40 (not -90!)
    assert int(n_obs.at[qdt, "CHRD"]) == 251
    assert 1.1 < float(beta_lag1.at[qdt, "CHRD"]) < 1.7


def test_qoq_split_zero_trade_gives_zero_delta_and_zero_active_weight_shift() -> None:
    """Across a 10:1 split + 100% rally with zero actual trades, delta_shares and active_w = 0."""
    rng = np.random.default_rng(7)
    cal = _make_cal("2022-01-03", periods=300)
    r_spy = rng.normal(0.0003, 0.01, size=len(cal))
    r_nvda = 1.5 * r_spy + rng.normal(0.002, 0.005, size=len(cal))
    r_msft = 1.0 * r_spy + rng.normal(0.0001, 0.003, size=len(cal))

    # NVDA splits 10:1 at cal[262] (between Q1 cal[230] and Q2 cal[270])
    mp = pd.concat(
        [
            _make_price_series(
                "NVDA", cal, r_nvda, start_price=500.0, split_idx=262, split_factor=10.0
            ),
            _make_price_series("MSFT", cal, r_msft, start_price=300.0),
        ],
        ignore_index=True,
    )
    spy = _make_price_series("SPY", cal, r_spy, start_price=400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "NVDA",
                "issuer_cik": "0001045810",
                "entity_name": "NVIDIA",
                "filed_date": cal[272],
                "end_date": cal[268],
                "shares_outstanding": 20_000_000_000.0,
                "form": "10-Q",
                "accession_number": "N1",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            },
            {
                "yahoo_ticker": "MSFT",
                "issuer_cik": "0000789019",
                "entity_name": "MICROSOFT",
                "filed_date": cal[272],
                "end_date": cal[268],
                "shares_outstanding": 7_000_000_000.0,
                "form": "10-Q",
                "accession_number": "M1",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            },
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "67066G104",
                "security_id": "BBG_NVDA",
                "yahoo_ticker": "NVDA",
                "figi_name": "NVIDIA",
                "is_us_common_stock": True,
            },
            {
                "cusip": "594918104",
                "security_id": "BBG_MSFT",
                "yahoo_ticker": "MSFT",
                "figi_name": "MICROSOFT",
                "is_us_common_stock": True,
            },
        ]
    )
    # Fund holds 100k pre-split NVDA & 200k MSFT in Q1, and 1M post-split NVDA & 200k MSFT in Q2
    # -> LITERALLY ZERO TRADES despite 10:1 split and huge relative price drift!
    ledger = PitHoldingsLedger.from_raw_tables(
        pd.DataFrame(
            [
                {
                    "accession_number": "Q1",
                    "filer_id": "fund_a",
                    "canonical_name": "Fund A",
                    "style": "growth",
                    "form_type": "13F-HR",
                    "amendment_type": "",
                    "period_of_report": cal[230],
                    "date_filed": cal[240],
                    "acceptance_date": cal[240],
                    "acceptance_datetime": "20221201160000",
                    "parse_status": "ok",
                },
                {
                    "accession_number": "Q2",
                    "filer_id": "fund_a",
                    "canonical_name": "Fund A",
                    "style": "growth",
                    "form_type": "13F-HR",
                    "amendment_type": "",
                    "period_of_report": cal[270],
                    "date_filed": cal[280],
                    "acceptance_date": cal[280],
                    "acceptance_datetime": "20230201160000",
                    "parse_status": "ok",
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "accession_number": "Q1",
                    "cusip": "67066G104",
                    "name_of_issuer": "NVDA",
                    "title_of_class": "COM",
                    "ssh_prnamt": 100_000.0,
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 50_000_000.0,
                },
                {
                    "accession_number": "Q1",
                    "cusip": "594918104",
                    "name_of_issuer": "MSFT",
                    "title_of_class": "COM",
                    "ssh_prnamt": 200_000.0,
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 60_000_000.0,
                },
                {
                    "accession_number": "Q2",
                    "cusip": "67066G104",
                    "name_of_issuer": "NVDA",
                    "title_of_class": "COM",
                    "ssh_prnamt": 1_000_000.0,
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 80_000_000.0,
                },
                {
                    "accession_number": "Q2",
                    "cusip": "594918104",
                    "name_of_issuer": "MSFT",
                    "title_of_class": "COM",
                    "ssh_prnamt": 200_000.0,
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 62_000_000.0,
                },
            ]
        ),
        trading_calendar=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)
    s3_mkt = Stage3MarketMetrics(store)
    snap = build_pit_stage3_snapshot(cal[285], ledger, store, s3_mkt, verify=True)

    nvda = snap[snap["yahoo_ticker"] == "NVDA"].iloc[0]
    assert np.isclose(nvda["delta_shares_pit"], 0.0, atol=1e-6)
    assert np.isclose(nvda["mean_active_weight_shift"], 0.0, atol=1e-12)
    assert int(nvda["n_buyers"]) == 0
    assert int(nvda["n_sellers"]) == 0


def test_dual_class_rollup_primary_selection_and_pct_gt_100_guard() -> None:
    """Equal-ratio GOOG+GOOGL roll up ownership %; only higher-ADV class is panel-eligible."""
    rng = np.random.default_rng(11)
    cal = _make_cal("2022-01-03", periods=290)
    r_spy = rng.normal(0.0004, 0.01, size=len(cal))
    r_g = 1.1 * r_spy + rng.normal(0.0, 0.004, size=len(cal))
    r_bad = 0.9 * r_spy + rng.normal(0.0, 0.005, size=len(cal))

    mp = pd.concat(
        [
            _make_price_series("GOOGL", cal, r_g, start_price=150.0, adv_usd=2_000_000_000.0),
            _make_price_series("GOOG", cal, r_g, start_price=151.0, adv_usd=1_500_000_000.0),
            _make_price_series("BADX", cal, r_bad, start_price=50.0, adv_usd=50_000_000.0),
        ],
        ignore_index=True,
    )
    spy = _make_price_series("SPY", cal, r_spy, start_price=400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "GOOGL",
                "issuer_cik": "0001652044",
                "entity_name": "ALPHABET",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 10_000_000_000.0,
                "form": "10-Q",
                "accession_number": "G1",
                "tag_source": "us-gaap:CommonStockSharesOutstanding",
                "tag_priority": 2,
                "n_classes_reported": 1,
            },
            {
                "yahoo_ticker": "GOOG",
                "issuer_cik": "0001652044",
                "entity_name": "ALPHABET",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 10_000_000_000.0,
                "form": "10-Q",
                "accession_number": "G1",
                "tag_source": "us-gaap:CommonStockSharesOutstanding",
                "tag_priority": 2,
                "n_classes_reported": 1,
            },
            {
                "yahoo_ticker": "BADX",
                "issuer_cik": "0009998888",
                "entity_name": "BAD XBRL CO",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 50_000_000.0,  # Bogus denominator < 100M owned -> >100%!
                "form": "10-Q",
                "accession_number": "X1",
                "tag_source": "us-gaap:CommonStockSharesOutstanding",
                "tag_priority": 2,
                "n_classes_reported": 1,
            },
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "02079K305",
                "security_id": "BBG_GOOGL",
                "yahoo_ticker": "GOOGL",
                "figi_name": "ALPHABET CL A",
                "is_us_common_stock": True,
            },
            {
                "cusip": "02079K107",
                "security_id": "BBG_GOOG",
                "yahoo_ticker": "GOOG",
                "figi_name": "ALPHABET CL C",
                "is_us_common_stock": True,
            },
            {
                "cusip": "999999101",
                "security_id": "BBG_BADX",
                "yahoo_ticker": "BADX",
                "figi_name": "BAD XBRL",
                "is_us_common_stock": True,
            },
        ]
    )
    ledger = PitHoldingsLedger.from_raw_tables(
        pd.DataFrame(
            [
                {
                    "accession_number": "F1",
                    "filer_id": "fund_a",
                    "canonical_name": "Fund A",
                    "style": "growth",
                    "form_type": "13F-HR",
                    "amendment_type": "",
                    "period_of_report": cal[245],
                    "date_filed": cal[255],
                    "acceptance_date": cal[255],
                    "acceptance_datetime": "20230101160000",
                    "parse_status": "ok",
                }
            ]
        ),
        pd.DataFrame(
            [
                {
                    "accession_number": "F1",
                    "cusip": "02079K305",
                    "name_of_issuer": "ALPHABET A",
                    "title_of_class": "CL A",
                    "ssh_prnamt": 300_000_000.0,  # 3% of 10B
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 45_000_000_000.0,
                },
                {
                    "accession_number": "F1",
                    "cusip": "02079K107",
                    "name_of_issuer": "ALPHABET C",
                    "title_of_class": "CL C",
                    "ssh_prnamt": 200_000_000.0,  # 2% of 10B -> combined = 5%!
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 30_200_000_000.0,
                },
                {
                    "accession_number": "F1",
                    "cusip": "999999101",
                    "name_of_issuer": "BADX",
                    "title_of_class": "COM",
                    "ssh_prnamt": 100_000_000.0,  # 200% of 50M -> MUST be nulled!
                    "ssh_prnamt_type": "SH",
                    "put_call": "",
                    "value_usd": 5_000_000_000.0,
                },
            ]
        ),
        trading_calendar=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)
    s3_mkt = Stage3MarketMetrics(store)
    snap = build_pit_stage3_snapshot(cal[270], ledger, store, s3_mkt, verify=True)

    googl = snap[snap["yahoo_ticker"] == "GOOGL"].iloc[0]
    goog = snap[snap["yahoo_ticker"] == "GOOG"].iloc[0]
    badx = snap[snap["yahoo_ticker"] == "BADX"].iloc[0]

    # Combined Alphabet ownership = (300M + 200M) / 10B = 5.0%
    assert np.isclose(googl["pct_shares_owned_class"], 0.03)
    assert np.isclose(googl["pct_shares_owned"], 0.05)
    assert np.isclose(goog["pct_shares_owned"], 0.05)
    # Higher-ADV GOOGL is primary & panel-eligible; GOOG is secondary & excluded from regression
    assert bool(googl["is_primary_class"]) is True
    assert bool(googl["is_panel_eligible"]) is True
    assert bool(goog["is_primary_class"]) is False
    assert bool(goog["is_panel_eligible"]) is False

    # BADX (>100% ownership) MUST be flagged invalid & nulled
    assert bool(badx["is_pct_owned_invalid"]) is True
    assert pd.isna(badx["pct_shares_owned"])
    assert bool(badx["is_panel_eligible"]) is False

    # Forward beta-adjusted return matches exact identity
    expected_resid = googl["fwd_day_ret"] - googl["beta_252d_lag1"] * googl["fwd_spy_ret"]
    assert np.isclose(googl["fwd_beta_adj_ret"], expected_resid, atol=1e-12)


def test_prev_and_fwd_returns_handle_splits_dividends_and_scrub_non_trading_days() -> None:
    """Verify prev_day_ret (t-1->t) & fwd_day_ret (t->t+1) across splits, divs, and weekends."""
    # Calendar has 5 trading days: Mon..Fri, with a weekend inserted defensively in market_prices
    cal = pd.DatetimeIndex(["2024-06-06", "2024-06-07", "2024-06-10", "2024-06-11", "2024-06-12"])
    # Stock splits 10:1 on Mon 2024-06-10 and pays a $10 special dividend on Tue 2024-06-11
    # Thu=1000, Fri=1000, Mon=105 (10:1 -> +5%), Tue=100+$10 div (+4.76%), Wed=110 (+10%)
    mp = pd.DataFrame(
        {
            "date": list(cal) + [pd.Timestamp("2024-06-08")],  # Stray Saturday row to be scrubbed!
            "yahoo_ticker": ["TST"] * 6,
            "raw_close": [1000.0, 1000.0, 105.0, 100.0, 110.0, 999.0],
            "split_adj_close": [100.0, 100.0, 105.0, 100.0, 110.0, 99.9],
            "raw_volume": [1e6] * 6,
            "dollar_volume": [1e8] * 6,
            "split_factor": [1.0, 1.0, 10.0, 1.0, 1.0, 1.0],
            "cum_split_to_latest": [10.0, 10.0, 1.0, 1.0, 1.0, 10.0],
            "raw_dividend": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
            "ret_total": [np.nan, 0.0, 0.05, (100.0 + 10.0) / 105.0 - 1.0, 0.10, 0.0],
            "adv_20d_usd": [1e8] * 6,
        }
    )
    spy = pd.DataFrame(
        {
            "date": cal,
            "yahoo_ticker": ["SPY"] * 5,
            "raw_close": [500.0, 505.0, 500.0, 502.0, 507.02],
            "split_adj_close": [500.0, 505.0, 500.0, 502.0, 507.02],
            "raw_volume": [1e8] * 5,
            "dollar_volume": [5e10] * 5,
            "split_factor": [1.0] * 5,
            "cum_split_to_latest": [1.0] * 5,
            "raw_dividend": [0.0, 0.0, 2.0, 0.0, 0.0],
            "ret_total": [np.nan, 0.01, (500.0 + 2.0) / 505.0 - 1.0, 0.004, 0.01],
            "adv_20d_usd": [5e10] * 5,
        }
    )
    so = pd.DataFrame(
        columns=[
            "yahoo_ticker",
            "issuer_cik",
            "filed_date",
            "end_date",
            "shares_outstanding",
            "tag_priority",
        ]
    )
    cm = pd.DataFrame(
        columns=["cusip", "security_id", "yahoo_ticker", "figi_name", "is_us_common_stock"]
    )

    store = PitMarketStore(mp, spy, so, cm)
    s3 = Stage3MarketMetrics(store, beta_window=3, beta_min_periods=2)

    # Stray Saturday MUST be scrubbed from trading calendar & ticker metrics
    assert pd.Timestamp("2024-06-08") not in s3.trading_calendar
    assert len(s3._ticker_metrics["TST"]["dates_ns"]) == 5

    # On Fri 2024-06-07: prev_day_ret (Thu->Fri) = 0%, fwd_day_ret (Fri->Mon 10:1 split) = +5%!
    m_fri = s3.get_stock_metrics("TST", pd.Timestamp("2024-06-07"))
    m_mon = s3.get_stock_metrics("TST", pd.Timestamp("2024-06-10"))
    m_tue = s3.get_stock_metrics("TST", pd.Timestamp("2024-06-11"))
    assert np.isclose(m_fri["prev_day_ret"], 0.0)
    assert np.isclose(m_fri["fwd_day_ret"], 0.05)
    assert np.isclose(m_mon["prev_day_ret"], 0.05)
    assert np.isclose(m_mon["fwd_day_ret"], (100.0 + 10.0) / 105.0 - 1.0)
    assert np.isclose(m_tue["prev_day_ret"], m_mon["fwd_day_ret"])
    assert np.isclose(m_tue["fwd_day_ret"], 0.10)

    # SPY ex-dividend return on Mon 2024-06-10 matches across Fri fwd_spy_ret & Mon prev_spy_ret
    spy_fri = s3.get_spy_returns(pd.Timestamp("2024-06-07"))
    spy_mon = s3.get_spy_returns(pd.Timestamp("2024-06-10"))
    expected_spy_mon = (500.0 + 2.0) / 505.0 - 1.0
    assert np.isclose(spy_fri["fwd_spy_ret"], expected_spy_mon)
    assert np.isclose(spy_mon["prev_spy_ret"], expected_spy_mon)
