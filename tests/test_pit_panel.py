"""Unit and adversarial QA tests for Stage 1 (security agg) and Stage 2 (PIT market data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pit.ledger import PitHoldingsLedger, PitLookaheadError
from src.pit.panel import (
    PitMarketStore,
    aggregate_positions_by_security,
    build_pit_security_snapshot,
)


def _make_trading_cal(start: str = "2022-01-03", periods: int = 400) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


def _build_synthetic_market_prices(
    ticker: str,
    dates: pd.DatetimeIndex,
    *,
    base_raw_close: float = 1000.0,
    split_date: pd.Timestamp | None = None,
    split_factor: float = 1.0,
    div_date: pd.Timestamp | None = None,
    div_amount: float = 0.0,
) -> pd.DataFrame:
    """Build synthetic `market_prices` rows matching `compute_pit_price_series` exact invariants."""
    n = len(dates)
    s_t = np.ones(n, dtype=np.float64)
    if split_date is not None:
        idx = dates.get_loc(split_date)
        s_t[idx] = split_factor

    rev = np.cumprod(s_t[::-1])[::-1]
    cum_fut = np.ones(n, dtype=np.float64)
    cum_fut[:-1] = rev[1:]

    # Constant economic price (0% daily return before dividends), stepping by split_factor
    split_adj_close = np.full(n, base_raw_close / rev[0], dtype=np.float64)
    raw_close = split_adj_close * cum_fut

    raw_div = np.zeros(n, dtype=np.float64)
    if div_date is not None:
        didx = dates.get_loc(div_date)
        raw_div[didx] = div_amount

    ret_total = np.full(n, np.nan, dtype=np.float64)
    ret_total[1:] = (raw_close[1:] * s_t[1:] + raw_div[1:]) / raw_close[:-1] - 1.0

    raw_vol = np.full(n, 500_000.0, dtype=np.float64)
    dollar_vol = raw_close * raw_vol

    return pd.DataFrame(
        {
            "date": dates,
            "yahoo_ticker": ticker,
            "raw_close": raw_close,
            "split_adj_close": split_adj_close,
            "raw_volume": raw_vol,
            "dollar_volume": dollar_vol,
            "split_factor": s_t,
            "cum_split_to_latest": cum_fut,
            "raw_dividend": raw_div,
            "ret_total": ret_total,
            "adv_20d_usd": dollar_vol,
        }
    )


def _make_ledger(
    filings: list[dict],
    holdings: list[dict],
    cal: pd.DatetimeIndex,
) -> PitHoldingsLedger:
    meta = pd.DataFrame(filings)
    raw_h = pd.DataFrame(holdings)
    return PitHoldingsLedger.from_raw_tables(meta, raw_h, trading_calendar=cal)


def test_split_ex_date_preserves_mcap_and_ownership_ratio_with_zero_lookahead() -> None:
    """Across a 10:1 split on T, mcap, position USD, and ownership % stay constant."""
    cal = _make_trading_cal("2022-01-03", periods=320)
    # Place split AFTER 260 trading days so security is seasoned (>252d)
    split_dt = cal[265]
    pre_dt = cal[264]
    # Also add a FUTURE 2:1 split at cal[300] to prove zero lookahead on pre_dt & split_dt!
    mp = _build_synthetic_market_prices(
        "NVDA", cal, base_raw_close=1000.0, split_date=split_dt, split_factor=10.0
    )
    # Manually inject a second future split on cal[300]
    s_t = mp["split_factor"].to_numpy().copy()
    s_t[300] = 2.0
    rev = np.cumprod(s_t[::-1])[::-1]
    cum_fut = np.ones(len(cal), dtype=np.float64)
    cum_fut[:-1] = rev[1:]
    mp["split_factor"] = s_t
    mp["cum_split_to_latest"] = cum_fut
    mp["split_adj_close"] = 1000.0 / rev[0]
    mp["raw_close"] = mp["split_adj_close"] * cum_fut
    mp["dollar_volume"] = mp["raw_close"] * mp["raw_volume"]
    mp["adv_20d_usd"] = mp["dollar_volume"]

    spy = _build_synthetic_market_prices(
        "SPY", cal, base_raw_close=400.0, div_date=cal[100], div_amount=2.0
    )
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "NVDA",
                "issuer_cik": "0001045810",
                "entity_name": "NVIDIA CORP",
                "filed_date": cal[240],
                "end_date": cal[235],
                "shares_outstanding": 2_000_000_000.0,  # 2B pre-split shares
                "form": "10-Q",
                "accession_number": "0001045810-23-000001",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            }
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "67066G104",
                "security_id": "BBG001S5TZJ6",
                "yahoo_ticker": "NVDA",
                "figi_name": "NVIDIA CORP",
                "is_us_common_stock": True,
            }
        ]
    )

    ledger = _make_ledger(
        filings=[
            {
                "accession_number": "A1",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[220],
                "date_filed": cal[245],
                "acceptance_date": cal[245],
                "acceptance_datetime": "20221201160000",
                "parse_status": "ok",
            }
        ],
        holdings=[
            {
                "accession_number": "A1",
                "cusip": "67066G104",
                "name_of_issuer": "NVIDIA CORP",
                "title_of_class": "COM",
                "ssh_prnamt": 10_000_000.0,  # 10M pre-split shares = 0.5% of 2B
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000_000_000.0,
            }
        ],
        cal=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)

    snap_pre = build_pit_security_snapshot(pre_dt, ledger, store, verify=True)
    snap_ex = build_pit_security_snapshot(split_dt, ledger, store, verify=True)

    r0 = snap_pre.iloc[0]
    r1 = snap_ex.iloc[0]

    # Pre-split: raw_close = 1000, shares_owned = 10M, shares_out = 2B (future 10x & 2x ignored!)
    assert np.isclose(r0["raw_close"], 1000.0)
    assert np.isclose(r0["total_shares_owned_pit"], 10_000_000.0)
    assert np.isclose(r0["shares_out_pit"], 2_000_000_000.0)
    assert np.isclose(r0["pit_mcap_usd"], 2_000_000_000_000.0)

    # Split ex-date: raw_close = 100, shares_owned = 100M, shares_out = 20B (future 2x ignored!)
    assert np.isclose(r1["raw_close"], 100.0)
    assert np.isclose(r1["total_shares_owned_pit"], 100_000_000.0)
    assert np.isclose(r1["shares_out_pit"], 20_000_000_000.0)
    assert np.isclose(r1["pit_mcap_usd"], 2_000_000_000_000.0)
    assert np.isclose(r1["total_position_usd_pit"], r0["total_position_usd_pit"])
    assert np.isclose(
        r1["total_shares_owned_pit"] / r1["shares_out_pit"],
        r0["total_shares_owned_pit"] / r0["shares_out_pit"],
    )
    assert bool(r1["is_seasoned"]) is True
    assert bool(r1["is_investable"]) is True


def test_split_straddling_across_two_filers_sums_in_exact_query_date_units() -> None:
    """When Fund A is on pre-split Q2 and Fund B is on post-split Q3, Stage 2 aligns both."""
    cal = _make_trading_cal("2022-01-03", periods=320)
    split_dt = cal[260]
    query_dt = cal[280]

    mp = _build_synthetic_market_prices(
        "GOOGL", cal, base_raw_close=2000.0, split_date=split_dt, split_factor=20.0
    )
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "GOOGL",
                "issuer_cik": "0001652044",
                "entity_name": "ALPHABET INC",
                # Filed AFTER split_dt for pre-split end_date -> GAAP ASC 260 already post-split
                "filed_date": cal[268],
                "end_date": cal[255],
                "shares_outstanding": 12_000_000_000.0,
                "form": "10-Q",
                "accession_number": "0001652044-23-000001",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            }
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "02079K305",
                "security_id": "BBG009S39JX6",
                "yahoo_ticker": "GOOGL",
                "figi_name": "ALPHABET INC-CL A",
                "is_us_common_stock": True,
            }
        ]
    )

    ledger = _make_ledger(
        filings=[
            # Fund A: Q2 pre-split report cal[250] -> 100k pre-split shares = 2M post-split
            {
                "accession_number": "FA_Q2",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[250],
                "date_filed": cal[255],
                "acceptance_date": cal[255],
                "acceptance_datetime": "20230101160000",
                "parse_status": "ok",
            },
            # Fund B: Q3 (post-split report cal[270], filed cal[275]) -> 3M post-split shares
            {
                "accession_number": "FB_Q3",
                "filer_id": "fund_b",
                "canonical_name": "Fund B",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[270],
                "date_filed": cal[275],
                "acceptance_date": cal[275],
                "acceptance_datetime": "20230201160000",
                "parse_status": "ok",
            },
        ],
        holdings=[
            {
                "accession_number": "FA_Q2",
                "cusip": "02079K305",
                "name_of_issuer": "ALPHABET",
                "title_of_class": "CL A",
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 200_000_000.0,
            },
            {
                "accession_number": "FB_Q3",
                "cusip": "02079K305",
                "name_of_issuer": "ALPHABET",
                "title_of_class": "CL A",
                "ssh_prnamt": 3_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 300_000_000.0,
            },
        ],
        cal=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)
    snap = build_pit_security_snapshot(query_dt, ledger, store, verify=True)
    r = snap.iloc[0]

    # Raw sum would have been bogus 3.1M; exact PIT split-aligned sum is (100k*20) + 3M = 5.0M!
    assert bool(r["has_split_straddle"]) is True
    assert np.isclose(r["total_shares_raw"], 3_100_000.0)
    assert np.isclose(r["total_shares_owned_pit"], 5_000_000.0)
    # GAAP ASC 260: XBRL filed_date (cal[268]) > split_dt -> NOT double-multiplied by 20x
    assert np.isclose(r["shares_out_split_factor"], 1.0)
    assert np.isclose(r["shares_out_pit"], 12_000_000_000.0)
    assert np.isclose(r["pit_mcap_usd"], 1_200_000_000_000.0)


def test_xbrl_same_day_lookahead_blocked_and_seasoning_gate_enforced() -> None:
    """XBRL filed on `query_date` is invisible until T+1; unseasoned (<252d) fails gate."""
    cal = _make_trading_cal("2022-01-03", periods=280)
    # Stock IPOs at cal[100] -> on cal[200] it has only 100 trading days (< 252!)
    ipo_cal = cal[100:]
    q_dt = cal[200]

    mp = _build_synthetic_market_prices("NEWCO", ipo_cal, base_raw_close=50.0)
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "NEWCO",
                "issuer_cik": "0009999999",
                "entity_name": "NEWCO INC",
                "filed_date": q_dt,  # Filed ON query_date -> MUST NOT be visible on q_dt!
                "end_date": cal[190],
                "shares_outstanding": 100_000_000.0,
                "form": "10-Q",
                "accession_number": "0009999999-22-000001",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            }
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "111111101",
                "security_id": "BBG00NEWCO01",
                "yahoo_ticker": "NEWCO",
                "figi_name": "NEWCO INC",
                "is_us_common_stock": True,
            }
        ]
    )
    ledger = _make_ledger(
        filings=[
            {
                "accession_number": "F1",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[180],
                "date_filed": cal[195],
                "acceptance_date": cal[195],
                "acceptance_datetime": "20221001160000",
                "parse_status": "ok",
            }
        ],
        holdings=[
            {
                "accession_number": "F1",
                "cusip": "111111101",
                "name_of_issuer": "NEWCO",
                "title_of_class": "COM",
                "ssh_prnamt": 1_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 50_000_000.0,
            }
        ],
        cal=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)

    snap_on_filed = build_pit_security_snapshot(q_dt, ledger, store, verify=True)
    snap_next_day = build_pit_security_snapshot(cal[201], ledger, store, verify=True)

    # On filed_date itself: shares_out MUST be NaN (strictly T+1 visibility)
    assert pd.isna(snap_on_filed.iloc[0]["shares_out_pit"])
    assert pd.isna(snap_on_filed.iloc[0]["shares_filed_date"])
    assert bool(snap_on_filed.iloc[0]["is_seasoned"]) is False
    assert snap_on_filed.iloc[0]["trading_days_history"] == 100
    assert bool(snap_on_filed.iloc[0]["is_investable"]) is False

    # Next trading day: shares_out IS visible, but still unseasoned (101 < 252) -> not investable
    assert np.isclose(snap_next_day.iloc[0]["shares_out_pit"], 100_000_000.0)
    assert np.isclose(snap_next_day.iloc[0]["pit_mcap_usd"], 5_000_000_000.0)
    assert bool(snap_next_day.iloc[0]["is_seasoned"]) is False
    assert bool(snap_next_day.iloc[0]["is_investable"]) is False


def test_unequal_dual_class_ratio_and_stale_xbrl_flagged_and_spy_forward_pit() -> None:
    """BRK-A/BRK-B unequal ratio nulled; forward-only spy_adj_close never changes retroactively."""
    cal = _make_trading_cal("2022-01-03", periods=280)
    q_dt = cal[265]

    mp = pd.concat(
        [
            _build_synthetic_market_prices("BRK-A", cal, base_raw_close=600_000.0),
            _build_synthetic_market_prices("BRK-B", cal, base_raw_close=400.0),
        ],
        ignore_index=True,
    )
    spy_short = _build_synthetic_market_prices(
        "SPY", cal[:270], base_raw_close=400.0, div_date=cal[100], div_amount=2.0
    )
    spy_long = _build_synthetic_market_prices(
        "SPY", cal, base_raw_close=400.0, div_date=cal[275], div_amount=10.0
    )
    # Add same earlier div on cal[100] to spy_long too, plus a future dividend at cal[275]
    spy_long.loc[100, "raw_dividend"] = 2.0
    spy_long.loc[100, "ret_total"] = (400.0 + 2.0) / 400.0 - 1.0

    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "BRK-A",
                "issuer_cik": "0001067983",
                "entity_name": "BERKSHIRE HATHAWAY",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 1_500_000.0,
                "form": "10-Q",
                "accession_number": "B1",
                "tag_source": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
                "tag_priority": 3,
                "n_classes_reported": 2,
            },
            {
                "yahoo_ticker": "BRK-B",
                "issuer_cik": "0001067983",
                "entity_name": "BERKSHIRE HATHAWAY",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 1_500_000.0,
                "form": "10-Q",
                "accession_number": "B1",
                "tag_source": "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
                "tag_priority": 3,
                "n_classes_reported": 2,
            },
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "084670702",
                "security_id": "BBG001S5PQL7",
                "yahoo_ticker": "BRK-B",
                "figi_name": "BERKSHIRE HATHAWAY-CL B",
                "is_us_common_stock": True,
            }
        ]
    )
    ledger = _make_ledger(
        filings=[
            {
                "accession_number": "F1",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[240],
                "date_filed": cal[255],
                "acceptance_date": cal[255],
                "acceptance_datetime": "20230101160000",
                "parse_status": "ok",
            }
        ],
        holdings=[
            {
                "accession_number": "F1",
                "cusip": "084670702",
                "name_of_issuer": "BERKSHIRE HATHAWAY",
                "title_of_class": "CL B",
                "ssh_prnamt": 2_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 800_000_000.0,
            }
        ],
        cal=cal,
    )

    store_short = PitMarketStore(mp, spy_short, so, cm)
    store_long = PitMarketStore(mp, spy_long, so, cm)

    snap_short = build_pit_security_snapshot(q_dt, ledger, store_short, verify=True)
    snap_long = build_pit_security_snapshot(q_dt, ledger, store_long, verify=True)

    # BRK-B detected as unequal dual-class ratio -> pit_mcap_usd nulled & not investable
    assert bool(snap_short.iloc[0]["is_dual_class_issuer"]) is True
    assert bool(snap_short.iloc[0]["has_unequal_class_ratio"]) is True
    assert pd.isna(snap_short.iloc[0]["pit_mcap_usd"])
    assert bool(snap_short.iloc[0]["is_investable"]) is False

    # Forward-only spy_adj_close on q_dt is 100% invariant to future dividend at cal[275]!
    assert np.isclose(snap_short.iloc[0]["spy_adj_close"], snap_long.iloc[0]["spy_adj_close"])
    assert snap_short.iloc[0]["spy_adj_close"] > snap_short.iloc[0]["spy_raw_close"]


def test_stage1_stitches_cusip_change_under_single_security_id() -> None:
    """Two filers holding old vs new CUSIP for the same `security_id` aggregate into 1 row."""
    cal = _make_trading_cal("2023-01-03", periods=60)
    cm = pd.DataFrame(
        [
            {
                "cusip": "13645T100",
                "security_id": "BBG001S7F4Z8",
                "yahoo_ticker": "CP",
                "figi_name": "CANADIAN PACIFIC KANSAS CITY",
                "is_us_common_stock": True,
            },
            {
                "cusip": "13646K108",
                "security_id": "BBG001S7F4Z8",
                "yahoo_ticker": "CP",
                "figi_name": "CANADIAN PACIFIC KANSAS CITY",
                "is_us_common_stock": True,
            },
        ]
    )
    ledger = _make_ledger(
        filings=[
            {
                "accession_number": "F1",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[5],
                "date_filed": cal[10],
                "acceptance_date": cal[10],
                "acceptance_datetime": "20230120160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "F2",
                "filer_id": "fund_b",
                "canonical_name": "Fund B",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[5],
                "date_filed": cal[12],
                "acceptance_date": cal[12],
                "acceptance_datetime": "20230124160000",
                "parse_status": "ok",
            },
        ],
        holdings=[
            {
                "accession_number": "F1",
                "cusip": "13645T100",
                "name_of_issuer": "CANADIAN PAC RY",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000.0,
            },
            {
                "accession_number": "F2",
                "cusip": "13646K108",
                "name_of_issuer": "CANADIAN PAC KANSAS",
                "title_of_class": "COM",
                "ssh_prnamt": 300.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 30_000.0,
            },
        ],
        cal=cal,
    )
    pit_h = ledger.get_latest_holdings(cal[20], verify=True)
    agg = aggregate_positions_by_security(pit_h, cm, verify=True)
    assert len(agg) == 1
    assert agg.iloc[0]["security_id"] == "BBG001S7F4Z8"
    assert agg.iloc[0]["n_filers"] == 2
    assert np.isclose(agg.iloc[0]["total_shares_raw"], 400.0)
    assert agg.iloc[0]["cusips"] == "13645T100,13646K108"


def test_stage2_verify_catches_injected_xbrl_lookahead() -> None:
    """Explicitly confirm `verify_stage2_pit_snapshot` raises `PitLookaheadError` on leaked XBRL."""
    from src.pit.panel import verify_stage2_pit_snapshot

    qdt = pd.Timestamp("2024-08-16")
    bad = pd.DataFrame(
        [
            {
                "query_date": qdt,
                "security_id": "BBG001",
                "yahoo_ticker": "AAPL",
                "n_filers": 5,
                "total_shares_raw": 1000.0,
                "latest_filing_date": pd.Timestamp("2024-08-14"),
                "latest_acceptance_date": pd.Timestamp("2024-08-14"),
                "latest_effective_date": pd.Timestamp("2024-08-15"),
                "shares_filed_date": qdt,  # ILLEGAL: same day as query_date!
                "shares_end_date": pd.Timestamp("2024-06-30"),
                "has_market_price": True,
                "raw_close": 220.0,
                "total_shares_owned_pit": 1000.0,
                "is_seasoned": True,
                "pit_mcap_usd": 3e12,
                "has_unequal_class_ratio": False,
                "is_investable": True,
            }
        ]
    )
    with pytest.raises(PitLookaheadError, match="XBRL LOOKAHEAD"):
        verify_stage2_pit_snapshot(bad, qdt)


def test_dynamic_spy_thresholds_are_strictly_pit_with_zero_lookahead() -> None:
    """Dynamic SPY thresholds use SMA_60 ending t-1; same-day & future SPY jumps never leak."""
    cal = _make_trading_cal("2014-01-02", periods=600)
    # Build SPY: flat at 200 through 2014-12-31, then jumps to 400 on cal[300] (2x!)
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=200.0)
    # Make SPY = 400 starting at index 300 (long after 2014-12-31)
    spy.loc[300:, "raw_close"] = 400.0
    spy.loc[300:, "split_adj_close"] = 400.0
    # Inject a huge same-day jump on cal[400] (1000.0) to prove t's own close does NOT leak into t!
    spy_with_future = spy.copy()
    spy_with_future.loc[400:, "raw_close"] = 1000.0
    spy_with_future.loc[400:, "split_adj_close"] = 1000.0

    mp = _build_synthetic_market_prices("AAA", cal, base_raw_close=50.0)
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

    store_base = PitMarketStore(mp, spy, so, cm)
    store_future = PitMarketStore(mp, spy_with_future, so, cm)

    # At cal[400]: trailing 60d ending cal[399] is all 400.0 vs 200.0 on 2014-12-31 -> exact 2.0x!
    th_base = store_base.get_dynamic_thresholds(cal[400])
    th_future = store_future.get_dynamic_thresholds(cal[400])

    assert np.isclose(th_base["spy_threshold_scale"], 2.0)
    assert np.isclose(th_base["min_mcap_threshold_usd"], 2_000_000_000.0)
    assert np.isclose(th_base["min_adv_threshold_usd"], 10_000_000.0)
    # Same-day & future jump to 1000.0 on cal[400+] MUST NOT alter cal[400]'s threshold!
    assert np.isclose(th_future["spy_threshold_scale"], th_base["spy_threshold_scale"])
    assert np.isclose(th_future["min_mcap_threshold_usd"], th_base["min_mcap_threshold_usd"])


def test_pit_xbrl_unit_scale_outlier_scrubbing_has_zero_lookahead() -> None:
    """Strictly PIT XBRL scale scrubber drops 1e6x/1e3x/1e-3x typos with zero lookahead."""
    cal = _make_trading_cal("2020-01-02", periods=120)
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=300.0)
    # Stock price = $100, raw_volume = 500k -> adv_20d_usd = $50M/day
    mp_mega = _build_synthetic_market_prices("MEGA", cal, base_raw_close=100.0)
    mp_mid = _build_synthetic_market_prices("MIDC", cal, base_raw_close=100.0)
    mp_down = _build_synthetic_market_prices("DOWN", cal, base_raw_close=100.0)
    mp = pd.concat([mp_mega, mp_mid, mp_down], ignore_index=True)

    so = pd.DataFrame(
        [
            # 1) MEGA: Legitimate $2.5T mega-cap (25B shares * $100, adv=$50M -> turn=2e-5? wait
            #    make MEGA ADV high enough or shares=20B -> $2T & set MEGA raw_volume=100M below)
            {
                "yahoo_ticker": "MEGA",
                "issuer_cik": "0000000001",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 20_000_000_000.0,  # $2.0T legit mega-cap
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # 2) MIDC: Clean Q1 filing (50M shares -> $5B mcap, adv=$50M -> 1% turnover)
            {
                "yahoo_ticker": "MIDC",
                "issuer_cik": "0000000002",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 50_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # MIDC Q2: 1,000,000x typo on DEI tag + clean US-GAAP tag in SAME filing
            {
                "yahoo_ticker": "MIDC",
                "issuer_cik": "0000000002",
                "filed_date": cal[50],
                "end_date": cal[45],
                "shares_outstanding": 50_000_000_000_000.0,  # $5,000T typo!
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "MIDC",
                "issuer_cik": "0000000002",
                "filed_date": cal[50],
                "end_date": cal[45],
                "shares_outstanding": 51_000_000.0,  # Clean fallback tag
                "form": "10-Q",
                "tag_source": "us-gaap:CommonStockSharesOutstanding",
                "tag_priority": 2,
            },
            # MIDC Q3: Sub-$4T 1,000x typo (50B shares -> $5T? No, 30B shares -> $3T < $4T cap,
            # but adv=$50M / $3T = 1.67e-5 < 2.5e-5 -> caught by turnover guard!)
            {
                "yahoo_ticker": "MIDC",
                "issuer_cik": "0000000002",
                "filed_date": cal[80],
                "end_date": cal[75],
                "shares_outstanding": 30_000_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # 3) DOWN: Clean Q1 (80M shares -> $8B), then 1e-3x typo in Q2 (80,000 shares -> $8M,
            #    adv=$50M / $8M = 6.25 > 2.5 -> caught by downward turnover guard!)
            {
                "yahoo_ticker": "DOWN",
                "issuer_cik": "0000000003",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 80_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "DOWN",
                "issuer_cik": "0000000003",
                "filed_date": cal[50],
                "end_date": cal[45],
                "shares_outstanding": 80_000.0,  # 1e-3x typo!
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
        ]
    )
    # Give MEGA realistic mega-cap liquidity ($5B/day ADV -> 0.25% daily turnover)
    mp.loc[mp["yahoo_ticker"] == "MEGA", "adv_20d_usd"] = 5_000_000_000.0
    cm = pd.DataFrame(
        columns=["cusip", "security_id", "yahoo_ticker", "figi_name", "is_us_common_stock"]
    )

    store = PitMarketStore(mp, spy, so, cm)

    # 1) Legitimate mega-cap kept untouched
    mega_res = store.get_pit_shares_outstanding("MEGA", cal[30])
    assert np.isclose(mega_res["shares_out_pit"], 20_000_000_000.0)

    # 2a) MIDC Q2 DEI 1e6x typo dropped -> seamlessly falls back to clean US-GAAP tag (51M)
    mid_q2 = store.get_pit_shares_outstanding("MIDC", cal[60])
    assert np.isclose(mid_q2["shares_out_pit"], 51_000_000.0)

    # 2b) MIDC Q3 sub-$4T 1e3x typo dropped -> carries forward clean Q2 (51M)
    mid_q3 = store.get_pit_shares_outstanding("MIDC", cal[90])
    assert np.isclose(mid_q3["shares_out_pit"], 51_000_000.0)

    # 3) DOWN Q2 1e-3x typo dropped -> carries forward clean Q1 (80M)
    down_q2 = store.get_pit_shares_outstanding("DOWN", cal[60])
    assert np.isclose(down_q2["shares_out_pit"], 80_000_000.0)

    # 4) Zero lookahead: wild price/ADV spike strictly AFTER cal[50] must NOT alter cal[50] scrub
    mp_future = mp.copy()
    mask_fut = (mp_future["yahoo_ticker"] == "MIDC") & (mp_future["date"] > cal[50])
    mp_future.loc[mask_fut, "raw_close"] = 10_000.0
    store_fut = PitMarketStore(mp_future, spy, so, cm)
    assert np.isclose(
        store_fut.get_pit_shares_outstanding("MIDC", cal[60])["shares_out_pit"],
        mid_q2["shares_out_pit"],
    )


def test_foreign_20f_40f_filers_excluded_from_investable_universe() -> None:
    """20-F and 40-F foreign private issuers fail `is_domestic_sec_filer` and `is_investable`."""
    cal = _make_trading_cal("2022-01-03", periods=280)
    q_dt = cal[265]
    mp = pd.concat(
        [
            _build_synthetic_market_prices("USCO", cal, base_raw_close=100.0),
            _build_synthetic_market_prices("FORN", cal, base_raw_close=100.0),
            _build_synthetic_market_prices("CANC", cal, base_raw_close=100.0),
        ],
        ignore_index=True,
    )
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "USCO",
                "issuer_cik": "0001000001",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 100_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "FORN",
                "issuer_cik": "0001000002",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 100_000_000.0,
                "form": "20-F",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "CANC",
                "issuer_cik": "0001000003",
                "filed_date": cal[250],
                "end_date": cal[245],
                "shares_outstanding": 100_000_000.0,
                "form": "40-F",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "100000101",
                "security_id": "BBG_USCO",
                "yahoo_ticker": "USCO",
                "figi_name": "US CO",
                "is_us_common_stock": True,
            },
            {
                "cusip": "100000102",
                "security_id": "BBG_FORN",
                "yahoo_ticker": "FORN",
                "figi_name": "FOREIGN CO",
                "is_us_common_stock": True,
            },
            {
                "cusip": "100000103",
                "security_id": "BBG_CANC",
                "yahoo_ticker": "CANC",
                "figi_name": "CANADIAN CO",
                "is_us_common_stock": True,
            },
        ]
    )
    ledger = _make_ledger(
        filings=[
            {
                "accession_number": "F1",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[240],
                "date_filed": cal[255],
                "acceptance_date": cal[255],
                "acceptance_datetime": "20230101160000",
                "parse_status": "ok",
            }
        ],
        holdings=[
            {
                "accession_number": "F1",
                "cusip": c,
                "name_of_issuer": name,
                "title_of_class": "COM",
                "ssh_prnamt": 1_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 100_000_000.0,
            }
            for c, name in [
                ("100000101", "US CO"),
                ("100000102", "FOREIGN CO"),
                ("100000103", "CANADIAN CO"),
            ]
        ],
        cal=cal,
    )
    store = PitMarketStore(mp, spy, so, cm)
    snap = build_pit_security_snapshot(q_dt, ledger, store, verify=True).set_index("yahoo_ticker")

    assert bool(snap.loc["USCO", "is_domestic_sec_filer"]) is True
    assert bool(snap.loc["USCO", "is_investable"]) is True
    for foreign_tkr in ("FORN", "CANC"):
        assert bool(snap.loc[foreign_tkr, "is_domestic_sec_filer"]) is False
        assert bool(snap.loc[foreign_tkr, "is_investable"]) is False


def test_pit_xbrl_sequential_qoq_jump_scrubber_catches_zts_and_preserves_splits() -> None:
    """Sequential QoQ XBRL scrubber catches ZTS-style 20x drop & 100x spike while keeping splits."""
    cal = _make_trading_cal("2020-01-02", periods=160)
    spy = _build_synthetic_market_prices("SPY", cal, base_raw_close=300.0)

    # ZTS-like stock: flat $120 price, 478M shares -> sudden 24.3M typo (~20x drop) -> 476M recovery
    mp_zts = _build_synthetic_market_prices("ZTSX", cal, base_raw_close=120.0)
    # CIGI-like stock: flat $100 price, 35M shares ($3.5B) -> sudden 3.5B upward typo (100x jump)
    mp_up = _build_synthetic_market_prices("UPTX", cal, base_raw_close=100.0)
    # Genuine 10:1 forward split on cal[60] -> 10x share increase MUST be preserved!
    mp_spl = _build_synthetic_market_prices(
        "SPLT", cal, base_raw_close=1000.0, split_date=cal[60], split_factor=10.0
    )
    # Genuine 1:10 reverse split on cal[60] -> 0.1x share decrease MUST be preserved!
    mp_rev = _build_synthetic_market_prices(
        "REVS", cal, base_raw_close=5.0, split_date=cal[60], split_factor=0.1
    )
    mp = pd.concat([mp_zts, mp_up, mp_spl, mp_rev], ignore_index=True)

    so = pd.DataFrame(
        [
            # ZTSX: clean Q1 -> 20x downward typo in Q2 (adv/mcap = 60M / 2.9B = 2%, misses Pass 1!)
            {
                "yahoo_ticker": "ZTSX",
                "issuer_cik": "0001",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 478_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "ZTSX",
                "issuer_cik": "0001",
                "filed_date": cal[50],
                "end_date": cal[45],
                "shares_outstanding": 24_300_000.0,  # 20x typo without reverse split!
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "ZTSX",
                "issuer_cik": "0001",
                "filed_date": cal[90],
                "end_date": cal[85],
                "shares_outstanding": 476_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # UPTX: clean Q1 ($3.5B) -> 100x upward typo in Q2 ($350B, turn=0.014% misses Pass 1)
            {
                "yahoo_ticker": "UPTX",
                "issuer_cik": "0002",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 35_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "UPTX",
                "issuer_cik": "0002",
                "filed_date": cal[50],
                "end_date": cal[45],
                "shares_outstanding": 3_500_000_000.0,  # 100x jump on $3.5B issuer!
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # SPLT: 100M pre-split -> 1B post-split after genuine 10:1 split on cal[60]
            {
                "yahoo_ticker": "SPLT",
                "issuer_cik": "0003",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 100_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "SPLT",
                "issuer_cik": "0003",
                "filed_date": cal[80],
                "end_date": cal[75],
                "shares_outstanding": 1_000_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            # REVS: 100M pre-reverse-split -> 10M post-reverse-split after 1:10 on cal[60]
            {
                "yahoo_ticker": "REVS",
                "issuer_cik": "0004",
                "filed_date": cal[20],
                "end_date": cal[15],
                "shares_outstanding": 100_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
            {
                "yahoo_ticker": "REVS",
                "issuer_cik": "0004",
                "filed_date": cal[80],
                "end_date": cal[75],
                "shares_outstanding": 10_000_000.0,
                "form": "10-Q",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
            },
        ]
    )
    cm = pd.DataFrame(
        columns=["cusip", "security_id", "yahoo_ticker", "figi_name", "is_us_common_stock"]
    )
    store = PitMarketStore(mp, spy, so, cm)

    # ZTSX Q2 20x drop scrubbed -> carries forward 478M on cal[60], updates to 476M on cal[100]
    assert np.isclose(
        store.get_pit_shares_outstanding("ZTSX", cal[60])["shares_out_pit"], 478_000_000.0
    )
    assert np.isclose(
        store.get_pit_shares_outstanding("ZTSX", cal[100])["shares_out_pit"], 476_000_000.0
    )

    # UPTX Q2 100x jump scrubbed -> carries forward 35M
    assert np.isclose(
        store.get_pit_shares_outstanding("UPTX", cal[60])["shares_out_pit"], 35_000_000.0
    )

    # Genuine forward & reverse splits preserved cleanly
    assert np.isclose(
        store.get_pit_shares_outstanding("SPLT", cal[90])["shares_out_pit"], 1_000_000_000.0
    )
    assert np.isclose(
        store.get_pit_shares_outstanding("REVS", cal[90])["shares_out_pit"], 10_000_000.0
    )

    # Zero lookahead: corrupting ZTSX Q3 on cal[90] must not alter cal[60] output
    so_fut = so.copy()
    so_fut.loc[so_fut["filed_date"] == cal[90], "shares_outstanding"] = 999_000_000_000.0
    store_fut = PitMarketStore(mp, spy, so_fut, cm)
    assert np.isclose(
        store_fut.get_pit_shares_outstanding("ZTSX", cal[60])["shares_out_pit"],
        478_000_000.0,
    )


