"""Unit & adversarial PIT tests for Signal 3: Smart Money Following (Report-Date Dollars)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.smart_money import (
    audit_report_date_dollars_qa,
    build_daily_report_date_dollars,
    build_smart_money_exposures,
    compute_state_report_date_dollars,
)
from src.pit.ledger import PitHoldingsLedger
from src.pit.panel import PitMarketStore


def _make_prices(
    ticker: str,
    dates: pd.DatetimeIndex,
    raw_closes: np.ndarray,
    split_factors: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(dates)
    s_t = np.ones(n, dtype=np.float64) if split_factors is None else split_factors.astype(float)
    rev = np.cumprod(s_t[::-1])[::-1]
    cum_fut = np.ones(n, dtype=np.float64)
    cum_fut[:-1] = rev[1:]
    raw_close = np.asarray(raw_closes, dtype=np.float64)
    split_adj_close = raw_close / cum_fut
    ret_total = np.full(n, np.nan, dtype=np.float64)
    ret_total[1:] = (raw_close[1:] * s_t[1:]) / raw_close[:-1] - 1.0
    raw_vol = np.full(n, 500_000.0, dtype=np.float64)
    dvol = raw_close * raw_vol
    return pd.DataFrame(
        {
            "date": dates,
            "yahoo_ticker": ticker,
            "raw_close": raw_close,
            "split_adj_close": split_adj_close,
            "raw_volume": raw_vol,
            "dollar_volume": dvol,
            "split_factor": s_t,
            "cum_split_to_latest": cum_fut,
            "raw_dividend": np.zeros(n, dtype=np.float64),
            "ret_total": ret_total,
            "adv_20d_usd": dvol,
        }
    )


def _empty_shares() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "yahoo_ticker",
            "issuer_cik",
            "filed_date",
            "end_date",
            "shares_outstanding",
            "tag_priority",
        ]
    )


def test_weekend_report_date_uses_friday_price_with_zero_lookahead() -> None:
    """Saturday period_of_report (2023-09-30) MUST price at Friday 2023-09-29, never Monday."""
    # Business days spanning Q3 end 2023-09-30 (Sat): Fri 2023-09-29 -> Mon 2023-10-02
    cal = pd.bdate_range("2023-09-25", "2023-11-20")
    closes = np.full(len(cal), 100.0)
    # Set Fri 2023-09-29 to $100 and Mon 2023-10-02 onwards to $250 (huge jump after quarter-end)
    mon_idx = cal.get_loc(pd.Timestamp("2023-10-02"))
    closes[mon_idx:] = 250.0

    mp = _make_prices("AAA", cal, closes)
    spy = _make_prices("SPY", cal, np.full(len(cal), 430.0))
    cm = pd.DataFrame(
        [
            {
                "cusip": "000000101",
                "security_id": "SEC_AAA",
                "yahoo_ticker": "AAA",
                "figi_name": "AAA CORP",
                "is_us_common_stock": True,
            }
        ]
    )
    store = PitMarketStore(mp, spy, _empty_shares(), cm)

    filings = pd.DataFrame(
        [
            {
                "filer_id": "F1",
                "canonical_name": "Fund One",
                "style": "fundamental",
                "period_of_report": "2023-09-30",  # Saturday!
                "date_filed": "2023-11-10",
                "acceptance_date": "2023-11-10",
                "acceptance_datetime": "20231110160000",
                "accession_number": "A1",
                "form_type": "13F-HR",
                "amendment_type": "",
                "parse_status": "ok",
            }
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "accession_number": "A1",
                "cusip": "000000101",
                "ssh_prnamt": 10_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 1_000_000.0,
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
            }
        ]
    )
    ledger = PitHoldingsLedger.from_raw_tables(filings, holdings, trading_calendar=cal)

    sd = compute_state_report_date_dollars(ledger, store, min_report_date="2023-01-01")
    assert len(sd) == 1
    row = sd.iloc[0]
    assert pd.Timestamp(row["price_date_rpt"]) == pd.Timestamp("2023-09-29")
    assert int(row["price_lag_days"]) == 1
    assert np.isclose(row["raw_close_rpt"], 100.0)
    assert np.isclose(row["dollars_rpt"], 1_000_000.0)

    qa = audit_report_date_dollars_qa(sd)
    assert np.isclose(qa["ratio_p50"], 1.0)
    assert qa["max_price_lag_days"] == 1


def test_staggered_filers_and_restatement_use_each_filers_own_report_date_price() -> None:
    """Two filers on Q2 vs Q3 price at their own report_date; 13F-HR/A replaces on T+1."""
    cal = pd.bdate_range("2023-06-26", "2023-11-22")
    closes = np.full(len(cal), 50.0)
    # Q2 end (2023-06-30) raw_close = $50; Q3 end (2023-09-29) raw_close = $80
    # Plus a 2:1 split on 2023-10-16 (after Q3 end, before filing!) -> raw_close drops to $40
    q3_idx = cal.get_loc(pd.Timestamp("2023-09-29"))
    split_idx = cal.get_loc(pd.Timestamp("2023-10-16"))
    closes[q3_idx:split_idx] = 80.0
    closes[split_idx:] = 40.0
    sf = np.ones(len(cal), dtype=np.float64)
    sf[split_idx] = 2.0

    mp = _make_prices("BBB", cal, closes, split_factors=sf)
    spy = _make_prices("SPY", cal, np.full(len(cal), 430.0))
    cm = pd.DataFrame(
        [
            {
                "cusip": "000000202",
                "security_id": "SEC_BBB",
                "yahoo_ticker": "BBB",
                "figi_name": "BBB CORP",
                "is_us_common_stock": True,
            }
        ]
    )
    store = PitMarketStore(mp, spy, _empty_shares(), cm)

    filings = pd.DataFrame(
        [
            # Filer A files Q2 on 2023-08-14 (1,000 sh * $50 = $50k), never files Q3 yet
            {
                "filer_id": "FA",
                "canonical_name": "Fund A",
                "style": "fundamental",
                "period_of_report": "2023-06-30",
                "date_filed": "2023-08-14",
                "acceptance_date": "2023-08-14",
                "acceptance_datetime": "20230814160000",
                "accession_number": "A_Q2",
                "form_type": "13F-HR",
                "amendment_type": "",
                "parse_status": "ok",
            },
            # Filer B files Q3 on 2023-11-10 (botched 10,000 sh * $80 = $800k)
            {
                "filer_id": "FB",
                "canonical_name": "Fund B",
                "style": "fundamental",
                "period_of_report": "2023-09-30",
                "date_filed": "2023-11-10",
                "acceptance_date": "2023-11-10",
                "acceptance_datetime": "20231110160000",
                "accession_number": "B_Q3_BAD",
                "form_type": "13F-HR",
                "amendment_type": "",
                "parse_status": "ok",
            },
            # Filer B amends Q3 via RESTATEMENT on 2023-11-14 (fixed 1,000 sh * $80 = $80k)
            {
                "filer_id": "FB",
                "canonical_name": "Fund B",
                "style": "fundamental",
                "period_of_report": "2023-09-30",
                "date_filed": "2023-11-14",
                "acceptance_date": "2023-11-14",
                "acceptance_datetime": "20231114160000",
                "accession_number": "B_Q3_FIX",
                "form_type": "13F-HR/A",
                "amendment_type": "RESTATEMENT",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "accession_number": "A_Q2",
                "cusip": "000000202",
                "ssh_prnamt": 1_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 50_000.0,
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
            },
            {
                "accession_number": "B_Q3_BAD",
                "cusip": "000000202",
                "ssh_prnamt": 10_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 800_000.0,
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
            },
            {
                "accession_number": "B_Q3_FIX",
                "cusip": "000000202",
                "ssh_prnamt": 1_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 80_000.0,
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
            },
        ]
    )
    ledger = PitHoldingsLedger.from_raw_tables(filings, holdings, trading_calendar=cal)
    sd = compute_state_report_date_dollars(ledger, store, min_report_date="2023-01-01")

    q_dates = pd.DatetimeIndex(
        [pd.Timestamp("2023-11-13"), pd.Timestamp("2023-11-14"), pd.Timestamp("2023-11-15")]
    )
    daily = build_daily_report_date_dollars(q_dates, ledger, sd).set_index("query_date")

    # On 2023-11-13 & 2023-11-14 (before restatement effective T+1 = 2023-11-15):
    #   FA (Q2: 1000 * $50 = $50k) + FB_BAD (Q3: 10000 * $80 = $800k, unaffected by Oct 16 split!)
    assert np.isclose(daily.loc[pd.Timestamp("2023-11-13"), "dollars_rpt"], 850_000.0)
    assert np.isclose(daily.loc[pd.Timestamp("2023-11-14"), "dollars_rpt"], 850_000.0)

    # On 2023-11-15 (T+1 after 13F-HR/A RESTATEMENT): FA ($50k) + FB_FIX ($80k) = $130k!
    assert np.isclose(daily.loc[pd.Timestamp("2023-11-15"), "dollars_rpt"], 130_000.0)


def test_smart_money_exposures_size_neutrality() -> None:
    """Size-neutralized report-date dollar factors have zero correlation with log(pit_mcap_usd)."""
    rng = np.random.default_rng(7)
    n = 120
    mcap = np.exp(rng.normal(23.0, 1.5, size=n))
    dollars = mcap * rng.uniform(0.005, 0.12, size=n)
    df = pd.DataFrame(
        {
            "query_date": pd.Timestamp("2022-06-15"),
            "security_id": [f"S{i}" for i in range(n)],
            "pit_mcap_usd": mcap,
            "issuer_dollars_invested_rpt": dollars,
            "mean_w_rpt": rng.uniform(0.005, 0.08, size=n),
            "max_w_rpt": rng.uniform(0.01, 0.12, size=n),
            "fwd_beta_adj_ret": rng.normal(0.0, 0.015, size=n),
        }
    )
    out = build_smart_money_exposures(df)
    lm = np.log(out["pit_mcap_usd"])
    # Raw dollars has strong positive size correlation; size-neutral versions are exact 0.0
    assert out["dollars_rpt_z"].corr(lm) > 0.5
    assert abs(out["dollars_rpt_sn"].corr(lm)) < 1e-12
    assert abs(out["log_dollars_rpt_sn"].corr(lm)) < 1e-12
    assert abs(out["active_ow_rpt_sn"].corr(lm)) < 1e-12
