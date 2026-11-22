"""Unit & adversarial PIT tests for Rolling Manager-Skill Smart Money (`manager_skill`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.manager_skill import (
    build_clean_stock_return_matrices,
    build_static_quarterly_manager_rankings,
    compute_drifted_state_quarter_returns,
)
from src.factors.smart_money import compute_state_report_date_dollars
from src.pit.compute import Stage3MarketMetrics
from src.pit.ledger import PitHoldingsLedger
from src.pit.panel import PitMarketStore


def _make_synthetic_market(cal: pd.DatetimeIndex) -> tuple[PitMarketStore, Stage3MarketMetrics]:
    """Build 6 synthetic US stocks + SPY with known clean returns and 1 injected splice."""
    rng = np.random.default_rng(42)
    n = len(cal)
    r_spy = rng.normal(0.0003, 0.009, size=n)
    r_spy[0] = 0.0

    def _mk(tkr: str, alpha_daily: float, beta: float = 1.0) -> pd.DataFrame:
        r = beta * r_spy + alpha_daily + rng.normal(0.0, 0.004, size=n)
        r[0] = 0.0
        px = 100.0 * np.cumprod(1.0 + r)
        rt = np.full(n, np.nan)
        rt[1:] = r[1:]
        return pd.DataFrame(
            {
                "date": cal,
                "yahoo_ticker": tkr,
                "raw_close": px,
                "split_adj_close": px,
                "raw_volume": np.full(n, 1_000_000.0),
                "dollar_volume": px * 1_000_000.0,
                "split_factor": np.ones(n),
                "cum_split_to_latest": np.ones(n),
                "raw_dividend": np.zeros(n),
                "ret_total": rt,
                "adv_20d_usd": px * 1_000_000.0,
            }
        )

    # 5 winners (+8 bps/d alpha) + 5 losers (-8 bps/d alpha) so each fund holds >= 5 stocks
    frames = [_mk(f"WIN{i}", +0.0008, beta=1.0) for i in range(5)] + [
        _mk(f"LOS{i}", -0.0008, beta=1.0) for i in range(5)
    ]
    mp = pd.concat(frames, ignore_index=True)
    # Inject a fake +1000% Chapter-11 splice in LOS0 on one day to verify masking!
    splice_dt = cal[320]
    mp.loc[(mp["yahoo_ticker"] == "LOS0") & (mp["date"] == splice_dt), "ret_total"] = 10.0

    spy = _mk("SPY", 0.0, beta=1.0)
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
        [
            {
                "cusip": f"10000010{i}",
                "security_id": f"SEC_WIN{i}",
                "yahoo_ticker": f"WIN{i}",
                "figi_name": f"WIN {i}",
                "is_us_common_stock": True,
            }
            for i in range(5)
        ]
        + [
            {
                "cusip": f"20000020{i}",
                "security_id": f"SEC_LOS{i}",
                "yahoo_ticker": f"LOS{i}",
                "figi_name": f"LOS {i}",
                "is_us_common_stock": True,
            }
            for i in range(5)
        ]
    )
    store = PitMarketStore(mp, spy, so, cm)
    return store, Stage3MarketMetrics(store)


def test_static_quarterly_ranking_excludes_current_basket_and_masks_splices() -> None:
    """Verify Double PIT Gate, splice masking, and strict exclusion of current quarter basket."""
    # Span 2019-01-01 -> 2022-06-30 so beta_252d_lag1 is warm from 2020-01-02 onwards
    cal = pd.bdate_range("2019-01-02", "2022-06-30")
    store, s3 = _make_synthetic_market(cal)

    # Create 6 historical quarterly filings for Fund_Good (holds WIN0..4) & Fund_Bad (holds LOS0..4)
    # Report dates: 2019-12-31, 2020-03-31, 2020-06-30, 2020-09-30, 2020-12-31, 2021-03-31
    rep_dates = pd.date_range("2019-12-31", "2021-03-31", freq="QE")
    filings_rows = []
    holdings_rows = []
    for k, rep in enumerate(rep_dates):
        fdt = (rep + pd.Timedelta(days=40)).normalize()
        for fid, cname, cusips in [
            ("fund_good", "Good Fund", [f"10000010{i}" for i in range(5)]),
            ("fund_bad", "Bad Fund", [f"20000020{i}" for i in range(5)]),
        ]:
            acc = f"{fid}_Q{k}"
            filings_rows.append(
                {
                    "accession_number": acc,
                    "filer_id": fid,
                    "canonical_name": cname,
                    "style": "quality_generalist",
                    "form_type": "13F-HR",
                    "amendment_type": "",
                    "period_of_report": rep,
                    "date_filed": fdt,
                    "acceptance_date": fdt,
                    "acceptance_datetime": fdt.strftime("%Y%m%d") + "160000",
                    "parse_status": "ok",
                }
            )
            for c in cusips:
                holdings_rows.append(
                    {
                        "accession_number": acc,
                        "cusip": c,
                        "name_of_issuer": c,
                        "title_of_class": "COM",
                        "ssh_prnamt": 10_000.0,
                        "ssh_prnamt_type": "SH",
                        "put_call": "",
                        "value_usd": 1_000_000.0,
                    }
                )

    # Also add a LATE filing for 2020-12-31 (filed 2021-05-01, AFTER 2021-Q2 start 2021-04-01!)
    # to prove that a late basket cannot leak into 2021-Q2 rankings!
    filings_rows.append(
        {
            "accession_number": "fund_late_Q4",
            "filer_id": "fund_late",
            "canonical_name": "Late Fund",
            "style": "quality_generalist",
            "form_type": "13F-HR",
            "amendment_type": "",
            "period_of_report": pd.Timestamp("2020-12-31"),
            "date_filed": pd.Timestamp("2021-05-01"),
            "acceptance_date": pd.Timestamp("2021-05-01"),
            "acceptance_datetime": "20210501160000",
            "parse_status": "ok",
        }
    )
    for i in range(5):
        holdings_rows.append(
            {
                "accession_number": "fund_late_Q4",
                "cusip": f"10000010{i}",
                "name_of_issuer": f"WIN{i}",
                "title_of_class": "COM",
                "ssh_prnamt": 10_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 1_000_000.0,
            }
        )

    ledger = PitHoldingsLedger.from_raw_tables(
        pd.DataFrame(filings_rows), pd.DataFrame(holdings_rows), trading_calendar=cal
    )
    ret_mats = build_clean_stock_return_matrices(
        store, s3, start_date="2020-01-02", end_date="2021-06-30"
    )
    # 1 injected +1000% splice MUST be caught and masked!
    assert ret_mats["n_extreme_stock_days_masked"] == 1

    sd = compute_state_report_date_dollars(ledger, store, min_report_date="2019-12-31")
    sq_ret, fund_qa = compute_drifted_state_quarter_returns(
        ledger, sd, ret_mats, min_report_date="2019-12-31", max_eval_end_date="2021-06-30"
    )
    assert fund_qa["states_evaluated"] >= 10

    # Evaluate on 2021-Q2 (2021-04-01) with 4-quarter lookback:
    # Trailing completed quarters end 2021-03-31 -> baskets 2019-12-31, 2020-03-31, 2020-06-30,
    # 2020-09-30 (strictly EXCLUDING current live basket 2020-12-31 / 2021-03-31!).
    sc, r_qa = build_static_quarterly_manager_rankings(
        ledger,
        sq_ret,
        ret_mats["cal"],
        lookback_qtrs=4,
        min_qtrs=3,
        start_eval_quarter="2021-04-01",
        end_eval_quarter="2021-04-01",
    )
    assert r_qa["pit_violations"] == 0
    assert set(sc["filer_id"]) == {"fund_good", "fund_bad"}
    # Late fund must NOT appear (its only filing wasn't filed until 2021-05-01 > 2021-04-01!)
    assert "fund_late" not in set(sc["filer_id"])

    # Because LOS0's +1000% splice was masked, fund_good MUST strictly beat fund_bad in IR_badj!
    ir_good = float(sc.loc[sc["filer_id"] == "fund_good", "ir_badj"].iloc[0])
    ir_bad = float(sc.loc[sc["filer_id"] == "fund_bad", "ir_badj"].iloc[0])
    assert ir_good > 0.0
    assert ir_bad < 0.0
    assert ir_good > ir_bad
