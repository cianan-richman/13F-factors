"""Unit & adversarial 13F-HR/A amendment delta tests (`src.factors.concentration`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.concentration import (
    build_concentration_shift_exposures,
)
from src.pit.compute import Stage3MarketMetrics, build_pit_stage3_snapshot
from src.pit.ledger import PitHoldingsLedger
from src.pit.panel import PitMarketStore


def _make_simple_market_and_xbrl(
    cal: pd.DatetimeIndex,
) -> tuple[PitMarketStore, Stage3MarketMetrics]:
    rng = np.random.default_rng(101)
    n = len(cal)
    r_spy = rng.normal(0.0004, 0.01, size=n)
    r_spy[0] = 0.0
    r_aaa = 1.0 * r_spy + rng.normal(0.0, 0.004, size=n)
    r_aaa[0] = 0.0
    r_bbb = 1.0 * r_spy + rng.normal(0.0, 0.004, size=n)
    r_bbb[0] = 0.0

    def _mk_px(tkr: str, rets: np.ndarray, p0: float) -> pd.DataFrame:
        tr = p0 * np.cumprod(1.0 + rets)
        ret_t = np.full(n, np.nan)
        ret_t[1:] = rets[1:]
        return pd.DataFrame(
            {
                "date": cal,
                "yahoo_ticker": tkr,
                "raw_close": tr,
                "split_adj_close": tr,
                "raw_volume": np.full(n, 2_000_000.0),
                "dollar_volume": tr * 2_000_000.0,
                "split_factor": np.ones(n),
                "cum_split_to_latest": np.ones(n),
                "raw_dividend": np.zeros(n),
                "ret_total": ret_t,
                "adv_20d_usd": tr * 2_000_000.0,
            }
        )

    mp = pd.concat([_mk_px("AAA", r_aaa, 50.0), _mk_px("BBB", r_bbb, 80.0)], ignore_index=True)
    spy = _mk_px("SPY", r_spy, 400.0)
    so = pd.DataFrame(
        [
            {
                "yahoo_ticker": "AAA",
                "issuer_cik": "0001111111",
                "entity_name": "AAA CORP",
                "filed_date": cal[200],
                "end_date": cal[195],
                "shares_outstanding": 10_000_000.0,  # 10M shares -> 100k = 1.0%
                "form": "10-Q",
                "accession_number": "SO_A",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            },
            {
                "yahoo_ticker": "BBB",
                "issuer_cik": "0002222222",
                "entity_name": "BBB CORP",
                "filed_date": cal[200],
                "end_date": cal[195],
                "shares_outstanding": 10_000_000.0,
                "form": "10-Q",
                "accession_number": "SO_B",
                "tag_source": "dei:EntityCommonStockSharesOutstanding",
                "tag_priority": 1,
                "n_classes_reported": 1,
            },
        ]
    )
    cm = pd.DataFrame(
        [
            {
                "cusip": "111111101",
                "security_id": "BBG_AAA",
                "yahoo_ticker": "AAA",
                "figi_name": "AAA CORP",
                "is_us_common_stock": True,
            },
            {
                "cusip": "222222102",
                "security_id": "BBG_BBB",
                "yahoo_ticker": "BBB",
                "figi_name": "BBB CORP",
                "is_us_common_stock": True,
            },
        ]
    )
    store = PitMarketStore(mp, spy, so, cm)
    s3 = Stage3MarketMetrics(store)
    return store, s3


def test_q_curr_and_q_prev_amendments_recompute_exact_delta_pct_own() -> None:
    """Verify BOTH Q_curr RESTATEMENT/CTR and retroactive Q_prev 13F-HR/A recompute exact deltas.

    Timeline on synthetic 10M-share issuer AAA (and CTR stock BBB):
      1. Q1 (cal[210]) filed cal[220]: AAA = 100k (1.0%), BBB = 50k (0.5%)
      2. Q2 (cal[255]) filed cal[265]: AAA = 200k (2.0%), BBB omitted (CTR)
         -> On cal[266]: delta(AAA) = +100k (+1.0%), delta(BBB) = -50k (-0.5%)
      3. Q2 RESTATEMENT filed cal[275]: AAA restated to 250k (2.5%)
         -> On cal[275] (day T): STILL +100k (+1.0%) — zero same-day lookahead!
         -> On cal[276] (T+1): delta(AAA) = +150k (+1.5% = 250k - 100k), NEVER +50k (250k - 200k)!
      4. Retroactive Q1 RESTATEMENT filed cal[285]: Q1 AAA restated from 100k -> 140k (1.4%)
         -> On cal[285] (day T): STILL +150k (+1.5%)!
         -> On cal[286] (T+1): Q_curr=250k, Q_prev=140k -> delta(AAA)=+110k (+1.1%)!
      5. Q2 NEW HOLDINGS (CTR reveal) filed cal[292]: reveals BBB = 80k (0.8%) in Q2
         -> On cal[293] (T+1): AAA stays +110k, BBB flips to +30k (+0.3% = 80k - 50k)!
    """
    cal = pd.bdate_range("2022-01-03", periods=310)
    store, s3 = _make_simple_market_and_xbrl(cal)

    meta = pd.DataFrame(
        [
            {
                "accession_number": "Q1_BASE",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[210],
                "date_filed": cal[220],
                "acceptance_date": cal[220],
                "acceptance_datetime": "20221107160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "Q2_BASE",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR",
                "amendment_type": "",
                "period_of_report": cal[255],
                "date_filed": cal[265],
                "acceptance_date": cal[265],
                "acceptance_datetime": "20230110160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "Q2_RESTATE",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR/A",
                "amendment_type": "RESTATEMENT",
                "period_of_report": cal[255],
                "date_filed": cal[275],
                "acceptance_date": cal[275],
                "acceptance_datetime": "20230124160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "Q1_RETRO_RESTATE",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR/A",
                "amendment_type": "RESTATEMENT",
                "period_of_report": cal[210],
                "date_filed": cal[285],
                "acceptance_date": cal[285],
                "acceptance_datetime": "20230207160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "Q2_CTR_REVEAL",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR/A",
                "amendment_type": "NEW HOLDINGS",
                "period_of_report": cal[255],
                "date_filed": cal[292],
                "acceptance_date": cal[292],
                "acceptance_datetime": "20230216160000",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "accession_number": "Q1_BASE",
                "cusip": "111111101",
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 5_000_000.0,
            },
            {
                "accession_number": "Q1_BASE",
                "cusip": "222222102",
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
                "ssh_prnamt": 50_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 4_000_000.0,
            },
            {
                "accession_number": "Q2_BASE",
                "cusip": "111111101",
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
                "ssh_prnamt": 200_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000_000.0,
            },
            {
                "accession_number": "Q2_RESTATE",
                "cusip": "111111101",
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
                "ssh_prnamt": 250_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 12_500_000.0,
            },
            {
                "accession_number": "Q1_RETRO_RESTATE",
                "cusip": "111111101",
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
                "ssh_prnamt": 140_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 7_000_000.0,
            },
            {
                "accession_number": "Q1_RETRO_RESTATE",
                "cusip": "222222102",
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
                "ssh_prnamt": 50_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 4_000_000.0,
            },
            {
                "accession_number": "Q2_CTR_REVEAL",
                "cusip": "222222102",
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
                "ssh_prnamt": 80_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 6_400_000.0,
            },
        ]
    )
    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    def _delta_pct(dt: pd.Timestamp, tkr: str) -> float:
        snap = build_pit_stage3_snapshot(dt, ledger, store, s3, verify=True)
        row = snap[snap["yahoo_ticker"] == tkr].iloc[0]
        return float(row["issuer_delta_shares_pit"] / row["shares_out_pit"])

    # 1. Post-Q2 base (200k - 100k = +1.0%)
    assert np.isclose(_delta_pct(cal[266], "AAA"), 0.010, atol=1e-12)
    # 2. On Q2_RESTATE filing date T (zero lookahead: still +1.0%), then T+1 (+1.5%, NOT +0.5%!)
    assert np.isclose(_delta_pct(cal[275], "AAA"), 0.010, atol=1e-12)
    assert np.isclose(_delta_pct(cal[276], "AAA"), 0.015, atol=1e-12)
    # 3. On Q1_RETRO_RESTATE filing date T (still +1.5%), then T+1 (+1.1% = 250k - 140k!)
    assert np.isclose(_delta_pct(cal[285], "AAA"), 0.015, atol=1e-12)
    assert np.isclose(_delta_pct(cal[286], "AAA"), 0.011, atol=1e-12)
    # 4. On Q2_CTR_REVEAL T+1: AAA stays +1.1% and newly revealed BBB has +0.3% (80k - 50k)
    assert np.isclose(_delta_pct(cal[293], "AAA"), 0.011, atol=1e-12)
    assert np.isclose(_delta_pct(cal[293], "BBB"), 0.003, atol=1e-12)


def test_concentration_exposures_standardized_and_size_neutral() -> None:
    """`delta_pct_own_sn` and alt shift exposures have mean=0, std=1, size corr=0 daily."""
    rng = np.random.default_rng(77)
    n_days, n_stocks = 8, 50
    cal = pd.bdate_range("2022-01-03", periods=n_days + 1)
    mcap = np.exp(rng.normal(23.0, 1.2, size=n_stocks))
    rows = []
    for t in range(n_days):
        so = mcap / 50.0
        pct = np.clip(rng.lognormal(-3.2, 0.6, size=n_stocks), 1e-4, 0.3)
        dsh = rng.normal(0.0, 0.01, size=n_stocks) * so
        for i in range(n_stocks):
            rows.append(
                {
                    "query_date": cal[t],
                    "fwd_ret_date": cal[t + 1],
                    "security_id": f"S{i:03d}",
                    "yahoo_ticker": f"T{i:03d}",
                    "n_filers": 5,
                    "pct_shares_owned": float(pct[i]),
                    "days_to_liquidate": float(pct[i] * 100.0),
                    "shares_out_pit": float(so[i]),
                    "pit_mcap_usd": float(mcap[i]),
                    "adv_20d_usd": float(mcap[i] * 0.01),
                    "issuer_delta_shares_pit": float(dsh[i]),
                    "issuer_total_shares_owned_pit": float(pct[i] * so[i]),
                    "n_flow_filers": 5,
                    "n_buyers": int(rng.integers(0, 4)),
                    "n_sellers": int(rng.integers(0, 4)),
                    "mean_active_weight_shift": float(rng.normal(0.0, 0.005)),
                    "delta_holder_hhi_pit": float(rng.normal(0.0, 0.1)),
                    "beta_252d_lag1": 1.0,
                    "fwd_day_ret": 0.001,
                    "fwd_spy_ret": 0.001,
                    "fwd_beta_adj_ret": 0.0,
                    "is_panel_eligible": True,
                }
            )
    out = build_concentration_shift_exposures(pd.DataFrame(rows))
    for _dt, grp in out.groupby("query_date"):
        lm = grp["log_mcap"].to_numpy(dtype=np.float64)
        for col in ("delta_pct_own_sn", "active_w_shift_sn", "delta_hhi_sn"):
            v = grp[col].to_numpy(dtype=np.float64)
            assert np.isclose(np.mean(v), 0.0, atol=1e-12)
            assert np.isclose(np.std(v, ddof=0), 1.0, atol=1e-12)
            assert np.isclose(np.corrcoef(v, lm)[0, 1], 0.0, atol=1e-11)


def test_ctr_reveal_masking_and_confidential_omitted_propagation() -> None:
    """Verify `is_confidential_omitted` propagates to Stage 3 and CTR spikes are masked."""
    cal = pd.bdate_range("2022-01-03", periods=310)
    store, s3 = _make_simple_market_and_xbrl(cal)

    meta = pd.DataFrame(
        [
            {
                "accession_number": "Q1_BASE",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR",
                "amendment_type": "",
                "is_confidential_omitted": True,
                "period_of_report": cal[210],
                "date_filed": cal[220],
                "acceptance_date": cal[220],
                "acceptance_datetime": "20221107160000",
                "parse_status": "ok",
            },
            {
                "accession_number": "Q1_CTR",
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "value",
                "form_type": "13F-HR/A",
                "amendment_type": "NEW HOLDINGS",
                "is_confidential_omitted": False,
                "period_of_report": cal[210],
                "date_filed": cal[270],
                "acceptance_date": cal[270],
                "acceptance_datetime": "20230117160000",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "accession_number": "Q1_BASE",
                "cusip": "111111101",
                "name_of_issuer": "AAA",
                "title_of_class": "COM",
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 5_000_000.0,
            },
            {
                "accession_number": "Q1_CTR",
                "cusip": "222222102",
                "name_of_issuer": "BBB",
                "title_of_class": "COM",
                "ssh_prnamt": 80_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 6_400_000.0,
            },
        ]
    )
    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    # Before CTR reveal: AAA carries has_confidential_omitted_filer=True
    snap_before = build_pit_stage3_snapshot(cal[268], ledger, store, s3, verify=True).set_index(
        "yahoo_ticker"
    )
    assert bool(snap_before.loc["AAA", "has_confidential_omitted_filer"]) is True
    assert bool(snap_before.loc["AAA", "has_ctr_reveal"]) is False

    # After CTR reveal T+1: confidential omission resolved, BBB tagged has_ctr_reveal=True
    snap_after = build_pit_stage3_snapshot(cal[271], ledger, store, s3, verify=True).set_index(
        "yahoo_ticker"
    )
    assert bool(snap_after.loc["AAA", "has_confidential_omitted_filer"]) is False
    assert bool(snap_after.loc["BBB", "has_ctr_reveal"]) is True

    # Exposure builder masks CTR-reveal spikes when exclude_ctr_reveals=True
    rng = np.random.default_rng(12)
    n_stocks = 30
    mcap = np.exp(rng.normal(23.0, 1.0, size=n_stocks))
    so = mcap / 50.0
    rows = []
    for i in range(n_stocks):
        is_ctr = i == 0
        delta_sh = float(0.15 * so[i] if is_ctr else 0.002 * so[i] * (i - 15))
        rows.append(
            {
                "query_date": cal[271],
                "fwd_ret_date": cal[272],
                "security_id": f"S{i:02d}",
                "yahoo_ticker": f"T{i:02d}",
                "n_filers": 5,
                "pct_shares_owned": 0.05,
                "days_to_liquidate": 5.0,
                "shares_out_pit": float(so[i]),
                "pit_mcap_usd": float(mcap[i]),
                "adv_20d_usd": float(mcap[i] * 0.01),
                "issuer_delta_shares_pit": delta_sh,
                "issuer_total_shares_owned_pit": float(0.05 * so[i]),
                "n_flow_filers": 5,
                "n_buyers": 3,
                "n_sellers": 1,
                "mean_active_weight_shift": 0.05 if is_ctr else 0.001,
                "delta_holder_hhi_pit": 0.1 if is_ctr else 0.01,
                "has_ctr_reveal": is_ctr,
                "has_confidential_omitted_filer": False,
                "beta_252d_lag1": 1.0,
                "fwd_day_ret": 0.001,
                "fwd_spy_ret": 0.001,
                "fwd_beta_adj_ret": 0.0,
                "is_panel_eligible": True,
            }
        )
    df_in = pd.DataFrame(rows)
    out_ex = build_concentration_shift_exposures(df_in, exclude_ctr_reveals=True)
    out_inc = build_concentration_shift_exposures(df_in, exclude_ctr_reveals=False)

    ctr_ex = out_ex[out_ex["security_id"] == "S00"].iloc[0]
    assert bool(ctr_ex["is_ctr_excluded"]) is True
    assert np.isclose(ctr_ex["delta_pct_own_raw"], 0.15)
    assert pd.isna(ctr_ex["delta_pct_own"])
    assert pd.isna(ctr_ex["delta_pct_own_sn"])
    assert pd.isna(ctr_ex["active_w_shift_sn"])
    assert pd.isna(ctr_ex["net_buyer_ratio_sn"])

    ctr_inc = out_inc[out_inc["security_id"] == "S00"].iloc[0]
    assert bool(ctr_inc["is_ctr_excluded"]) is False
    assert np.isclose(ctr_inc["delta_pct_own"], 0.15)
    assert pd.notna(ctr_inc["delta_pct_own_sn"])

