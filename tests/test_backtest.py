"""Unit & invariant tests for the rack-based backtester (`src.backtest.engine`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.backtest.engine import (
    BacktestConfig,
    BacktestRacks,
    compute_quarterly_rebalance_mask,
    compute_spy_10d_regime_scale,
    run_rack_backtest,
)


def _make_synthetic_racks(n_days: int = 10) -> BacktestRacks:
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    fwd_dates = pd.bdate_range("2022-01-04", periods=n_days)
    secs = pd.Index(["S1", "S2"])

    # Constant expected return: S1 = -2.0 bps/d, S2 = -1.0 bps/d
    er = np.tile(np.array([[-2.0, -1.0]]), (n_days, 1))
    # Known stock total returns & SPY returns
    fwd_spy = np.full(n_days, 0.01, dtype=np.float64)
    prev_spy = np.full(n_days, 0.005, dtype=np.float64)
    beta = np.tile(np.array([[1.2, 0.8]]), (n_days, 1))
    # Stock returns = beta * spy - 0.002 (so beta-adj return = -20 bps/d -> short makes +20 bps/d!)
    fwd_raw = beta * fwd_spy[:, None] - 0.002
    fwd_badj = fwd_raw - beta * fwd_spy[:, None]
    prev_raw = np.full((n_days, 2), 0.01, dtype=np.float64)
    # S1 has $10M ADV (5% cap = $500k), S2 has $500k ADV (5% cap = $25k!)
    adv = np.tile(np.array([[10_000_000.0, 500_000.0]]), (n_days, 1))

    return BacktestRacks(
        dates=dates,
        fwd_ret_dates=fwd_dates,
        securities=secs,
        expected_ret_bps=er,
        fwd_beta_adj_ret=fwd_badj,
        fwd_day_ret=fwd_raw,
        prev_day_ret=prev_raw,
        beta_lag1=beta,
        adv_20d_usd=adv,
        fwd_spy_ret=fwd_spy,
        prev_spy_ret=prev_spy,
    )


def test_exact_beta_hedge_identity_and_adv_cap() -> None:
    """Verify 5% ADV cap and exact identity: pnl_raw == stock_raw_pnl + spy_hedge_pnl."""
    racks = _make_synthetic_racks(n_days=6)
    # S1 target=-$100k (<$500k cap), S2 target=-$50k -> capped at -$25k (5% of $500k ADV)
    cfg = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=5.0,
        spy_hedge_cost_bps=1.0,
        borrow_bps_pa=0.0,
        rebalance_mode="daily",
    )
    out = run_rack_backtest(racks, cfg)
    df = out["daily"]
    st = out["stats"]

    # Day 0 stock GMV = $125k, SPY hedge = +$140k -> Total Portfolio GMV = $265k!
    assert np.isclose(df["gmv"].iloc[0], 125_000.0)
    assert np.isclose(df["total_gmv"].iloc[0], 265_000.0)
    assert np.isclose(st["avg_stock_gmv"], 125_000.0)
    assert np.isclose(st["avg_spy_hedge_abs"], 140_000.0)
    assert np.isclose(st["avg_daily_gmv"], 265_000.0)
    assert np.isclose(st["capital_base_usd"], 265_000.0)
    assert np.isclose(st["ann_two_way_turnover_x"], 2.0 * st["ann_turnover_x"])
    assert np.isclose(st["pct_days_adv_capped"], 1.0)

    # Exact algebraic identity on every single day: pnl_raw == pnl_stock_raw + pnl_spy_hedge
    diff = np.abs(df["pnl_raw"] - (df["pnl_stock_raw"] + df["pnl_spy_hedge"]))
    assert float(diff.max()) < 1e-10

    # Short -$125k earning -20 bps/d beta-adj return => exact +$250/day gross PnL!
    assert np.allclose(df["pnl_raw"].to_numpy(), 250.0)

    # Day 0 entry cost: 5 bps * $125,000 = $62.50
    # SPY hedge on day 0: -( -100k*1.2 - 25k*0.8 ) = +$140,000 -> 1 bp * $140k = $14.00
    assert np.isclose(df["exec_cost"].iloc[0], 62.50)
    assert np.isclose(df["spy_cost"].iloc[0], 14.00)
    assert np.isclose(df["pnl"].iloc[0], 250.0 - 62.50 - 14.00)

    # Day 1 drift: prev_day_ret = +1% -> drifted = [-101k, -25.25k], rebalance back to [-100k, -25k]
    # Traded notional = $1,000 + $250 = $1,250 -> 5 bps * $1,250 = $0.625!
    assert np.isclose(df["traded_notional"].iloc[1], 1_250.0)
    assert np.isclose(df["exec_cost"].iloc[1], 0.625)


def test_buffered_and_quarterly_eliminate_daily_drift_churn_and_apply_borrow() -> None:
    """Verify buffered & quarterly modes avoid drift trades and deduct exact borrow_bps_pa."""
    racks = _make_synthetic_racks(n_days=10)

    # Buffered with 5% per-position band (> 1% daily drift) -> ZERO trades on day 1!
    cfg_buf = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=5.0,
        spy_hedge_cost_bps=0.0,
        borrow_bps_pa=252.0,  # 1 bp/day exact -> $12.50/day on $125k short book
        rebalance_mode="buffered",
        position_buffer_pct=0.05,
    )
    out_buf = run_rack_backtest(racks, cfg_buf)
    df_buf = out_buf["daily"]
    assert np.isclose(df_buf["traded_notional"].iloc[1], 0.0)
    assert np.isclose(df_buf["exec_cost"].iloc[1], 0.0)
    assert np.isclose(df_buf["borrow_cost"].iloc[0], 12.50)

    # Quarterly mask strictly triggers on day 0 + Day +46 post-quarter-end
    cal = pd.bdate_range("2022-01-03", "2022-12-30")
    q_mask = compute_quarterly_rebalance_mask(cal, offset_days=46)
    rebal_dates = cal[q_mask]
    # Day 0 + Q4'21 (Feb 15) + Q1'22 (May 16) + Q2'22 (Aug 15) + Q3'22 (Nov 15) = 5 dates
    assert len(rebal_dates) == 5
    assert pd.Timestamp("2022-01-03") in rebal_dates
    assert pd.Timestamp("2022-02-15") in rebal_dates
    assert pd.Timestamp("2022-05-16") in rebal_dates
    assert pd.Timestamp("2022-08-15") in rebal_dates
    assert pd.Timestamp("2022-11-15") in rebal_dates


def test_spy_10d_regime_scale_ramp_cap_and_two_tail_logic() -> None:
    """Verify exact -1% -> -3% linear ramp, -3% cap, > +3% trigger, and combined union."""
    r = np.array([-0.10, -0.03, -0.02, -0.01, 0.0, 0.029, 0.031, 0.06, np.nan])
    s_unc = compute_spy_10d_regime_scale(r, "unconditional")
    s_off = compute_spy_10d_regime_scale(r, "risk_off_ramp")
    s_up = compute_spy_10d_regime_scale(r, "rally_gt_3pct")
    s_cmb = compute_spy_10d_regime_scale(r, "combined_two_tail")

    assert np.allclose(s_unc, 1.0)
    assert np.allclose(s_off, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(s_up, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])
    assert np.allclose(s_cmb, [1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0])


def test_fixed_capital_base_preserves_sharpe_and_honest_return() -> None:
    """Fixed full-scale capital base keeps Return honest on cash days & Sharpe invariant."""
    racks = _make_synthetic_racks(n_days=10)
    # Zero out second half of days (simulate 50% cash regime gating!)
    racks.expected_ret_bps[5:, :] = 0.0
    cfg_auto = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=0.0,
        spy_hedge_cost_bps=0.0,
        rebalance_mode="daily",
    )
    cfg_fixed = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=0.0,
        spy_hedge_cost_bps=0.0,
        rebalance_mode="daily",
        capital_base_usd=265_000.0,  # exact 1x full-scale Total GMV ($125k stocks + $140k SPY)
    )
    st_auto = run_rack_backtest(racks, cfg_auto)["stats"]
    st_fixed = run_rack_backtest(racks, cfg_fixed)["stats"]

    assert np.isclose(st_fixed["pct_active_days"], 0.5)
    assert np.isclose(st_fixed["sharpe"], st_auto["sharpe"])
    # Fixed full-scale capital base avoids 2x Return inflation from cash days!
    assert np.isclose(st_fixed["ann_return"], 0.5 * st_auto["ann_return"])


def test_buffered_relative_band_and_dollar_neutrality_guard() -> None:
    """Asymmetric dollar-neutral book (2 large shorts vs. 40 small longs) stays strictly neutral."""
    n_days = 25
    n_short = 2
    n_long = 40
    n_sec = n_short + n_long
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    fwd_dates = pd.bdate_range("2022-01-04", periods=n_days)
    secs = pd.Index([f"S{i}" for i in range(n_short)] + [f"L{i}" for i in range(n_long)])

    # Target: 2 shorts @ -$100k (-$200k) vs. 40 longs @ +$5k (+$200k) -> exact $0 net target!
    er_row = np.array([-2.0] * n_short + [0.1] * n_long, dtype=np.float64)
    er = np.tile(er_row, (n_days, 1))
    # Force a sign flip on L0 on day 20 to verify immediate crossing through zero
    er[20:, n_short] = -0.1
    er[20:, n_short + 1] = 0.2  # preserve exact $0 net target on days 20..24

    # Persistent +2%/day drift on longs, 0%/day on shorts (adversarial long-drift stress!)
    prev_raw = np.zeros((n_days, n_sec), dtype=np.float64)
    prev_raw[:, n_short:] = 0.02
    fwd_spy = np.zeros(n_days, dtype=np.float64)
    prev_spy = np.zeros(n_days, dtype=np.float64)
    beta = np.ones((n_days, n_sec), dtype=np.float64)
    fwd_raw = np.zeros((n_days, n_sec), dtype=np.float64)
    fwd_badj = np.zeros((n_days, n_sec), dtype=np.float64)
    adv = np.full((n_days, n_sec), 50_000_000.0, dtype=np.float64)

    racks = BacktestRacks(
        dates=dates,
        fwd_ret_dates=fwd_dates,
        securities=secs,
        expected_ret_bps=er,
        fwd_beta_adj_ret=fwd_badj,
        fwd_day_ret=fwd_raw,
        prev_day_ret=prev_raw,
        beta_lag1=beta,
        adv_20d_usd=adv,
        fwd_spy_ret=fwd_spy,
        prev_spy_ret=prev_spy,
    )

    cfg = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=5.0,
        spy_hedge_cost_bps=1.0,
        rebalance_mode="buffered",
        position_buffer_pct=0.20,
        max_net_exposure_pct=0.02,
    )
    out = run_rack_backtest(racks, cfg)
    df = out["daily"]
    st = out["stats"]

    # Hard dollar-neutrality invariant: |long_gmv - short_gmv| / gmv <= 2% on EVERY single day
    net_frac = (df["long_gmv"] - df["short_gmv"]).abs() / df["gmv"]
    assert float(net_frac.max()) <= 0.02 + 1e-9
    assert st["max_abs_net_exposure_pct"] <= 0.02 + 1e-9
    # Guard must have actively fired against the +2%/day correlated long drift
    assert st["n_net_recenter_days"] > 0
    # Day 1 drift is +2% on $200k longs (+$4k -> net_frac = 4/404 = 0.99% <= 2%, within 20% band)
    # so day 1 incurs ZERO trades!
    assert np.isclose(df["traded_notional"].iloc[1], 0.0)
    # Day 20 sign flip on L0 (+$5k -> -$5k) must trigger trading cleanly
    assert df["traded_notional"].iloc[20] > 0.0


def test_tiered_large_and_mid_cap_execution_costs() -> None:
    """Verify tiered 5 bps (>= $10B PIT MCap) vs. 12 bps (< $10B PIT MCap) one-way costs."""
    racks = _make_synthetic_racks(n_days=4)
    # S1 = $50B large-cap (5 bps), S2 = $4B mid-cap (12 bps)
    racks.pit_mcap_usd = np.tile(np.array([[50_000_000_000.0, 4_000_000_000.0]]), (4, 1))

    cfg = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=5.0,
        mid_cap_one_way_bps=12.0,
        large_cap_threshold_usd=10_000_000_000.0,
        spy_hedge_cost_bps=0.0,
        borrow_bps_pa=0.0,
        rebalance_mode="daily",
    )
    out = run_rack_backtest(racks, cfg)
    df = out["daily"]
    st = out["stats"]

    # Day 0: S1 trades $100k @ 5 bps = $50.00; S2 trades $25k (ADV-capped) @ 12 bps = $30.00 -> $80
    assert np.isclose(df["exec_cost"].iloc[0], 80.0)
    assert np.isclose(df["mid_cap_traded_notional"].iloc[0], 25_000.0)
    # Effective blended one-way cost = (100k*5 + 25k*12) / 125k = 6.4 bps exact
    assert np.isclose(st["effective_one_way_cost_bps"], 6.4)
    assert np.isclose(st["pct_traded_mid_cap"], 0.20)


def test_tradable_has_zero_t_plus_1_lookahead_on_unexpected_halt() -> None:
    """Day-t trades must NEVER peek at t+1 fwd_day_ret / fwd_beta_adj_ret NaNs."""
    racks_clean = _make_synthetic_racks(n_days=5)
    racks_clean.has_price = np.ones((5, 2), dtype=np.float64)

    # Unexpected trading halt on S1 on t+1 after day-2 MOC (day 2 still priced & tradable!)
    racks_halt = _make_synthetic_racks(n_days=5)
    racks_halt.has_price = np.ones((5, 2), dtype=np.float64)
    racks_halt.fwd_day_ret[2, 0] = np.nan
    racks_halt.fwd_beta_adj_ret[2, 0] = np.nan
    # Then on day 3 S1 is halted (has_price = 0) -> exits cleanly on day 3
    racks_halt.has_price[3, 0] = 0.0

    cfg = BacktestConfig(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=5.0,
        spy_hedge_cost_bps=1.0,
        rebalance_mode="daily",
    )
    df_clean = run_rack_backtest(racks_clean, cfg)["daily"]
    df_halt = run_rack_backtest(racks_halt, cfg)["daily"]

    # Day-2 MOC positions, GMV, SPY hedge, and traded notional MUST be bit-for-bit identical!
    for col in ("gmv", "total_gmv", "spy_hedge_notional", "traded_notional", "exec_cost"):
        assert np.isclose(df_halt[col].iloc[2], df_clean[col].iloc[2])

    # Algebraic PnL identity still holds exactly on the halt day (live SPY hedge moves by f_spy)
    diff = np.abs(df_halt["pnl_raw"] - (df_halt["pnl_stock_raw"] + df_halt["pnl_spy_hedge"]))
    assert float(diff.max()) < 1e-10

    # On day 3 (when has_price[3, 0] == 0 is observable), S1 exits -> stock GMV drops to $25k (S2)
    assert np.isclose(df_halt["gmv"].iloc[3], 25_000.0)


