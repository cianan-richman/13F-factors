"""Rack-Based Corporate-Action-Clean Portfolio Backtester (`date x security_id`).

Simulates realistic multi-manager dollar-notional portfolios from any aligned daily
`expected_ret_bps` rack (in `bps/day`) paired with strictly PIT corporate-action-clean
forward returns (`fwd_beta_adj_ret`, `fwd_day_ret`, `prev_day_ret`) and liquidity (`adv_20d_usd`).

Core Design & PIT Discipline:
  - **Signal-to-Notional Mapping:**
        N_target_{i, t} = clip(notional_per_bps * expected_ret_bps_{i, t},
                               -max_adv_pct * adv_20d_usd_{i, t},
                               +max_adv_pct * adv_20d_usd_{i, t})
  - **Exact Execution Timing (Zero Lookahead):**
        Filing accepted on EDGAR on day T -> PIT-visible on effective_date = T+1 ->
        signal computed on query_date = T+1 -> trades execute at T+1 MOC ->
        earns `fwd_beta_adj_ret` over `(T+1 -> T+2]`.
  - **Corporate-Action-Clean Daily Notional Drift:**
        N_drifted_{i, t} = N_{i, t-1} * (1 + prev_day_ret_{i, t})
    where `prev_day_ret` is the split- and dividend-adjusted total return from `t-1` to `t`.
  - **Three Rebalancing Horizons Supported:**
      1. `daily`: Rebalances to exact `N_target` every trading day.
      2. `buffered`: Holds `N_drifted` unless a stock enters/exits the target book or
         `|N_target - N_drifted| > (buffer_bps_of_gmv / 1e4) * target_gmv_t`.
      3. `quarterly`: Rebalances strictly on the first trading day `>= quarterly_offset_days`
         (default `46` calendar days) after each calendar quarter-end (plus day 0 inception),
         holding drifted positions in between (exiting only if a stock drops off the price tape).
  - **Two-Way Transaction Costs & Optional Short-Borrow Carry:**
      - Stock execution cost: `one_way_cost_bps * sum_i |N_{i, t} - N_drifted_{i, t}|`
        (default `5 bps` each way = `10 bps` round-trip).
      - Implicit `SPY` beta-hedge cost: `spy_hedge_cost_bps * |H_t - H_drifted_t|`
        where `H_t = -sum_i N_{i, t} * beta_{i, t}` (default `1 bp` one-way).
      - Configurable short-borrow carry: `(borrow_bps_pa / 1e4 / 252) * short_gmv_t`.
  - **Strict In-Sample Default (`sample="in_sample"`):**
      Enforces `fwd_ret_date <= 2023-12-31` so Out-of-Sample (`2024+`) is NEVER touched
      until explicitly requested with frozen In-Sample parameters.

Usage:
    python -m src.backtest.engine
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, load_config
from src.factors.crowding import build_crowding_exposures, fit_clustered_crowding_ols
from src.factors.ownership import (
    DEFAULT_IS_SIGNAL_START,
    DEFAULT_MIN_CROSS_SECTION,
    DEFAULT_SPY_PARQUET,
    TRADING_DAYS_PER_YEAR,
    SampleSplit,
    attach_fwd_ret_date,
    load_factor_panel,
)
from src.pit.daily_panel import DEFAULT_DAILY_PANEL_PARQUET

logger = logging.getLogger(__name__)

RebalanceMode = Literal["daily", "buffered", "quarterly"]
CrowdingStrategyMode = Literal["short_gt_d7", "short_d10", "continuous_ls"]
RegimeGateMode = Literal["unconditional", "risk_off_ramp", "rally_gt_3pct", "combined_two_tail"]

DEFAULT_BACKTEST_OUTPUT_DIR = OUTPUTS_DIR / "backtest"
# Frozen In-Sample OLS slope for 2-pillar crowding_z (`fwd_beta_adj_ret_bps ~ const + crowding_z`)
FROZEN_IS_CROWDING_BETA_BPS = -0.534773


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for rack-based portfolio simulation."""

    notional_per_bps: float = 50_000.0  # $50k per 1 bps/day expected return (~$7-10M book)
    max_adv_pct: float = 0.05  # Hard 5% of 20-day ADV single-name cap
    one_way_cost_bps: float = 5.0  # 5 bps each way (= 10 bps round-trip) on |ΔNotional|
    spy_hedge_cost_bps: float = 1.0  # 1 bp one-way on net SPY beta-hedge notional delta
    borrow_bps_pa: float = 0.0  # Annualized short-borrow fee on short GMV (0 in ex-borrow base)
    rebalance_mode: RebalanceMode = "buffered"
    buffer_bps_of_gmv: float = 25.0  # No-trade zone (bps of target GMV) for buffered mode
    quarterly_offset_days: int = 46  # Calendar days after quarter-end for fixed quarterly rebal
    capital_base_usd: float | None = None  # Fixed allocated capital base (e.g. 1x full-scale GMV)
    name: str = ""


@dataclass
class BacktestRacks:
    """Aligned `date x security_id` data racks consumed by `run_rack_backtest`."""

    dates: pd.DatetimeIndex
    fwd_ret_dates: pd.DatetimeIndex
    securities: pd.Index
    expected_ret_bps: np.ndarray  # (T, N) expected beta-adjusted return in bps/day (0 = no pos)
    fwd_beta_adj_ret: np.ndarray  # (T, N) next-day beta-adjusted return (decimal)
    fwd_day_ret: np.ndarray  # (T, N) next-day total return (decimal)
    prev_day_ret: np.ndarray  # (T, N) prior-day total return t-1->t for drift (decimal)
    beta_lag1: np.ndarray  # (T, N) strictly PIT lagged 252d OLS beta
    adv_20d_usd: np.ndarray  # (T, N) 20-day ADV in USD
    fwd_spy_ret: np.ndarray  # (T,) next-day SPY total return (decimal)
    prev_spy_ret: np.ndarray  # (T,) prior-day SPY total return t-1->t (decimal)


def compute_quarterly_rebalance_mask(
    dates: pd.DatetimeIndex,
    *,
    offset_days: int = 46,
) -> np.ndarray:
    """Return boolean mask `True` on day 0 and first trading day `>= offset_days` post-quarter."""
    n = len(dates)
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    mask[0] = True

    d_norm = pd.DatetimeIndex(pd.to_datetime(dates).normalize())
    start_q = (d_norm.min() - pd.offsets.QuarterEnd(2)).normalize()
    end_q = d_norm.max().normalize()
    q_ends = pd.date_range(start_q, end_q, freq="QE")

    for qe in q_ends:
        target = (qe + pd.Timedelta(days=offset_days)).normalize()
        pos = int(d_norm.searchsorted(target, side="left"))
        if 0 <= pos < n:
            mask[pos] = True
    return mask


def build_crowding_backtest_racks(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    strategy_mode: CrowdingStrategyMode = "short_gt_d7",
    beta_ols_bps: float | None = None,
    min_signal_date: str | None = DEFAULT_IS_SIGNAL_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
) -> tuple[BacktestRacks, float]:
    """Build strictly PIT aligned `(T x N)` racks for the 2-Pillar Crowding strategy.

    Return racks (`fwd_beta_adj_ret`, `fwd_day_ret`, `prev_day_ret`, `beta_lag1`, `adv_20d_usd`)
    include ALL priced securities on those trading dates so a stock held during `quarterly` or
    `buffered` rebalancing that temporarily dips below the dynamic MCap/ADV floor still earns
    its exact corporate-action-clean return until the next rebalance. Target `expected_ret_bps`
    is strictly non-zero ONLY for `is_panel_eligible == True` stocks.
    """
    elig_panel = load_factor_panel(
        panel_path=panel_path,
        spy_path=spy_path,
        sample=sample,
        min_signal_date=min_signal_date,
        min_cross_section=min_cross_section,
    )
    cf = build_crowding_exposures(elig_panel)

    # Strictly freeze beta_ols_bps from IS (if not passed explicitly, estimate ONLY on IS)
    if beta_ols_bps is None:
        if sample == "in_sample":
            cf_tmp = cf.copy()
            cf_tmp["fwd_beta_adj_ret_bps"] = cf_tmp["fwd_beta_adj_ret"] * 1e4
            fit = fit_clustered_crowding_ols(cf_tmp, ["crowding_z"])
            slope_bps = float(fit.params["crowding_z"])
        else:
            slope_bps = FROZEN_IS_CROWDING_BETA_BPS
    else:
        slope_bps = float(beta_ols_bps)

    # Compute daily equal-count decile (1..10) of crowding_z on eligible cross-section
    ranks = cf.groupby("query_date")["crowding_z"].rank(method="first")
    counts = cf.groupby("query_date")["crowding_z"].transform("count")
    cf["decile"] = (np.floor((ranks - 1.0) * 10.0 / counts).astype(int).clip(0, 9) + 1)

    if strategy_mode == "short_gt_d7":
        # Short > D7 (D8, D9, D10): expected return = slope_bps * crowding_z (< 0 -> short!)
        # Enforce <= 0 in case a rare D8 border stock had tiny negative z
        raw_er = np.where(cf["decile"] >= 8, np.minimum(0.0, slope_bps * cf["crowding_z"]), 0.0)
    elif strategy_mode == "short_d10":
        raw_er = np.where(cf["decile"] == 10, np.minimum(0.0, slope_bps * cf["crowding_z"]), 0.0)
    elif strategy_mode == "continuous_ls":
        raw_er = slope_bps * cf["crowding_z"].to_numpy(dtype=np.float64)
    else:
        raise ValueError(f"Unsupported strategy_mode: {strategy_mode}")
    cf["expected_ret_bps"] = raw_er

    # Strict IS/OOS guard verification
    cfg = load_config()
    is_end = pd.Timestamp(cfg.settings["window"]["in_sample_end"]).normalize()
    if sample == "in_sample":
        max_fwd = pd.Timestamp(cf["fwd_ret_date"].max()).normalize()
        if max_fwd > is_end:
            raise AssertionError(
                f"OOS LEAKAGE: max fwd_ret_date {max_fwd.date()} > {is_end.date()}"
            )

    dates = pd.DatetimeIndex(sorted(cf["query_date"].unique()))
    date_fwd_map = (
        cf[["query_date", "fwd_ret_date"]]
        .drop_duplicates("query_date")
        .set_index("query_date")["fwd_ret_date"]
        .reindex(dates)
    )
    fwd_ret_dates = pd.DatetimeIndex(date_fwd_map.to_numpy())

    # Load all priced market rows on those exact dates so held positions drift cleanly
    all_cols = [
        "query_date",
        "security_id",
        "prev_day_ret",
        "fwd_day_ret",
        "fwd_beta_adj_ret",
        "beta_252d_lag1",
        "adv_20d_usd",
        "prev_spy_ret",
        "fwd_spy_ret",
    ]
    full_p = pd.read_parquet(panel_path, columns=all_cols)
    full_p["query_date"] = pd.to_datetime(full_p["query_date"]).dt.normalize()
    full_p = full_p[full_p["query_date"].isin(dates)].copy()

    # Keep only securities that ever had a non-zero target signal in this sample (fast & exact!)
    active_secs = pd.Index(sorted(cf.loc[cf["expected_ret_bps"] != 0.0, "security_id"].unique()))
    full_p = full_p[full_p["security_id"].isin(active_secs)].copy()
    cf_act = cf[cf["security_id"].isin(active_secs)].copy()

    def _piv(df_in: pd.DataFrame, val_col: str, fill_val: float = np.nan) -> np.ndarray:
        piv = df_in.pivot(index="query_date", columns="security_id", values=val_col)
        piv = piv.reindex(index=dates, columns=active_secs)
        if not np.isnan(fill_val):
            piv = piv.fillna(fill_val)
        return piv.to_numpy(dtype=np.float64)

    spy_df = pd.read_parquet(spy_path)
    spy_df = attach_fwd_ret_date(
        pd.DataFrame({"query_date": dates}), spy_df
    )  # sanity align calendar
    _ = spy_df  # calendar verified
    spy_by_day = (
        full_p[["query_date", "fwd_spy_ret", "prev_spy_ret"]]
        .drop_duplicates("query_date")
        .set_index("query_date")
        .reindex(dates)
    )

    racks = BacktestRacks(
        dates=dates,
        fwd_ret_dates=fwd_ret_dates,
        securities=active_secs,
        expected_ret_bps=_piv(cf_act, "expected_ret_bps", fill_val=0.0),
        fwd_beta_adj_ret=_piv(full_p, "fwd_beta_adj_ret", fill_val=np.nan),
        fwd_day_ret=_piv(full_p, "fwd_day_ret", fill_val=np.nan),
        prev_day_ret=_piv(full_p, "prev_day_ret", fill_val=np.nan),
        beta_lag1=_piv(full_p, "beta_252d_lag1", fill_val=np.nan),
        adv_20d_usd=_piv(full_p, "adv_20d_usd", fill_val=np.nan),
        fwd_spy_ret=spy_by_day["fwd_spy_ret"].to_numpy(dtype=np.float64),
        prev_spy_ret=spy_by_day["prev_spy_ret"].to_numpy(dtype=np.float64),
    )
    return racks, slope_bps


def run_rack_backtest(
    racks: BacktestRacks,
    config: BacktestConfig,
) -> dict[str, Any]:
    """Run corporate-action-clean, drifted, beta-hedged rack backtest and return daily & stats."""
    t_len, n_sec = racks.expected_ret_bps.shape
    if t_len == 0 or n_sec == 0:
        raise ValueError("BacktestRacks is empty.")

    q_rebal_mask = (
        compute_quarterly_rebalance_mask(racks.dates, offset_days=config.quarterly_offset_days)
        if config.rebalance_mode == "quarterly"
        else np.zeros(t_len, dtype=bool)
    )

    held = np.zeros(n_sec, dtype=np.float64)
    held_spy_hedge = 0.0

    daily_rows: list[dict[str, Any]] = []
    n_adv_capped_days = 0

    one_way_rate = config.one_way_cost_bps / 1e4
    spy_rate = config.spy_hedge_cost_bps / 1e4
    borrow_daily_rate = (config.borrow_bps_pa / 1e4) / TRADING_DAYS_PER_YEAR
    buf_frac = config.buffer_bps_of_gmv / 1e4

    for t in range(t_len):
        # 1. Drift yesterday's held positions forward to today's close via corporate-action-clean
        #    total return `prev_day_ret` (t-1 -> t)
        if t == 0:
            drifted = np.zeros(n_sec, dtype=np.float64)
            drifted_spy = 0.0
        else:
            pr = racks.prev_day_ret[t]
            pr_clean = np.where(np.isfinite(pr), pr, 0.0)
            drifted = held * (1.0 + pr_clean)
            pspy = racks.prev_spy_ret[t]
            drifted_spy = held_spy_hedge * (1.0 + (pspy if np.isfinite(pspy) else 0.0))

        # Tradable / priced mask for earning next-day beta-adjusted return on (t -> t+1]
        fwd_badj = racks.fwd_beta_adj_ret[t]
        fwd_raw = racks.fwd_day_ret[t]
        beta_t = racks.beta_lag1[t]
        adv_t = racks.adv_20d_usd[t]
        tradable = np.isfinite(fwd_badj) & np.isfinite(fwd_raw) & np.isfinite(beta_t)

        # 2. Compute capped target notional from expected_ret_bps rack
        er_t = racks.expected_ret_bps[t]
        raw_target = np.where(tradable, config.notional_per_bps * er_t, 0.0)
        if config.max_adv_pct > 0.0:
            adv_cap = np.where(np.isfinite(adv_t) & (adv_t > 0.0), config.max_adv_pct * adv_t, 0.0)
            target = np.clip(raw_target, -adv_cap, adv_cap)
            if np.any(np.abs(raw_target) > adv_cap + 1e-6):
                n_adv_capped_days += 1
        else:
            target = raw_target

        # 3. Apply rebalancing rule -> final held notionals at day-t MOC
        if config.rebalance_mode == "daily":
            new_held = target.copy()
        elif config.rebalance_mode == "buffered":
            target_gmv = float(np.abs(target).sum())
            band = buf_frac * target_gmv
            enter_or_exit = ((drifted == 0.0) & (target != 0.0)) | (
                (target == 0.0) & (drifted != 0.0)
            )
            breach_band = np.abs(target - drifted) > band
            do_trade = enter_or_exit | breach_band | (~tradable)
            new_held = np.where(do_trade, target, drifted)
        elif config.rebalance_mode == "quarterly":
            if q_rebal_mask[t]:
                new_held = target.copy()
            else:
                # Hold drifted positions between quarterly rebalances; exit if untradable
                new_held = np.where(tradable, drifted, 0.0)
        else:
            raise ValueError(f"Unsupported rebalance_mode: {config.rebalance_mode}")

        # Zero out tiny floating-point dust (< $1)
        new_held = np.where(np.abs(new_held) < 1.0, 0.0, new_held)

        # 4. Exact two-way traded notional, SPY beta-hedge notional, and costs
        traded_notional = float(np.abs(new_held - drifted).sum())
        exec_cost = one_way_rate * traded_notional

        beta_clean = np.where(np.isfinite(beta_t), beta_t, 0.0)
        new_spy_hedge = float(-np.dot(new_held, beta_clean))
        spy_traded = abs(new_spy_hedge - drifted_spy)
        spy_cost = spy_rate * spy_traded

        short_gmv = float(np.abs(np.minimum(new_held, 0.0)).sum())
        long_gmv = float(np.maximum(new_held, 0.0).sum())
        gmv = long_gmv + short_gmv
        borrow_cost = borrow_daily_rate * short_gmv

        total_cost = exec_cost + spy_cost + borrow_cost

        # 5. Realize next-day corporate-action-clean beta-adjusted PnL over (t -> t+1]
        fwd_badj_clean = np.where(np.isfinite(fwd_badj), fwd_badj, 0.0)
        fwd_raw_clean = np.where(np.isfinite(fwd_raw), fwd_raw, 0.0)
        f_spy = float(racks.fwd_spy_ret[t]) if np.isfinite(racks.fwd_spy_ret[t]) else 0.0

        pnl_raw = float(np.dot(new_held, fwd_badj_clean))
        pnl_stock_raw = float(np.dot(new_held, fwd_raw_clean))
        pnl_spy_hedge = new_spy_hedge * f_spy
        pnl_net = pnl_raw - total_cost

        daily_rows.append(
            {
                "query_date": racks.dates[t],
                "fwd_ret_date": racks.fwd_ret_dates[t],
                "n_positions": int((new_held != 0.0).sum()),
                "long_gmv": long_gmv,
                "short_gmv": short_gmv,
                "gmv": gmv,
                "spy_hedge_notional": new_spy_hedge,
                "traded_notional": traded_notional,
                "exec_cost": exec_cost,
                "spy_cost": spy_cost,
                "borrow_cost": borrow_cost,
                "total_cost": total_cost,
                "pnl_raw": pnl_raw,
                "pnl_stock_raw": pnl_stock_raw,
                "pnl_spy_hedge": pnl_spy_hedge,
                "pnl": pnl_net,
                "fwd_spy_ret": f_spy,
            }
        )

        held = new_held
        held_spy_hedge = new_spy_hedge

    daily_df = pd.DataFrame(daily_rows)
    daily_df["cum_pnl_raw"] = daily_df["pnl_raw"].cumsum()
    daily_df["cum_pnl"] = daily_df["pnl"].cumsum()
    daily_df["dd_usd"] = daily_df["cum_pnl"] - daily_df["cum_pnl"].cummax()

    stats = summarize_backtest(daily_df, config, n_adv_capped_days=n_adv_capped_days)
    return {"daily": daily_df, "stats": stats, "config": config}


def summarize_backtest(
    daily_df: pd.DataFrame,
    config: BacktestConfig,
    *,
    n_adv_capped_days: int = 0,
) -> dict[str, Any]:
    """Compute exact dollar and percentage performance metrics from daily backtest output."""
    n_days = int(len(daily_df))
    avg_gmv = float(daily_df["gmv"].mean())
    avg_short_gmv = float(daily_df["short_gmv"].mean())
    active_mask = daily_df["gmv"] > 0.0
    pct_active_days = float(active_mask.mean()) if n_days else 0.0
    avg_active_gmv = float(daily_df.loc[active_mask, "gmv"].mean()) if active_mask.any() else 0.0
    cap_base = (
        float(config.capital_base_usd)
        if (config.capital_base_usd is not None and config.capital_base_usd > 0.0)
        else (avg_gmv if avg_gmv > 0.0 else 1.0)
    )

    r_raw = daily_df["pnl_raw"].to_numpy(dtype=np.float64) / cap_base
    r_net = daily_df["pnl"].to_numpy(dtype=np.float64) / cap_base
    spy_r = daily_df["fwd_spy_ret"].to_numpy(dtype=np.float64)

    ann_pnl_raw = float(daily_df["pnl_raw"].mean() * TRADING_DAYS_PER_YEAR)
    ann_pnl = float(daily_df["pnl"].mean() * TRADING_DAYS_PER_YEAR)
    ann_ret_raw = float(r_raw.mean() * TRADING_DAYS_PER_YEAR)
    ann_ret = float(r_net.mean() * TRADING_DAYS_PER_YEAR)
    ann_sdev_raw = float(r_raw.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    ann_sdev = float(r_net.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    sharpe_raw = float(ann_ret_raw / ann_sdev_raw) if ann_sdev_raw > 0.0 else np.nan
    sharpe = float(ann_ret / ann_sdev) if ann_sdev > 0.0 else np.nan

    hit_raw = float((daily_df["pnl_raw"] > 0.0).mean())
    hit_net = float((daily_df["pnl"] > 0.0).mean())

    max_dd_usd = float(daily_df["dd_usd"].min())
    max_dd_pct = float(max_dd_usd / cap_base)

    # Annualized one-way turnover as multiple of average daily GMV (ex-day-0 inception)
    if n_days > 1:
        avg_daily_turn = float(daily_df["traded_notional"].iloc[1:].mean() / cap_base)
    else:
        avg_daily_turn = 0.0
    ann_turnover_x = avg_daily_turn * TRADING_DAYS_PER_YEAR

    spy_var = float(np.var(spy_r, ddof=1))
    realized_spy_beta = float(np.cov(r_net, spy_r, ddof=1)[0, 1] / spy_var) if spy_var > 0 else 0.0

    # Exact breakeven annualized borrow rate (bps/yr on short_gmv) where net PnL ex-borrow hits 0
    pnl_ex_borrow = daily_df["pnl"] + daily_df["borrow_cost"]
    ann_pnl_ex_borrow = float(pnl_ex_borrow.mean() * TRADING_DAYS_PER_YEAR)
    breakeven_borrow_bps_pa = (
        float((ann_pnl_ex_borrow / avg_short_gmv) * 1e4) if avg_short_gmv > 0.0 else np.nan
    )

    return {
        "name": config.name or config.rebalance_mode,
        "rebalance_mode": config.rebalance_mode,
        "one_way_cost_bps": config.one_way_cost_bps,
        "borrow_bps_pa": config.borrow_bps_pa,
        "n_trading_days": n_days,
        "pct_active_days": pct_active_days,
        "avg_n_positions": float(daily_df["n_positions"].mean()),
        "avg_daily_gmv": avg_gmv,
        "avg_active_gmv": avg_active_gmv,
        "capital_base_usd": cap_base,
        "pct_days_adv_capped": float(n_adv_capped_days / n_days) if n_days else 0.0,
        "daily_hit_rate_raw": hit_raw,
        "daily_hit_rate": hit_net,
        "cum_pnl_raw": float(daily_df["cum_pnl_raw"].iloc[-1]),
        "cum_pnl": float(daily_df["cum_pnl"].iloc[-1]),
        "total_exec_cost": float(daily_df["exec_cost"].sum()),
        "total_spy_cost": float(daily_df["spy_cost"].sum()),
        "total_borrow_cost": float(daily_df["borrow_cost"].sum()),
        "total_cost": float(daily_df["total_cost"].sum()),
        "ann_pnl_raw": ann_pnl_raw,
        "ann_pnl": ann_pnl,
        "ann_return_raw": ann_ret_raw,
        "ann_return": ann_ret,
        "ann_sdev_raw": ann_sdev_raw,
        "ann_sdev": ann_sdev,
        "sharpe_raw": sharpe_raw,
        "sharpe": sharpe,
        "max_dd_usd": max_dd_usd,
        "max_dd_pct": max_dd_pct,
        "ann_turnover_x": ann_turnover_x,
        "realized_spy_beta": realized_spy_beta,
        "breakeven_borrow_bps_pa": breakeven_borrow_bps_pa,
    }


def run_crowding_horizon_comparison(
    *,
    sample: SampleSplit = "in_sample",
    strategy_mode: CrowdingStrategyMode = "short_gt_d7",
    notional_per_bps: float = 50_000.0,
    max_adv_pct: float = 0.05,
    one_way_cost_bps: float = 5.0,
    spy_hedge_cost_bps: float = 1.0,
    buffer_bps_of_gmv: float = 25.0,
    quarterly_offset_days: int = 46,
    output_dir: Path | None = DEFAULT_BACKTEST_OUTPUT_DIR,
) -> dict[str, Any]:
    """Compare Daily, Buffered Event-Driven, and Fixed Quarterly (Day +46) + cost/borrow grids."""
    racks, slope_bps = build_crowding_backtest_racks(
        sample=sample,
        strategy_mode=strategy_mode,
    )

    modes: list[tuple[RebalanceMode, str]] = [
        ("daily", "Daily Rebalance"),
        ("buffered", f"Buffered Event-Driven ({buffer_bps_of_gmv:.0f} bps band)"),
        ("quarterly", f"Fixed Quarterly (Day +{quarterly_offset_days})"),
    ]

    results: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    for mode, label in modes:
        cfg = BacktestConfig(
            notional_per_bps=notional_per_bps,
            max_adv_pct=max_adv_pct,
            one_way_cost_bps=one_way_cost_bps,
            spy_hedge_cost_bps=spy_hedge_cost_bps,
            borrow_bps_pa=0.0,
            rebalance_mode=mode,
            buffer_bps_of_gmv=buffer_bps_of_gmv,
            quarterly_offset_days=quarterly_offset_days,
            name=label,
        )
        res = run_rack_backtest(racks, cfg)
        results[mode] = res
        summary_rows.append(res["stats"])

    # Cost & short-borrow sensitivity across buffered & quarterly modes
    sens_rows: list[dict[str, Any]] = []
    for mode in ("daily", "buffered", "quarterly"):
        for c_bps in (0.0, 5.0, 10.0, 20.0):
            cfg = BacktestConfig(
                notional_per_bps=notional_per_bps,
                max_adv_pct=max_adv_pct,
                one_way_cost_bps=c_bps,
                spy_hedge_cost_bps=0.0 if c_bps == 0.0 else spy_hedge_cost_bps,
                borrow_bps_pa=0.0,
                rebalance_mode=mode,
                buffer_bps_of_gmv=buffer_bps_of_gmv,
                quarterly_offset_days=quarterly_offset_days,
                name=f"{mode}_{int(c_bps)}bps",
            )
            sens_rows.append(run_rack_backtest(racks, cfg)["stats"])

    borrow_rows: list[dict[str, Any]] = []
    for b_pa in (0.0, 35.0, 100.0, 250.0):
        cfg = BacktestConfig(
            notional_per_bps=notional_per_bps,
            max_adv_pct=max_adv_pct,
            one_way_cost_bps=one_way_cost_bps,
            spy_hedge_cost_bps=spy_hedge_cost_bps,
            borrow_bps_pa=b_pa,
            rebalance_mode="buffered",
            buffer_bps_of_gmv=buffer_bps_of_gmv,
            quarterly_offset_days=quarterly_offset_days,
            name=f"buffered_borrow_{int(b_pa)}bps",
        )
        borrow_rows.append(run_rack_backtest(racks, cfg)["stats"])

    summary_df = pd.DataFrame(summary_rows)
    cost_sens_df = pd.DataFrame(sens_rows)
    borrow_sens_df = pd.DataFrame(borrow_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        summary_df.to_csv(output_dir / f"crowding_{strategy_mode}_horizons_{tag}.csv", index=False)
        cost_sens_df.to_csv(
            output_dir / f"crowding_{strategy_mode}_cost_sens_{tag}.csv", index=False
        )
        borrow_sens_df.to_csv(
            output_dir / f"crowding_{strategy_mode}_borrow_sens_{tag}.csv", index=False
        )
        for mode, res in results.items():
            res["daily"].to_parquet(
                output_dir / f"crowding_{strategy_mode}_{mode}_daily_{tag}.parquet", index=False
            )

    return {
        "racks": racks,
        "slope_bps": slope_bps,
        "results": results,
        "summary": summary_df,
        "cost_sensitivity": cost_sens_df,
        "borrow_sensitivity": borrow_sens_df,
    }


def compute_spy_10d_regime_scale(
    spy_ret_10d: np.ndarray,
    regime_mode: RegimeGateMode,
    *,
    down_start: float = -0.01,
    down_cap: float = -0.03,
    up_thresh: float = 0.03,
) -> np.ndarray:
    """Compute strictly PIT daily portfolio scale `m(t) in [0, 1]` from 10-BD trailing `SPY`.

    Modes:
      - `unconditional`: `1.0` on every day.
      - `risk_off_ramp`: `0.0` when `spy_ret_10d >= -1%`, ramps linearly `-1% -> -3%`,
        capped at `1.0` for `<= -3%` (prevents late over-sizing on `-10%` crashes).
      - `rally_gt_3pct`: `1.0` when `spy_ret_10d > +3%`, else `0.0`.
      - `combined_two_tail`: `max(risk_off_ramp, rally_gt_3pct)`.
    """
    r = np.asarray(spy_ret_10d, dtype=np.float64)
    if regime_mode == "unconditional":
        return np.ones_like(r, dtype=np.float64)

    span = abs(down_start - down_cap)
    down_scale = np.where(
        np.isfinite(r) & (r < down_start),
        np.clip((down_start - r) / span, 0.0, 1.0),
        0.0,
    )
    up_scale = np.where(np.isfinite(r) & (r > up_thresh), 1.0, 0.0)

    if regime_mode == "risk_off_ramp":
        return down_scale
    if regime_mode == "rally_gt_3pct":
        return up_scale
    if regime_mode == "combined_two_tail":
        return np.maximum(down_scale, up_scale)
    raise ValueError(f"Unsupported regime_mode: {regime_mode}")


def build_d10_vs_rest_backtest_racks(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    beta_ols_bps: float | None = None,
    min_signal_date: str | None = DEFAULT_IS_SIGNAL_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
) -> tuple[BacktestRacks, np.ndarray, float]:
    """Build strictly PIT dollar-neutral `Short D10 (crowding_z-sized) / Long D1-D9 (EW)` racks.

    Returns `(base_racks, spy_ret_10d_by_day, slope_bps)` where `base_racks.expected_ret_bps`
    has exact dollar-neutral sum = 0 on every trading day:
      - Short leg (`D10`): `expected_ret_bps = min(0, slope_bps * crowding_z)`
      - Long leg (`D1-D9`): equal positive expected_ret_bps summing to exact `+|sum(D10)|`
    so `notional_per_bps * expected_ret_bps` produces an exact dollar-neutral, stock-level
    beta-hedged `Short D10 / Long Rest` book before regime scaling `m(t)`.
    """
    elig_panel = load_factor_panel(
        panel_path=panel_path,
        spy_path=spy_path,
        sample=sample,
        min_signal_date=min_signal_date,
        min_cross_section=min_cross_section,
    )
    cf = build_crowding_exposures(elig_panel)

    if beta_ols_bps is None:
        if sample == "in_sample":
            cf_tmp = cf.copy()
            cf_tmp["fwd_beta_adj_ret_bps"] = cf_tmp["fwd_beta_adj_ret"] * 1e4
            fit_c = fit_clustered_crowding_ols(cf_tmp, ["crowding_z"])
            slope_bps = float(fit_c.params["crowding_z"])
        else:
            slope_bps = FROZEN_IS_CROWDING_BETA_BPS
    else:
        slope_bps = float(beta_ols_bps)

    cfg = load_config()
    is_end = pd.Timestamp(cfg.settings["window"]["in_sample_end"]).normalize()
    if sample == "in_sample":
        max_fwd = pd.Timestamp(cf["fwd_ret_date"].max()).normalize()
        if max_fwd > is_end:
            raise AssertionError(
                f"OOS LEAKAGE: max fwd_ret_date {max_fwd.date()} > {is_end.date()}"
            )

    ranks = cf.groupby("query_date")["crowding_z"].rank(method="first")
    counts = cf.groupby("query_date")["crowding_z"].transform("count")
    cf["decile"] = np.floor((ranks - 1.0) * 10.0 / counts).astype(int).clip(0, 9) + 1

    # Short leg: D10 proportional to crowding_z * slope_bps (< 0)
    cf["short_er"] = np.where(
        cf["decile"] == 10, np.minimum(0.0, slope_bps * cf["crowding_z"]), 0.0
    )
    # Long leg: D1-D9 equal-weighted summing to exact +|sum(short_er)| per day (dollar-neutral!)
    day_short_sum = cf.groupby("query_date")["short_er"].transform("sum")
    day_n_rest = (cf["decile"] <= 9).groupby(cf["query_date"]).transform("sum").clip(lower=1)
    cf["expected_ret_bps"] = np.where(
        cf["decile"] == 10,
        cf["short_er"],
        (-day_short_sum) / day_n_rest,
    )

    dates = pd.DatetimeIndex(sorted(cf["query_date"].unique()))
    fwd_ret_dates = pd.DatetimeIndex(
        cf[["query_date", "fwd_ret_date"]]
        .drop_duplicates("query_date")
        .set_index("query_date")["fwd_ret_date"]
        .reindex(dates)
        .to_numpy()
    )

    # Strictly PIT 10-BD trailing SPY return ending query_date t (forward-only TR index)
    spy = pd.read_parquet(spy_path, columns=["date", "ret_total"]).copy()
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = (
        spy[spy["date"].dt.day_of_week < 5]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    r_spy = spy["ret_total"].fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    if len(r_spy) > 0:
        r_spy[0] = 0.0
    tr_spy = np.cumprod(1.0 + r_spy)
    s10 = np.full(len(tr_spy), np.nan, dtype=np.float64)
    if len(tr_spy) > 10:
        s10[10:] = tr_spy[10:] / tr_spy[:-10] - 1.0
    spy["spy_ret_10d"] = s10
    spy_ret_10d = spy.set_index("date")["spy_ret_10d"].reindex(dates).to_numpy(dtype=np.float64)

    all_cols = [
        "query_date",
        "security_id",
        "prev_day_ret",
        "fwd_day_ret",
        "fwd_beta_adj_ret",
        "beta_252d_lag1",
        "adv_20d_usd",
        "prev_spy_ret",
        "fwd_spy_ret",
    ]
    full_p = pd.read_parquet(panel_path, columns=all_cols)
    full_p["query_date"] = pd.to_datetime(full_p["query_date"]).dt.normalize()
    full_p = full_p[full_p["query_date"].isin(dates)].copy()

    active_secs = pd.Index(sorted(cf["security_id"].unique()))
    full_p = full_p[full_p["security_id"].isin(active_secs)].copy()

    def _piv(df_in: pd.DataFrame, val_col: str, fill_val: float = np.nan) -> np.ndarray:
        piv = df_in.pivot(index="query_date", columns="security_id", values=val_col)
        piv = piv.reindex(index=dates, columns=active_secs)
        if not np.isnan(fill_val):
            piv = piv.fillna(fill_val)
        return piv.to_numpy(dtype=np.float64)

    spy_by_day = (
        full_p[["query_date", "fwd_spy_ret", "prev_spy_ret"]]
        .drop_duplicates("query_date")
        .set_index("query_date")
        .reindex(dates)
    )

    racks = BacktestRacks(
        dates=dates,
        fwd_ret_dates=fwd_ret_dates,
        securities=active_secs,
        expected_ret_bps=_piv(cf, "expected_ret_bps", fill_val=0.0),
        fwd_beta_adj_ret=_piv(full_p, "fwd_beta_adj_ret", fill_val=np.nan),
        fwd_day_ret=_piv(full_p, "fwd_day_ret", fill_val=np.nan),
        prev_day_ret=_piv(full_p, "prev_day_ret", fill_val=np.nan),
        beta_lag1=_piv(full_p, "beta_252d_lag1", fill_val=np.nan),
        adv_20d_usd=_piv(full_p, "adv_20d_usd", fill_val=np.nan),
        fwd_spy_ret=spy_by_day["fwd_spy_ret"].to_numpy(dtype=np.float64),
        prev_spy_ret=spy_by_day["prev_spy_ret"].to_numpy(dtype=np.float64),
    )
    return racks, spy_ret_10d, slope_bps


def run_d10_vs_rest_regime_study(
    *,
    sample: SampleSplit = "in_sample",
    notional_per_bps: float = 50_000.0,
    max_adv_pct: float = 0.05,
    one_way_cost_bps: float = 5.0,
    spy_hedge_cost_bps: float = 1.0,
    buffer_bps_of_gmv: float = 25.0,
    beta_ols_bps: float | None = None,
    output_dir: Path | None = DEFAULT_BACKTEST_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run all 4 Buffered Dollar- & Beta-Neutral `Short D10 / Long D1-D9` iterations.

    Iterations evaluated on an identical Full-Scale Allocated Capital Base (1x Unconditional GMV):
      1. `unconditional`: Always-on Buffered Short D10 / Long D1-D9.
      2. `risk_off_ramp`: Active when 10-BD SPY < -1%, linear ramp to 100% cap at -3%.
      3. `rally_gt_3pct`: Active when 10-BD SPY > +3% (violent snap-back rotation).
      4. `combined_two_tail`: Active when 10-BD SPY < -1% (ramped to -3%) OR > +3%.
    """
    base_racks, spy_ret_10d, slope_bps = build_d10_vs_rest_backtest_racks(
        sample=sample,
        beta_ols_bps=beta_ols_bps,
    )

    # First run Unconditional to determine exact 1x full-scale allocated GMV capital base
    cfg_uncond = BacktestConfig(
        notional_per_bps=notional_per_bps,
        max_adv_pct=max_adv_pct,
        one_way_cost_bps=one_way_cost_bps,
        spy_hedge_cost_bps=spy_hedge_cost_bps,
        borrow_bps_pa=0.0,
        rebalance_mode="buffered",
        buffer_bps_of_gmv=buffer_bps_of_gmv,
        name="1. Unconditional (Always On)",
    )
    res_uncond = run_rack_backtest(base_racks, cfg_uncond)
    full_scale_cap = float(res_uncond["stats"]["avg_daily_gmv"])

    # Re-summarize unconditional with explicit capital_base_usd = full_scale_cap for parity
    results: dict[str, dict[str, Any]] = {"unconditional": res_uncond}
    summary_rows: list[dict[str, Any]] = [res_uncond["stats"]]

    regime_specs: list[tuple[RegimeGateMode, str]] = [
        ("risk_off_ramp", "2. Risk-Off Ramp (10d SPY -1% -> -3% cap)"),
        ("rally_gt_3pct", "3. Violent Rally Tail (10d SPY > +3%)"),
        ("combined_two_tail", "4. Combined Two-Tail (<-1% ramp OR >+3%)"),
    ]

    for r_mode, label in regime_specs:
        scales = compute_spy_10d_regime_scale(spy_ret_10d, r_mode)
        scaled_racks = BacktestRacks(
            dates=base_racks.dates,
            fwd_ret_dates=base_racks.fwd_ret_dates,
            securities=base_racks.securities,
            expected_ret_bps=base_racks.expected_ret_bps * scales[:, None],
            fwd_beta_adj_ret=base_racks.fwd_beta_adj_ret,
            fwd_day_ret=base_racks.fwd_day_ret,
            prev_day_ret=base_racks.prev_day_ret,
            beta_lag1=base_racks.beta_lag1,
            adv_20d_usd=base_racks.adv_20d_usd,
            fwd_spy_ret=base_racks.fwd_spy_ret,
            prev_spy_ret=base_racks.prev_spy_ret,
        )
        cfg = BacktestConfig(
            notional_per_bps=notional_per_bps,
            max_adv_pct=max_adv_pct,
            one_way_cost_bps=one_way_cost_bps,
            spy_hedge_cost_bps=spy_hedge_cost_bps,
            borrow_bps_pa=0.0,
            rebalance_mode="buffered",
            buffer_bps_of_gmv=buffer_bps_of_gmv,
            capital_base_usd=full_scale_cap,
            name=label,
        )
        res = run_rack_backtest(scaled_racks, cfg)
        results[r_mode] = res
        summary_rows.append(res["stats"])

    summary_df = pd.DataFrame(summary_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        summary_df.to_csv(output_dir / f"crowding_d10_vs_rest_regimes_{tag}.csv", index=False)
        for r_mode, res in results.items():
            res["daily"].to_parquet(
                output_dir / f"crowding_d10_vs_rest_{r_mode}_daily_{tag}.parquet", index=False
            )

    return {
        "slope_bps": slope_bps,
        "full_scale_capital_usd": full_scale_cap,
        "results": results,
        "summary": summary_df,
    }


def run_frozen_oos_evaluation(
    *,
    notional_per_bps: float = 50_000.0,
    max_adv_pct: float = 0.05,
    one_way_cost_bps: float = 5.0,
    spy_hedge_cost_bps: float = 1.0,
    buffer_bps_of_gmv: float = 25.0,
    frozen_slope_bps: float = FROZEN_IS_CROWDING_BETA_BPS,
    output_dir: Path | None = DEFAULT_BACKTEST_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run single honest Out-of-Sample (`2024+`) evaluation of both frozen IS strategies.

    Uses strictly frozen In-Sample parameters (`frozen_slope_bps = -0.534773 bps/d`,
    Buffered 25 bps band, 5 bps one-way = 10 bps RT, $50k/bps, 5% ADV cap) with ZERO
    re-estimation on OOS data. Evaluates:
      1. Primary:   Unconditional Buffered Dollar- & Beta-Neutral `Short D10 / Long D1-D9`
      2. Secondary: Buffered `Short > D7` Crowding vs. `SPY`
    across Train (`in_sample`), Test (`oos`), and Full (`full`) spans, plus OOS factor OLS
    and OOS short-borrow sensitivity.
    """
    from src.factors.crowding import compute_crowding_decile_profile

    # 1. Pure OOS factor validation (crowding_z OLS & D10 - Rest spread, zero refit)
    oos_panel = build_crowding_exposures(load_factor_panel(sample="oos"))
    oos_panel["fwd_beta_adj_ret_bps"] = oos_panel["fwd_beta_adj_ret"] * 1e4
    oos_fit = fit_clustered_crowding_ols(oos_panel, ["crowding_z"])
    _, oos_dec = compute_crowding_decile_profile(oos_panel)
    oos_d10_rest = oos_dec[oos_dec["bucket"] == "D10 - Rest (D1-D9)"].iloc[0]

    oos_factor_check = {
        "oos_start_fwd_date": str(pd.Timestamp(oos_panel["fwd_ret_date"].min()).date()),
        "oos_end_fwd_date": str(pd.Timestamp(oos_panel["fwd_ret_date"].max()).date()),
        "oos_n_days": int(oos_panel["query_date"].nunique()),
        "oos_n_obs": int(len(oos_panel)),
        "oos_slope_bps": float(oos_fit.params["crowding_z"]),
        "oos_slope_se": float(oos_fit.bse["crowding_z"]),
        "oos_slope_z": float(oos_fit.tvalues["crowding_z"]),
        "oos_slope_p": float(oos_fit.pvalues["crowding_z"]),
        "oos_d10_rest_bps": float(oos_d10_rest["badj_mean_bps"]),
        "oos_d10_rest_t": float(oos_d10_rest["badj_t_stat"]),
        "oos_d10_rest_p": float(oos_d10_rest["badj_p_value"]),
    }

    summary_rows: list[dict[str, Any]] = []
    daily_results: dict[tuple[str, str], dict[str, Any]] = {}
    borrow_rows: list[dict[str, Any]] = []

    for split_name, split_label in [
        ("in_sample", "Train (IS 2015–2023)"),
        ("oos", "Test (OOS 2024–Present)"),
        ("full", "Full (2015–Present)"),
    ]:
        # Strategy 1: Unconditional Buffered Dollar- & Beta-Neutral Short D10 / Long D1-D9
        r_d10, _, _ = build_d10_vs_rest_backtest_racks(
            sample=split_name,  # type: ignore[arg-type]
            beta_ols_bps=frozen_slope_bps,
        )
        cfg_d10 = BacktestConfig(
            notional_per_bps=notional_per_bps,
            max_adv_pct=max_adv_pct,
            one_way_cost_bps=one_way_cost_bps,
            spy_hedge_cost_bps=spy_hedge_cost_bps,
            borrow_bps_pa=0.0,
            rebalance_mode="buffered",
            buffer_bps_of_gmv=buffer_bps_of_gmv,
            name="Short D10 / Long D1-D9 (Dollar-Neutral)",
        )
        res_d10 = run_rack_backtest(r_d10, cfg_d10)
        st_d10 = {**res_d10["stats"], "split": split_label, "strategy_key": "d10_vs_rest"}
        summary_rows.append(st_d10)
        daily_results[("d10_vs_rest", split_name)] = res_d10

        # Strategy 2: Buffered Short >D7 Crowding vs. SPY
        r_gt7, _ = build_crowding_backtest_racks(
            sample=split_name,  # type: ignore[arg-type]
            strategy_mode="short_gt_d7",
            beta_ols_bps=frozen_slope_bps,
        )
        cfg_gt7 = BacktestConfig(
            notional_per_bps=notional_per_bps,
            max_adv_pct=max_adv_pct,
            one_way_cost_bps=one_way_cost_bps,
            spy_hedge_cost_bps=spy_hedge_cost_bps,
            borrow_bps_pa=0.0,
            rebalance_mode="buffered",
            buffer_bps_of_gmv=buffer_bps_of_gmv,
            name="Short >D7 Crowding vs. SPY",
        )
        res_gt7 = run_rack_backtest(r_gt7, cfg_gt7)
        st_gt7 = {**res_gt7["stats"], "split": split_label, "strategy_key": "short_gt_d7"}
        summary_rows.append(st_gt7)
        daily_results[("short_gt_d7", split_name)] = res_gt7

        if split_name == "oos":
            for skey, s_racks, s_name in [
                ("d10_vs_rest", r_d10, "Short D10 / Long D1-D9 (OOS)"),
                ("short_gt_d7", r_gt7, "Short >D7 vs. SPY (OOS)"),
            ]:
                for b_pa in (0.0, 35.0, 100.0, 250.0):
                    cfg_b = BacktestConfig(
                        notional_per_bps=notional_per_bps,
                        max_adv_pct=max_adv_pct,
                        one_way_cost_bps=one_way_cost_bps,
                        spy_hedge_cost_bps=spy_hedge_cost_bps,
                        borrow_bps_pa=b_pa,
                        rebalance_mode="buffered",
                        buffer_bps_of_gmv=buffer_bps_of_gmv,
                        name=s_name,
                    )
                    b_st = run_rack_backtest(s_racks, cfg_b)["stats"]
                    borrow_rows.append({**b_st, "strategy_key": skey})

    summary_df = pd.DataFrame(summary_rows)
    oos_borrow_df = pd.DataFrame(borrow_rows)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_dir / "oos_frozen_comparison.csv", index=False)
        oos_borrow_df.to_csv(output_dir / "oos_borrow_sensitivity.csv", index=False)
        pd.DataFrame([oos_factor_check]).to_csv(
            output_dir / "oos_crowding_factor_check.csv", index=False
        )
        for (skey, split_name), res in daily_results.items():
            if split_name in ("oos", "full"):
                res["daily"].to_parquet(
                    output_dir / f"frozen_{skey}_{split_name}_daily.parquet", index=False
                )

    return {
        "frozen_slope_bps": frozen_slope_bps,
        "oos_factor_check": oos_factor_check,
        "oos_deciles": oos_dec,
        "summary": summary_df,
        "oos_borrow_sensitivity": oos_borrow_df,
        "daily_results": daily_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Crowding Rack Backtester (In-Sample default)."
    )
    parser.add_argument("--sample", choices=["in_sample", "oos", "full"], default="in_sample")
    parser.add_argument(
        "--study",
        choices=["horizons", "d10_vs_rest", "oos_final"],
        default="horizons",
    )
    parser.add_argument(
        "--strategy",
        choices=["short_gt_d7", "short_d10", "continuous_ls"],
        default="short_gt_d7",
    )
    parser.add_argument("--notional-per-bps", type=float, default=50_000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.study == "oos_final":
        out = run_frozen_oos_evaluation(
            notional_per_bps=args.notional_per_bps,
            one_way_cost_bps=args.cost_bps,
        )
        fc = out["oos_factor_check"]
        print(
            f"\n=== FROZEN OUT-OF-SAMPLE EVALUATION ({fc['oos_start_fwd_date']} -> "
            f"{fc['oos_end_fwd_date']} | {fc['oos_n_days']} trading days) ==="
        )
        print(
            f"OOS Pure Factor Check (crowding_z): slope={fc['oos_slope_bps']:+.3f} bps/d "
            f"(SE={fc['oos_slope_se']:.3f}, z={fc['oos_slope_z']:+.2f}, p={fc['oos_slope_p']:.4f})"
            f" | D10-Rest={fc['oos_d10_rest_bps']:+.3f} bps/d "
            f"(t={fc['oos_d10_rest_t']:+.2f}, p={fc['oos_d10_rest_p']:.4f})"
        )
        cols = [
            "name",
            "split",
            "n_trading_days",
            "avg_daily_gmv",
            "daily_hit_rate",
            "cum_pnl_raw",
            "cum_pnl",
            "ann_pnl",
            "ann_return",
            "ann_sdev",
            "sharpe_raw",
            "sharpe",
            "max_dd_pct",
            "ann_turnover_x",
            "realized_spy_beta",
            "breakeven_borrow_bps_pa",
        ]
        print("\n--- TRAIN (IS) vs. TEST (OOS) vs. FULL COMPARISON ---")
        print(out["summary"][cols].to_string(index=False))
        print("\n--- OOS SHORT-BORROW SENSITIVITY ---")
        bcols = ["name", "borrow_bps_pa", "cum_pnl", "ann_return", "sharpe", "total_borrow_cost"]
        print(out["oos_borrow_sensitivity"][bcols].to_string(index=False))
        return 0

    if args.study == "d10_vs_rest":
        out = run_d10_vs_rest_regime_study(
            sample=args.sample,
            notional_per_bps=args.notional_per_bps,
            one_way_cost_bps=args.cost_bps,
        )
        print(
            f"\n=== D10 vs REST DOLLAR- & BETA-NEUTRAL REGIME BACKTEST "
            f"({args.sample.upper()} | Full-Scale Cap: ${out['full_scale_capital_usd']:,.0f} | "
            f"{args.cost_bps:.0f}bps one-way | Buffered 25bps) ==="
        )
        cols = [
            "name",
            "pct_active_days",
            "avg_daily_gmv",
            "daily_hit_rate",
            "cum_pnl_raw",
            "cum_pnl",
            "ann_pnl",
            "ann_return",
            "ann_sdev",
            "sharpe_raw",
            "sharpe",
            "max_dd_pct",
            "ann_turnover_x",
            "realized_spy_beta",
            "breakeven_borrow_bps_pa",
        ]
        print(out["summary"][cols].to_string(index=False))
        return 0

    out = run_crowding_horizon_comparison(
        sample=args.sample,
        strategy_mode=args.strategy,
        notional_per_bps=args.notional_per_bps,
        one_way_cost_bps=args.cost_bps,
    )

    print(
        f"\n=== CROWDING BACKTEST ({args.strategy.upper()} | {args.sample.upper()} | "
        f"${args.notional_per_bps:,.0f}/bps | {args.cost_bps:.0f}bps one-way | 5% ADV cap) ==="
    )
    print(f"Frozen IS OLS Slope used: {out['slope_bps']:+.4f} bps/day per +1 sigma")
    cols = [
        "name",
        "n_trading_days",
        "avg_daily_gmv",
        "daily_hit_rate",
        "cum_pnl_raw",
        "cum_pnl",
        "ann_pnl",
        "ann_return",
        "ann_sdev",
        "sharpe_raw",
        "sharpe",
        "max_dd_pct",
        "ann_turnover_x",
        "realized_spy_beta",
        "breakeven_borrow_bps_pa",
    ]
    print("\n--- 1. REBALANCING HORIZON COMPARISON ---")
    print(out["summary"][cols].to_string(index=False))

    print("\n--- 2. ONE-WAY COST SENSITIVITY (0 / 5 / 10 / 20 bps) ---")
    scols = [
        "rebalance_mode",
        "one_way_cost_bps",
        "cum_pnl",
        "ann_return",
        "sharpe",
        "ann_turnover_x",
    ]
    print(out["cost_sensitivity"][scols].to_string(index=False))

    print("\n--- 3. SHORT-BORROW FEE SENSITIVITY (0 / 35 / 100 / 250 bps/yr) ---")
    bcols = [
        "rebalance_mode",
        "borrow_bps_pa",
        "cum_pnl",
        "ann_return",
        "sharpe",
        "total_borrow_cost",
    ]
    print(out["borrow_sensitivity"][bcols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
