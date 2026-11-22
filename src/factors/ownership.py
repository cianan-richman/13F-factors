"""Pure Institutional Ownership (`pct_shares_owned`) Cross-Sectional Factor Engine.

Constructs and evaluates our first clean 13F positioning signal — aggregate percentage of
shares outstanding owned by our 100 audited fundamental managers (`pct_shares_owned`) —
using daily cross-sectional OLS factor returns and equal-weighted quantile sorts strictly
inside the In-Sample window by default (`fwd_ret_date <= 2023-12-31`, zero OOS touch).

Mathematical conventions:
  - Signal date `t` (`query_date`): all inputs strictly PIT as of close of day `t`
    (13F filings accepted <= t-1, XBRL shares filed < t, dynamic thresholds using SPY up to t-1,
    masked 252-day OLS beta estimated strictly over `[t-252, t-1]`).
  - Realized return date `t+1` (`fwd_ret_date`): next consecutive US trading day on the
    scrubbed `SPY` calendar (`cal_step == 1`).
  - Stock-level beta-adjusted forward return:
        r_resid_{i,t+1} = r_{i,t+1} - beta_{i,t} * r_spy_{t+1}
  - Daily cross-sectional OLS factor return (slope of next-day beta-adjusted return on
    standardized day-`t` exposure `z_{i,t}`):
        r_resid_{i,t+1} = alpha_t + f_beta_adj_{t+1} * z_{i,t} + eps_{i,t+1}
        w_{i,t} = (z_{i,t} - mean(z_t)) / sum_j (z_{j,t} - mean(z_t))^2
        f_beta_adj_{t+1} = sum_i w_{i,t} * r_resid_{i,t+1}
                         = f_raw_{t+1} - (sum_i w_{i,t} * beta_{i,t}) * r_spy_{t+1}
    All statistics evaluate the pure regression slope `f_beta_adj` (return per +1 sigma
    exposure) without imposing trading direction, leverage scaling, or portfolio rules.

Usage:
    python -m src.factors.ownership
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, load_config
from src.pit.daily_panel import DEFAULT_DAILY_PANEL_PARQUET

logger = logging.getLogger(__name__)

SampleSplit = Literal["in_sample", "oos", "full"]

DEFAULT_SPY_PARQUET = PROCESSED_DATA_DIR / "spy_prices.parquet"
DEFAULT_FACTORS_OUTPUT_DIR = OUTPUTS_DIR / "factors"
# First trading day after the statutory 45-day Q4 2014 13F deadline (2015-02-17)
DEFAULT_IS_SIGNAL_START = "2015-02-18"
DEFAULT_MIN_CROSS_SECTION = 100
DEFAULT_WINSOR_LIMITS = (0.01, 0.99)
TRADING_DAYS_PER_YEAR = 252

_STD_NORMAL = NormalDist(0.0, 1.0)


def _norm_ppf(probs: np.ndarray) -> np.ndarray:
    """Vectorized inverse standard normal CDF (uses scipy if installed, else Python stdlib)."""
    try:
        from scipy.special import ndtri  # type: ignore[import-untyped]

        return np.asarray(ndtri(probs), dtype=np.float64)
    except Exception:
        return np.fromiter((_STD_NORMAL.inv_cdf(float(p)) for p in probs), dtype=np.float64)


def attach_fwd_ret_date(
    panel: pd.DataFrame,
    spy_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact next US trading day (`fwd_ret_date`) from the scrubbed SPY calendar."""
    spy_s = spy_prices.copy()
    spy_s["date"] = pd.to_datetime(spy_s["date"]).dt.normalize()
    spy_dates = pd.DatetimeIndex(
        spy_s.loc[spy_s["date"].dt.day_of_week < 5, "date"].drop_duplicates().sort_values()
    )
    next_date_map = pd.Series(spy_dates[1:], index=spy_dates[:-1])

    out = panel.copy()
    out["query_date"] = pd.to_datetime(out["query_date"]).dt.normalize()
    out["fwd_ret_date"] = out["query_date"].map(next_date_map)
    return out


def load_factor_panel(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    min_signal_date: str | None = DEFAULT_IS_SIGNAL_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    require_fwd_returns: bool = True,
) -> pd.DataFrame:
    """Load eligible daily PIT panel strictly respecting IS/OOS return-date boundaries.

    Critical zero-OOS-leakage guarantee:
      For `sample="in_sample"`, we filter on `fwd_ret_date <= in_sample_end` (NOT merely
      `query_date <= in_sample_end`), so `query_date = 2023-12-29` (whose forward return is
      earned on `2024-01-02`, the first OOS trading day) is strictly excluded from IS.

    Pass `require_fwd_returns=False` when building backtest signal racks so day-`t` portfolio
    eligibility and cross-sectional z-scores never condition on `t+1` return availability.
    """
    cfg = load_config()
    window = cfg.settings["window"]
    is_end = pd.Timestamp(window["in_sample_end"]).normalize()
    oos_start = pd.Timestamp(window["oos_start"]).normalize()

    cols = [
        "query_date",
        "security_id",
        "yahoo_ticker",
        "n_filers",
        "pct_shares_owned",
        "days_to_liquidate",
        "pit_mcap_usd",
        "adv_20d_usd",
        "beta_252d_lag1",
        "fwd_day_ret",
        "fwd_spy_ret",
        "fwd_beta_adj_ret",
        "is_panel_eligible",
    ]
    panel = pd.read_parquet(panel_path, columns=cols)
    spy = pd.read_parquet(spy_path, columns=["date"])
    panel = attach_fwd_ret_date(panel, spy)

    # Strict sample boundary filter on realized forward-return date
    if sample == "in_sample":
        panel = panel[panel["fwd_ret_date"].notna() & (panel["fwd_ret_date"] <= is_end)].copy()
        if not panel.empty and panel["fwd_ret_date"].max() > is_end:
            raise ValueError("IS leakage guard triggered: fwd_ret_date > in_sample_end!")
    elif sample == "oos":
        panel = panel[panel["fwd_ret_date"].notna() & (panel["fwd_ret_date"] >= oos_start)].copy()
    elif sample == "full":
        panel = panel[panel["fwd_ret_date"].notna()].copy()
    else:
        raise ValueError(f"Unsupported sample split: {sample!r}")

    if min_signal_date is not None:
        panel = panel[panel["query_date"] >= pd.Timestamp(min_signal_date).normalize()].copy()

    # Keep strictly eligible rows with valid day-t signal, size, and beta (and t+1 returns if OLS)
    valid = (
        panel["is_panel_eligible"].fillna(False)
        & panel["pct_shares_owned"].notna()
        & (panel["pct_shares_owned"] > 0.0)
        & panel["pit_mcap_usd"].notna()
        & (panel["pit_mcap_usd"] > 0.0)
        & panel["beta_252d_lag1"].notna()
    )
    if require_fwd_returns:
        valid = (
            valid
            & panel["fwd_day_ret"].notna()
            & panel["fwd_spy_ret"].notna()
            & panel["fwd_beta_adj_ret"].notna()
        )
    panel = panel.loc[valid].copy()

    # Enforce minimum cross-section breadth per day
    if min_cross_section > 1 and not panel.empty:
        day_counts = panel.groupby("query_date")["security_id"].transform("count")
        panel = panel.loc[day_counts >= min_cross_section].copy()

    # Verify exact beta-adjusted return algebraic identity on every single row
    if not panel.empty:
        expected_badj = panel["fwd_day_ret"] - panel["beta_252d_lag1"] * panel["fwd_spy_ret"]
        max_err = float(np.max(np.abs(panel["fwd_beta_adj_ret"] - expected_badj)))
        if max_err > 1e-10:
            raise ValueError(
                f"Beta-adjusted return identity violated: max_err={max_err:.3e}"
            )

    return panel.sort_values(["query_date", "security_id"]).reset_index(drop=True)


def _standardize_vec(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-std standardization (returns NaNs if zero variance or < 3 obs)."""
    arr = np.asarray(x, dtype=np.float64)
    if len(arr) < 3:
        return np.full_like(arr, np.nan)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=0))
    if not np.isfinite(sd) or sd <= 1e-12:
        return np.full_like(arr, np.nan)
    return (arr - mu) / sd


def _orthogonalize_against(target: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Cross-sectional OLS residual of `target` on `[1, control]`, re-standardized to (0, 1)."""
    t = np.asarray(target, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    tc = t - np.mean(t)
    cc = c - np.mean(c)
    denom = float(np.dot(cc, cc))
    if denom <= 1e-12:
        return _standardize_vec(t)
    slope = float(np.dot(tc, cc)) / denom
    resid = tc - slope * cc
    return _standardize_vec(resid)


def build_pct_ownership_exposures(
    panel: pd.DataFrame,
    *,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.DataFrame:
    """Attach strictly cross-sectional day-`t` standardized `pct_shares_owned` exposures.

    Constructs four transparent exposure variants on each `query_date` using ONLY day-`t` data:
      - `pct_own_winsor_z`: 1%/99% cross-sectionally winsorized `pct_shares_owned` z-score
      - `pct_own_rank_z`: Gaussianized cross-sectional rank z-score (`Phi^-1((r - 0.5) / N)`)
      - `pct_own_size_neutral_z`: `pct_own_winsor_z` orthogonalized against `log(pit_mcap_usd)`
      - `pct_own_rank_size_neutral_z`: `pct_own_rank_z` orthogonalized against `log(pit_mcap_usd)`
    """
    if panel.empty:
        out = panel.copy()
        for col in (
            "log_mcap",
            "pct_own_winsor_z",
            "pct_own_rank_z",
            "pct_own_size_neutral_z",
            "pct_own_rank_size_neutral_z",
        ):
            out[col] = np.nan
        return out

    df = panel.sort_values(["query_date", "security_id"]).reset_index(drop=True).copy()
    df["log_mcap"] = np.log(df["pit_mcap_usd"].to_numpy(dtype=np.float64))

    n_rows = len(df)
    winsor_z = np.full(n_rows, np.nan, dtype=np.float64)
    rank_z = np.full(n_rows, np.nan, dtype=np.float64)
    size_neut_z = np.full(n_rows, np.nan, dtype=np.float64)
    rank_size_neut_z = np.full(n_rows, np.nan, dtype=np.float64)

    lo_q, hi_q = winsor_limits
    pct_all = df["pct_shares_owned"].to_numpy(dtype=np.float64)
    lmcap_all = df["log_mcap"].to_numpy(dtype=np.float64)

    # Identify contiguous day slices (panel is sorted by query_date)
    dates_ns = df["query_date"].to_numpy(dtype="datetime64[ns]")
    change_idx = np.flatnonzero(dates_ns[1:] != dates_ns[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n_rows]))

    for s, e in zip(starts, ends, strict=False):
        x = pct_all[s:e]
        lm = lmcap_all[s:e]
        n = e - s
        if n < 3:
            continue

        # 1. Winsorized z-score
        lo, hi = np.quantile(x, [lo_q, hi_q])
        x_w = np.clip(x, lo, hi)
        wz = _standardize_vec(x_w)

        # 2. Gaussianized rank z-score (average ranks on ties)
        ranks = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
        probs = (ranks - 0.5) / float(n)
        rz = _standardize_vec(_norm_ppf(probs))

        # 3. Size-neutralized variants (cross-sectional OLS residual vs log_mcap)
        sn_wz = _orthogonalize_against(wz, lm)
        sn_rz = _orthogonalize_against(rz, lm)

        winsor_z[s:e] = wz
        rank_z[s:e] = rz
        size_neut_z[s:e] = sn_wz
        rank_size_neut_z[s:e] = sn_rz

    df["pct_own_winsor_z"] = winsor_z
    df["pct_own_rank_z"] = rank_z
    df["pct_own_size_neutral_z"] = size_neut_z
    df["pct_own_rank_size_neutral_z"] = rank_size_neut_z
    return df


def compute_daily_factor_returns(
    panel_with_z: pd.DataFrame,
    signal_col: str = "pct_own_winsor_z",
) -> pd.DataFrame:
    """Compute daily cross-sectional OLS factor returns and regression diagnostics.

    For each day `t`, estimates the cross-sectional slope `f_{t+1}` of next-day return on
    standardized day-`t` exposure `z_{i,t}`:
        w_{i,t} = (z_{i,t} - mean(z_t)) / sum_j (z_{j,t} - mean(z_t))^2
    which satisfies `sum w = 0` and `sum w*z = 1` (unit +1 sigma exposure), and verifies the
    exact algebraic identity:
        f_beta_adj == f_raw - ex_ante_beta * fwd_spy_ret.

    Also computes daily cross-sectional R^2, Spearman Rank IC, and cross-sectional
    correlations with `log_mcap` and `beta_252d_lag1`.
    """
    if panel_with_z.empty:
        return pd.DataFrame()

    df = panel_with_z.dropna(
        subset=[signal_col, "fwd_day_ret", "fwd_spy_ret", "fwd_beta_adj_ret", "beta_252d_lag1"]
    ).sort_values(["query_date", "security_id"])

    z_all = df[signal_col].to_numpy(dtype=np.float64)
    r_raw_all = df["fwd_day_ret"].to_numpy(dtype=np.float64)
    r_badj_all = df["fwd_beta_adj_ret"].to_numpy(dtype=np.float64)
    r_spy_all = df["fwd_spy_ret"].to_numpy(dtype=np.float64)
    beta_all = df["beta_252d_lag1"].to_numpy(dtype=np.float64)
    lmcap_all = df["log_mcap"].to_numpy(dtype=np.float64)
    q_dates = df["query_date"].to_numpy(dtype="datetime64[ns]")
    fwd_dates = df["fwd_ret_date"].to_numpy(dtype="datetime64[ns]")

    n_rows = len(df)
    change_idx = np.flatnonzero(q_dates[1:] != q_dates[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n_rows]))

    rows: list[dict[str, Any]] = []
    for s, e in zip(starts, ends, strict=False):
        n = e - s
        if n < 3:
            continue
        z = z_all[s:e]
        r_raw = r_raw_all[s:e]
        r_badj = r_badj_all[s:e]
        beta = beta_all[s:e]
        lm = lmcap_all[s:e]
        spy_r = float(r_spy_all[s])

        zc = z - float(np.mean(z))
        ss_z = float(np.dot(zc, zc))
        if ss_z <= 1e-12:
            continue

        # Unit-exposure OLS slope weights: sum(w) = 0, sum(w * z) = 1
        w = zc / ss_z
        f_raw = float(np.dot(w, r_raw))
        f_badj = float(np.dot(w, r_badj))
        ex_ante_beta = float(np.dot(w, beta))

        # Exact algebraic identity check: stock-level beta hedge == portfolio-level beta hedge
        hedge_err = abs(f_badj - (f_raw - ex_ante_beta * spy_r))
        if hedge_err > 1e-10:
            raise ValueError(
                f"Factor beta-hedge identity error on {pd.Timestamp(q_dates[s]).date()}: "
                f"{hedge_err:.3e}"
            )

        # Daily cross-sectional R^2 (beta-adjusted) and Spearman Rank IC
        cs_corr = _safe_corr(z, r_badj)
        cs_r2 = float(cs_corr * cs_corr) if np.isfinite(cs_corr) else np.nan
        rz = pd.Series(z).rank(method="average").to_numpy(dtype=np.float64)
        rr_badj = pd.Series(r_badj).rank(method="average").to_numpy(dtype=np.float64)
        rr_raw = pd.Series(r_raw).rank(method="average").to_numpy(dtype=np.float64)
        rank_ic = _safe_corr(rz, rr_badj)
        rank_ic_raw = _safe_corr(rz, rr_raw)

        rows.append(
            {
                "query_date": pd.Timestamp(q_dates[s]),
                "fwd_ret_date": pd.Timestamp(fwd_dates[s]),
                "signal": signal_col,
                "n_stocks": int(n),
                "fwd_spy_ret": spy_r,
                "f_raw": f_raw,
                "f_beta_adj": f_badj,
                "ex_ante_beta": ex_ante_beta,
                "cs_r2": cs_r2,
                "rank_ic": rank_ic,
                "rank_ic_raw": rank_ic_raw,
                "size_corr": _safe_corr(z, lm),
                "beta_corr": _safe_corr(z, beta),
            }
        )

    return pd.DataFrame(rows)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    denom = float(np.sqrt(np.dot(ac, ac) * np.dot(bc, bc)))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(ac, bc) / denom)


def compute_quantile_portfolios(
    panel_with_z: pd.DataFrame,
    *,
    sort_col: str = "pct_shares_owned",
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Compute daily equal-weighted quantile portfolio returns (`raw` and `beta-adjusted`).

    Assigns stocks on each `query_date` into `n_buckets` equal-sized buckets (`1..n_buckets`)
    using rank-based cuts (guaranteed tie-safe), and returns daily bucket returns, ex-ante
    average betas, and the `Top - Bottom` equal-weighted long-short spread.
    """
    if panel_with_z.empty:
        return pd.DataFrame()

    df = panel_with_z.dropna(
        subset=[sort_col, "fwd_day_ret", "fwd_spy_ret", "fwd_beta_adj_ret", "beta_252d_lag1"]
    ).copy()

    # Tie-safe rank-based quantile assignment strictly within each query_date
    ranks = df.groupby("query_date")[sort_col].rank(method="first")
    counts = df.groupby("query_date")[sort_col].transform("count")
    df["bucket"] = (
        np.floor((ranks - 1.0) * n_buckets / counts).astype(int).clip(0, n_buckets - 1) + 1
    )

    agg = (
        df.groupby(["query_date", "fwd_ret_date", "bucket"], as_index=False)
        .agg(
            n_stocks=("security_id", "count"),
            mean_pct_owned=("pct_shares_owned", "mean"),
            mean_n_filers=("n_filers", "mean"),
            mean_beta=("beta_252d_lag1", "mean"),
            ew_raw_ret=("fwd_day_ret", "mean"),
            ew_beta_adj_ret=("fwd_beta_adj_ret", "mean"),
            fwd_spy_ret=("fwd_spy_ret", "first"),
        )
        .sort_values(["query_date", "bucket"])
        .reset_index(drop=True)
    )
    return agg


def _two_sided_t_pvalue(t_stat: float, df: int) -> float:
    """Compute exact two-sided Student-t p-value (falls back to normal if scipy unavailable)."""
    if not np.isfinite(t_stat) or df < 1:
        return np.nan
    try:
        from scipy.stats import t as student_t  # type: ignore[import-untyped]

        return float(2.0 * student_t.sf(abs(t_stat), df=df))
    except Exception:
        return float(2.0 * (1.0 - _STD_NORMAL.cdf(abs(t_stat))))


def compute_return_stats(
    ret: pd.Series | np.ndarray,
    spy_ret: pd.Series | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute transparent daily return statistics with exact two-sided Student-t p-value."""
    r = np.asarray(ret, dtype=np.float64)
    ok = np.isfinite(r)
    r = r[ok]
    n = len(r)
    if n < 2:
        return {
            "n_days": float(n),
            "mean_daily_bps": np.nan,
            "std_daily_bps": np.nan,
            "ann_return": np.nan,
            "cagr": np.nan,
            "ann_vol": np.nan,
            "sharpe": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "hit_rate": np.nan,
            "max_drawdown": np.nan,
            "realized_spy_beta": np.nan,
            "spy_r2": np.nan,
        }

    mean_d = float(np.mean(r))
    std_d = float(np.std(r, ddof=1))
    ann_ret = mean_d * TRADING_DAYS_PER_YEAR
    ann_vol = std_d * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else np.nan
    t_stat = mean_d / (std_d / np.sqrt(n)) if std_d > 1e-12 else np.nan
    p_value = _two_sided_t_pvalue(t_stat, df=n - 1) if np.isfinite(t_stat) else np.nan
    hit_rate = float(np.mean(r > 0.0))

    # Compounded equity curve & max drawdown (guard against <= -100% single-day blowups)
    wealth = np.cumprod(np.maximum(1.0 + r, 1e-9))
    cagr = float(wealth[-1] ** (TRADING_DAYS_PER_YEAR / n) - 1.0)
    peak = np.maximum.accumulate(wealth)
    max_dd = float(np.min(wealth / peak - 1.0))

    realized_beta = np.nan
    spy_r2 = np.nan
    if spy_ret is not None:
        s = np.asarray(spy_ret, dtype=np.float64)[ok]
        ok_s = np.isfinite(s)
        if np.sum(ok_s) >= 2:
            rv = float(np.var(r[ok_s], ddof=1))
            sv = float(np.var(s[ok_s], ddof=1))
            if sv > 1e-12:
                cov_rs = float(np.cov(r[ok_s], s[ok_s], ddof=1)[0, 1])
                realized_beta = cov_rs / sv
                if rv > 1e-12:
                    corr_rs = cov_rs / np.sqrt(rv * sv)
                    spy_r2 = float(corr_rs * corr_rs)

    return {
        "n_days": float(n),
        "mean_daily_bps": mean_d * 1e4,
        "std_daily_bps": std_d * 1e4,
        "ann_return": ann_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "t_stat": t_stat,
        "p_value": p_value,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "realized_spy_beta": realized_beta,
        "spy_r2": spy_r2,
    }


def summarize_quantile_table(
    bucket_df: pd.DataFrame,
    *,
    label_prefix: str = "Q",
) -> pd.DataFrame:
    """Summarize each quantile bucket + Top-Bottom spread on both raw and beta-adjusted returns."""
    if bucket_df.empty:
        return pd.DataFrame()

    n_b = int(bucket_df["bucket"].max())
    rows: list[dict[str, Any]] = []
    for b in range(1, n_b + 1):
        sub = bucket_df[bucket_df["bucket"] == b].sort_values("query_date")
        st_raw = compute_return_stats(sub["ew_raw_ret"], sub["fwd_spy_ret"])
        st_adj = compute_return_stats(sub["ew_beta_adj_ret"], sub["fwd_spy_ret"])
        rows.append(
            {
                "bucket": f"{label_prefix}{b}",
                "avg_n": float(sub["n_stocks"].mean()),
                "avg_pct_owned": float(sub["mean_pct_owned"].mean()),
                "avg_n_filers": float(sub["mean_n_filers"].mean()),
                "ex_ante_beta": float(sub["mean_beta"].mean()),
                "raw_ann_ret": st_raw["ann_return"],
                "raw_spy_beta": st_raw["realized_spy_beta"],
                "badj_ann_ret": st_adj["ann_return"],
                "badj_ann_vol": st_adj["ann_vol"],
                "badj_sharpe": st_adj["sharpe"],
                "badj_t_stat": st_adj["t_stat"],
                "badj_p_value": st_adj["p_value"],
                "badj_max_dd": st_adj["max_drawdown"],
                "badj_spy_beta": st_adj["realized_spy_beta"],
            }
        )

    # Top - Bottom Long-Short spread
    top = bucket_df[bucket_df["bucket"] == n_b].set_index("query_date")
    bot = bucket_df[bucket_df["bucket"] == 1].set_index("query_date")
    common = top.index.intersection(bot.index)
    ls_raw = top.loc[common, "ew_raw_ret"] - bot.loc[common, "ew_raw_ret"]
    ls_adj = top.loc[common, "ew_beta_adj_ret"] - bot.loc[common, "ew_beta_adj_ret"]
    spy_s = top.loc[common, "fwd_spy_ret"]
    st_ls_raw = compute_return_stats(ls_raw, spy_s)
    st_ls_adj = compute_return_stats(ls_adj, spy_s)
    rows.append(
        {
            "bucket": f"{label_prefix}{n_b}-{label_prefix}1",
            "avg_n": float((top.loc[common, "n_stocks"] + bot.loc[common, "n_stocks"]).mean()),
            "avg_pct_owned": float(
                (top.loc[common, "mean_pct_owned"] - bot.loc[common, "mean_pct_owned"]).mean()
            ),
            "avg_n_filers": float(
                (top.loc[common, "mean_n_filers"] - bot.loc[common, "mean_n_filers"]).mean()
            ),
            "ex_ante_beta": float(
                (top.loc[common, "mean_beta"] - bot.loc[common, "mean_beta"]).mean()
            ),
            "raw_ann_ret": st_ls_raw["ann_return"],
            "raw_spy_beta": st_ls_raw["realized_spy_beta"],
            "badj_ann_ret": st_ls_adj["ann_return"],
            "badj_ann_vol": st_ls_adj["ann_vol"],
            "badj_sharpe": st_ls_adj["sharpe"],
            "badj_t_stat": st_ls_adj["t_stat"],
            "badj_p_value": st_ls_adj["p_value"],
            "badj_max_dd": st_ls_adj["max_drawdown"],
            "badj_spy_beta": st_ls_adj["realized_spy_beta"],
        }
    )
    return pd.DataFrame(rows)


def summarize_annual_and_regimes(
    daily_df: pd.DataFrame,
    ret_col: str = "f_beta_adj",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build year-by-year and market-regime regression diagnostics for a daily factor return."""
    if daily_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = daily_df.copy()
    df["year"] = df["fwd_ret_date"].dt.year

    ann_rows: list[dict[str, Any]] = []
    for yr, grp in df.groupby("year", sort=True):
        st = compute_return_stats(grp[ret_col], grp["fwd_spy_ret"])
        ann_rows.append(
            {
                "year": int(yr),
                "n_days": int(st["n_days"]),
                "ann_return": st["ann_return"],
                "ann_vol": st["ann_vol"],
                "sharpe": st["sharpe"],
                "t_stat": st["t_stat"],
                "p_value": st["p_value"],
                "mean_rank_ic": float(grp["rank_ic"].mean()),
                "realized_spy_beta": st["realized_spy_beta"],
            }
        )
    annual_df = pd.DataFrame(ann_rows)

    # Market & stress regime diagnostics (strictly IS historical windows)
    spy_p10 = float(df["fwd_spy_ret"].quantile(0.10))
    spy_p90 = float(df["fwd_spy_ret"].quantile(0.90))
    regimes: list[tuple[str, pd.Series]] = [
        ("SPY Up Days (>0%)", df["fwd_spy_ret"] > 0.0),
        ("SPY Down Days (<0%)", df["fwd_spy_ret"] < 0.0),
        ("SPY Worst 10% Stress Days", df["fwd_spy_ret"] <= spy_p10),
        ("SPY Best 10% Rally Days", df["fwd_spy_ret"] >= spy_p90),
        (
            "2018 Q4 Selloff (Oct-Dec 2018)",
            (df["fwd_ret_date"] >= "2018-10-01") & (df["fwd_ret_date"] <= "2018-12-31"),
        ),
        (
            "2020 Q1 COVID Crash (Feb-Mar 2020)",
            (df["fwd_ret_date"] >= "2020-02-19") & (df["fwd_ret_date"] <= "2020-03-31"),
        ),
        (
            "2021 Q1 Factor/Meme Squeeze (Jan-Mar 2021)",
            (df["fwd_ret_date"] >= "2021-01-01") & (df["fwd_ret_date"] <= "2021-03-31"),
        ),
        (
            "2022 Rate/Growth De-Risking (Jan-Dec 2022)",
            (df["fwd_ret_date"] >= "2022-01-01") & (df["fwd_ret_date"] <= "2022-12-31"),
        ),
    ]
    reg_rows: list[dict[str, Any]] = []
    for name, mask in regimes:
        sub = df.loc[mask]
        if len(sub) < 5:
            continue
        r = sub[ret_col].to_numpy(dtype=np.float64)
        st = compute_return_stats(r, sub["fwd_spy_ret"])
        cum_ret = float(np.sum(r))
        mean_bps = float(np.mean(r) * 1e4)
        reg_rows.append(
            {
                "regime": name,
                "n_days": len(sub),
                "mean_daily_bps": mean_bps,
                "cum_factor_ret": cum_ret,
                "t_stat": st["t_stat"],
                "p_value": st["p_value"],
                "pct_positive_days": st["hit_rate"],
                "spy_cum_return": float(np.prod(1.0 + sub["fwd_spy_ret"].to_numpy()) - 1.0),
            }
        )
    regime_df = pd.DataFrame(reg_rows)
    return annual_df, regime_df


def run_pct_ownership_study(
    *,
    sample: SampleSplit = "in_sample",
    output_dir: Path | None = DEFAULT_FACTORS_OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Run pure `pct_shares_owned` daily cross-sectional regression study and save outputs."""
    panel = load_factor_panel(sample=sample)
    panel_z = build_pct_ownership_exposures(panel)

    signal_cols = [
        "pct_own_winsor_z",
        "pct_own_rank_z",
        "pct_own_size_neutral_z",
        "pct_own_rank_size_neutral_z",
    ]
    daily_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for scol in signal_cols:
        ddf = compute_daily_factor_returns(panel_z, signal_col=scol)
        daily_frames.append(ddf)

        st_raw = compute_return_stats(ddf["f_raw"], ddf["fwd_spy_ret"])
        st_badj = compute_return_stats(ddf["f_beta_adj"], ddf["fwd_spy_ret"])
        ic_vals = ddf["rank_ic"].dropna().to_numpy(dtype=np.float64)
        ic_n = len(ic_vals)
        ic_mean = float(np.mean(ic_vals)) if ic_n >= 2 else np.nan
        ic_std = float(np.std(ic_vals, ddof=1)) if ic_n >= 2 else np.nan
        ic_t = ic_mean / (ic_std / np.sqrt(ic_n)) if ic_std > 1e-12 else np.nan
        ic_p = _two_sided_t_pvalue(ic_t, df=ic_n - 1) if np.isfinite(ic_t) else np.nan

        summary_rows.append(
            {
                "signal": scol,
                "sample": sample,
                "n_days": int(st_badj["n_days"]),
                "avg_n_stocks": float(ddf["n_stocks"].mean()),
                "mean_size_corr": float(ddf["size_corr"].mean()),
                "mean_beta_corr": float(ddf["beta_corr"].mean()),
                "ex_ante_beta": float(ddf["ex_ante_beta"].mean()),
                "raw_mean_bps": st_raw["mean_daily_bps"],
                "raw_std_bps": st_raw["std_daily_bps"],
                "raw_t_stat": st_raw["t_stat"],
                "raw_p_value": st_raw["p_value"],
                "raw_spy_beta": st_raw["realized_spy_beta"],
                "raw_spy_r2": st_raw["spy_r2"],
                "badj_mean_bps": st_badj["mean_daily_bps"],
                "badj_std_bps": st_badj["std_daily_bps"],
                "badj_t_stat": st_badj["t_stat"],
                "badj_p_value": st_badj["p_value"],
                "badj_spy_beta": st_badj["realized_spy_beta"],
                "badj_spy_r2": st_badj["spy_r2"],
                "mean_rank_ic": ic_mean,
                "ic_t_stat": ic_t,
                "ic_p_value": ic_p,
                "mean_cs_r2": float(ddf["cs_r2"].mean()),
            }
        )

    daily_all = pd.concat(daily_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Cross-sectional quantile diagnostics on raw pct_shares_owned (linearity / tail check)
    q5_daily = compute_quantile_portfolios(panel_z, sort_col="pct_shares_owned", n_buckets=5)
    q10_daily = compute_quantile_portfolios(panel_z, sort_col="pct_shares_owned", n_buckets=10)
    q5_summary = summarize_quantile_table(q5_daily, label_prefix="Q")
    q10_summary = summarize_quantile_table(q10_daily, label_prefix="D")

    # Year-by-year & regime tables on pure OLS factor return `f_beta_adj`
    primary_daily = daily_all[daily_all["signal"] == "pct_own_winsor_z"].copy()
    annual_df, regime_df = summarize_annual_and_regimes(primary_daily, ret_col="f_beta_adj")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        daily_all.to_parquet(
            output_dir / f"pct_ownership_daily_returns_{tag}.parquet", index=False
        )
        summary_df.to_csv(output_dir / f"pct_ownership_summary_{tag}.csv", index=False)
        annual_df.to_csv(output_dir / f"pct_ownership_annual_{tag}.csv", index=False)
        regime_df.to_csv(output_dir / f"pct_ownership_regimes_{tag}.csv", index=False)
        q5_summary.to_csv(output_dir / f"pct_ownership_quintiles_{tag}.csv", index=False)
        q10_summary.to_csv(output_dir / f"pct_ownership_deciles_{tag}.csv", index=False)

    return {
        "daily": daily_all,
        "summary": summary_df,
        "annual": annual_df,
        "regimes": regime_df,
        "quintiles": q5_summary,
        "deciles": q10_summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pure pct_shares_owned daily cross-sectional factor returns."
    )
    parser.add_argument(
        "--sample",
        choices=("in_sample", "oos", "full"),
        default="in_sample",
        help="Sample split (default: in_sample; strictly excludes OOS).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    res = run_pct_ownership_study(sample=args.sample)

    d0 = res["daily"][res["daily"]["signal"] == "pct_own_winsor_z"]
    print(
        f"\n=== PURE pct_shares_owned REGRESSION VALIDATION ({args.sample.upper()}) ===\n"
        f"  Signal dates (t):    {d0['query_date'].min().date()} -> "
        f"{d0['query_date'].max().date()} ({len(d0):,} trading days)\n"
        f"  Return dates (t+1):  {d0['fwd_ret_date'].min().date()} -> "
        f"{d0['fwd_ret_date'].max().date()}\n"
        f"  Avg daily breadth:   {d0['n_stocks'].mean():.1f} stocks/day\n"
    )

    print("--- 1. GLOBAL DAILY CROSS-SECTIONAL OLS SUMMARY (bps/day per +1 sigma) ---")
    print(res["summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
