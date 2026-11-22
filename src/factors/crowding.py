"""Two-Variable Crowding Factor Validation (`pct_own` and `days_to_exit`).

Focuses cleanly on the two structural proxies of institutional crowding strictly inside the
In-Sample window by default (`fwd_ret_date <= 2023-12-31`, zero OOS touch):
  1. `pct_own` (`pct_shares_owned`): Aggregate share of issuer equity held by our 100 audited
     fundamental managers (Ownership Concentration / Float Absorption).
  2. `days_to_exit` (`days_to_liquidate`): Total 100-fund position USD divided by 20-day
     issuer ADV (Liquidity / Exit Bottleneck — days of normal market volume required to unwind).

Both variables are 1%/99% cross-sectionally winsorized and size-neutralized against
`log(pit_mcap_usd)` strictly on day `t` (`pct_own_sn`, `days_to_exit_sn`). We evaluate:
  - Univariate date-clustered OLS for `pct_own_sn` and `days_to_exit_sn` individually
  - Multilinear date-clustered OLS (`fwd_beta_adj_ret_bps ~ const + pct_own_sn + days_to_exit_sn`)
    plus exact cross-variable correlation and VIF collinearity diagnostics
  - Equal-weight standardized crowding score (`crowding_z = 0.5*pct_own_sn + 0.5*days_to_exit_sn`)

Usage:
    python -m src.factors.crowding
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.factors.ownership import (
    DEFAULT_FACTORS_OUTPUT_DIR,
    DEFAULT_IS_SIGNAL_START,
    DEFAULT_MIN_CROSS_SECTION,
    DEFAULT_SPY_PARQUET,
    DEFAULT_WINSOR_LIMITS,
    SampleSplit,
    _orthogonalize_against,
    _standardize_vec,
    compute_daily_factor_returns,
    compute_return_stats,
    load_factor_panel,
)

logger = logging.getLogger(__name__)


def build_crowding_exposures(
    panel: pd.DataFrame,
    *,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.DataFrame:
    """Attach strictly day-`t` cross-sectional `pct_own` and `days_to_exit` crowding exposures.

    Requires valid `pct_shares_owned > 0`, `days_to_liquidate > 0`, and `pit_mcap_usd > 0`.
    Exposes intuitive aliases `pct_own` and `days_to_exit`, their 1%/99% winsorized z-scores
    (`pct_own_z`, `days_to_exit_z`), exact size-neutralized z-scores (`pct_own_sn`,
    `days_to_exit_sn`), and equal-weight combined crowding score (`crowding_z`).
    """
    if panel.empty:
        return panel.copy()

    valid = (
        panel["pct_shares_owned"].notna()
        & (panel["pct_shares_owned"] > 0.0)
        & panel["days_to_liquidate"].notna()
        & (panel["days_to_liquidate"] > 0.0)
        & panel["pit_mcap_usd"].notna()
        & (panel["pit_mcap_usd"] > 0.0)
    )
    df = panel.loc[valid].sort_values(["query_date", "security_id"]).reset_index(drop=True).copy()
    df["pct_own"] = df["pct_shares_owned"].to_numpy(dtype=np.float64)
    df["days_to_exit"] = df["days_to_liquidate"].to_numpy(dtype=np.float64)
    df["log_mcap"] = np.log(df["pit_mcap_usd"].to_numpy(dtype=np.float64))

    n_rows = len(df)
    cols = (
        "pct_own_z",
        "pct_own_sn",
        "days_to_exit_z",
        "days_to_exit_sn",
        "log_mcap_z",
        "crowding_z",
    )
    out = {c: np.full(n_rows, np.nan, dtype=np.float64) for c in cols}

    lo_q, hi_q = winsor_limits
    pct_all = df["pct_own"].to_numpy(dtype=np.float64)
    dte_all = df["days_to_exit"].to_numpy(dtype=np.float64)
    lm_all = df["log_mcap"].to_numpy(dtype=np.float64)

    dates_ns = df["query_date"].to_numpy(dtype="datetime64[ns]")
    change_idx = np.flatnonzero(dates_ns[1:] != dates_ns[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n_rows]))

    def _winsor_z(arr: np.ndarray) -> np.ndarray:
        lo, hi = np.quantile(arr, [lo_q, hi_q])
        return _standardize_vec(np.clip(arr, lo, hi))

    for s, e in zip(starts, ends, strict=False):
        if e - s < 5:
            continue
        lm = lm_all[s:e]
        pz = _winsor_z(pct_all[s:e])
        dz = _winsor_z(dte_all[s:e])
        lmz = _standardize_vec(lm)

        psn = _orthogonalize_against(pz, lm)
        dsn = _orthogonalize_against(dz, lm)

        out["pct_own_z"][s:e] = pz
        out["pct_own_sn"][s:e] = psn
        out["days_to_exit_z"][s:e] = dz
        out["days_to_exit_sn"][s:e] = dsn
        out["log_mcap_z"][s:e] = lmz
        out["crowding_z"][s:e] = _standardize_vec((psn + dsn) / 2.0)

    for c, vals in out.items():
        df[c] = vals
    return df


def compute_crowding_collinearity(panel_c: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return correlation matrix and VIF table for size-neutral `pct_own_sn` & `days_to_exit_sn`."""
    cols = ["pct_own_sn", "days_to_exit_sn"]
    sub = panel_c[cols].dropna()
    corr_df = sub.corr()

    X = sm.add_constant(sub)
    x_arr = X.to_numpy(dtype=np.float64)
    vif_rows = [
        {"feature": col, "vif": float(variance_inflation_factor(x_arr, idx))}
        for idx, col in enumerate(X.columns)
        if col != "const"
    ]
    return corr_df, pd.DataFrame(vif_rows)


def fit_clustered_crowding_ols(
    panel_c: pd.DataFrame,
    x_cols: list[str],
    *,
    y_col: str = "fwd_beta_adj_ret_bps",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit date-clustered `statsmodels.OLS` (`cov_type='cluster'` grouped by `query_date`)."""
    df = panel_c.copy()
    if y_col not in df.columns:
        if y_col == "fwd_beta_adj_ret_bps":
            df[y_col] = df["fwd_beta_adj_ret"] * 1e4
        elif y_col == "fwd_day_ret_bps":
            df[y_col] = df["fwd_day_ret"] * 1e4
    sub = df.dropna(subset=[y_col, "query_date", *x_cols])
    X = sm.add_constant(sub[x_cols])
    return sm.OLS(sub[y_col], X).fit(
        cov_type="cluster", cov_kwds={"groups": sub["query_date"]}
    )


def compute_crowding_decile_profile(
    panel_c: pd.DataFrame,
    *,
    sort_col: str = "crowding_z",
    n_buckets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute daily equal-weighted decile (`D1..D10`) returns and tail spread summary.

    Sorts eligible stocks strictly within each `query_date` into `n_buckets` equal-count
    deciles (`D1` least crowded -> `D10` most crowded). Returns:
      - `daily_deciles`: per-day, per-decile equal-weighted raw & beta-adjusted returns
      - `decile_summary`: global IS daily mean (`bps/day`), std, t-stat, p-value, SPY beta,
        and physical crowding anatomy (`pct_own`, `days_to_exit`, `pit_mcap_usd`) for
        `D1..D10`, `D10 - D1`, `Top2 - Bot2 (D9-10 vs D1-2)`, and `D10 - Rest (D1-D9)`.
    """
    req = [
        "query_date",
        "fwd_ret_date",
        sort_col,
        "pct_own",
        "days_to_exit",
        "pit_mcap_usd",
        "beta_252d_lag1",
        "fwd_day_ret",
        "fwd_spy_ret",
        "fwd_beta_adj_ret",
    ]
    df = panel_c.dropna(subset=req).copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    ranks = df.groupby("query_date")[sort_col].rank(method="first")
    counts = df.groupby("query_date")[sort_col].transform("count")
    df["decile"] = (
        np.floor((ranks - 1.0) * n_buckets / counts).astype(int).clip(0, n_buckets - 1) + 1
    )

    daily_dec = (
        df.groupby(["query_date", "fwd_ret_date", "decile"], as_index=False)
        .agg(
            n_stocks=("security_id", "count"),
            mean_z=(sort_col, "mean"),
            mean_pct_own=("pct_own", "mean"),
            mean_days_to_exit=("days_to_exit", "mean"),
            median_days_to_exit=("days_to_exit", "median"),
            median_mcap_b=("pit_mcap_usd", lambda s: float(s.median()) / 1e9),
            mean_beta=("beta_252d_lag1", "mean"),
            ew_raw_ret=("fwd_day_ret", "mean"),
            ew_beta_adj_ret=("fwd_beta_adj_ret", "mean"),
            fwd_spy_ret=("fwd_spy_ret", "first"),
        )
        .sort_values(["query_date", "decile"])
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for d in range(1, n_buckets + 1):
        sub = daily_dec[daily_dec["decile"] == d].sort_values("query_date")
        st_raw = compute_return_stats(sub["ew_raw_ret"], sub["fwd_spy_ret"])
        st_adj = compute_return_stats(sub["ew_beta_adj_ret"], sub["fwd_spy_ret"])
        rows.append(
            {
                "bucket": f"D{d}",
                "avg_n": float(sub["n_stocks"].mean()),
                "mean_z": float(sub["mean_z"].mean()),
                "avg_pct_own": float(sub["mean_pct_own"].mean()),
                "avg_days_to_exit": float(sub["mean_days_to_exit"].mean()),
                "med_days_to_exit": float(sub["median_days_to_exit"].mean()),
                "med_mcap_b": float(sub["median_mcap_b"].mean()),
                "raw_mean_bps": st_raw["mean_daily_bps"],
                "badj_mean_bps": st_adj["mean_daily_bps"],
                "badj_std_bps": st_adj["std_daily_bps"],
                "badj_t_stat": st_adj["t_stat"],
                "badj_p_value": st_adj["p_value"],
                "badj_spy_beta": st_adj["realized_spy_beta"],
            }
        )

    # Pivot daily beta-adjusted & raw returns by decile for clean tail-spread stats
    piv_adj = daily_dec.pivot(index="query_date", columns="decile", values="ew_beta_adj_ret")
    piv_raw = daily_dec.pivot(index="query_date", columns="decile", values="ew_raw_ret")
    spy_by_day = daily_dec.groupby("query_date")["fwd_spy_ret"].first().reindex(piv_adj.index)

    spread_defs = [
        ("D10 - D1", piv_adj[n_buckets] - piv_adj[1], piv_raw[n_buckets] - piv_raw[1]),
        (
            "D9-10 - D1-2",
            piv_adj[[n_buckets - 1, n_buckets]].mean(axis=1) - piv_adj[[1, 2]].mean(axis=1),
            piv_raw[[n_buckets - 1, n_buckets]].mean(axis=1) - piv_raw[[1, 2]].mean(axis=1),
        ),
        (
            "D10 - Rest (D1-D9)",
            piv_adj[n_buckets] - piv_adj[list(range(1, n_buckets))].mean(axis=1),
            piv_raw[n_buckets] - piv_raw[list(range(1, n_buckets))].mean(axis=1),
        ),
    ]
    for label, s_adj, s_raw in spread_defs:
        st_raw = compute_return_stats(s_raw, spy_by_day)
        st_adj = compute_return_stats(s_adj, spy_by_day)
        rows.append(
            {
                "bucket": label,
                "avg_n": np.nan,
                "mean_z": np.nan,
                "avg_pct_own": np.nan,
                "avg_days_to_exit": np.nan,
                "med_days_to_exit": np.nan,
                "med_mcap_b": np.nan,
                "raw_mean_bps": st_raw["mean_daily_bps"],
                "badj_mean_bps": st_adj["mean_daily_bps"],
                "badj_std_bps": st_adj["std_daily_bps"],
                "badj_t_stat": st_adj["t_stat"],
                "badj_p_value": st_adj["p_value"],
                "badj_spy_beta": st_adj["realized_spy_beta"],
            }
        )

    return daily_dec, pd.DataFrame(rows)


def run_crowding_study(
    *,
    sample: SampleSplit = "in_sample",
    min_signal_date: str | None = DEFAULT_IS_SIGNAL_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    output_dir: Path | None = DEFAULT_FACTORS_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run strictly PIT `pct_own` & `days_to_exit` univariate and multilinear OLS study."""
    panel = load_factor_panel(
        sample=sample,
        min_signal_date=min_signal_date,
        min_cross_section=min_cross_section,
    )
    panel_c = build_crowding_exposures(panel)
    panel_c["fwd_day_ret_bps"] = panel_c["fwd_day_ret"] * 1e4
    panel_c["fwd_beta_adj_ret_bps"] = panel_c["fwd_beta_adj_ret"] * 1e4

    corr_df, vif_df = compute_crowding_collinearity(panel_c)

    fit_pct = fit_clustered_crowding_ols(panel_c, ["pct_own_sn"])
    fit_dte = fit_clustered_crowding_ols(panel_c, ["days_to_exit_sn"])
    fit_multi = fit_clustered_crowding_ols(panel_c, ["pct_own_sn", "days_to_exit_sn"])
    fit_comb = fit_clustered_crowding_ols(panel_c, ["crowding_z"])

    # Summary comparison table across univariate, multilinear, and equal-weight combined models
    summary_rows: list[dict[str, Any]] = []
    for name, col, fit in [
        ("Univariate: pct_own (Size-Neutral)", "pct_own_sn", fit_pct),
        ("Univariate: days_to_exit (Size-Neutral)", "days_to_exit_sn", fit_dte),
        ("Equal-Weight Composite: 0.5*pct_own + 0.5*days_to_exit", "crowding_z", fit_comb),
    ]:
        ddf = compute_daily_factor_returns(panel_c, signal_col=col)
        spy_v = float(np.var(ddf["fwd_spy_ret"], ddof=1))
        spy_b = float(np.cov(ddf["f_beta_adj"], ddf["fwd_spy_ret"], ddof=1)[0, 1] / spy_v)
        summary_rows.append(
            {
                "model": name,
                "variable": col,
                "coef_bps": float(fit.params[col]),
                "clustered_se": float(fit.bse[col]),
                "z_stat": float(fit.tvalues[col]),
                "p_value": float(fit.pvalues[col]),
                "joint_f_p_value": float(fit.f_pvalue),
                "spy_beta_check": spy_b,
            }
        )

    for col in ("pct_own_sn", "days_to_exit_sn"):
        summary_rows.append(
            {
                "model": "Multilinear OLS: pct_own + days_to_exit",
                "variable": col,
                "coef_bps": float(fit_multi.params[col]),
                "clustered_se": float(fit_multi.bse[col]),
                "z_stat": float(fit_multi.tvalues[col]),
                "p_value": float(fit_multi.pvalues[col]),
                "joint_f_p_value": float(fit_multi.f_pvalue),
                "spy_beta_check": np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    deciles_daily, deciles_summary = compute_crowding_decile_profile(panel_c)
    regime_buckets_df, regime_interaction_df, decile_regime_df = (
        compute_crowding_spy_regime_analysis(panel_c)
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        summary_df.to_csv(output_dir / f"crowding_ols_summary_{tag}.csv", index=False)
        corr_df.to_csv(output_dir / f"crowding_corr_{tag}.csv")
        vif_df.to_csv(output_dir / f"crowding_vif_{tag}.csv", index=False)
        deciles_summary.to_csv(output_dir / f"crowding_deciles_{tag}.csv", index=False)
        regime_buckets_df.to_csv(output_dir / f"crowding_spy_regime_{tag}.csv", index=False)
        regime_interaction_df.to_csv(
            output_dir / f"crowding_spy_interaction_{tag}.csv", index=False
        )
        decile_regime_df.to_csv(
            output_dir / f"crowding_decile_spy10d_regime_{tag}.csv", index=False
        )

    return {
        "panel": panel_c,
        "summary": summary_df,
        "corr": corr_df,
        "vif": vif_df,
        "deciles_daily": deciles_daily,
        "deciles_summary": deciles_summary,
        "regime_buckets": regime_buckets_df,
        "regime_interaction": regime_interaction_df,
        "decile_regime_10d": decile_regime_df,
        "fits": {
            "pct_own_sn": fit_pct,
            "days_to_exit_sn": fit_dte,
            "multilinear": fit_multi,
            "crowding_z": fit_comb,
        },
    }


def compute_crowding_spy_regime_analysis(
    panel_c: pd.DataFrame,
    *,
    spy_prices: pd.DataFrame | None = None,
    spy_path: Path = DEFAULT_SPY_PARQUET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate next-day (`t -> t+1`) crowding alpha conditioned on strictly PIT trailing `SPY`.

    Strictly Point-in-Time guarantee:
      Uses forward-only compounded `SPY` total returns ending on `query_date` `t`
      (`spy_ret_1d`: `t-1 -> t`, `spy_ret_5d`: `t-5 -> t`, `spy_ret_10d`: `t-10bd -> t`), all
      fully observable at day-`t` close BEFORE executing at day-`t` MOC and earning `t -> t+1`
      (`fwd_beta_adj_ret`). Returns:
        1. `bucket_df`: Binary (`<0%` vs `>=0%`) & magnitude buckets across `1d`, `5d`, `10d`.
        2. `interaction_df`: Continuous HC3 OLS interaction stats.
        3. `decile_regime_df`: Full `D1..D10` crowding decile profile across `10bd` Risk-Off
           (`<0%`), Risk-On (`>=0%`), and Deep Stress (`<= -3%`).
    """
    spy = (
        spy_prices.copy()
        if spy_prices is not None
        else pd.read_parquet(spy_path, columns=["date", "ret_total"])
    )
    spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
    spy = (
        spy[spy["date"].dt.day_of_week < 5]
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    r_arr = spy["ret_total"].fillna(0.0).to_numpy(dtype=np.float64, copy=True)
    if len(r_arr) > 0:
        r_arr[0] = 0.0
    tr = np.cumprod(1.0 + r_arr)
    spy["spy_ret_1d"] = spy["ret_total"].astype(np.float64)
    for w in (5, 10):
        col = f"spy_ret_{w}d"
        if len(tr) > w:
            vals = np.full(len(tr), np.nan, dtype=np.float64)
            vals[w:] = tr[w:] / tr[:-w] - 1.0
            spy[col] = vals
        else:
            spy[col] = np.nan

    spy_feat = spy.rename(columns={"date": "query_date"})[
        ["query_date", "spy_ret_1d", "spy_ret_5d", "spy_ret_10d"]
    ]

    # Daily crowding_z cross-sectional OLS slope & full D1..D10 decile matrix
    fd = compute_daily_factor_returns(panel_c, signal_col="crowding_z")
    df_d = panel_c[["query_date", "crowding_z", "fwd_beta_adj_ret"]].dropna().copy()
    ranks = df_d.groupby("query_date")["crowding_z"].rank(method="first")
    counts = df_d.groupby("query_date")["crowding_z"].transform("count")
    df_d["dec"] = np.floor((ranks - 1.0) * 10.0 / counts).astype(int).clip(0, 9) + 1

    dec_wide = df_d.groupby(["query_date", "dec"])["fwd_beta_adj_ret"].mean().unstack() * 1e4
    for d in range(1, 11):
        if d not in dec_wide.columns:
            dec_wide[d] = np.nan
    dec_wide = dec_wide[list(range(1, 11))].copy()
    dec_wide["gt7_bps"] = dec_wide[[8, 9, 10]].mean(axis=1)
    dec_wide["gt7_rest_bps"] = dec_wide["gt7_bps"] - dec_wide[list(range(1, 8))].mean(axis=1)
    dec_wide["d10_rest_bps"] = dec_wide[10] - dec_wide[list(range(1, 10))].mean(axis=1)

    daily = (
        fd[["query_date", "f_beta_adj"]]
        .merge(spy_feat, on="query_date", how="left")
        .merge(dec_wide.reset_index(), on="query_date", how="left")
    )
    daily["slope_bps"] = daily["f_beta_adj"] * 1e4

    horizon_specs: list[tuple[str, str, list[float], list[str]]] = [
        (
            "spy_ret_1d",
            "1-Day Trailing SPY (t-1 -> t)",
            [-100.0, -1.5, -0.5, 0.5, 1.5, 100.0],
            ["<= -1.5%", "-1.5% to -0.5%", "-0.5% to +0.5%", "+0.5% to +1.5%", "> +1.5%"],
        ),
        (
            "spy_ret_5d",
            "5-BD Trailing SPY (t-5bd -> t)",
            [-100.0, -3.0, -1.0, 1.0, 3.0, 100.0],
            ["<= -3.0%", "-3.0% to -1.0%", "-1.0% to +1.0%", "+1.0% to +3.0%", "> +3.0%"],
        ),
        (
            "spy_ret_10d",
            "10-BD Trailing SPY (t-10bd -> t)",
            [-100.0, -3.0, -1.0, 1.0, 3.0, 100.0],
            ["<= -3.0%", "-3.0% to -1.0%", "-1.0% to +1.0%", "+1.0% to +3.0%", "> +3.0%"],
        ),
    ]

    def _mean_t(s: pd.Series) -> tuple[float, float]:
        sv = s.dropna().to_numpy(dtype=np.float64)
        if len(sv) < 2:
            return (float(sv[0]) if len(sv) == 1 else np.nan), np.nan
        mu = float(sv.mean())
        sd = float(sv.std(ddof=1))
        t = float(mu / (sd / np.sqrt(len(sv)))) if sd > 0.0 else np.nan
        return mu, t

    bucket_rows: list[dict[str, Any]] = []
    inter_rows: list[dict[str, Any]] = []

    for col, h_label, bins, labels in horizon_specs:
        sub_all = daily.dropna(
            subset=[col, "slope_bps", "gt7_bps", "gt7_rest_bps", "d10_rest_bps"]
        ).copy()
        if sub_all.empty:
            continue

        # 1. Binary Risk-Off (< 0%) vs. Risk-On (>= 0%) split
        for b_lbl, mask in [
            ("< 0% (Risk-Off)", sub_all[col] < 0.0),
            (">= 0% (Risk-On)", sub_all[col] >= 0.0),
        ]:
            s = sub_all.loc[mask]
            sm_mu, sm_t = _mean_t(s["slope_bps"])
            g7_mu, g7_t = _mean_t(s["gt7_bps"])
            g7r_mu, g7r_t = _mean_t(s["gt7_rest_bps"])
            dr_mu, dr_t = _mean_t(s["d10_rest_bps"])
            bucket_rows.append(
                {
                    "horizon": col,
                    "horizon_label": h_label,
                    "split_type": "binary",
                    "bucket": b_lbl,
                    "n_days": int(len(s)),
                    "crowd_slope_bps": sm_mu,
                    "crowd_slope_t": sm_t,
                    "gt_d7_bps": g7_mu,
                    "gt_d7_t": g7_t,
                    "gt_d7_rest_bps": g7r_mu,
                    "gt_d7_rest_t": g7r_t,
                    "d10_rest_bps": dr_mu,
                    "d10_rest_t": dr_t,
                }
            )

        # 2. Pre-specified round magnitude buckets
        cut = pd.cut(sub_all[col] * 100.0, bins=bins, labels=labels, right=True)
        for lbl in labels:
            s = sub_all.loc[cut == lbl]
            sm_mu, sm_t = _mean_t(s["slope_bps"])
            g7_mu, g7_t = _mean_t(s["gt7_bps"])
            g7r_mu, g7r_t = _mean_t(s["gt7_rest_bps"])
            dr_mu, dr_t = _mean_t(s["d10_rest_bps"])
            bucket_rows.append(
                {
                    "horizon": col,
                    "horizon_label": h_label,
                    "split_type": "magnitude",
                    "bucket": str(lbl),
                    "n_days": int(len(s)),
                    "crowd_slope_bps": sm_mu,
                    "crowd_slope_t": sm_t,
                    "gt_d7_bps": g7_mu,
                    "gt_d7_t": g7_t,
                    "gt_d7_rest_bps": g7r_mu,
                    "gt_d7_rest_t": g7r_t,
                    "d10_rest_bps": dr_mu,
                    "d10_rest_t": dr_t,
                }
            )

        # 3. Continuous HC3 OLS interaction (bps change in crowding factor per +1% trailing SPY)
        if len(sub_all) >= 20:
            X = sm.add_constant(sub_all[col] * 100.0)
            f1 = sm.OLS(sub_all["slope_bps"], X).fit(cov_type="HC3")
            f2 = sm.OLS(sub_all["d10_rest_bps"], X).fit(cov_type="HC3")
            inter_rows.append(
                {
                    "horizon": col,
                    "horizon_label": h_label,
                    "n_days": int(len(sub_all)),
                    "slope_per_1pct_spy": float(f1.params[col]),
                    "slope_z": float(f1.tvalues[col]),
                    "slope_p": float(f1.pvalues[col]),
                    "d10_rest_per_1pct_spy": float(f2.params[col]),
                    "d10_rest_z": float(f2.tvalues[col]),
                    "d10_rest_p": float(f2.pvalues[col]),
                }
            )

    # Full D1..D10 decile breakdown across 10-BD trailing SPY regimes (Risk-Off / Risk-On / Deep)
    d10_sub = daily.dropna(subset=["spy_ret_10d"]).copy()
    off = d10_sub[d10_sub["spy_ret_10d"] < 0.0]
    on = d10_sub[d10_sub["spy_ret_10d"] >= 0.0]
    deep = d10_sub[d10_sub["spy_ret_10d"] <= -0.03]

    dec_reg_rows: list[dict[str, Any]] = []
    for key, lbl in (
        [(d, f"D{d}") for d in range(1, 11)]
        + [
            ("gt7_bps", "> D7 (D8-D10)"),
            ("gt7_rest_bps", "> D7 - Rest (D1-D7)"),
            ("d10_rest_bps", "D10 - Rest (D1-D9)"),
        ]
    ):
        mu_off, t_off = _mean_t(off[key])
        mu_on, t_on = _mean_t(on[key])
        mu_dp, t_dp = _mean_t(deep[key])
        dec_reg_rows.append(
            {
                "bucket": lbl,
                "n_days_off": int(len(off)),
                "risk_off_bps": mu_off,
                "risk_off_t": t_off,
                "n_days_on": int(len(on)),
                "risk_on_bps": mu_on,
                "risk_on_t": t_on,
                "diff_off_minus_on_bps": mu_off - mu_on,
                "n_days_deep": int(len(deep)),
                "deep_stress_bps": mu_dp,
                "deep_stress_t": t_dp,
            }
        )

    return pd.DataFrame(bucket_rows), pd.DataFrame(inter_rows), pd.DataFrame(dec_reg_rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pct_own & days_to_exit Crowding OLS validation (default: in_sample)."
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
    res = run_crowding_study(sample=args.sample)

    print(f"\n=== CROWDING OLS VALIDATION ({args.sample.upper()}) ===")
    print(res["summary"].to_string(index=False))
    print("\n--- CORRELATION & VIF (pct_own_sn vs days_to_exit_sn) ---")
    print(res["corr"].round(4).to_string())
    print(res["vif"].to_string(index=False))
    print("\n--- MULTILINEAR OLS SUMMARY (fwd_beta_adj_ret_bps ~ pct_own_sn + days_to_exit_sn) ---")
    print(res["fits"]["multilinear"].summary())
    print("\n--- DECILE & TAIL PROFILE (crowding_z: D1 Least Crowded -> D10 Most Crowded) ---")
    print(res["deciles_summary"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
