"""Unit tests for pure `pct_shares_owned` factor engine (`src.factors.ownership`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.ownership import (
    attach_fwd_ret_date,
    build_pct_ownership_exposures,
    compute_daily_factor_returns,
    compute_quantile_portfolios,
    compute_return_stats,
    summarize_quantile_table,
)


def _make_synthetic_panel(n_days: int = 40, n_stocks: int = 60) -> pd.DataFrame:
    """Synthetic multi-day eligible panel with known true factor alpha and market beta."""
    rng = np.random.default_rng(2026)
    cal = pd.bdate_range("2022-01-03", periods=n_days + 1)
    spy_rets = rng.normal(0.0005, 0.012, size=n_days)

    rows: list[dict[str, object]] = []
    # Fixed per-stock attributes
    base_mcap = np.exp(rng.normal(23.0, 1.4, size=n_stocks))
    base_beta = rng.uniform(0.5, 1.8, size=n_stocks)

    for t in range(n_days):
        # Construct pct_shares_owned with mild negative size correlation
        raw_pct = np.clip(
            0.05 - 0.008 * (np.log(base_mcap) - 23.0) + rng.lognormal(-3.5, 0.7, size=n_stocks),
            1e-4,
            0.50,
        )
        # True beta-adjusted alpha: +0.001 * standardized pct_shares_owned + noise
        z_true = (raw_pct - raw_pct.mean()) / raw_pct.std()
        eps = rng.normal(0.0, 0.01, size=n_stocks)
        r_badj = 0.0010 * z_true + eps
        r_raw = base_beta * spy_rets[t] + r_badj

        for i in range(n_stocks):
            rows.append(
                {
                    "query_date": cal[t],
                    "fwd_ret_date": cal[t + 1],
                    "security_id": f"SEC_{i:03d}",
                    "yahoo_ticker": f"T{i:03d}",
                    "n_filers": int(rng.integers(1, 15)),
                    "pct_shares_owned": float(raw_pct[i]),
                    "days_to_liquidate": float(raw_pct[i] * 20.0),
                    "pit_mcap_usd": float(base_mcap[i]),
                    "adv_20d_usd": float(base_mcap[i] * 0.01),
                    "beta_252d_lag1": float(base_beta[i]),
                    "fwd_day_ret": float(r_raw[i]),
                    "fwd_spy_ret": float(spy_rets[t]),
                    "fwd_beta_adj_ret": float(r_badj[i]),
                    "is_panel_eligible": True,
                }
            )
    return pd.DataFrame(rows)


def test_exposures_zero_mean_unit_std_and_exact_size_orthogonality() -> None:
    """All exposure variants have mean=0, std=1 daily; size-neutral has corr(z, log_mcap)=0."""
    panel = _make_synthetic_panel(n_days=10, n_stocks=50)
    out = build_pct_ownership_exposures(panel)

    for _dt, grp in out.groupby("query_date"):
        for col in (
            "pct_own_winsor_z",
            "pct_own_rank_z",
            "pct_own_size_neutral_z",
            "pct_own_rank_size_neutral_z",
        ):
            v = grp[col].to_numpy(dtype=np.float64)
            assert np.isclose(np.mean(v), 0.0, atol=1e-12)
            assert np.isclose(np.std(v, ddof=0), 1.0, atol=1e-12)

        # Size-neutral exposures MUST have exact 0 correlation with log_mcap every single day
        lm = grp["log_mcap"].to_numpy(dtype=np.float64)
        for sn_col in ("pct_own_size_neutral_z", "pct_own_rank_size_neutral_z"):
            corr = np.corrcoef(grp[sn_col].to_numpy(), lm)[0, 1]
            assert np.isclose(corr, 0.0, atol=1e-11)


def test_daily_ols_factor_return_matches_exact_slope_and_beta_hedge_identity() -> None:
    """OLS factor return equals Cov(z, r)/Var(z) and f_beta_adj == f_raw - ex_ante_beta*r_spy."""
    panel = _make_synthetic_panel(n_days=25, n_stocks=60)
    out = build_pct_ownership_exposures(panel)
    daily = compute_daily_factor_returns(out, signal_col="pct_own_winsor_z")

    assert len(daily) == 25
    # Verify exact match to numpy polyfit OLS slope on every day
    for _, row in daily.iterrows():
        grp = out[out["query_date"] == row["query_date"]]
        z = grp["pct_own_winsor_z"].to_numpy()
        r_badj = grp["fwd_beta_adj_ret"].to_numpy()
        r_raw = grp["fwd_day_ret"].to_numpy()
        slope_badj = float(np.polyfit(z, r_badj, 1)[0])
        slope_raw = float(np.polyfit(z, r_raw, 1)[0])

        assert np.isclose(row["f_beta_adj"], slope_badj, atol=1e-12)
        assert np.isclose(row["f_raw"], slope_raw, atol=1e-12)
        assert np.isclose(
            row["f_beta_adj"],
            row["f_raw"] - row["ex_ante_beta"] * row["fwd_spy_ret"],
            atol=1e-12,
        )


def test_beta_adjustment_strips_out_realized_spy_beta() -> None:
    """When high-ownership stocks systematically have higher beta, beta-adj removes SPY beta."""
    rng = np.random.default_rng(99)
    n_days, n_stocks = 120, 60
    cal = pd.bdate_range("2022-01-03", periods=n_days + 1)
    spy_rets = rng.normal(0.0004, 0.015, size=n_days)

    # Deliberately make high pct_shares_owned stocks have beta = 1.8 and low have beta = 0.6
    pct_fixed = np.linspace(0.005, 0.25, n_stocks)
    beta_fixed = np.linspace(0.6, 1.8, n_stocks)

    rows = []
    for t in range(n_days):
        eps = rng.normal(0.0, 0.005, size=n_stocks)
        r_raw = beta_fixed * spy_rets[t] + eps
        r_badj = r_raw - beta_fixed * spy_rets[t]
        for i in range(n_stocks):
            rows.append(
                {
                    "query_date": cal[t],
                    "fwd_ret_date": cal[t + 1],
                    "security_id": f"S{i}",
                    "yahoo_ticker": f"T{i}",
                    "n_filers": 5,
                    "pct_shares_owned": float(pct_fixed[i]),
                    "pit_mcap_usd": 1e10,
                    "beta_252d_lag1": float(beta_fixed[i]),
                    "fwd_day_ret": float(r_raw[i]),
                    "fwd_spy_ret": float(spy_rets[t]),
                    "fwd_beta_adj_ret": float(r_badj[i]),
                    "is_panel_eligible": True,
                }
            )
    panel = build_pct_ownership_exposures(pd.DataFrame(rows))
    daily = compute_daily_factor_returns(panel, signal_col="pct_own_winsor_z")

    spy_var = np.var(daily["fwd_spy_ret"], ddof=1)
    b_raw = float(np.cov(daily["f_raw"], daily["fwd_spy_ret"])[0, 1] / spy_var)
    b_adj = float(np.cov(daily["f_beta_adj"], daily["fwd_spy_ret"])[0, 1] / spy_var)
    # Raw OLS slope has large positive SPY beta (~+0.35); beta-adjusted has ~0.0!
    assert b_raw > 0.25
    assert abs(b_adj) < 0.02


def test_attach_fwd_ret_date_strictly_excludes_2023_12_29_from_is() -> None:
    """`2023-12-29` forward return lands on `2024-01-02` (OOS) and MUST be mapped accurately."""
    spy = pd.DataFrame(
        {"date": pd.to_datetime(["2023-12-28", "2023-12-29", "2024-01-02", "2024-01-03"])}
    )
    panel = pd.DataFrame({"query_date": pd.to_datetime(["2023-12-28", "2023-12-29"])})
    out = attach_fwd_ret_date(panel, spy)
    assert out.loc[0, "fwd_ret_date"] == pd.Timestamp("2023-12-29")
    assert out.loc[1, "fwd_ret_date"] == pd.Timestamp("2024-01-02")


def test_quantile_portfolios_and_summary() -> None:
    """Equal-weighted quintile sort produces Q1..Q5 + Q5-Q1 spread cleanly."""
    panel = build_pct_ownership_exposures(_make_synthetic_panel(n_days=20, n_stocks=50))
    q5 = compute_quantile_portfolios(panel, sort_col="pct_shares_owned", n_buckets=5)
    summ = summarize_quantile_table(q5, label_prefix="Q")
    assert list(summ["bucket"]) == ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5-Q1"]
    assert "badj_p_value" in summ.columns
    assert np.all(np.isclose(summ.iloc[:5]["avg_n"], 10.0))
    # Monotonic average pct_shares_owned across Q1 -> Q5
    assert list(summ.iloc[:5]["avg_pct_owned"]) == sorted(summ.iloc[:5]["avg_pct_owned"])


def test_return_stats_exact_sharpe_tstat_and_pvalue() -> None:
    """Verify exact arithmetic ann_return, ddof=1 ann_vol, Sharpe, t-stat, and Student-t p-value."""
    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(7)
    r = rng.normal(0.0005, 0.008, size=252)
    st = compute_return_stats(r)
    ref = ttest_1samp(r, 0.0)

    assert np.isclose(st["mean_daily_bps"], np.mean(r) * 1e4, atol=1e-12)
    assert np.isclose(st["std_daily_bps"], np.std(r, ddof=1) * 1e4, atol=1e-12)
    assert np.isclose(st["ann_return"], np.mean(r) * 252.0, atol=1e-12)
    assert np.isclose(st["ann_vol"], np.std(r, ddof=1) * np.sqrt(252.0), atol=1e-12)
    assert np.isclose(st["sharpe"], st["ann_return"] / st["ann_vol"], atol=1e-12)
    assert np.isclose(st["t_stat"], float(ref.statistic), atol=1e-12)
    assert np.isclose(st["p_value"], float(ref.pvalue), atol=1e-12)
