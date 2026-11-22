"""Unit tests for `pct_own` & `days_to_exit` Crowding validation (`src.factors.crowding`)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.crowding import (
    build_crowding_exposures,
    compute_crowding_collinearity,
    compute_crowding_decile_profile,
    fit_clustered_crowding_ols,
)


def _make_synthetic_crowding_panel(n_days: int = 30, n_stocks: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cal = pd.bdate_range("2022-01-03", periods=n_days + 1)
    spy_rets = rng.normal(0.0004, 0.01, size=n_days)

    base_mcap = np.exp(rng.normal(23.0, 1.3, size=n_stocks))
    base_beta = rng.uniform(0.6, 1.6, size=n_stocks)

    rows: list[dict[str, object]] = []
    for t in range(n_days):
        pct = np.clip(rng.lognormal(-3.2, 0.7, size=n_stocks), 1e-4, 0.40)
        dte = pct * rng.lognormal(4.5, 0.4, size=n_stocks)
        eps = rng.normal(0.0, 0.01, size=n_stocks)
        r_badj = -0.0008 * ((pct - pct.mean()) / pct.std()) + eps
        r_raw = base_beta * spy_rets[t] + r_badj

        for i in range(n_stocks):
            rows.append(
                {
                    "query_date": cal[t],
                    "fwd_ret_date": cal[t + 1],
                    "security_id": f"SEC_{i:03d}",
                    "yahoo_ticker": f"T{i:03d}",
                    "n_filers": 5,
                    "pct_shares_owned": float(pct[i]),
                    "days_to_liquidate": float(dte[i]),
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


def test_crowding_exposures_standardized_and_exact_size_neutral() -> None:
    """`pct_own_sn`, `days_to_exit_sn`, and `crowding_z` have mean=0, std=1, size corr=0 daily."""
    panel = _make_synthetic_crowding_panel(n_days=10, n_stocks=50)
    out = build_crowding_exposures(panel)

    assert "pct_own" in out.columns and "days_to_exit" in out.columns
    for _dt, grp in out.groupby("query_date"):
        lm = grp["log_mcap"].to_numpy(dtype=np.float64)
        for col in ("pct_own_sn", "days_to_exit_sn", "crowding_z"):
            v = grp[col].to_numpy(dtype=np.float64)
            assert np.isclose(np.mean(v), 0.0, atol=1e-12)
            assert np.isclose(np.std(v, ddof=0), 1.0, atol=1e-12)
            assert np.isclose(np.corrcoef(v, lm)[0, 1], 0.0, atol=1e-11)


def test_collinearity_and_multilinear_clustered_ols_run_cleanly() -> None:
    """Correlation, VIF, and date-clustered multilinear OLS produce valid finite statistics."""
    panel = build_crowding_exposures(_make_synthetic_crowding_panel(n_days=25, n_stocks=60))
    corr_df, vif_df = compute_crowding_collinearity(panel)
    fit_m = fit_clustered_crowding_ols(panel, ["pct_own_sn", "days_to_exit_sn"])

    assert corr_df.shape == (2, 2)
    assert set(vif_df["feature"]) == {"pct_own_sn", "days_to_exit_sn"}
    assert (vif_df["vif"] >= 1.0).all()
    assert set(fit_m.params.index) == {"const", "pct_own_sn", "days_to_exit_sn"}
    assert np.isfinite(fit_m.pvalues).all()
    assert np.isfinite(fit_m.f_pvalue)


def test_crowding_decile_profile_produces_d1_to_d10_and_tail_spreads() -> None:
    """Decile profile yields D1..D10 plus D10-D1, D9-10 - D1-2, and D10 - Rest spreads."""
    panel = build_crowding_exposures(_make_synthetic_crowding_panel(n_days=20, n_stocks=50))
    daily_dec, summ = compute_crowding_decile_profile(panel)

    assert len(daily_dec) == 20 * 10
    expected_buckets = [f"D{i}" for i in range(1, 11)] + [
        "D10 - D1",
        "D9-10 - D1-2",
        "D10 - Rest (D1-D9)",
    ]
    assert list(summ["bucket"]) == expected_buckets
    # Mean crowding_z must increase monotonically from D1 -> D10
    z_means = summ.iloc[:10]["mean_z"].to_list()
    assert z_means == sorted(z_means)
    assert np.isfinite(summ["badj_p_value"]).all()


def test_crowding_spy_regime_analysis_has_zero_lookahead() -> None:
    """Trailing 1d/5d/20d SPY regime buckets depend strictly on SPY <= query_date t."""
    from src.factors.crowding import compute_crowding_spy_regime_analysis

    panel = build_crowding_exposures(_make_synthetic_crowding_panel(n_days=30, n_stocks=60))
    dates = pd.bdate_range("2021-12-01", periods=65)
    rng = np.random.default_rng(123)
    spy_base = pd.DataFrame({"date": dates, "ret_total": rng.normal(0.0, 0.012, size=len(dates))})

    # Inject a huge +50% SPY shock strictly AFTER the last panel query_date!
    max_qdt = panel["query_date"].max()
    spy_fut = spy_base.copy()
    spy_fut.loc[spy_fut["date"] > max_qdt, "ret_total"] = 0.50

    b1, i1, d1 = compute_crowding_spy_regime_analysis(panel, spy_prices=spy_base)
    b2, i2, d2 = compute_crowding_spy_regime_analysis(panel, spy_prices=spy_fut)

    assert not b1.empty
    assert set(b1["horizon"].unique()) == {"spy_ret_1d", "spy_ret_5d", "spy_ret_10d"}
    assert "gt_d7_rest_bps" in b1.columns
    assert len(d1) == 13  # D1..D10 + >D7 + >D7-Rest + D10-Rest
    # Future SPY return strictly after max_qdt MUST NOT change a single bucket or decile!
    pd.testing.assert_frame_equal(b1, b2)
    pd.testing.assert_frame_equal(i1, i2)
    pd.testing.assert_frame_equal(d1, d2)
