"""Rolling PIT Manager-Skill-Weighted Smart Money Factor Validation.

Refines Signal 3 ("Smart Money Following") by ranking our 100 audited fundamental managers
strictly Point-in-Time using a **static quarterly sliding window** of trailing completed
quarters (primary: **8 completed quarters / 2 years**, excluding the current live quarter
basket) scored by **Beta-Adjusted Information Ratio (`IR_badj`)**.

Why freeze manager rankings strictly on completed prior quarters (instead of daily rolling)?
  1. **Severs intra-quarter stock-momentum contamination:** Re-ranking managers every day
     using their *current* quarter's basket mechanically boosts every manager holding a
     stock that rallied last month — quietly turning "Manager Skill" into disguised
     short-term stock price momentum (`ret_20d`/`ret_63d`).
  2. **Eliminates asynchronous filing-window distortion & rank churn:** Evaluating all active
     managers synchronously on the exact same trailing 8 completed calendar quarters (once
     each historical basket is PIT-visible AND its evaluation quarter has fully closed) keeps
     manager skill rankings static within each quarter while live holdings still update
     strictly on each fund's `T+1` `effective_date`.

Double PIT Gate (Zero Lookahead):
  A historical quarterly basket `Q_q` (drifted buy-and-hold over `(Q_q, Q_{q+1}]`) enters
  Fund `f`'s trailing `N`-quarter skill score for evaluation quarter `Q_m` (starting on
  trading day `t_start`) IF AND ONLY IF:
    1. `valid_from <= t_start < valid_to` (uses ONLY the latest amended state for `Q_q`
       already visible on EDGAR by `t_start`; future `13F-HR/A`s never rewrite history), AND
    2. `Q_{q+1} < t_start` (the evaluation quarter is 100% completed in the past), AND
    3. `Q_q` is strictly older than the current live quarter basket (excludes current basket).

Usage:
    python -m src.factors.manager_skill
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from src.factors.crowding import build_crowding_exposures, compute_crowding_decile_profile
from src.factors.ownership import (
    DEFAULT_FACTORS_OUTPUT_DIR,
    DEFAULT_MIN_CROSS_SECTION,
    DEFAULT_SPY_PARQUET,
    DEFAULT_WINSOR_LIMITS,
    SampleSplit,
    _orthogonalize_against,
    _standardize_vec,
    compute_daily_factor_returns,
    load_factor_panel,
)
from src.factors.smart_money import compute_state_report_date_dollars
from src.pit.compute import DEFAULT_BETA_MAX_DAILY_RET, Stage3MarketMetrics
from src.pit.daily_panel import DEFAULT_DAILY_PANEL_PARQUET, _extract_state_signatures_by_date
from src.pit.ledger import DEFAULT_MAX_STALENESS_DAYS, PitHoldingsLedger
from src.pit.panel import PitMarketStore

logger = logging.getLogger(__name__)

DEFAULT_SKILL_LOOKBACK_QTRS = 8  # 2-year static quarterly sliding window
DEFAULT_SKILL_MIN_QTRS = 6
DEFAULT_MIN_PRICED_HOLDINGS = 5
DEFAULT_MIN_SKILL_DAYS = 120


def build_clean_stock_return_matrices(
    market_store: PitMarketStore,
    stage3_market: Stage3MarketMetrics,
    *,
    start_date: str = "2015-01-01",
    end_date: str = "2023-12-31",
) -> dict[str, Any]:
    """Build strictly PIT daily raw & beta-adjusted stock return matrices over `[start, end]`.

    Masks Yahoo Finance bankruptcy/reorg single-day price discontinuities (`ret_total > 5.0`
    or `<= -0.99`, identical to `DEFAULT_BETA_MAX_DAILY_RET` in `src.pit.compute`) so a single
    corrupt penny-stock splice cannot distort a fund's 2-year Information Ratio.
    """
    cal = stage3_market.trading_calendar
    cal_sub = cal[(cal >= pd.Timestamp(start_date)) & (cal <= pd.Timestamp(end_date))]
    spy_ret = market_store.spy_prices.set_index("date")["ret_total"].reindex(cal_sub)

    mp = market_store.market_prices[market_store.market_prices["date"].isin(cal_sub)].copy()
    ext_mask = (mp["ret_total"] > DEFAULT_BETA_MAX_DAILY_RET) | (mp["ret_total"] <= -0.99)
    n_masked = int(ext_mask.sum())
    mp.loc[ext_mask, "ret_total"] = np.nan

    ret_wide = mp.pivot(index="date", columns="yahoo_ticker", values="ret_total").reindex(cal_sub)
    beta_wide = stage3_market._beta_lag1_wide.reindex(index=cal_sub, columns=ret_wide.columns)
    badj_wide = ret_wide.sub(beta_wide.mul(spy_ret, axis=0))

    return {
        "cal": cal_sub,
        "spy_ret": spy_ret,
        "ret_wide": ret_wide,
        "badj_wide": badj_wide,
        "n_extreme_stock_days_masked": n_masked,
    }


def compute_drifted_state_quarter_returns(
    ledger: PitHoldingsLedger,
    state_dollars: pd.DataFrame,
    ret_mats: dict[str, Any],
    *,
    min_report_date: str = "2014-12-31",
    max_eval_end_date: str = "2023-12-31",
    min_priced_holdings: int = DEFAULT_MIN_PRICED_HOLDINGS,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Compute drifted buy-and-hold daily raw & beta-adjusted returns per `state_id`.

    Each state_id is immutable per `(filer_id, period_of_report, amendment_version)` and is
    evaluated strictly over `(period_of_report, next_quarter_end]` starting from its report-date
    dollar weights and drifting daily by realized `ret_total` (zero fake daily rebalancing).
    """
    cal_sub: pd.DatetimeIndex = ret_mats["cal"]
    cal_ns = cal_sub.to_numpy(dtype="datetime64[ns]")
    ret_wide: pd.DataFrame = ret_mats["ret_wide"]
    badj_wide: pd.DataFrame = ret_mats["badj_wide"]
    spy_ret: pd.Series = ret_mats["spy_ret"]

    ret_np = ret_wide.to_numpy(dtype=np.float64)
    badj_np = badj_wide.to_numpy(dtype=np.float64)
    tkr_to_col = {str(t): i for i, t in enumerate(ret_wide.columns)}

    st = ledger.states_df[
        [
            "state_id",
            "filer_id",
            "canonical_name",
            "style",
            "period_of_report",
            "effective_date",
            "total_reported_value_usd",
        ]
    ].copy()
    st["period_of_report"] = pd.to_datetime(st["period_of_report"]).dt.normalize()
    st["effective_date"] = pd.to_datetime(st["effective_date"]).dt.normalize()
    st = st[st["period_of_report"] >= pd.Timestamp(min_report_date)].copy()

    sd_by_sid = {int(sid): grp for sid, grp in state_dollars.groupby("state_id", sort=False)}
    max_end_ts = pd.Timestamp(max_eval_end_date).normalize()

    state_q_returns: dict[int, dict[str, Any]] = {}
    n_eligible_states = 0
    all_raw_list: list[np.ndarray] = []
    all_badj_list: list[np.ndarray] = []
    cov_list: list[float] = []

    for _, sr in st.iterrows():
        q_start = pd.Timestamp(sr["period_of_report"]).normalize()
        q_end = (q_start + pd.offsets.QuarterEnd(1)).normalize()
        if q_end > max_end_ts:
            continue
        n_eligible_states += 1

        sid = int(sr["state_id"])
        grp = sd_by_sid.get(sid)
        if grp is None or grp.empty:
            continue

        i0 = int(np.searchsorted(cal_ns, np.datetime64(q_start, "ns"), side="right"))
        i1 = int(np.searchsorted(cal_ns, np.datetime64(q_end, "ns"), side="right"))
        if i1 <= i0:
            continue

        sub = grp.dropna(subset=["dollars_rpt"])
        sub = sub[sub["dollars_rpt"] > 0.0]
        cols = [tkr_to_col.get(str(t)) for t in sub["yahoo_ticker"]]
        keep = [c is not None for c in cols]
        if sum(keep) < min_priced_holdings:
            continue

        c_idx = np.array([c for c in cols if c is not None], dtype=np.int64)
        dol = sub.loc[keep, "dollars_rpt"].to_numpy(dtype=np.float64)
        if len(np.unique(c_idx)) < len(c_idx):
            tmp_s = pd.Series(dol).groupby(c_idx).sum()
            c_idx = tmp_s.index.to_numpy(dtype=np.int64)
            dol = tmp_s.to_numpy(dtype=np.float64)
        if len(c_idx) < min_priced_holdings:
            continue

        r_sub = ret_np[i0:i1, :][:, c_idx]
        b_sub = badj_np[i0:i1, :][:, c_idx]
        w = dol / dol.sum()
        n_days = i1 - i0
        f_raw = np.full(n_days, np.nan, dtype=np.float64)
        f_badj = np.full(n_days, np.nan, dtype=np.float64)

        for d in range(n_days):
            rd = r_sub[d]
            bd = b_sub[d]
            ok_r = np.isfinite(rd)
            ok_b = np.isfinite(bd)
            if ok_r.any() and w[ok_r].sum() > 0.0:
                f_raw[d] = float(np.dot(w[ok_r], rd[ok_r]) / w[ok_r].sum())
            if ok_b.any() and w[ok_b].sum() > 0.0:
                f_badj[d] = float(np.dot(w[ok_b], bd[ok_b]) / w[ok_b].sum())
            mult = np.where(ok_r, 1.0 + rd, 1.0)
            w = w * np.clip(mult, 0.0, None)
            ws = float(w.sum())
            if ws > 0.0:
                w = w / ws

        tot_rep = (
            float(sr["total_reported_value_usd"])
            if pd.notna(sr["total_reported_value_usd"])
            else np.nan
        )
        cov = float(dol.sum() / tot_rep) if (tot_rep and tot_rep > 0.0) else np.nan
        state_q_returns[sid] = {
            "dates_ns": cal_ns[i0:i1],
            "r_raw": f_raw,
            "r_badj": f_badj,
            "basket_q": q_start,
            "eval_end": q_end,
            "effective_date": pd.Timestamp(sr["effective_date"]).normalize(),
            "n_priced": len(c_idx),
            "priced_cov": cov,
        }
        all_raw_list.append(f_raw[np.isfinite(f_raw)])
        all_badj_list.append(f_badj[np.isfinite(f_badj)])
        if np.isfinite(cov):
            cov_list.append(cov)

    # Realized fund SPY beta QA check across all funds (raw vs beta-adjusted)
    spy_map = dict(zip(cal_ns, spy_ret.to_numpy(dtype=np.float64), strict=False))
    raw_betas: list[float] = []
    badj_betas: list[float] = []
    latest_per_q = (
        st.sort_values("effective_date")
        .drop_duplicates(subset=["filer_id", "period_of_report"], keep="last")
        .groupby("filer_id")
    )
    for _, grp in latest_per_q:
        rs, bs, ss = [], [], []
        for sid in grp["state_id"]:
            rec = state_q_returns.get(int(sid))
            if rec is None:
                continue
            sp = np.array([spy_map[d] for d in rec["dates_ns"]], dtype=np.float64)
            rs.append(rec["r_raw"])
            bs.append(rec["r_badj"])
            ss.append(sp)
        if not rs:
            continue
        ra = np.concatenate(rs)
        ba = np.concatenate(bs)
        sa = np.concatenate(ss)
        ok = np.isfinite(ra) & np.isfinite(ba) & np.isfinite(sa)
        if int(ok.sum()) >= 250:
            sv = float(np.var(sa[ok], ddof=1))
            if sv > 0.0:
                raw_betas.append(float(np.cov(ra[ok], sa[ok], ddof=1)[0, 1] / sv))
                badj_betas.append(float(np.cov(ba[ok], sa[ok], ddof=1)[0, 1] / sv))

    qa = {
        "n_extreme_stock_days_masked": int(ret_mats["n_extreme_stock_days_masked"]),
        "states_evaluated": len(state_q_returns),
        "states_total": n_eligible_states,
        "states_coverage_pct": (
            float(len(state_q_returns) / n_eligible_states) if n_eligible_states else 0.0
        ),
        "median_priced_usd_cov": float(np.median(cov_list)) if cov_list else np.nan,
        "n_funds_beta_checked": len(raw_betas),
        "median_raw_fund_spy_beta": float(np.median(raw_betas)) if raw_betas else np.nan,
        "median_badj_fund_spy_beta": float(np.median(badj_betas)) if badj_betas else np.nan,
        "p05_badj_fund_spy_beta": float(np.quantile(badj_betas, 0.05)) if badj_betas else np.nan,
        "p95_badj_fund_spy_beta": float(np.quantile(badj_betas, 0.95)) if badj_betas else np.nan,
    }
    return state_q_returns, qa


def build_static_quarterly_manager_rankings(
    ledger: PitHoldingsLedger,
    state_q_returns: dict[int, dict[str, Any]],
    trading_cal: pd.DatetimeIndex,
    *,
    lookback_qtrs: int = DEFAULT_SKILL_LOOKBACK_QTRS,
    min_qtrs: int = DEFAULT_SKILL_MIN_QTRS,
    start_eval_quarter: str = "2017-01-01",
    end_eval_quarter: str = "2023-10-01",
    max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
    min_days: int = DEFAULT_MIN_SKILL_DAYS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct static quarterly manager skill rankings strictly via the Double PIT Gate.

    For each evaluation quarter `Q_m` starting on first trading day `t_start`:
      - Selects the trailing `lookback_qtrs` completed calendar quarters ending strictly
        before `Q_m` (excluding the current live quarter basket!).
      - Enforces the Double PIT Gate:
          (a) `valid_from <= t_start < valid_to` (latest amended state visible as of `t_start`),
          (b) `eval_end < t_start` and `max_return_date < t_start` (100% completed in the past),
          (c) Fund is actively filing as of `t_start` (`staleness <= max_staleness_days`), keeping
              closed 2015-era peers in the cross-section strictly while alive.
      - Ranks active funds by annualized **Beta-Adjusted Information Ratio (`ir_badj`)**, plus
        diagnostic alternatives (`cum_alpha_badj` and style-relative `rank_ir_style`).
    """
    st = ledger.states_df[
        [
            "state_id",
            "filer_id",
            "canonical_name",
            "style",
            "period_of_report",
            "effective_date",
            "valid_from",
            "valid_to",
        ]
    ].copy()
    for col in ("period_of_report", "effective_date", "valid_from", "valid_to"):
        st[col] = pd.to_datetime(st[col]).dt.normalize()

    cal_ns = trading_cal.to_numpy(dtype="datetime64[ns]")
    eval_q_starts = pd.date_range(start_eval_quarter, end_eval_quarter, freq="QS")

    rows: list[dict[str, Any]] = []
    pit_violations = 0

    for qs in eval_q_starts:
        t_pos = int(np.searchsorted(cal_ns, np.datetime64(qs, "ns"), side="left"))
        if t_pos >= len(cal_ns):
            continue
        t_dt = pd.Timestamp(cal_ns[t_pos]).normalize()

        last_eval_end = (pd.Timestamp(qs) - pd.Timedelta(days=1)).normalize()
        # Trailing lookback_qtrs completed baskets: Q_{m-2}, Q_{m-3}, ..., Q_{m-1-lookback_qtrs}
        # Note: basket Q_{m-1} (period_of_report == last_eval_end) is the CURRENT live basket
        # and is STRICTLY EXCLUDED from the skill window!
        basket_qs = [
            (last_eval_end - pd.offsets.QuarterEnd(k)).normalize()
            for k in range(1, lookback_qtrs + 1)
        ]

        vis_hist = st[
            st["period_of_report"].isin(basket_qs)
            & (st["valid_from"] <= t_dt)
            & (st["valid_to"].isna() | (st["valid_to"] > t_dt))
        ]
        # Active filers on t_dt (anti-survivorship: closed peers included while alive!)
        act_all = st[(st["valid_from"] <= t_dt) & (st["valid_to"].isna() | (st["valid_to"] > t_dt))]
        latest_rep = act_all.groupby("filer_id")["period_of_report"].max()
        active_fids = set(latest_rep[(t_dt - latest_rep).dt.days <= max_staleness_days].index)

        for fid, grp in vis_hist.groupby("filer_id"):
            if fid not in active_fids:
                continue
            fb_list: list[np.ndarray] = []
            q_used = 0
            for _, r in grp.sort_values("period_of_report").iterrows():
                sid = int(r["state_id"])
                rec = state_q_returns.get(sid)
                if rec is None:
                    continue
                # Hard Double PIT Gate assertions
                if (
                    rec["effective_date"] > t_dt
                    or rec["eval_end"] >= t_dt
                    or pd.Timestamp(rec["dates_ns"][-1]) >= t_dt
                    or rec["basket_q"] >= last_eval_end
                ):
                    pit_violations += 1
                    continue
                fb_list.append(rec["r_badj"])
                q_used += 1
            if q_used < min_qtrs:
                continue
            arr = np.concatenate(fb_list)
            arr = arr[np.isfinite(arr)]
            if len(arr) < min_days:
                continue
            sd_v = float(arr.std(ddof=1))
            ir = float((arr.mean() / sd_v) * np.sqrt(252.0)) if sd_v > 0.0 else np.nan
            rows.append(
                {
                    "eval_q_start": pd.Timestamp(qs).normalize(),
                    "asof_t": t_dt,
                    "filer_id": str(fid),
                    "canonical_name": str(grp["canonical_name"].iloc[-1]),
                    "style": str(grp["style"].iloc[-1]),
                    "q_used": q_used,
                    "n_days": int(len(arr)),
                    "ir_badj": ir,
                    "cum_alpha_badj": float(arr.sum()),
                }
            )

    if pit_violations > 0:
        raise AssertionError(f"MANAGER SKILL PIT VIOLATION: {pit_violations} rows failed gate!")

    df_sc = pd.DataFrame(rows)
    if df_sc.empty:
        return df_sc, {"pit_violations": 0, "n_quarters": 0, "avg_funds_per_qtr": 0.0}

    df_sc["rank_ir"] = df_sc.groupby("eval_q_start")["ir_badj"].transform(
        lambda s: 2.0 * s.rank(pct=True) - 1.0
    )
    df_sc["rank_alpha"] = df_sc.groupby("eval_q_start")["cum_alpha_badj"].transform(
        lambda s: 2.0 * s.rank(pct=True) - 1.0
    )
    df_sc["rank_ir_style"] = df_sc.groupby(["eval_q_start", "style"])["ir_badj"].transform(
        lambda s: 2.0 * s.rank(pct=True) - 1.0 if len(s) >= 4 else 0.0
    )
    df_sc["q_bucket"] = df_sc.groupby("eval_q_start")["ir_badj"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    )

    closed_all = set(st.loc[st["style"] == "closed_peer", "filer_id"].unique())
    closed_scored = set(df_sc.loc[df_sc["filer_id"].isin(closed_all), "filer_id"].unique())
    meta_qa = {
        "pit_violations": pit_violations,
        "n_quarters": int(df_sc["eval_q_start"].nunique()),
        "avg_funds_per_qtr": float(df_sc.groupby("eval_q_start")["filer_id"].nunique().mean()),
        "closed_peers_scored": len(closed_scored),
        "closed_peers_total": len(closed_all),
    }
    return df_sc, meta_qa


def audit_manager_skill_persistence(
    manager_scores: pd.DataFrame,
    ledger: PitHoldingsLedger,
    state_q_returns: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Measure cross-sectional fund-level 1-quarter forward IR persistence (Spearman Rank IC)."""
    st = ledger.states_df[["state_id", "filer_id", "period_of_report", "effective_date"]].copy()
    st["period_of_report"] = pd.to_datetime(st["period_of_report"]).dt.normalize()
    st["effective_date"] = pd.to_datetime(st["effective_date"]).dt.normalize()
    latest_st = (
        st.sort_values("effective_date")
        .drop_duplicates(subset=["filer_id", "period_of_report"], keep="last")
        .set_index(["filer_id", "period_of_report"])["state_id"]
        .to_dict()
    )

    fwd_rows: list[dict[str, Any]] = []
    for _, r in manager_scores.iterrows():
        fid = str(r["filer_id"])
        b_rep = (pd.Timestamp(r["eval_q_start"]) - pd.Timedelta(days=1)).normalize()
        sid = latest_st.get((fid, b_rep))
        if sid is None:
            continue
        rec = state_q_returns.get(int(sid))
        if rec is None:
            continue
        fb = rec["r_badj"][np.isfinite(rec["r_badj"])]
        if len(fb) < 40:
            continue
        sd_v = float(fb.std(ddof=1))
        fwd_ir = float((fb.mean() / sd_v) * np.sqrt(252.0)) if sd_v > 0.0 else np.nan
        fwd_rows.append(
            {
                "eval_q_start": r["eval_q_start"],
                "filer_id": fid,
                "ir_badj": float(r["ir_badj"]),
                "rank_ir_style": float(r["rank_ir_style"]),
                "fwd_1q_ir": fwd_ir,
            }
        )

    fdf = pd.DataFrame(fwd_rows)
    ics_all: list[float] = []
    ics_style: list[float] = []
    for _, g in fdf.groupby("eval_q_start"):
        if len(g) < 20:
            continue
        ics_all.append(float(spearmanr(g["ir_badj"], g["fwd_1q_ir"]).statistic))
        ics_style.append(float(spearmanr(g["rank_ir_style"], g["fwd_1q_ir"]).statistic))

    def _ic_stats(vals: list[float]) -> tuple[float, float, float]:
        if len(vals) < 2:
            return np.nan, np.nan, np.nan
        arr = np.asarray(vals, dtype=np.float64)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1))
        t = float(mu / (sd / np.sqrt(len(arr)))) if sd > 0.0 else np.nan
        return mu, sd, t

    mu_a, sd_a, t_a = _ic_stats(ics_all)
    mu_s, sd_s, t_s = _ic_stats(ics_style)
    return {
        "n_eval_quarters": len(ics_all),
        "overall_ic_mean": mu_a,
        "overall_ic_std": sd_a,
        "overall_ic_t": t_a,
        "style_rel_ic_mean": mu_s,
        "style_rel_ic_std": sd_s,
        "style_rel_ic_t": t_s,
    }


def _cs_size_neutral_z(
    df: pd.DataFrame,
    col: str,
    *,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    lo_q, hi_q = winsor_limits
    log_mcap = np.log(df["pit_mcap_usd"].astype(np.float64).clip(lower=1.0))
    for _, idx in df.groupby("query_date", sort=False).groups.items():
        x = df.loc[idx, col].to_numpy(dtype=np.float64)
        s = log_mcap.loc[idx].to_numpy(dtype=np.float64)
        ok = np.isfinite(x) & np.isfinite(s)
        if int(ok.sum()) < 20:
            continue
        xv = np.clip(x[ok], *np.quantile(x[ok], [lo_q, hi_q]))
        out.loc[idx[ok]] = _standardize_vec(_orthogonalize_against(xv, s[ok]))
    return out


def map_manager_skill_to_stocks(
    cf_panel: pd.DataFrame,
    manager_scores: pd.DataFrame,
    ledger: PitHoldingsLedger,
    state_dollars: pd.DataFrame,
    panel_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Map static quarterly manager skill ranks onto daily eligible stocks (dual-class safe)."""
    sc_lookup = {
        (pd.Timestamp(qs).normalize(), str(fid)): (
            float(r_ir),
            float(r_al),
            float(r_st),
            int(qb),
        )
        for qs, fid, r_ir, r_al, r_st, qb in zip(
            manager_scores["eval_q_start"],
            manager_scores["filer_id"],
            manager_scores["rank_ir"],
            manager_scores["rank_alpha"],
            manager_scores["rank_ir_style"],
            manager_scores["q_bucket"],
            strict=False,
        )
    }

    min_qs = pd.Timestamp(manager_scores["eval_q_start"].min()).normalize()
    sub_cf = cf_panel[cf_panel["query_date"] >= min_qs].copy()
    dates = pd.DatetimeIndex(sorted(sub_cf["query_date"].unique()))
    blocks = _extract_state_signatures_by_date(dates, ledger)

    state_hold_map = {
        int(sid): grp[["security_id", "filer_id", "dollars_rpt", "w_rpt"]].dropna(
            subset=["dollars_rpt", "w_rpt"]
        )
        for sid, grp in state_dollars.groupby("state_id", sort=False)
    }

    parts: list[pd.DataFrame] = []
    for sig, b_dates in blocks:
        b_df = pd.DataFrame({"dt": pd.to_datetime(b_dates)})
        b_df["qs"] = b_df["dt"].dt.to_period("Q").dt.start_time
        c_list = [state_hold_map[c_sid] for _, c_sid, _ in sig if c_sid in state_hold_map]
        if not c_list:
            continue
        base = pd.concat(c_list, ignore_index=True)
        fids = base["filer_id"].to_numpy()
        for qs, q_grp in b_df.groupby("qs"):
            qs_ts = pd.Timestamp(qs).normalize()
            tuples = [sc_lookup.get((qs_ts, str(f)), (np.nan, np.nan, np.nan, 0)) for f in fids]
            r_ir = np.fromiter((t[0] for t in tuples), dtype=np.float64, count=len(fids))
            r_al = np.fromiter((t[1] for t in tuples), dtype=np.float64, count=len(fids))
            r_st = np.fromiter((t[2] for t in tuples), dtype=np.float64, count=len(fids))
            qb = np.fromiter((t[3] for t in tuples), dtype=np.int64, count=len(fids))

            ok = np.isfinite(r_ir)
            if not ok.any():
                continue
            b_ok = base.loc[ok, ["security_id", "dollars_rpt", "w_rpt"]].copy()
            w_arr = b_ok["w_rpt"].to_numpy(dtype=np.float64)
            d_arr = b_ok["dollars_rpt"].to_numpy(dtype=np.float64)
            b_ok["w_ir"] = w_arr * r_ir[ok]
            b_ok["w_al"] = w_arr * r_al[ok]
            b_ok["w_st"] = w_arr * r_st[ok]
            b_ok["d_ir"] = d_arr * r_ir[ok]
            b_ok["d_q4"] = np.where(qb[ok] == 4, d_arr, 0.0)
            b_ok["d_q1"] = np.where(qb[ok] == 1, d_arr, 0.0)

            agg = b_ok.groupby("security_id", as_index=False).agg(
                sum_w=("w_rpt", "sum"),
                sum_d=("dollars_rpt", "sum"),
                sum_w_ir=("w_ir", "sum"),
                sum_w_al=("w_al", "sum"),
                sum_w_st=("w_st", "sum"),
                sum_d_ir=("d_ir", "sum"),
                d_q4=("d_q4", "sum"),
                d_q1=("d_q1", "sum"),
                n_scored=("w_rpt", "count"),
            )
            for dt in q_grp["dt"]:
                tmp = agg.copy()
                tmp["query_date"] = pd.Timestamp(dt).normalize()
                parts.append(tmp)

    daily_sk = pd.concat(parts, ignore_index=True)

    # Equal-ratio dual-class issuer rollup before primary-class filter
    pm = panel_meta[panel_meta["query_date"].isin(dates)].merge(
        daily_sk, on=["query_date", "security_id"], how="left"
    )
    eq_dc = (
        pm["is_dual_class_issuer"].astype(bool)
        & (~pm["has_unequal_class_ratio"].astype(bool))
        & pm["issuer_cik"].astype(str).ne("")
    )
    sum_cols = [
        "sum_w",
        "sum_d",
        "sum_w_ir",
        "sum_w_al",
        "sum_w_st",
        "sum_d_ir",
        "d_q4",
        "d_q1",
        "n_scored",
    ]
    dc_s = pm.loc[eq_dc].groupby(["query_date", "issuer_cik"], as_index=False)[sum_cols].sum()
    pm = pm.merge(dc_s, on=["query_date", "issuer_cik"], how="left", suffixes=("", "_dc"))
    for c in sum_cols:
        pm[c] = np.where(eq_dc, pm[f"{c}_dc"], pm[c])

    pm = pm[pm["is_panel_eligible"].astype(bool)].copy()
    pm["skill_conv"] = np.where(pm["sum_w"] > 0.0, pm["sum_w_ir"] / pm["sum_w"], np.nan)
    pm["skill_alpha"] = np.where(pm["sum_w"] > 0.0, pm["sum_w_al"] / pm["sum_w"], np.nan)
    pm["skill_style"] = np.where(pm["sum_w"] > 0.0, pm["sum_w_st"] / pm["sum_w"], np.nan)
    pm["skill_dollar"] = np.where(pm["sum_d"] > 0.0, pm["sum_d_ir"] / pm["sum_d"], np.nan)

    keep = [
        "query_date",
        "security_id",
        "skill_conv",
        "skill_alpha",
        "skill_style",
        "skill_dollar",
        "d_q4",
        "d_q1",
        "n_scored",
    ]
    out = sub_cf.merge(pm[keep], on=["query_date", "security_id"], how="left")
    out["skill_q4_q1"] = (out["d_q4"] - out["d_q1"]) / out["pit_mcap_usd"]
    out["fwd_beta_adj_ret_bps"] = out["fwd_beta_adj_ret"].astype(np.float64) * 1e4

    for col in ("skill_conv", "skill_dollar", "skill_q4_q1", "skill_alpha", "skill_style"):
        out[f"{col}_sn"] = _cs_size_neutral_z(out, col)
    return out


def run_manager_skill_study(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    output_dir: Path | None = DEFAULT_FACTORS_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run complete static quarterly sliding-window Manager-Skill Smart Money validation."""
    store = PitMarketStore.from_parquet()
    s3 = Stage3MarketMetrics(store)
    ledger = PitHoldingsLedger.from_parquet()

    end_dt = "2023-12-31" if sample == "in_sample" else "2026-12-31"
    max_rep = "2023-09-30" if sample == "in_sample" else None
    end_eval_q = "2023-10-01" if sample == "in_sample" else "2026-10-01"

    ret_mats = build_clean_stock_return_matrices(
        store, s3, start_date="2015-01-01", end_date=end_dt
    )
    sd = compute_state_report_date_dollars(
        ledger, store, min_report_date="2014-12-31", max_report_date=max_rep
    )
    state_q_returns, fund_qa = compute_drifted_state_quarter_returns(
        ledger, sd, ret_mats, min_report_date="2014-12-31", max_eval_end_date=end_dt
    )

    fp = load_factor_panel(
        panel_path=panel_path,
        spy_path=spy_path,
        sample=sample,
        min_cross_section=DEFAULT_MIN_CROSS_SECTION,
    )
    cf = build_crowding_exposures(fp)
    p_meta = pd.read_parquet(
        panel_path,
        columns=[
            "query_date",
            "security_id",
            "issuer_cik",
            "is_dual_class_issuer",
            "has_unequal_class_ratio",
            "is_panel_eligible",
        ],
    )
    p_meta["query_date"] = pd.to_datetime(p_meta["query_date"]).dt.normalize()

    # Primary: 8-quarter (2-year) static quarterly sliding window (2017-Q1 -> 2023-Q4)
    sc8, rank_qa8 = build_static_quarterly_manager_rankings(
        ledger,
        state_q_returns,
        ret_mats["cal"],
        lookback_qtrs=8,
        min_qtrs=6,
        start_eval_quarter="2017-01-01",
        end_eval_quarter=end_eval_q,
    )
    persistence_qa = audit_manager_skill_persistence(sc8, ledger, state_q_returns)
    panel_8q = map_manager_skill_to_stocks(cf, sc8, ledger, sd, p_meta)

    specs: list[tuple[str, str]] = [
        ("skill_conv_sn", "Primary (8Q): Conviction-Weighted Beta-Adj IR Skill"),
        ("skill_dollar_sn", "Alt 1 (8Q): Dollar-Weighted Beta-Adj IR Skill"),
        ("skill_q4_q1_sn", "Alt 2 (8Q): Top-Quartile − Bottom-Quartile Ownership %"),
        ("skill_alpha_sn", "Alt 3 (8Q): Conviction-Weighted Cumulative Alpha"),
        ("skill_style_sn", "Alt 4 (8Q): Style-Relative Beta-Adj IR Skill"),
    ]

    fits: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for col, label in specs:
        sub = panel_8q[[col, "fwd_beta_adj_ret_bps", "query_date", "crowding_z"]].dropna()
        fit = sm.OLS(sub["fwd_beta_adj_ret_bps"], sm.add_constant(sub[[col]])).fit(
            cov_type="cluster", cov_kwds={"groups": sub["query_date"]}
        )
        fits[col] = fit
        fd = compute_daily_factor_returns(panel_8q, col)
        spy_b = float(
            np.cov(fd["f_beta_adj"], fd["fwd_spy_ret"], ddof=1)[0, 1]
            / np.var(fd["fwd_spy_ret"], ddof=1)
        )
        summary_rows.append(
            {
                "window": "8Q (2yr)",
                "signal": col,
                "label": label,
                "n_obs": int(len(sub)),
                "n_days": int(sub["query_date"].nunique()),
                "corr_crowding": float(sub[col].corr(sub["crowding_z"])),
                "coef_bps": float(fit.params[col]),
                "clustered_se": float(fit.bse[col]),
                "z_stat": float(fit.tvalues[col]),
                "p_value": float(fit.pvalues[col]),
                "spy_beta_check": spy_b,
            }
        )

    # Lookback sensitivity: 4Q (1yr) & 12Q (3yr) for Primary & Style-Relative IR
    sens_rows: list[dict[str, Any]] = [
        r for r in summary_rows if r["signal"] in ("skill_conv_sn", "skill_style_sn")
    ]
    for nq, minq, sq, wlbl in [
        (4, 3, "2016-01-01", "4Q (1yr)"),
        (12, 8, "2018-01-01", "12Q (3yr)"),
    ]:
        sc_w, _ = build_static_quarterly_manager_rankings(
            ledger,
            state_q_returns,
            ret_mats["cal"],
            lookback_qtrs=nq,
            min_qtrs=minq,
            start_eval_quarter=sq,
            end_eval_quarter=end_eval_q,
        )
        p_w = map_manager_skill_to_stocks(cf, sc_w, ledger, sd, p_meta)
        for col, lbl_sfx in [
            ("skill_conv_sn", "Conviction-Weighted Beta-Adj IR Skill"),
            ("skill_style_sn", "Style-Relative Beta-Adj IR Skill"),
        ]:
            sub = p_w[[col, "fwd_beta_adj_ret_bps", "query_date", "crowding_z"]].dropna()
            fit = sm.OLS(sub["fwd_beta_adj_ret_bps"], sm.add_constant(sub[[col]])).fit(
                cov_type="cluster", cov_kwds={"groups": sub["query_date"]}
            )
            fd = compute_daily_factor_returns(p_w, col)
            spy_b = float(
                np.cov(fd["f_beta_adj"], fd["fwd_spy_ret"], ddof=1)[0, 1]
                / np.var(fd["fwd_spy_ret"], ddof=1)
            )
            sens_rows.append(
                {
                    "window": wlbl,
                    "signal": col,
                    "label": f"{wlbl}: {lbl_sfx}",
                    "n_obs": int(len(sub)),
                    "n_days": int(sub["query_date"].nunique()),
                    "corr_crowding": float(sub[col].corr(sub["crowding_z"])),
                    "coef_bps": float(fit.params[col]),
                    "clustered_se": float(fit.bse[col]),
                    "z_stat": float(fit.tvalues[col]),
                    "p_value": float(fit.pvalues[col]),
                    "spy_beta_check": spy_b,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    sensitivity_df = pd.DataFrame(sens_rows).sort_values(["signal", "window"]).reset_index(
        drop=True
    )

    # Decile profile (D1 Bottom Skill -> D10 Top Skill) on primary 8Q Conviction-Weighted IR
    _, deciles_df = compute_crowding_decile_profile(
        panel_8q.dropna(subset=["skill_conv_sn"]), sort_col="skill_conv_sn"
    )

    qa_combined = {
        **fund_qa,
        **rank_qa8,
        **persistence_qa,
        "stock_coverage_ge1_pct": float(panel_8q["skill_conv"].notna().mean()),
        "stock_coverage_ge2_pct": float((panel_8q["n_scored"] >= 2).mean()),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        pd.DataFrame([qa_combined]).to_csv(
            output_dir / f"manager_skill_qa_audit_{tag}.csv", index=False
        )
        summary_df.to_csv(output_dir / f"manager_skill_summary_{tag}.csv", index=False)
        sensitivity_df.to_csv(output_dir / f"manager_skill_sensitivity_{tag}.csv", index=False)
        deciles_df.to_csv(output_dir / f"manager_skill_deciles_{tag}.csv", index=False)

    return {
        "panel": panel_8q,
        "manager_scores": sc8,
        "qa_audit": qa_combined,
        "summary": summary_df,
        "sensitivity": sensitivity_df,
        "fits": fits,
        "deciles": deciles_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Rolling PIT Manager-Skill Smart Money validation."
    )
    parser.add_argument("--sample", choices=["in_sample", "oos", "full"], default="in_sample")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    res = run_manager_skill_study(sample=args.sample)
    qa = res["qa_audit"]
    print(f"\n=== MANAGER-SKILL SMART MONEY VALIDATION ({args.sample.upper()}) ===")
    print(
        f"QA Audit: masked_stock_days={qa['n_extreme_stock_days_masked']}, "
        f"states={qa['states_evaluated']}/{qa['states_total']} ({qa['states_coverage_pct']:.1%}), "
        f"fund_SPY_beta: {qa['median_raw_fund_spy_beta']:.3f} -> "
        f"{qa['median_badj_fund_spy_beta']:+.3f}, "
        f"PIT_violations={qa['pit_violations']}, "
        f"closed_peers={qa['closed_peers_scored']}/{qa['closed_peers_total']}"
    )
    print(
        f"Manager 1Q Persistence IC: overall={qa['overall_ic_mean']:+.4f} "
        f"(std={qa['overall_ic_std']:.3f}, t={qa['overall_ic_t']:+.2f}) | "
        f"style_rel={qa['style_rel_ic_mean']:+.4f} "
        f"(std={qa['style_rel_ic_std']:.3f}, t={qa['style_rel_ic_t']:+.2f})"
    )
    print("\n--- 8Q STATIC QUARTERLY MANAGER-SKILL OLS SUMMARY ---")
    cols = [
        "signal",
        "corr_crowding",
        "coef_bps",
        "clustered_se",
        "z_stat",
        "p_value",
        "spy_beta_check",
    ]
    print(res["summary"][cols].to_string(index=False))
    print("\n--- LOOKBACK SENSITIVITY (4Q / 8Q / 12Q) ---")
    print(res["sensitivity"][["window"] + cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
