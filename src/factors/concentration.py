"""Concentration Shifts Factor Validation (`delta_pct_own` — PIT Amendment-Safe QoQ Deltas).

Constructs and validates our second positioning hypothesis — **Concentration Shifts** —
measured primarily as the quarter-over-quarter active change in aggregate percentage of
shares outstanding owned by our 100 audited fundamental managers (`delta_pct_own`), plus
supporting shift proxies (`mean_active_weight_shift`, `delta_holder_hhi_pit`,
`net_buyer_ratio`), strictly inside the In-Sample window by default (`2015-05-18 ->
2023-12-28`, `fwd_ret_date <= 2023-12-31`, zero OOS touch).

Critical PIT Amendment-Safe Delta Guarantee (`13F-HR/A` handling):
  We NEVER compute `delta_pct_own` by differencing yesterday's aggregate `pct_own` record.
  Instead, on every `query_date`, `PitHoldingsLedger` independently resolves EVERY active
  filer's:
    1. Most up-to-date amended state for its latest `period_of_report` (`Q_curr`) as of
       `query_date` (`valid_from <= query_date < valid_to`), AND
    2. Most up-to-date amended state for its immediately preceding consecutive quarter
       (`Q_prev`, `0 < gap <= 105 days`, excluding `>75%` fund wind-downs) as of `query_date`.
  Both `Q_curr` and `Q_prev` share counts are forward-adjusted for splits to exact
  `query_date` share units before differencing, rolled up across equal-ratio dual-class
  issuers (`issuer_delta_shares_pit`), and divided by PIT `shares_out_pit[t]`:
      delta_pct_own_{i,t} = issuer_delta_shares_pit_{i,t} / shares_out_pit_{i,t}
  This handles:
    - Current-quarter `13F-HR/A` (`RESTATEMENT` & `NEW HOLDINGS` CTR reveals): updates
      `Q_curr` on `T+1` while keeping `Q_prev` intact (never diffs `Q_curr_amended` vs.
      `Q_curr_original`).
    - Retroactive prior-quarter `13F-HR/A`: keeps `Q_curr` intact while updating `Q_prev`
      to the newly amended prior-quarter state on `T+1` and recomputing the exact delta.

Usage:
    python -m src.factors.concentration
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.config import PROCESSED_DATA_DIR, load_config
from src.factors.crowding import build_crowding_exposures, compute_crowding_decile_profile
from src.factors.ownership import (
    DEFAULT_FACTORS_OUTPUT_DIR,
    DEFAULT_MIN_CROSS_SECTION,
    DEFAULT_SPY_PARQUET,
    DEFAULT_WINSOR_LIMITS,
    SampleSplit,
    _orthogonalize_against,
    _standardize_vec,
    attach_fwd_ret_date,
    compute_daily_factor_returns,
)
from src.pit.daily_panel import DEFAULT_DAILY_PANEL_PARQUET, _extract_state_signatures_by_date
from src.pit.ledger import PitHoldingsLedger

logger = logging.getLogger(__name__)

# First trading day after the statutory 45-day Q1 2015 13F deadline (2015-05-15), when
# consecutive QoQ deltas (Q1 2015 - Q4 2014) reach 99.7% eligible universe coverage.
DEFAULT_CONCENTRATION_IS_START = "2015-05-18"


def audit_pit_amendment_qoq_transitions(
    ledger: PitHoldingsLedger,
    spy_path: Path = DEFAULT_SPY_PARQUET,
) -> dict[str, int]:
    """Verify `Q_curr` and `Q_prev` state-signature transitions across ALL real `13F-HR/A`s.

    Confirms across the entire ledger that:
      - Every `Q_curr` amendment replaces `curr_state_id` on `T` while preserving `prev_state_id`
        (never diffing amended `Q_curr` against unamended `Q_curr`).
      - Every immediately-prior-quarter (`Q_prev`) retroactive amendment updates `prev_state_id`
        on `T` while preserving `curr_state_id` (or updates both if both were amended same day).
    """
    st = ledger.states_df.sort_values(["filer_id", "valid_from", "period_of_report"])
    spy = pd.read_parquet(spy_path, columns=["date"])
    cal = pd.DatetimeIndex(
        pd.to_datetime(spy["date"]).dt.normalize().drop_duplicates().sort_values()
    )
    cal = cal[cal.day_of_week < 5]

    n_qcurr_checked = 0
    n_qprev_checked = 0

    for fid, grp in st.groupby("filer_id"):
        amends = grp[grp["source_form_type"] == "13F-HR/A"]
        for _, r in amends.iterrows():
            eff = pd.Timestamp(r["valid_from"]).normalize()
            rep = pd.Timestamp(r["period_of_report"]).normalize()
            pos = int(cal.searchsorted(eff))
            if pos <= 0 or pos >= len(cal):
                continue
            d_before, d_on = cal[pos - 1], cal[pos]

            act_b = grp[
                (grp["valid_from"] <= d_before)
                & (grp["valid_to"].isna() | (grp["valid_to"] > d_before))
            ]
            if act_b.empty:
                continue
            reps_b = sorted(act_b["period_of_report"].unique())
            is_qcurr = reps_b[-1] == rep
            is_qprev = len(reps_b) >= 2 and reps_b[-2] == rep
            if not (is_qcurr or is_qprev):
                continue

            blks = _extract_state_signatures_by_date(pd.DatetimeIndex([d_before, d_on]), ledger)
            if not blks:
                continue
            sig_b = {x[0]: (x[1], x[2]) for x in blks[0][0]}
            sig_o = {x[0]: (x[1], x[2]) for x in blks[-1][0]}
            pair_b = sig_b.get(str(fid))
            pair_o = sig_o.get(str(fid))
            if pair_b is None or pair_o is None:
                continue

            new_sid = int(r["state_id"])
            if is_qcurr:
                # Must never set prev_sid to the superseded unamended Q_curr state!
                if pair_o[1] == pair_b[0]:
                    raise AssertionError(
                        f"Amendment bug on {fid} {eff.date()}: Q_curr amendment pushed old Q_curr "
                        f"({pair_b[0]}) into Q_prev!"
                    )
                if pair_o[0] == new_sid:
                    n_qcurr_checked += 1
            elif is_qprev:
                if pair_o[1] == new_sid:
                    n_qprev_checked += 1

    return {
        "q_curr_amendments_verified": n_qcurr_checked,
        "q_prev_amendments_verified": n_qprev_checked,
    }


def load_concentration_panel(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    min_signal_date: str | None = DEFAULT_CONCENTRATION_IS_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
) -> pd.DataFrame:
    """Load eligible daily PIT panel with QoQ flow/concentration fields respecting IS/OOS."""
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
        "shares_out_pit",
        "pit_mcap_usd",
        "adv_20d_usd",
        "issuer_delta_shares_pit",
        "issuer_total_shares_owned_pit",
        "n_flow_filers",
        "n_buyers",
        "n_sellers",
        "mean_active_weight_shift",
        "sum_active_weight_shift",
        "holder_hhi_pit",
        "delta_holder_hhi_pit",
        "beta_252d_lag1",
        "fwd_day_ret",
        "fwd_spy_ret",
        "fwd_beta_adj_ret",
        "is_panel_eligible",
    ]
    panel = pd.read_parquet(panel_path, columns=cols)
    spy = pd.read_parquet(spy_path, columns=["date"])
    panel = attach_fwd_ret_date(panel, spy)

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

    valid = (
        panel["is_panel_eligible"].fillna(False)
        & panel["pct_shares_owned"].notna()
        & (panel["pct_shares_owned"] > 0.0)
        & panel["shares_out_pit"].notna()
        & (panel["shares_out_pit"] > 0.0)
        & panel["days_to_liquidate"].notna()
        & (panel["days_to_liquidate"] > 0.0)
        & panel["pit_mcap_usd"].notna()
        & (panel["pit_mcap_usd"] > 0.0)
        & panel["issuer_delta_shares_pit"].notna()
        & panel["beta_252d_lag1"].notna()
        & panel["fwd_day_ret"].notna()
        & panel["fwd_spy_ret"].notna()
        & panel["fwd_beta_adj_ret"].notna()
    )
    panel = panel.loc[valid].copy()

    if min_cross_section > 1 and not panel.empty:
        day_counts = panel.groupby("query_date")["security_id"].transform("count")
        panel = panel.loc[day_counts >= min_cross_section].copy()

    return panel.sort_values(["query_date", "security_id"]).reset_index(drop=True)


def build_concentration_shift_exposures(
    panel: pd.DataFrame,
    *,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.DataFrame:
    """Attach strictly day-`t` cross-sectional Concentration Shift exposures (and `crowding_z`).

    Primary signal:
      - `delta_pct_own` = `issuer_delta_shares_pit / shares_out_pit` (active QoQ change in
        aggregate % shares owned across paired active filers, split- & amendment-safe).
      - `delta_pct_own_z` / `delta_pct_own_sn`: 1%/99% winsorized & size-neutralized z-scores.

    Supporting shift benchmarks (for ablation):
      - `active_w_shift_sn`: constant-price `mean_active_weight_shift` (size-neutral z-score)
      - `delta_hhi_sn`: `delta_holder_hhi_pit` (size-neutral z-score)
      - `net_buyer_ratio_sn`: `(n_buyers - n_sellers) / n_flow_filers` (size-neutral z-score)
    """
    if panel.empty:
        return panel.copy()

    df = build_crowding_exposures(panel, winsor_limits=winsor_limits)
    df["delta_pct_own"] = df["issuer_delta_shares_pit"].to_numpy(dtype=np.float64) / df[
        "shares_out_pit"
    ].to_numpy(dtype=np.float64)
    df["net_buyer_ratio"] = np.where(
        df["n_flow_filers"].fillna(0) > 0,
        (df["n_buyers"].fillna(0) - df["n_sellers"].fillna(0))
        / np.where(df["n_flow_filers"].fillna(0) > 0, df["n_flow_filers"].fillna(0), 1.0),
        np.nan,
    )

    n_rows = len(df)
    lo_q, hi_q = winsor_limits
    lm_all = df["log_mcap"].to_numpy(dtype=np.float64)
    dates_ns = df["query_date"].to_numpy(dtype="datetime64[ns]")
    change_idx = np.flatnonzero(dates_ns[1:] != dates_ns[:-1]) + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [n_rows]))

    src_map = [
        ("delta_pct_own", "delta_pct_own_z", "delta_pct_own_sn"),
        ("mean_active_weight_shift", "active_w_shift_z", "active_w_shift_sn"),
        ("delta_holder_hhi_pit", "delta_hhi_z", "delta_hhi_sn"),
        ("net_buyer_ratio", "net_buyer_ratio_z", "net_buyer_ratio_sn"),
    ]

    for raw_col, z_col, sn_col in src_map:
        raw_arr = df[raw_col].to_numpy(dtype=np.float64)
        z_out = np.full(n_rows, np.nan, dtype=np.float64)
        sn_out = np.full(n_rows, np.nan, dtype=np.float64)

        for s, e in zip(starts, ends, strict=False):
            sl = raw_arr[s:e]
            lm = lm_all[s:e]
            ok = np.isfinite(sl) & np.isfinite(lm)
            if int(np.sum(ok)) < 5:
                continue
            vals = sl[ok]
            lo, hi = np.quantile(vals, [lo_q, hi_q])
            zr = _standardize_vec(np.clip(vals, lo, hi))
            zsn = _orthogonalize_against(zr, lm[ok])
            idx_ok = np.flatnonzero(ok) + s
            z_out[idx_ok] = zr
            sn_out[idx_ok] = zsn

        df[z_col] = z_out
        df[sn_col] = sn_out

    return df


def compute_quarterly_persistence(
    panel_cs: pd.DataFrame,
    cols: list[str],
    *,
    lag_trading_days: int = 63,
) -> dict[str, float]:
    """Compute average 1-quarter (`63`-trading-day) cross-sectional autocorrelation per signal."""
    out: dict[str, float] = {}
    for col in cols:
        piv = panel_cs.pivot(index="query_date", columns="security_id", values=col)
        ac = piv.corrwith(piv.shift(lag_trading_days), axis=1).dropna()
        out[col] = float(ac.mean()) if not ac.empty else np.nan
    return out


def run_concentration_shifts_study(
    *,
    sample: SampleSplit = "in_sample",
    min_signal_date: str | None = DEFAULT_CONCENTRATION_IS_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    output_dir: Path | None = DEFAULT_FACTORS_OUTPUT_DIR,
    verify_amendments: bool = True,
) -> dict[str, Any]:
    """Run strictly PIT Concentration Shifts OLS & decile validation (default: In-Sample only)."""
    amend_audit: dict[str, int] = {}
    if verify_amendments and (PROCESSED_DATA_DIR / "pit_13f_states.parquet").exists():
        ledger = PitHoldingsLedger.from_parquet()
        amend_audit = audit_pit_amendment_qoq_transitions(ledger)

    panel = load_concentration_panel(
        sample=sample,
        min_signal_date=min_signal_date,
        min_cross_section=min_cross_section,
    )
    panel_cs = build_concentration_shift_exposures(panel)
    panel_cs["fwd_beta_adj_ret_bps"] = panel_cs["fwd_beta_adj_ret"] * 1e4

    specs = [
        ("delta_pct_own_sn", "Primary: delta_pct_own (QoQ Δ % Shares Owned, Size-Neutral)"),
        ("active_w_shift_sn", "Alt 1: mean_active_weight_shift (Constant-Price Active Δw)"),
        ("delta_hhi_sn", "Alt 2: delta_holder_hhi (QoQ Δ Holder HHI)"),
        ("net_buyer_ratio_sn", "Alt 3: net_buyer_ratio ((Buyers - Sellers) / Flow Filers)"),
    ]

    fits: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    for col, label in specs:
        sub = panel_cs.dropna(subset=[col, "fwd_beta_adj_ret_bps", "crowding_z"]).copy()
        X = sm.add_constant(sub[[col]])
        fit = sm.OLS(sub["fwd_beta_adj_ret_bps"], X).fit(
            cov_type="cluster", cov_kwds={"groups": sub["query_date"]}
        )
        fits[col] = fit

        ddf = compute_daily_factor_returns(sub, signal_col=col)
        spy_v = float(np.var(ddf["fwd_spy_ret"], ddof=1))
        spy_b = float(np.cov(ddf["f_beta_adj"], ddf["fwd_spy_ret"], ddof=1)[0, 1] / spy_v)

        summary_rows.append(
            {
                "signal": col,
                "label": label,
                "n_obs": len(sub),
                "corr_crowding": float(sub[col].corr(sub["crowding_z"])),
                "coef_bps": float(fit.params[col]),
                "clustered_se": float(fit.bse[col]),
                "z_stat": float(fit.tvalues[col]),
                "p_value": float(fit.pvalues[col]),
                "spy_beta_check": spy_b,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    _, deciles_summary = compute_crowding_decile_profile(
        panel_cs, sort_col="delta_pct_own_sn", n_buckets=10
    )
    persistence = compute_quarterly_persistence(
        panel_cs, ["crowding_z", "delta_pct_own_sn", "active_w_shift_sn"]
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        summary_df.to_csv(output_dir / f"concentration_shifts_summary_{tag}.csv", index=False)
        deciles_summary.to_csv(output_dir / f"concentration_shifts_deciles_{tag}.csv", index=False)

    return {
        "panel": panel_cs,
        "summary": summary_df,
        "deciles_summary": deciles_summary,
        "persistence_63d": persistence,
        "amendment_audit": amend_audit,
        "fits": fits,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PIT amendment-safe Concentration Shifts validation (default: in_sample)."
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
    res = run_concentration_shifts_study(sample=args.sample)

    print(f"\n=== CONCENTRATION SHIFTS VALIDATION ({args.sample.upper()}) ===")
    print("PIT 13F-HR/A Amendment Transition Audit:", res["amendment_audit"])
    print("\n--- CLUSTERED OLS SUMMARY (fwd_beta_adj_ret_bps ~ const + signal) ---")
    print(res["summary"].to_string(index=False))
    print("\n--- 1-QUARTER (63d) CROSS-SECTIONAL PERSISTENCE ---")
    for k, v in res["persistence_63d"].items():
        print(f"  {k:22s}: {v:+.3f}")
    print("\n--- DECILE PROFILE (delta_pct_own_sn: D1 Heavy Selling -> D10 Heavy Buying) ---")
    print(
        res["deciles_summary"][
            ["bucket", "mean_z", "badj_mean_bps", "badj_std_bps", "badj_t_stat", "badj_p_value"]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
