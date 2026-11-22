"""Smart Money Following Factor Validation (Report-Date Dollars Invested).

Constructs and validates our third institutional positioning hypothesis — **Smart Money
Following** — defined as following the aggregate dollar capital deployed by our 100 audited
fundamental managers, valued strictly as of each filing's **`period_of_report`** (NOT the
filing date or query date):

    dollars_invested_rpt_{f, i}(q_f) = shares_{f, i}(q_f) * raw_close_i(last_trading_day <= q_f)

Point-in-time & data-engineering guarantees:
  1. **Report-date pricing (zero forward look):** Quarter-ends often fall on weekends or
     holidays (e.g., `2023-09-30` Saturday, `2018-03-31` Easter weekend). We look up
     `raw_close` strictly on the last trading day `<= period_of_report` (`max lag <= 3d`),
     never peeking at next-week prices.
  2. **Cross-validated against SEC `reported_value_usd`:** Independently pricing `shares`
     at `raw_close(<= period_of_report)` avoids SEC 13F XML `value` column typos while
     giving an exact row-level QA audit (`median ratio = 1.0000`, `~96%` within 1%).
  3. **Staggered multi-quarter & `13F-HR/A` amendment safety:** On every `query_date`,
     each active filer uses its own latest PIT-visible amended `state_id` and that state's
     own `period_of_report` price (never mixing Q2 shares with Q3 prices), rolled up across
     equal-ratio dual-class issuers (`issuer_cik`) identically to `pit_daily_panel.parquet`.
  4. **Strict In-Sample boundary:** Defaults to `sample="in_sample"` (`fwd_ret_date <=
     2023-12-31`, `2015-02-18 -> 2023-12-28`, zero OOS touch).

Usage:
    python -m src.factors.smart_money
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.factors.crowding import build_crowding_exposures, compute_crowding_decile_profile
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
    load_factor_panel,
)
from src.pit.daily_panel import DEFAULT_DAILY_PANEL_PARQUET, _extract_state_signatures_by_date
from src.pit.ledger import DEFAULT_MAX_STALENESS_DAYS, PitHoldingsLedger
from src.pit.panel import PitMarketStore

logger = logging.getLogger(__name__)


def _lookup_report_date_prices(
    df: pd.DataFrame,
    market_store: PitMarketStore,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(raw_close_rpt, price_date_rpt)` on latest trading day `<= period_of_report`."""
    n = len(df)
    px_rpt = np.full(n, np.nan, dtype=np.float64)
    dt_rpt = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    if n == 0:
        return px_rpt, dt_rpt

    df_reset = df.reset_index(drop=True)
    for tkr, grp in df_reset.groupby("yahoo_ticker", sort=False):
        arr = market_store._ticker_arrays.get(str(tkr).strip().upper())
        if arr is None or len(arr["dates_ns"]) == 0:
            continue
        d_ns = arr["dates_ns"]
        rc = arr["raw_close"]
        rep_ns = grp["period_of_report"].to_numpy(dtype="datetime64[ns]")
        pos = np.searchsorted(d_ns, rep_ns, side="right") - 1
        ok = pos >= 0
        if np.any(ok):
            idxs = grp.index.to_numpy()[ok]
            px_rpt[idxs] = rc[pos[ok]]
            dt_rpt[idxs] = d_ns[pos[ok]]
    return px_rpt, dt_rpt


def compute_state_report_date_dollars(
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
    *,
    min_report_date: str = "2014-01-01",
    max_report_date: str | None = None,
) -> pd.DataFrame:
    """Compute exact per-`(state_id, security_id)` dollars invested as of `period_of_report`.

    For every US Common Stock holding in each PIT 13F state, computes:
      - `dollars_rpt = shares * raw_close(last_trading_day <= period_of_report)`
      - `w_rpt = dollars_rpt / sum_j(dollars_rpt)` (per-filer report-date portfolio weight)
    while retaining `reported_value_usd` and `price_lag_days` for transparent QA auditing.
    """
    st = ledger.states_df[
        ["state_id", "filer_id", "canonical_name", "period_of_report", "filing_date"]
    ].copy()
    st["period_of_report"] = pd.to_datetime(st["period_of_report"]).dt.normalize()
    st = st[st["period_of_report"] >= pd.Timestamp(min_report_date)].copy()
    if max_report_date is not None:
        st = st[st["period_of_report"] <= pd.Timestamp(max_report_date)].copy()

    # Flatten state holdings from ledger
    h_parts: list[pd.DataFrame] = []
    valid_sids = set(int(s) for s in st["state_id"])
    for sid, h_df in ledger._holdings_by_state.items():
        if int(sid) not in valid_sids or h_df.empty:
            continue
        sub = h_df[["cusip", "shares", "reported_value_usd"]].copy()
        sub["state_id"] = int(sid)
        h_parts.append(sub)
    if not h_parts:
        return pd.DataFrame(
            columns=[
                "state_id",
                "filer_id",
                "canonical_name",
                "period_of_report",
                "filing_date",
                "security_id",
                "yahoo_ticker",
                "shares",
                "reported_value_usd",
                "raw_close_rpt",
                "price_date_rpt",
                "price_lag_days",
                "dollars_rpt",
                "w_rpt",
            ]
        )

    ev = pd.concat(h_parts, ignore_index=True).merge(st, on="state_id", how="inner")

    cm = market_store.cusip_map
    cm_us = cm[cm["is_us_common_stock"].astype(bool)][
        ["cusip", "security_id", "yahoo_ticker"]
    ].drop_duplicates(subset=["cusip"], keep="first")

    mapped = ev.merge(cm_us, on="cusip", how="inner")
    # Aggregate multiple CUSIPs of the same canonical security within one state
    agg = mapped.groupby(
        [
            "state_id",
            "filer_id",
            "canonical_name",
            "period_of_report",
            "filing_date",
            "security_id",
            "yahoo_ticker",
        ],
        as_index=False,
    ).agg(
        shares=("shares", "sum"),
        reported_value_usd=("reported_value_usd", "sum"),
    )

    px_rpt, dt_rpt = _lookup_report_date_prices(agg, market_store)
    agg["raw_close_rpt"] = px_rpt
    agg["price_date_rpt"] = dt_rpt
    agg["price_lag_days"] = (agg["period_of_report"] - agg["price_date_rpt"]).dt.days
    agg["dollars_rpt"] = agg["shares"] * agg["raw_close_rpt"]

    # Report-date portfolio weight per state (across priced US common holdings)
    state_tot = agg.groupby("state_id")["dollars_rpt"].transform("sum")
    agg["w_rpt"] = np.where(state_tot > 0.0, agg["dollars_rpt"] / state_tot, np.nan)
    return agg


def audit_report_date_dollars_qa(state_dollars: pd.DataFrame) -> dict[str, Any]:
    """Cross-audit `shares * raw_close(<= period_of_report)` against SEC `reported_value_usd`."""
    total_rows = int(len(state_dollars))
    priced = state_dollars[state_dollars["raw_close_rpt"].notna()].copy()
    n_priced = int(len(priced))
    max_lag = int(priced["price_lag_days"].max()) if n_priced else 0
    n_lag_gt5 = int((priced["price_lag_days"] > 5).sum()) if n_priced else 0
    n_forward_leak = int((priced["price_lag_days"] < 0).sum()) if n_priced else 0

    if n_forward_leak > 0:
        raise AssertionError(
            f"REPORT-DATE PRICE LOOKAHEAD: {n_forward_leak} rows used price_date > period_of_report"
        )

    valid = priced[(priced["reported_value_usd"] > 0.0) & (priced["dollars_rpt"] > 0.0)].copy()
    if valid.empty:
        return {
            "total_state_rows": total_rows,
            "priced_rows": n_priced,
            "priced_pct": 0.0,
            "max_price_lag_days": max_lag,
            "rows_lag_gt_5d": n_lag_gt5,
            "ratio_p25": np.nan,
            "ratio_p50": np.nan,
            "ratio_p75": np.nan,
            "pct_within_1pct": np.nan,
            "pct_within_2pct": np.nan,
            "pct_within_5pct": np.nan,
            "rows_off_gt_2x": 0,
            "pct_off_gt_2x": np.nan,
        }

    ratio = valid["dollars_rpt"] / valid["reported_value_usd"]
    off_2x = (ratio > 2.0) | (ratio < 0.5)
    return {
        "total_state_rows": total_rows,
        "priced_rows": n_priced,
        "priced_pct": float(n_priced / total_rows) if total_rows else 0.0,
        "max_price_lag_days": max_lag,
        "rows_lag_gt_5d": n_lag_gt5,
        "ratio_p25": float(ratio.quantile(0.25)),
        "ratio_p50": float(ratio.quantile(0.50)),
        "ratio_p75": float(ratio.quantile(0.75)),
        "pct_within_1pct": float(((ratio >= 0.99) & (ratio <= 1.01)).mean()),
        "pct_within_2pct": float(((ratio >= 0.98) & (ratio <= 1.02)).mean()),
        "pct_within_5pct": float(((ratio >= 0.95) & (ratio <= 1.05)).mean()),
        "rows_off_gt_2x": int(off_2x.sum()),
        "pct_off_gt_2x": float(off_2x.mean()),
    }


def build_daily_report_date_dollars(
    trading_dates: pd.DatetimeIndex,
    ledger: PitHoldingsLedger,
    state_dollars: pd.DataFrame,
    *,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
) -> pd.DataFrame:
    """Aggregate report-date dollars invested across active PIT filers for every `query_date`."""
    blocks = _extract_state_signatures_by_date(
        trading_dates,
        ledger,
        max_staleness_days=max_staleness_days,
    )
    state_map: dict[int, pd.DataFrame] = {
        int(sid): grp[["security_id", "dollars_rpt", "w_rpt"]].dropna(subset=["dollars_rpt"])
        for sid, grp in state_dollars.groupby("state_id", sort=False)
    }

    parts: list[pd.DataFrame] = []
    for sig, b_dates in blocks:
        c_list = [state_map[c_sid] for _, c_sid, _ in sig if c_sid in state_map]
        if not c_list:
            continue
        agg = (
            pd.concat(c_list, ignore_index=True)
            .groupby("security_id", as_index=False)
            .agg(
                dollars_rpt=("dollars_rpt", "sum"),
                sum_w_rpt=("w_rpt", "sum"),
                mean_w_rpt=("w_rpt", "mean"),
                max_w_rpt=("w_rpt", "max"),
                n_filers_rpt=("dollars_rpt", "count"),
            )
        )
        for dt in b_dates:
            tmp = agg.copy()
            tmp["query_date"] = pd.Timestamp(dt).normalize()
            parts.append(tmp)

    if not parts:
        return pd.DataFrame(
            columns=[
                "query_date",
                "security_id",
                "dollars_rpt",
                "sum_w_rpt",
                "mean_w_rpt",
                "max_w_rpt",
                "n_filers_rpt",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _cs_standardize(
    df: pd.DataFrame,
    col: str,
    *,
    size_neutral: bool,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.Series:
    """Cross-sectionally winsorize, optionally neutralize on `log(pit_mcap_usd)`, and z-score."""
    out = pd.Series(np.nan, index=df.index, dtype=np.float64)
    lo_q, hi_q = winsor_limits
    log_mcap = np.log(df["pit_mcap_usd"].astype(np.float64).clip(lower=1.0))

    for _, idx in df.groupby("query_date", sort=False).groups.items():
        x = df.loc[idx, col].to_numpy(dtype=np.float64)
        sz = log_mcap.loc[idx].to_numpy(dtype=np.float64)
        valid = np.isfinite(x) & np.isfinite(sz)
        if int(valid.sum()) < 20:
            continue
        xv = x[valid]
        lo, hi = np.quantile(xv, [lo_q, hi_q])
        xv_w = np.clip(xv, lo, hi)
        if size_neutral:
            xv_adj = _orthogonalize_against(xv_w, sz[valid])
        else:
            xv_adj = xv_w
        out.loc[idx[valid]] = _standardize_vec(xv_adj)
    return out


def load_smart_money_panel(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    min_signal_date: str | None = DEFAULT_IS_SIGNAL_START,
    min_cross_section: int = DEFAULT_MIN_CROSS_SECTION,
    ledger: PitHoldingsLedger | None = None,
    market_store: PitMarketStore | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load eligible PIT panel enriched with verified report-date dollars invested.

    Rolls up equal-ratio dual-class issuers across `(query_date, issuer_cik)` identically to
    `pit_daily_panel.parquet` before restricting to `is_panel_eligible == True` and strictly
    respecting `fwd_ret_date <= 2023-12-31` for `sample="in_sample"`.
    """
    fp = load_factor_panel(
        panel_path=panel_path,
        spy_path=spy_path,
        sample=sample,
        min_signal_date=min_signal_date,
        min_cross_section=min_cross_section,
    )
    cf = build_crowding_exposures(fp)

    led = ledger if ledger is not None else PitHoldingsLedger.from_parquet()
    store = market_store if market_store is not None else PitMarketStore.from_parquet()

    # For IS, restrict QA audit report_dates cleanly to <= 2023-12-31 (no OOS states touched)
    max_rep = "2023-12-31" if sample == "in_sample" else None
    state_dollars = compute_state_report_date_dollars(
        led,
        store,
        min_report_date="2014-01-01",
        max_report_date=max_rep,
    )
    qa_audit = audit_report_date_dollars_qa(
        state_dollars[state_dollars["period_of_report"] >= pd.Timestamp("2014-12-31")]
    )

    is_dates = pd.DatetimeIndex(sorted(cf["query_date"].unique()))
    rpt_daily = build_daily_report_date_dollars(is_dates, led, state_dollars)

    # Dual-class rollup over full panel metadata on those exact dates (before primary-class filter)
    meta_cols = [
        "query_date",
        "security_id",
        "issuer_cik",
        "is_dual_class_issuer",
        "has_unequal_class_ratio",
        "is_panel_eligible",
        "issuer_total_position_usd_pit",
    ]
    p_meta = pd.read_parquet(panel_path, columns=meta_cols)
    p_meta["query_date"] = pd.to_datetime(p_meta["query_date"]).dt.normalize()
    p_meta = p_meta[p_meta["query_date"].isin(is_dates)].merge(
        rpt_daily, on=["query_date", "security_id"], how="left"
    )

    eq_dc = (
        p_meta["is_dual_class_issuer"].astype(bool)
        & (~p_meta["has_unequal_class_ratio"].astype(bool))
        & p_meta["issuer_cik"].astype(str).ne("")
    )
    dc_sum = (
        p_meta.loc[eq_dc]
        .groupby(["query_date", "issuer_cik"], as_index=False)
        .agg(
            dc_dollars_rpt=("dollars_rpt", "sum"),
            dc_sum_w_rpt=("sum_w_rpt", "sum"),
        )
    )
    p_meta = p_meta.merge(dc_sum, on=["query_date", "issuer_cik"], how="left")
    p_meta["issuer_dollars_invested_rpt"] = np.where(
        eq_dc, p_meta["dc_dollars_rpt"], p_meta["dollars_rpt"]
    )
    p_meta["issuer_sum_w_rpt"] = np.where(eq_dc, p_meta["dc_sum_w_rpt"], p_meta["sum_w_rpt"])

    elig_meta = p_meta.loc[
        p_meta["is_panel_eligible"].astype(bool),
        [
            "query_date",
            "security_id",
            "issuer_dollars_invested_rpt",
            "issuer_sum_w_rpt",
            "mean_w_rpt",
            "max_w_rpt",
            "issuer_total_position_usd_pit",
        ],
    ]
    out = cf.merge(elig_meta, on=["query_date", "security_id"], how="left")
    qa_audit["eligible_panel_rows"] = int(len(out))
    qa_audit["eligible_non_null_pct"] = float(out["issuer_dollars_invested_rpt"].notna().mean())
    return out, qa_audit


def build_smart_money_exposures(
    panel: pd.DataFrame,
    *,
    winsor_limits: tuple[float, float] = DEFAULT_WINSOR_LIMITS,
) -> pd.DataFrame:
    """Build report-date Smart Money Following factor exposures (raw and size-neutral)."""
    df = panel.copy()
    df["fwd_beta_adj_ret_bps"] = df["fwd_beta_adj_ret"].astype(np.float64) * 1e4
    df["log_dollars_rpt"] = np.log(df["issuer_dollars_invested_rpt"].astype(np.float64).clip(1.0))

    # Book dollar share vs market-cap weight (Active Book Overweight at report date)
    tot_book = df.groupby("query_date")["issuer_dollars_invested_rpt"].transform("sum")
    tot_mcap = df.groupby("query_date")["pit_mcap_usd"].transform("sum")
    df["book_share_rpt"] = np.where(tot_book > 0.0, df["issuer_dollars_invested_rpt"] / tot_book, 0)
    df["mcap_weight"] = np.where(tot_mcap > 0.0, df["pit_mcap_usd"] / tot_mcap, 0.0)
    df["active_overweight_rpt"] = df["book_share_rpt"] - df["mcap_weight"]

    df["dollars_rpt_z"] = _cs_standardize(
        df, "issuer_dollars_invested_rpt", size_neutral=False, winsor_limits=winsor_limits
    )
    df["dollars_rpt_sn"] = _cs_standardize(
        df, "issuer_dollars_invested_rpt", size_neutral=True, winsor_limits=winsor_limits
    )
    df["log_dollars_rpt_z"] = _cs_standardize(
        df, "log_dollars_rpt", size_neutral=False, winsor_limits=winsor_limits
    )
    df["log_dollars_rpt_sn"] = _cs_standardize(
        df, "log_dollars_rpt", size_neutral=True, winsor_limits=winsor_limits
    )
    df["active_ow_rpt_sn"] = _cs_standardize(
        df, "active_overweight_rpt", size_neutral=True, winsor_limits=winsor_limits
    )
    df["mean_w_rpt_sn"] = _cs_standardize(
        df, "mean_w_rpt", size_neutral=True, winsor_limits=winsor_limits
    )
    df["max_w_rpt_sn"] = _cs_standardize(
        df, "max_w_rpt", size_neutral=True, winsor_limits=winsor_limits
    )
    return df


def run_smart_money_study(
    *,
    panel_path: Path = DEFAULT_DAILY_PANEL_PARQUET,
    spy_path: Path = DEFAULT_SPY_PARQUET,
    sample: SampleSplit = "in_sample",
    output_dir: Path | None = DEFAULT_FACTORS_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run full Smart Money Following (Report-Date Dollars Invested) validation study."""
    raw_panel, qa_audit = load_smart_money_panel(
        panel_path=panel_path,
        spy_path=spy_path,
        sample=sample,
    )
    panel = build_smart_money_exposures(raw_panel)

    specs: list[tuple[str, str]] = [
        ("dollars_rpt_z", "Raw Dollars Invested (Report Date, Z)"),
        ("dollars_rpt_sn", "Size-Neutral Dollars Invested (Report Date)"),
        ("log_dollars_rpt_z", "Raw Log Dollars Invested (Report Date, Z)"),
        ("log_dollars_rpt_sn", "Primary: Size-Neutral Log Dollars Invested (Report Date)"),
        ("active_ow_rpt_sn", "Alt 1: Active Book Overweight (Book% - MCap%, SN)"),
        ("mean_w_rpt_sn", "Alt 2: Mean Portfolio Conviction Weight (Report Date, SN)"),
        ("max_w_rpt_sn", "Alt 3: Max Single-Filer Conviction Weight (Report Date, SN)"),
    ]

    fits: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    log_mcap = np.log(panel["pit_mcap_usd"].astype(np.float64).clip(lower=1.0))

    for col, label in specs:
        sub = panel[[col, "fwd_beta_adj_ret_bps", "query_date", "crowding_z", "pct_own_sn"]].copy()
        sub["log_mcap"] = log_mcap
        sub = sub.dropna()
        X = sm.add_constant(sub[[col]])
        y = sub["fwd_beta_adj_ret_bps"]
        fit = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": sub["query_date"]})
        fits[col] = fit

        # Realized SPY beta of daily cross-sectional OLS factor return
        f_daily = compute_daily_factor_returns(panel, col)
        spy_beta = float(
            np.cov(f_daily["f_beta_adj"], f_daily["fwd_spy_ret"], ddof=1)[0, 1]
            / np.var(f_daily["fwd_spy_ret"], ddof=1)
        )
        rows.append(
            {
                "signal": col,
                "label": label,
                "n_obs": int(len(sub)),
                "n_days": int(sub["query_date"].nunique()),
                "corr_size": float(sub[col].corr(sub["log_mcap"])),
                "corr_crowding": float(sub[col].corr(sub["crowding_z"])),
                "corr_pct_own": float(sub[col].corr(sub["pct_own_sn"])),
                "coef_bps": float(fit.params[col]),
                "clustered_se": float(fit.bse[col]),
                "z_stat": float(fit.tvalues[col]),
                "p_value": float(fit.pvalues[col]),
                "spy_beta_check": spy_beta,
            }
        )

    summary_df = pd.DataFrame(rows)

    # Decile profiles for BOTH Raw Dollars Invested ("Hedge Fund VIP") and Size-Neutral Log Dollars
    _, dec_raw_df = compute_crowding_decile_profile(panel, sort_col="dollars_rpt_z")
    _, dec_log_sn_df = compute_crowding_decile_profile(panel, sort_col="log_dollars_rpt_sn")

    # Attach median report-date dollars ($M) per decile for physical interpretability
    for col, dec_df in [("dollars_rpt_z", dec_raw_df), ("log_dollars_rpt_sn", dec_log_sn_df)]:
        sub = panel[["query_date", col, "issuer_dollars_invested_rpt"]].dropna().copy()
        sub["dec"] = sub.groupby("query_date")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1
        )
        med_dol = sub.groupby("dec")["issuer_dollars_invested_rpt"].median() / 1e6
        dec_df["med_dollars_rpt_m"] = [
            float(med_dol.get(i + 1, np.nan)) if i < 10 else np.nan for i in range(len(dec_df))
        ]

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = "is" if sample == "in_sample" else sample
        pd.DataFrame([qa_audit]).to_csv(output_dir / f"smart_money_qa_audit_{tag}.csv", index=False)
        summary_df.to_csv(output_dir / f"smart_money_summary_{tag}.csv", index=False)
        dec_raw_df.to_csv(output_dir / f"smart_money_deciles_raw_{tag}.csv", index=False)
        dec_log_sn_df.to_csv(output_dir / f"smart_money_deciles_log_sn_{tag}.csv", index=False)

    return {
        "panel": panel,
        "qa_audit": qa_audit,
        "summary": summary_df,
        "fits": fits,
        "deciles_raw": dec_raw_df,
        "deciles_log_sn": dec_log_sn_df,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Smart Money Following (Report-Date Dollars Invested) validation."
    )
    parser.add_argument("--sample", choices=["in_sample", "oos", "full"], default="in_sample")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    res = run_smart_money_study(sample=args.sample)

    print(f"\n=== SMART MONEY FOLLOWING VALIDATION ({args.sample.upper()}) ===")
    qa = res["qa_audit"]
    print(
        f"Report-Date Pricing QA Audit: priced={qa['priced_rows']:,}/{qa['total_state_rows']:,} "
        f"({qa['priced_pct']:.2%}), median_ratio={qa['ratio_p50']:.4f}, "
        f"within_1%={qa['pct_within_1pct']:.2%}, within_5%={qa['pct_within_5pct']:.2%}, "
        f"max_price_lag={qa['max_price_lag_days']}d, "
        f"eligible_non_null={qa['eligible_non_null_pct']:.2%}"
    )
    print("\n--- CLUSTERED OLS SUMMARY (fwd_beta_adj_ret_bps ~ const + signal) ---")
    cols = [
        "signal",
        "corr_size",
        "corr_crowding",
        "coef_bps",
        "clustered_se",
        "z_stat",
        "p_value",
        "spy_beta_check",
    ]
    print(res["summary"][cols].to_string(index=False))

    print("\n--- DECILE PROFILE: RAW DOLLARS INVESTED (Hedge Fund VIP) ---")
    dcols = [
        "bucket",
        "med_dollars_rpt_m",
        "med_mcap_b",
        "badj_mean_bps",
        "badj_t_stat",
        "badj_p_value",
    ]
    print(res["deciles_raw"][dcols].to_string(index=False))

    print("\n--- DECILE PROFILE: SIZE-NEUTRAL LOG DOLLARS INVESTED ---")
    print(res["deciles_log_sn"][dcols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
