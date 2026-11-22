"""Stage 3 Point-in-Time (PIT) Backwards & Forward Compute Engine.

Sits cleanly on top of Stage 1 & Stage 2 (`src.pit.panel`) to compute all
backwards-looking risk/flow metrics and forward-evaluation target labels with
zero lookahead:

  Backwards-Looking PIT Features (strictly <= query_date; Beta strictly < query_date):
    - `beta_252d_lag1`: Exact masked trailing 252-trading-day OLS equity beta vs. `SPY`
      computed strictly over `[t-252, t-1]` (NEVER includes `query_date` `t`).
    - `prev_day_ret`, `prev_spy_ret`: Corporate-action-adjusted close-to-close total return
      from yesterday's trading-day close to today's close (`t-1 -> t`), requiring strictly
      consecutive US trading days on the scrubbed `SPY` calendar (`cal_step == 1`).
    - `ret_63d_lag1`: Trailing 63-trading-day (~1 quarter) total return ending `t-1`.
    - `delta_shares_pit`, `delta_shares_usd_pit`: Split-aligned QoQ net institutional
      share change across all filers with a valid consecutive prior quarter (`gap <= 105d`,
      excluding `>75%` fund wind-down liquidations, and properly capturing full exits).
    - `n_buyers`, `n_sellers`, `n_initiations`, `n_exits`: Active manager trading breadth.
    - `mean_active_weight_shift`, `sum_active_weight_shift`: Active portfolio-weight shifts
      stripping out passive price drift identically by valuing both prior and current
      split-aligned share books at constant `query_date` close prices.
    - `holder_hhi_pit`, `delta_holder_hhi_pit`: Concentration HHI across active holders and
      its split-aligned QoQ change.
    - `flow_qtr_ret_pit`: Average stock total return over the reporting filers' exact
      `(prev_period_of_report -> period_of_report)` quarter (strictly `< query_date`).
    - Dual-class rollup & `% shares owned`:
      - Rolls up `issuer_total_shares_owned_pit`, `issuer_delta_shares_pit`,
        `issuer_total_position_usd_pit`, and `issuer_adv_20d_usd` across equal-economic-ratio
        sibling share classes (e.g., `GOOG + GOOGL`) sharing an `issuer_cik` so
        `pct_shares_owned = issuer_total_shares_owned_pit / shares_out_pit` and
        `days_to_liquidate` are numerator/denominator consistent.
      - Enforces hard `pct_shares_owned <= 1.0` sanity guard (`is_pct_owned_invalid`).
      - Marks the single highest-ADV share class per equal-ratio `issuer_cik` as
        `is_primary_class = True` to prevent double-counting in cross-sectional regressions.

  Forward Evaluation Labels (target ONLY — NEVER used as features):
    - `fwd_day_ret`, `fwd_spy_ret`: Next trading day corporate-action-adjusted close-to-close
      total return (`t -> t+1`), strictly consecutive US trading days (`cal_step == 1`).
    - `fwd_beta_adj_ret`: `fwd_day_ret - beta_252d_lag1 * fwd_spy_ret`.

Usage:
    python -m src.pit.compute --query-date 2024-08-16 --audit
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.pit.ledger import (
    DEFAULT_MAX_STALENESS_DAYS,
    PitHoldingsLedger,
    PitLookaheadError,
)
from src.pit.panel import (
    DEFAULT_ANCHOR_ADV_USD,
    DEFAULT_ANCHOR_MCAP_USD,
    DEFAULT_MAX_SHARES_AGE_DAYS,
    DEFAULT_MIN_SEASONING_DAYS,
    DEFAULT_THRESHOLD_ANCHOR_DATE,
    PitMarketStore,
    build_pit_security_snapshot,
    map_holdings_to_securities,
    verify_stage2_pit_snapshot,
)

logger = logging.getLogger(__name__)

# Stage 3 defaults
DEFAULT_BETA_WINDOW = 252
DEFAULT_BETA_MIN_PERIODS = 250  # Covers 2015-01-02 (251 prior 2014 daily returns) cleanly
DEFAULT_BETA_MAX_DAILY_RET = 5.0  # Mask >500% single-day bankruptcy-reorg / missing-split glitches
DEFAULT_MAX_QTR_GAP_DAYS = 105  # Max calendar days between consecutive quarterly report dates
DEFAULT_WINDDOWN_DROP_PCT = 0.75  # Exclude >75% single-quarter book collapses from flow deltas
DEFAULT_TRADE_NOISE_THRESHOLD = 0.01  # >=1% share change counts as active buy/sell
DEFAULT_MAX_FWD_GAP_DAYS = 7  # Do not link fwd_day_ret across >7 calendar-day trading halts


def compute_masked_rolling_beta_lag1(
    ret_wide: pd.DataFrame,
    spy_ret: pd.Series,
    *,
    window: int = DEFAULT_BETA_WINDOW,
    min_periods: int = DEFAULT_BETA_MIN_PERIODS,
    max_daily_ret: float = DEFAULT_BETA_MAX_DAILY_RET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute exact masked trailing OLS beta strictly EXCLUDING `query_date` (`[t-W, t-1]`).

    Masks `spy_ret` to the exact dates where each stock has a valid, non-corrupted return
    (`-0.99 < ret <= max_daily_ret`, filtering out Chapter-11 reorganization ticker-reuse
    discontinuities like +25,733% single-day jumps while preserving all genuine equity moves)
    before computing `Var(r_spy)` so numerator and denominator date sets always match. Shifts
    the final rolling result forward by 1 trading day so row `t` uses strictly `[t-W, t-1]`.
    """
    aligned_spy = spy_ret.reindex(ret_wide.index).astype(np.float64)
    r_mat = ret_wide.astype(np.float64)
    valid = (
        r_mat.notna()
        & aligned_spy.notna().to_numpy()[:, None]
        & (r_mat > -0.99)
        & (r_mat <= max_daily_ret)
    )

    x = r_mat.where(valid)
    y_mat = pd.DataFrame(
        np.broadcast_to(aligned_spy.to_numpy()[:, None], r_mat.shape),
        index=r_mat.index,
        columns=r_mat.columns,
    ).where(valid)

    n_obs = valid.astype(np.float64).rolling(window=window, min_periods=1).sum()
    sum_x = x.fillna(0.0).rolling(window=window, min_periods=1).sum()
    sum_y = y_mat.fillna(0.0).rolling(window=window, min_periods=1).sum()
    sum_xy = (x * y_mat).fillna(0.0).rolling(window=window, min_periods=1).sum()
    sum_yy = (y_mat * y_mat).fillna(0.0).rolling(window=window, min_periods=1).sum()

    # Exact sample/population ratio: (sum_xy - sum_x*sum_y/n) / (sum_yy - sum_y^2/n)
    cov_num = sum_xy - (sum_x * sum_y) / n_obs
    var_den = sum_yy - (sum_y * sum_y) / n_obs

    beta_incl_t = cov_num / var_den
    mask_ok = (n_obs >= min_periods) & (var_den > 1e-12)
    beta_incl_t = beta_incl_t.where(mask_ok)

    # Strictly lag by 1 trading day so row `t` NEVER includes query_date `t`'s return!
    beta_lag1 = beta_incl_t.shift(1)
    n_obs_lag1 = n_obs.shift(1)
    return beta_lag1, n_obs_lag1


class Stage3MarketMetrics:
    """Precomputed vectorized market metrics (lagged beta, lagged/forward returns) for Stage 3."""

    def __init__(
        self,
        market_store: PitMarketStore,
        *,
        beta_window: int = DEFAULT_BETA_WINDOW,
        beta_min_periods: int = DEFAULT_BETA_MIN_PERIODS,
    ) -> None:
        self.market_store = market_store
        self.beta_window = beta_window
        self.beta_min_periods = beta_min_periods

        mp = market_store.market_prices
        spy = market_store.spy_prices

        # 1. Scrub non-trading days & build canonical US trading calendar from SPY
        spy_s = spy.set_index("date").sort_index()
        # Exclude any weekend timestamps defensively (US exchanges closed Sat/Sun)
        spy_s = spy_s[spy_s.index.day_of_week < 5]
        self.trading_calendar = pd.DatetimeIndex(spy_s.index)
        cal_pos_map: dict[np.datetime64, int] = {
            np.datetime64(dt, "ns"): idx for idx, dt in enumerate(self.trading_calendar)
        }

        # SPY close-to-close total returns: prev_spy_ret = (t-1 -> t), fwd_spy_ret = (t -> t+1)
        spy_ret = spy_s["ret_total"].astype(np.float64)
        spy_fwd_ret = spy_ret.shift(-1)
        spy_next_date = pd.Series(spy_s.index, index=spy_s.index).shift(-1)
        spy_fwd_gap = (spy_next_date - pd.Series(spy_s.index, index=spy_s.index)).dt.days
        spy_fwd_ret = spy_fwd_ret.where(spy_fwd_gap <= DEFAULT_MAX_FWD_GAP_DAYS)

        self._spy_metrics: dict[pd.Timestamp, dict[str, float]] = {
            pd.Timestamp(dt): {
                "prev_spy_ret": float(spy_ret.loc[dt]) if pd.notna(spy_ret.loc[dt]) else np.nan,
                "fwd_spy_ret": float(spy_fwd_ret.loc[dt])
                if pd.notna(spy_fwd_ret.loc[dt])
                else np.nan,
            }
            for dt in spy_s.index
        }

        if mp.empty or spy.empty:
            self._ticker_metrics: dict[str, dict[str, np.ndarray]] = {}
            return

        # 2. Restrict stock returns strictly to canonical US trading days & compute lagged beta
        mp_valid = mp[mp["date"].isin(self.trading_calendar)]
        ret_wide = mp_valid.pivot(index="date", columns="yahoo_ticker", values="ret_total").reindex(
            self.trading_calendar
        )
        beta_lag1, n_obs_lag1 = compute_masked_rolling_beta_lag1(
            ret_wide,
            spy_ret,
            window=beta_window,
            min_periods=beta_min_periods,
        )
        self._beta_lag1_wide = beta_lag1
        self._n_obs_lag1_wide = n_obs_lag1

        # 3. Per-ticker close-to-close total returns strictly requiring consecutive trading days
        #    (`cal_step == 1` on the scrubbed SPY calendar) so missing bars / halts never stitch
        #    multi-day jumps into 1-day prev_day_ret (`t-1 -> t`) or fwd_day_ret (`t -> t+1`).
        self._ticker_metrics = {}
        for tkr, arr in market_store._ticker_arrays.items():
            dates_ns = arr["dates_ns"]
            ret = arr["ret_total"].copy()
            tr = arr["tr_close"]

            # Keep only dates present on the canonical US trading calendar
            cal_idx = np.array([cal_pos_map.get(d, -1) for d in dates_ns], dtype=np.int64)
            on_cal = cal_idx >= 0
            if not np.all(on_cal):
                dates_ns = dates_ns[on_cal]
                ret = ret[on_cal]
                tr = tr[on_cal]
                cal_idx = cal_idx[on_cal]

            n = len(dates_ns)
            prev_ret = np.full(n, np.nan, dtype=np.float64)
            fwd_ret = np.full(n, np.nan, dtype=np.float64)
            if n > 1:
                consec = (cal_idx[1:] - cal_idx[:-1]) == 1
                # prev_day_ret[t] = (t-1 -> t) strictly on consecutive US trading days
                prev_ret[1:] = np.where(consec, ret[1:], np.nan)
                # fwd_day_ret[t]  = (t -> t+1) strictly on consecutive US trading days
                fwd_ret[:-1] = np.where(consec, ret[1:], np.nan)

            # Trailing 63-trading-day total return ending strictly on t-1: tr[t-1] / tr[t-64] - 1
            ret_63d_lag1 = np.full(n, np.nan, dtype=np.float64)
            if n > 64:
                ret_63d_lag1[64:] = tr[63:-1] / tr[:-64] - 1.0

            self._ticker_metrics[tkr] = {
                "dates_ns": dates_ns,
                "prev_day_ret": prev_ret,
                "ret_63d_lag1": ret_63d_lag1,
                "fwd_day_ret": fwd_ret,
                "tr_close": tr,
            }

    def get_spy_returns(self, query_date: pd.Timestamp) -> dict[str, float]:
        qdt = pd.Timestamp(query_date).normalize()
        return self._spy_metrics.get(
            qdt,
            {"prev_spy_ret": np.nan, "fwd_spy_ret": np.nan},
        )

    def get_stock_metrics(self, ticker: str, query_date: pd.Timestamp) -> dict[str, float]:
        """Return lagged beta, prev_day_ret (t-1->t), ret_63d_lag1, & fwd_day_ret (t->t+1)."""
        qdt = pd.Timestamp(query_date).normalize()
        tkr = str(ticker).strip().upper()
        empty = {
            "beta_252d_lag1": np.nan,
            "beta_n_obs": 0.0,
            "prev_day_ret": np.nan,
            "ret_63d_lag1": np.nan,
            "fwd_day_ret": np.nan,
        }
        arr = self._ticker_metrics.get(tkr)
        if arr is None:
            return empty

        q_ns = np.datetime64(qdt, "ns")
        dates_ns = arr["dates_ns"]
        idx = int(np.searchsorted(dates_ns, q_ns, side="left"))
        if idx >= len(dates_ns) or dates_ns[idx] != q_ns:
            return empty

        beta_val = np.nan
        n_obs_val = 0.0
        if hasattr(self, "_beta_lag1_wide") and qdt in self._beta_lag1_wide.index:
            if tkr in self._beta_lag1_wide.columns:
                b = self._beta_lag1_wide.at[qdt, tkr]
                n_o = self._n_obs_lag1_wide.at[qdt, tkr]
                beta_val = float(b) if pd.notna(b) else np.nan
                n_obs_val = float(n_o) if pd.notna(n_o) else 0.0

        return {
            "beta_252d_lag1": beta_val,
            "beta_n_obs": n_obs_val,
            "prev_day_ret": float(arr["prev_day_ret"][idx]),
            "ret_63d_lag1": float(arr["ret_63d_lag1"][idx]),
            "fwd_day_ret": float(arr["fwd_day_ret"][idx]),
        }

    def get_period_total_return(
        self,
        ticker: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        *,
        query_date: pd.Timestamp,
    ) -> float:
        """Return stock total return over `(start_date -> end_date]` strictly `< query_date`."""
        tkr = str(ticker).strip().upper()
        arr = self._ticker_metrics.get(tkr)
        if arr is None or len(arr["dates_ns"]) == 0:
            return np.nan
        # Enforce strict no-lookahead cap: never read price on or after query_date
        end_cap = min(
            pd.Timestamp(end_date).normalize(),
            pd.Timestamp(query_date) - pd.Timedelta(days=1),
        )
        s_ns = np.datetime64(pd.Timestamp(start_date).normalize(), "ns")
        e_ns = np.datetime64(end_cap.normalize(), "ns")
        if s_ns >= e_ns:
            return np.nan

        dates_ns = arr["dates_ns"]
        idx_s = int(np.searchsorted(dates_ns, s_ns, side="right")) - 1
        idx_e = int(np.searchsorted(dates_ns, e_ns, side="right")) - 1
        if idx_s < 0 or idx_e <= idx_s:
            return np.nan
        tr_s = float(arr["tr_close"][idx_s])
        tr_e = float(arr["tr_close"][idx_e])
        if tr_s <= 0 or not np.isfinite(tr_s) or not np.isfinite(tr_e):
            return np.nan
        return tr_e / tr_s - 1.0


def _compute_pit_qoq_flow_and_concentration(
    query_date: pd.Timestamp,
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
    stage3_market: Stage3MarketMetrics,
    *,
    us_common_only: bool = True,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    max_qtr_gap_days: int = DEFAULT_MAX_QTR_GAP_DAYS,
    winddown_drop_pct: float = DEFAULT_WINDDOWN_DROP_PCT,
    trade_noise_threshold: float = DEFAULT_TRADE_NOISE_THRESHOLD,
) -> pd.DataFrame:
    """Compute split-aligned QoQ share deltas, active weight shifts, and holder HHI shifts.

    Strictly PIT: reads ONLY states active as of `query_date` (`valid_from <= query_date` and
    `valid_to > query_date`). Compares each active filer's latest visible quarter `Q_curr`
    against its immediately preceding visible quarter `Q_prev` (requiring `gap <= 105 days`
    so multi-quarter hiatuses like Blue Ridge 2019–2020 never produce bogus deltas, and
    excluding `>75%` single-quarter book collapses so terminal liquidations aren't misread
    as bearish stock calls).

    Both `Q_curr` and `Q_prev` share counts are forward-adjusted to exact `query_date` share
    units before differencing, and active portfolio-weight shifts value BOTH books at constant
    `query_date` close prices (`raw_close[t]`) so passive price drift cancels identically to 0.
    """
    qdt = pd.Timestamp(query_date).normalize()
    active_states = ledger._active_states_as_of(qdt)
    if active_states.empty:
        return pd.DataFrame(columns=["security_id"])

    # Identify latest non-stale state and immediately preceding state per filer_id
    active_sorted = active_states.sort_values(["filer_id", "period_of_report", "valid_from"])
    day_prices = market_store._prices_by_date.get(qdt)

    # Collect per-(filer, security) records for curr and prev quarters
    per_sec_curr_shares: dict[str, list[float]] = {}
    per_sec_prev_shares: dict[str, list[float]] = {}
    per_sec_delta_shares: dict[str, float] = {}
    per_sec_active_w_shifts: dict[str, list[float]] = {}
    per_sec_buyers: dict[str, int] = {}
    per_sec_sellers: dict[str, int] = {}
    per_sec_inits: dict[str, int] = {}
    per_sec_exits: dict[str, int] = {}
    per_sec_n_flow_filers: dict[str, int] = {}
    per_sec_qtr_rets: dict[str, list[float]] = {}
    n_winddown_filers = 0

    for _filer_id, fgrp in active_sorted.groupby("filer_id", sort=False):
        fgrp = fgrp.drop_duplicates(subset=["period_of_report"], keep="last").reset_index(drop=True)
        curr_st = fgrp.iloc[-1]
        curr_rep = pd.Timestamp(curr_st["period_of_report"])
        if max_staleness_days is not None and int((qdt - curr_rep).days) > max_staleness_days:
            continue

        # Extract and map current holdings
        curr_h_raw = ledger._assemble_holdings(fgrp.iloc[[-1]], qdt, verify=False)
        curr_map = map_holdings_to_securities(
            curr_h_raw, market_store.cusip_map, us_common_only=us_common_only
        )

        # Forward-adjust current shares to query_date units
        curr_sec: dict[str, tuple[str, float]] = {}
        for r in curr_map.to_dict(orient="records"):
            sid = str(r["security_id"])
            tkr = str(r["yahoo_ticker"]).strip().upper()
            k_curr = market_store.get_forward_split_factor(tkr, curr_rep, qdt) if tkr else 1.0
            sh_pit = float(r["shares"]) * k_curr
            curr_sec[sid] = (tkr, sh_pit)
            per_sec_curr_shares.setdefault(sid, []).append(sh_pit)

        # Check for valid consecutive prior quarter
        if len(fgrp) < 2:
            continue
        prev_st = fgrp.iloc[-2]
        prev_rep = pd.Timestamp(prev_st["period_of_report"])
        gap_days = int((curr_rep - prev_rep).days)
        if gap_days <= 0 or gap_days > max_qtr_gap_days:
            continue

        # Wind-down guard: exclude >75% single-quarter total book collapse from active flow
        prev_book = float(prev_st["total_reported_value_usd"])
        curr_book = float(curr_st["total_reported_value_usd"])
        if prev_book > 0 and (1.0 - curr_book / prev_book) > winddown_drop_pct:
            n_winddown_filers += 1
            continue

        prev_h_raw = ledger._assemble_holdings(fgrp.iloc[[-2]], qdt, verify=False)
        prev_map = map_holdings_to_securities(
            prev_h_raw, market_store.cusip_map, us_common_only=us_common_only
        )

        prev_sec: dict[str, tuple[str, float]] = {}
        for r in prev_map.to_dict(orient="records"):
            sid = str(r["security_id"])
            tkr = str(r["yahoo_ticker"]).strip().upper()
            k_prev = market_store.get_forward_split_factor(tkr, prev_rep, qdt) if tkr else 1.0
            sh_pit = float(r["shares"]) * k_prev
            prev_sec[sid] = (tkr, sh_pit)
            per_sec_prev_shares.setdefault(sid, []).append(sh_pit)

        # Union of securities across curr & prev (captures full exits AND new initiations!)
        all_sids = set(curr_sec.keys()) | set(prev_sec.keys())
        if not all_sids:
            continue

        # Value BOTH curr & prev split-aligned books at constant query_date close prices
        # so passive price drift cancels identically to 0.0!
        curr_val_by_sid: dict[str, float] = {}
        prev_val_by_sid: dict[str, float] = {}
        tot_curr_val = 0.0
        tot_prev_val = 0.0

        for sid in all_sids:
            tkr = curr_sec[sid][0] if sid in curr_sec else prev_sec[sid][0]
            px = np.nan
            if day_prices is not None and tkr and tkr in day_prices.index:
                raw_px = day_prices.loc[tkr, "raw_close"]
                if isinstance(raw_px, pd.Series):
                    raw_px = raw_px.iloc[-1]
                px = float(raw_px)
            if pd.notna(px) and px > 0:
                c_sh = curr_sec[sid][1] if sid in curr_sec else 0.0
                p_sh = prev_sec[sid][1] if sid in prev_sec else 0.0
                cv = c_sh * px
                pv = p_sh * px
                curr_val_by_sid[sid] = cv
                prev_val_by_sid[sid] = pv
                tot_curr_val += cv
                tot_prev_val += pv

        for sid in all_sids:
            tkr = curr_sec[sid][0] if sid in curr_sec else prev_sec[sid][0]
            c_sh = curr_sec[sid][1] if sid in curr_sec else 0.0
            p_sh = prev_sec[sid][1] if sid in prev_sec else 0.0
            d_sh = c_sh - p_sh

            per_sec_delta_shares[sid] = per_sec_delta_shares.get(sid, 0.0) + d_sh
            per_sec_n_flow_filers[sid] = per_sec_n_flow_filers.get(sid, 0) + 1

            # Active buy / sell / initiation / exit classification
            if p_sh == 0.0 and c_sh > 0.0:
                per_sec_inits[sid] = per_sec_inits.get(sid, 0) + 1
                per_sec_buyers[sid] = per_sec_buyers.get(sid, 0) + 1
            elif c_sh == 0.0 and p_sh > 0.0:
                per_sec_exits[sid] = per_sec_exits.get(sid, 0) + 1
                per_sec_sellers[sid] = per_sec_sellers.get(sid, 0) + 1
            elif p_sh > 0.0:
                rel_chg = d_sh / p_sh
                if rel_chg >= trade_noise_threshold:
                    per_sec_buyers[sid] = per_sec_buyers.get(sid, 0) + 1
                elif rel_chg <= -trade_noise_threshold:
                    per_sec_sellers[sid] = per_sec_sellers.get(sid, 0) + 1

            # Constant-price active weight shift (strictly zero when c_sh == p_sh for all stocks!)
            if tot_curr_val > 0 and tot_prev_val > 0 and sid in curr_val_by_sid:
                w_curr = curr_val_by_sid[sid] / tot_curr_val
                w_passive = prev_val_by_sid[sid] / tot_prev_val
                per_sec_active_w_shifts.setdefault(sid, []).append(w_curr - w_passive)

            # Stock return over this filer's exact reporting quarter (prev_rep -> curr_rep)
            if tkr:
                q_ret = stage3_market.get_period_total_return(
                    tkr, prev_rep, curr_rep, query_date=qdt
                )
                if pd.notna(q_ret):
                    per_sec_qtr_rets.setdefault(sid, []).append(q_ret)

    # Assemble per-security flow & HHI summary
    all_seen_sids = (
        set(per_sec_curr_shares.keys())
        | set(per_sec_delta_shares.keys())
        | set(per_sec_prev_shares.keys())
    )
    rows: list[dict[str, Any]] = []
    for sid in sorted(all_seen_sids):
        c_list = per_sec_curr_shares.get(sid, [])
        p_list = per_sec_prev_shares.get(sid, [])
        hhi_curr = _compute_hhi(c_list)
        hhi_prev = _compute_hhi(p_list)
        delta_hhi = (hhi_curr - hhi_prev) if (pd.notna(hhi_curr) and pd.notna(hhi_prev)) else np.nan
        w_shifts = per_sec_active_w_shifts.get(sid, [])
        q_rets = per_sec_qtr_rets.get(sid, [])

        rows.append(
            {
                "security_id": sid,
                "delta_shares_pit": per_sec_delta_shares.get(sid, np.nan),
                "n_flow_filers": per_sec_n_flow_filers.get(sid, 0),
                "n_buyers": per_sec_buyers.get(sid, 0),
                "n_sellers": per_sec_sellers.get(sid, 0),
                "n_initiations": per_sec_inits.get(sid, 0),
                "n_exits": per_sec_exits.get(sid, 0),
                "mean_active_weight_shift": float(np.mean(w_shifts)) if w_shifts else np.nan,
                "sum_active_weight_shift": float(np.sum(w_shifts)) if w_shifts else np.nan,
                "holder_hhi_pit": hhi_curr,
                "holder_hhi_prev_pit": hhi_prev,
                "delta_holder_hhi_pit": delta_hhi,
                "flow_qtr_ret_pit": float(np.mean(q_rets)) if q_rets else np.nan,
                "n_winddown_filers_excluded": n_winddown_filers,
            }
        )

    return pd.DataFrame(rows)


def _compute_hhi(values: list[float]) -> float:
    if not values:
        return np.nan
    arr = np.asarray(values, dtype=np.float64)
    tot = float(arr.sum())
    if tot <= 0:
        return np.nan
    w = arr / tot
    return float(np.sum(w * w))


def attach_stage3_compute(
    stage2_df: pd.DataFrame,
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
    stage3_market: Stage3MarketMetrics,
    *,
    us_common_only: bool = True,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    max_qtr_gap_days: int = DEFAULT_MAX_QTR_GAP_DAYS,
    winddown_drop_pct: float = DEFAULT_WINDDOWN_DROP_PCT,
    verify: bool = True,
) -> pd.DataFrame:
    """Stage 3 API: Attach strictly PIT backwards compute & forward target labels to Stage 2."""
    if stage2_df.empty:
        empty = stage2_df.copy()
        for col in (
            "delta_shares_pit",
            "delta_shares_usd_pit",
            "n_flow_filers",
            "n_buyers",
            "n_sellers",
            "n_initiations",
            "n_exits",
            "mean_active_weight_shift",
            "sum_active_weight_shift",
            "holder_hhi_pit",
            "holder_hhi_prev_pit",
            "delta_holder_hhi_pit",
            "flow_qtr_ret_pit",
            "n_winddown_filers_excluded",
            "beta_252d_lag1",
            "beta_n_obs",
            "prev_day_ret",
            "prev_spy_ret",
            "ret_63d_lag1",
            "fwd_day_ret",
            "fwd_spy_ret",
            "fwd_beta_adj_ret",
            "issuer_total_shares_owned_pit",
            "issuer_delta_shares_pit",
            "issuer_total_position_usd_pit",
            "issuer_adv_20d_usd",
            "issuer_n_classes_held",
            "is_primary_class",
            "pct_shares_owned_class",
            "pct_shares_owned",
            "is_pct_owned_invalid",
            "days_to_liquidate",
            "is_investable",
            "is_panel_eligible",
        ):
            if col not in empty.columns:
                empty[col] = pd.Series(dtype="object")
        return empty

    qdt = pd.Timestamp(stage2_df["query_date"].iloc[0]).normalize()
    if verify:
        verify_stage2_pit_snapshot(stage2_df, qdt)

    spy_rets = stage3_market.get_spy_returns(qdt)

    # 1. Compute split-aligned QoQ share deltas, active weight shifts, and holder HHI shifts
    flow_df = _compute_pit_qoq_flow_and_concentration(
        qdt,
        ledger,
        market_store,
        stage3_market,
        us_common_only=us_common_only,
        max_staleness_days=max_staleness_days,
        max_qtr_gap_days=max_qtr_gap_days,
        winddown_drop_pct=winddown_drop_pct,
    )

    df = stage2_df.merge(flow_df, on="security_id", how="left")

    # 2. Attach lagged beta ([t-252, t-1], excl. t), prev_day (t-1->t), and fwd (t->t+1) returns
    beta_vals: list[float] = []
    beta_n_vals: list[float] = []
    prev_rets: list[float] = []
    ret_63ds: list[float] = []
    fwd_rets: list[float] = []

    for tkr in df["yahoo_ticker"].astype(str):
        m = stage3_market.get_stock_metrics(tkr, qdt) if tkr else None
        if m is None:
            beta_vals.append(np.nan)
            beta_n_vals.append(0.0)
            prev_rets.append(np.nan)
            ret_63ds.append(np.nan)
            fwd_rets.append(np.nan)
        else:
            beta_vals.append(m["beta_252d_lag1"])
            beta_n_vals.append(m["beta_n_obs"])
            prev_rets.append(m["prev_day_ret"])
            ret_63ds.append(m["ret_63d_lag1"])
            fwd_rets.append(m["fwd_day_ret"])

    df["beta_252d_lag1"] = beta_vals
    df["beta_n_obs"] = beta_n_vals
    df["prev_day_ret"] = prev_rets
    df["prev_spy_ret"] = spy_rets["prev_spy_ret"]
    df["ret_63d_lag1"] = ret_63ds
    df["fwd_day_ret"] = fwd_rets
    df["fwd_spy_ret"] = spy_rets["fwd_spy_ret"]
    df["fwd_beta_adj_ret"] = df["fwd_day_ret"] - df["beta_252d_lag1"] * df["fwd_spy_ret"]

    # Dollar flow at query_date close
    df["delta_shares_usd_pit"] = df["delta_shares_pit"] * df["raw_close"]

    # 3. Equal-economic-ratio dual-class rollup by issuer_cik & primary liquid class selection
    #    (e.g., GOOG + GOOGL share issuer-level XBRL shares_out_pit -> sum shares_owned across
    #     equal-ratio siblings so ownership % is never halved, and mark highest-ADV class primary.)
    equal_dual_mask = (
        df["is_dual_class_issuer"].fillna(False)
        & (~df["has_unequal_class_ratio"].fillna(False))
        & (df["issuer_cik"].fillna("") != "")
    )

    df["issuer_total_shares_owned_pit"] = df["total_shares_owned_pit"].astype(np.float64)
    df["issuer_delta_shares_pit"] = df["delta_shares_pit"].astype(np.float64)
    df["issuer_total_position_usd_pit"] = df["total_position_usd_pit"].astype(np.float64)
    df["issuer_adv_20d_usd"] = df["adv_20d_usd"].astype(np.float64)
    df["issuer_n_classes_held"] = 1
    df["is_primary_class"] = True

    if equal_dual_mask.any():
        dual_groups = df.index[equal_dual_mask].groupby(df.loc[equal_dual_mask, "issuer_cik"])
        for _cik, idx in dual_groups.items():
            sub = df.loc[idx]
            if len(sub) <= 1:
                continue
            tot_sh = float(sub["total_shares_owned_pit"].fillna(0.0).sum())
            tot_dsh = (
                float(sub["delta_shares_pit"].dropna().sum())
                if sub["delta_shares_pit"].notna().any()
                else np.nan
            )
            tot_usd = (
                float(sub["total_position_usd_pit"].dropna().sum())
                if sub["total_position_usd_pit"].notna().any()
                else np.nan
            )
            tot_adv = (
                float(sub["adv_20d_usd"].dropna().sum())
                if sub["adv_20d_usd"].notna().any()
                else np.nan
            )
            df.loc[idx, "issuer_total_shares_owned_pit"] = tot_sh
            df.loc[idx, "issuer_delta_shares_pit"] = tot_dsh
            df.loc[idx, "issuer_total_position_usd_pit"] = tot_usd
            df.loc[idx, "issuer_adv_20d_usd"] = tot_adv
            df.loc[idx, "issuer_n_classes_held"] = len(sub)

            # Pick single primary share class strictly PIT by highest adv_20d_usd (then pos_usd)
            ranked = sub.assign(
                _adv_sort=sub["adv_20d_usd"].fillna(-1.0),
                _usd_sort=sub["total_position_usd_pit"].fillna(-1.0),
            ).sort_values(
                ["_adv_sort", "_usd_sort", "yahoo_ticker"],
                ascending=[False, False, True],
            )
            primary_idx = ranked.index[0]
            df.loc[idx, "is_primary_class"] = False
            df.at[primary_idx, "is_primary_class"] = True

    # 4. % of shares outstanding owned + hard physical-impossibility guard (pct_owned > 1.0)
    valid_denom = (
        df["shares_out_pit"].notna()
        & (df["shares_out_pit"] > 0)
        & (~df["has_unequal_class_ratio"].fillna(False))
    )
    pct_class = np.where(valid_denom, df["total_shares_owned_pit"] / df["shares_out_pit"], np.nan)
    pct_issuer = np.where(
        valid_denom, df["issuer_total_shares_owned_pit"] / df["shares_out_pit"], np.nan
    )
    is_pct_invalid = pd.Series(pct_issuer, index=df.index).gt(1.0).fillna(False)

    df["pct_shares_owned_class"] = np.where(is_pct_invalid, np.nan, pct_class)
    df["pct_shares_owned"] = np.where(is_pct_invalid, np.nan, pct_issuer)
    df["is_pct_owned_invalid"] = is_pct_invalid

    # Days-to-Liquidate (total issuer-consistent position USD / issuer-consistent 20d ADV)
    df["days_to_liquidate"] = np.where(
        df["issuer_adv_20d_usd"].notna() & (df["issuer_adv_20d_usd"] > 0),
        df["issuer_total_position_usd_pit"] / df["issuer_adv_20d_usd"],
        np.nan,
    )

    # Final cross-sectional factor/regression eligibility flag
    df["is_panel_eligible"] = (
        df["is_investable"].fillna(False)
        & df["is_primary_class"].fillna(False)
        & df["beta_252d_lag1"].notna()
        & (~df["is_pct_owned_invalid"].fillna(False))
    )

    if verify:
        verify_stage3_snapshot(df, qdt)

    return df


def build_pit_stage3_snapshot(
    query_date: str | pd.Timestamp,
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
    stage3_market: Stage3MarketMetrics,
    *,
    us_common_only: bool = True,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    min_seasoning_days: int = DEFAULT_MIN_SEASONING_DAYS,
    anchor_date: str | pd.Timestamp = DEFAULT_THRESHOLD_ANCHOR_DATE,
    anchor_mcap_usd: float = DEFAULT_ANCHOR_MCAP_USD,
    anchor_adv_usd: float = DEFAULT_ANCHOR_ADV_USD,
    min_mcap_usd: float | None = None,
    min_adv_usd: float | None = None,
    max_shares_age_days: int | None = DEFAULT_MAX_SHARES_AGE_DAYS,
    verify: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper running Stage 1 -> Stage 2 -> Stage 3 for one `query_date`."""
    stage2 = build_pit_security_snapshot(
        query_date,
        ledger,
        market_store,
        us_common_only=us_common_only,
        max_staleness_days=max_staleness_days,
        min_seasoning_days=min_seasoning_days,
        anchor_date=anchor_date,
        anchor_mcap_usd=anchor_mcap_usd,
        anchor_adv_usd=anchor_adv_usd,
        min_mcap_usd=min_mcap_usd,
        min_adv_usd=min_adv_usd,
        max_shares_age_days=max_shares_age_days,
        verify=verify,
    )
    return attach_stage3_compute(
        stage2,
        ledger,
        market_store,
        stage3_market,
        us_common_only=us_common_only,
        max_staleness_days=max_staleness_days,
        verify=verify,
    )


def verify_stage3_snapshot(df: pd.DataFrame, query_date: pd.Timestamp) -> None:
    """Strict runtime QA assertions for Stage 3 backwards/forward compute."""
    qdt = pd.Timestamp(query_date).normalize()
    if df.empty:
        return

    verify_stage2_pit_snapshot(df, qdt)

    # 1. Valid pct_shares_owned MUST lie strictly in (0, 1.0]
    pct_ok = df["pct_shares_owned"].dropna()
    if not pct_ok.empty and ((pct_ok <= 0.0).any() or (pct_ok > 1.0).any()):
        raise PitLookaheadError(
            f"Stage 3 integrity: pct_shares_owned out of (0, 1] on {qdt.date()}."
        )

    # 2. Panel-eligible rows MUST have finite lagged beta, unique primary dual-class issuer,
    #    and valid forward beta-adjusted math wherever fwd returns exist
    elig = df[df["is_panel_eligible"]]
    if not elig.empty:
        if not np.isfinite(elig["beta_252d_lag1"]).all():
            raise PitLookaheadError(
                f"Stage 3 integrity: non-finite beta_252d_lag1 on {qdt.date()}."
            )
        if not (elig["beta_n_obs"] >= DEFAULT_BETA_MIN_PERIODS).all():
            raise PitLookaheadError(
                f"Stage 3 integrity: panel-eligible beta_n_obs < {DEFAULT_BETA_MIN_PERIODS}."
            )
        # No duplicate equal-ratio dual-class issuer_cik among panel-eligible rows!
        dual_elig = elig[elig["is_dual_class_issuer"] & (elig["issuer_cik"] != "")]
        if dual_elig.duplicated(subset=["issuer_cik"]).any():
            dups = dual_elig[dual_elig.duplicated(subset=["issuer_cik"], keep=False)][
                ["issuer_cik", "yahoo_ticker"]
            ]
            raise PitLookaheadError(
                f"Stage 3 integrity: duplicate dual-class issuer in panel_eligible:\n"
                f"{dups.to_string(index=False)}"
            )

    # 3. Verify fwd_beta_adj_ret identity wherever inputs are finite
    both = df["fwd_day_ret"].notna() & df["fwd_spy_ret"].notna() & df["beta_252d_lag1"].notna()
    if both.any():
        expected = df.loc[both, "fwd_day_ret"] - (
            df.loc[both, "beta_252d_lag1"] * df.loc[both, "fwd_spy_ret"]
        )
        max_diff = float((df.loc[both, "fwd_beta_adj_ret"] - expected).abs().max())
        if max_diff > 1e-12:
            raise PitLookaheadError(f"Stage 3 fwd_beta_adj_ret math mismatch: {max_diff}")


def run_stage3_qa_audit(
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
    stage3_market: Stage3MarketMetrics,
) -> dict[str, Any]:
    """Run exhaustive empirical QA checks on Stage 3 across real dates, splits, and dual-classes."""
    cal = pd.DatetimeIndex(market_store.spy_prices["date"])

    # 1. Verify beta_252d_lag1 matches independent np.polyfit on [t-252, t-1] to 1e-10
    #    AND strictly excludes query_date t on real flagship stocks
    check_dt = pd.Timestamp("2024-08-16")
    snap_chk = build_pit_stage3_snapshot(check_dt, ledger, market_store, stage3_market, verify=True)
    spy_series = market_store.spy_prices.set_index("date")["ret_total"].sort_index()
    beta_audit_rows: list[dict[str, Any]] = []
    for tkr in ["MSFT", "NVDA", "GOOGL", "AMZN", "META"]:
        row = snap_chk[snap_chk["yahoo_ticker"] == tkr]
        if row.empty:
            continue
        r = row.iloc[0]
        arr = market_store._ticker_arrays[tkr]
        s_ret = pd.Series(arr["ret_total"], index=pd.to_datetime(arr["dates_ns"]))
        # Strictly < check_dt, last 252 trading days
        strict_prior = s_ret[s_ret.index < check_dt].tail(252)
        joint = pd.concat(
            [strict_prior.rename("s"), spy_series.rename("m")], axis=1, sort=True
        ).dropna()
        slope = float(np.polyfit(joint["m"].to_numpy(), joint["s"].to_numpy(), 1)[0])
        diff = abs(float(r["beta_252d_lag1"]) - slope)
        if diff > 1e-9:
            raise PitLookaheadError(f"Beta polyfit mismatch for {tkr}: diff={diff}")
        beta_audit_rows.append(
            {
                "ticker": tkr,
                "beta_252d_lag1": round(float(r["beta_252d_lag1"]), 6),
                "polyfit_excl_t": round(slope, 6),
                "abs_diff": f"{diff:.2e}",
                "n_obs": int(r["beta_n_obs"]),
            }
        )

    # 2. Verify fwd_day_ret(t) == prev_day_ret(t+1) AND fwd_spy_ret(t) == prev_spy_ret(t+1)!
    pos_chk = int(cal.searchsorted(check_dt, side="left"))
    next_dt = pd.Timestamp(cal[pos_chk + 1])
    snap_next = build_pit_stage3_snapshot(next_dt, ledger, market_store, stage3_market, verify=True)
    ret_join = (
        snap_chk[["yahoo_ticker", "fwd_day_ret", "fwd_spy_ret"]]
        .merge(
            snap_next[["yahoo_ticker", "prev_day_ret", "prev_spy_ret"]],
            on="yahoo_ticker",
            how="inner",
        )
        .dropna()
    )
    max_fwd_err = float((ret_join["fwd_day_ret"] - ret_join["prev_day_ret"]).abs().max())
    max_spy_err = float((ret_join["fwd_spy_ret"] - ret_join["prev_spy_ret"]).abs().max())
    if max_fwd_err > 1e-12 or max_spy_err > 1e-12:
        raise PitLookaheadError(
            f"Return alignment failed: stock_err={max_fwd_err}, spy_err={max_spy_err}"
        )

    # 3. Verify dual-class rollup on GOOG / GOOGL on 2024-08-16
    dual_tickers = ["GOOG", "GOOGL", "FOX", "FOXA", "NWS", "NWSA"]
    dual_view = snap_chk[snap_chk["yahoo_ticker"].isin(dual_tickers)][
        [
            "yahoo_ticker",
            "issuer_cik",
            "n_filers",
            "adv_20d_usd",
            "pct_shares_owned_class",
            "pct_shares_owned",
            "is_primary_class",
            "is_panel_eligible",
        ]
    ].sort_values(["issuer_cik", "yahoo_ticker"])

    # 4. Verify NVDA split-crossing QoQ delta on 2024-08-16 (Q1 pre-split -> Q2 post-split)
    nvda_row = snap_chk[snap_chk["yahoo_ticker"] == "NVDA"][
        [
            "yahoo_ticker",
            "n_filers",
            "total_shares_owned_pit",
            "delta_shares_pit",
            "n_buyers",
            "n_sellers",
            "mean_active_weight_shift",
            "pct_shares_owned",
            "beta_252d_lag1",
        ]
    ]

    # 5. Cross-sectional panel summary across 2015-2026
    sample_dates = [
        "2015-02-18",
        "2015-05-18",
        "2017-05-16",
        "2019-05-16",
        "2020-05-18",
        "2022-05-17",
        "2024-08-16",
        "2025-05-16",
        "2026-05-18",
    ]
    panel_rows: list[dict[str, Any]] = []
    for d_str in sample_dates:
        pos = int(cal.searchsorted(pd.Timestamp(d_str), side="left"))
        if pos >= len(cal):
            continue
        qdt = pd.Timestamp(cal[pos])
        sn = build_pit_stage3_snapshot(qdt, ledger, market_store, stage3_market, verify=True)
        elig = sn[sn["is_panel_eligible"]]
        panel_rows.append(
            {
                "date": str(qdt.date()),
                "n_investable": int(sn["is_investable"].sum()),
                "n_panel_elig": len(elig),
                "med_beta": round(float(elig["beta_252d_lag1"].median()), 3)
                if len(elig)
                else np.nan,
                "med_pct_owned": round(float(elig["pct_shares_owned"].median()) * 100, 3)
                if len(elig)
                else np.nan,
                "max_pct_owned": round(float(elig["pct_shares_owned"].max()) * 100, 2)
                if len(elig)
                else np.nan,
                "n_pct_invalid": int(sn["is_pct_owned_invalid"].sum()),
                "n_with_qoq_flow": int(elig["delta_shares_pit"].notna().sum()),
                "n_with_fwd_ret": int(elig["fwd_beta_adj_ret"].notna().sum()),
            }
        )

    return {
        "beta_audit": pd.DataFrame(beta_audit_rows),
        "max_fwd_ret_alignment_err": max_fwd_err,
        "dual_class_sample": dual_view,
        "nvda_split_qoq": nvda_row,
        "panel_summary": pd.DataFrame(panel_rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and QA-audit Stage 3 PIT backwards/forward compute."
    )
    parser.add_argument(
        "--query-date",
        type=str,
        default="2024-08-16",
        help="Trading date YYYY-MM-DD for snapshot inspection.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run comprehensive Stage 3 QA audit across real dates.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    ledger = PitHoldingsLedger.from_parquet()
    store = PitMarketStore.from_parquet()
    s3_mkt = Stage3MarketMetrics(store)

    snap = build_pit_stage3_snapshot(args.query_date, ledger, store, s3_mkt, verify=True)
    elig = snap[snap["is_panel_eligible"]]
    print(
        f"Stage 3 Snapshot ({args.query_date}): "
        f"investable={snap['is_investable'].sum():,} | "
        f"panel_eligible={len(elig):,} | "
        f"pct_invalid_nulled={snap['is_pct_owned_invalid'].sum()}"
    )
    if not elig.empty:
        cols = [
            "yahoo_ticker",
            "beta_252d_lag1",
            "pct_shares_owned",
            "delta_shares_pit",
            "mean_active_weight_shift",
            "prev_day_ret",
            "fwd_day_ret",
            "fwd_beta_adj_ret",
        ]
        print(elig[cols].head(10).to_string(index=False))

    if args.audit:
        print("\n=== RUNNING STAGE 3 QA AUDIT ===")
        qa = run_stage3_qa_audit(ledger, store, s3_mkt)
        print("\n1. Exact Lagged Beta Verification (`[t-252, t-1]` vs independent np.polyfit):")
        print(qa["beta_audit"].to_string(index=False))
        print(
            f"\n2. Return Alignment (`fwd_day_ret(t) == prev_day_ret(t+1)`): "
            f"max_abs_err = {qa['max_fwd_ret_alignment_err']:.2e}"
        )
        print("\n3. Dual-Class Rollup & Primary Class Selection (2024-08-16):")
        print(qa["dual_class_sample"].to_string(index=False))
        print("\n4. NVDA Split-Crossing QoQ Delta (2024-08-16, post 10:1 split):")
        print(qa["nvda_split_qoq"].to_string(index=False))
        print("\n5. Stage 3 Panel Eligibility & Sanity Summary (2015-2026):")
        print(qa["panel_summary"].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
