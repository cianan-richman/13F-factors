"""Staged Point-in-Time (PIT) Security Aggregation and Market-Data Panel API.

Implements a clean, auditable two-stage pipeline sitting directly on top of
`src.pit.ledger.PitHoldingsLedger`:

  Stage 1 — Security Mapping & Aggregation (`map_holdings_to_securities`,
            `aggregate_positions_by_security`):
      Maps filer-level PIT holdings `(filer_id, cusip)` to canonical securities
      (`security_id` = `share_class_figi`, `yahoo_ticker`), filters to US Common
      Stocks by default, stitches CUSIP changes, and aggregates raw holdings by
      security for a given `query_date`.

  Stage 2 — PIT Market Data, Split Alignment, SPY Close & Universe Gates
            (`PitMarketStore`, `attach_pit_market_data`, `build_pit_security_snapshot`):
      Joins strictly point-in-time close prices (`raw_close`, `split_adj_close`,
      forward-only total-return `tr_close`), forward-only adjusted `SPY` close
      (`spy_adj_close`), 20-day ADV, trading-day seasoning (`>= 252` prior trading
      days), and PIT XBRL `shares_outstanding` (`filed_date < query_date`).
      Applies strictly forward split adjustments (zero lookahead):
        - 13F per-filer shares: splits strictly in `(period_of_report, query_date]`
          (empirically verified: 13F share counts are on `period_of_report` basis).
        - XBRL shares outstanding: splits strictly in `(filed_date, query_date]`
          (empirically verified: under GAAP ASC 260, 10-Q/10-K share counts already
          reflect splits occurring between `end_date` and `filed_date`).
      Re-aggregates split-aligned filer shares to security level so filers straddling
      a split across different report dates are always summed in exact `query_date`
      units, computes `pit_mcap_usd = raw_close * shares_out_pit`, flags dual-class
      issuers and stale XBRL share counts, and runs runtime zero-lookahead QA checks.

Usage:
    python -m src.pit.panel --query-date 2024-06-10 --audit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ingest.cusip_mapper import DEFAULT_CUSIP_MAP_PARQUET
from src.ingest.market_data import DEFAULT_MARKET_PRICES_PARQUET, DEFAULT_SPY_PARQUET
from src.ingest.shares_outstanding import DEFAULT_SHARES_PARQUET
from src.pit.ledger import (
    DEFAULT_MAX_STALENESS_DAYS,
    PitHoldingsLedger,
    PitLookaheadError,
    verify_no_lookahead,
)

logger = logging.getLogger(__name__)

# Universe & QA defaults (strictly point-in-time)
DEFAULT_MIN_SEASONING_DAYS = 252  # > 1 trading year strictly prior to query_date
DEFAULT_THRESHOLD_ANCHOR_DATE = "2014-12-31"  # Strictly pre-IS anchor (zero lookahead)
DEFAULT_THRESHOLD_SPY_SMA_DAYS = 60  # Smooth trailing SPY raw_close window ending t-1
DEFAULT_ANCHOR_MCAP_USD = 1_000_000_000.0  # $1.0B on 2014-12-31 (scaled forward by SPY price)
DEFAULT_ANCHOR_ADV_USD = 5_000_000.0  # $5.0M on 2014-12-31 (constant 0.5% turnover floor)
DEFAULT_MAX_SHARES_AGE_DAYS = 460  # Accommodates annual 20-F filers; drops multi-yr stale XBRL
DEFAULT_UNEQUAL_CLASS_PRICE_RATIO = 2.5  # Detects BRK-A/BRK-B-style unequal economic ratios
# Strictly PIT SEC XBRL unit-scale typo guards (catches 1e3x/1e6x/1e9x & 1e-3x inline scale typos)
DEFAULT_XBRL_MAX_MCAP_ANCHOR_USD = 4_000_000_000_000.0  # $4T on 2014-12-31, scaled by SPY SMA60
DEFAULT_XBRL_UPWARD_TYPO_MIN_MCAP_USD = 50_000_000_000.0  # Check tiny-turnover typos above $50B
DEFAULT_XBRL_UPWARD_TYPO_MIN_ADV_USD = 5_000_000.0  # Liquid US-traded floor (avoids OTC F-shares)
DEFAULT_XBRL_UPWARD_TYPO_MAX_TURNOVER = 2.5e-5  # <0.0025%/day implies 1000x upward share typo
DEFAULT_XBRL_DOWNWARD_TYPO_MIN_TURNOVER = 2.5  # >250%/day sustained 20d ADV implies 1e-3x typo


class PitMarketStore:
    """In-memory indexed store for fast, strictly PIT market, SPY, and XBRL lookups."""

    def __init__(
        self,
        market_prices: pd.DataFrame,
        spy_prices: pd.DataFrame,
        shares_out: pd.DataFrame,
        cusip_map: pd.DataFrame,
    ) -> None:
        self.cusip_map = _prepare_cusip_map(cusip_map)

        # Prepare per-ticker sorted price arrays for O(log N) split & price lookups
        mp = market_prices.copy()
        mp["date"] = pd.to_datetime(mp["date"]).dt.normalize()
        mp["yahoo_ticker"] = mp["yahoo_ticker"].astype(str).str.strip().str.upper()
        mp = mp.sort_values(["yahoo_ticker", "date"]).reset_index(drop=True)

        # Compute strictly forward-only total-return close (tr_close) and trading_days_history
        # per ticker without ever touching future dates.
        mp["trading_days_history"] = mp.groupby("yahoo_ticker", sort=False).cumcount()
        ret_filled = mp["ret_total"].fillna(0.0).to_numpy(dtype=np.float64)
        # Forward-compounded growth index per ticker anchored at first raw_close
        tr_vals = np.empty(len(mp), dtype=np.float64)
        self._ticker_arrays: dict[str, dict[str, np.ndarray]] = {}
        self._ticker_has_splits: set[str] = set()

        for tkr, grp in mp.groupby("yahoo_ticker", sort=False):
            idx = grp.index.to_numpy()
            r = ret_filled[idx].copy()
            r[0] = 0.0
            base_px = float(grp["raw_close"].iloc[0])
            tr_series = base_px * np.cumprod(1.0 + r)
            tr_vals[idx] = tr_series

            dates_ns = grp["date"].to_numpy(dtype="datetime64[ns]")
            s_fac = grp["split_factor"].to_numpy(dtype=np.float64)
            f_fut = grp["cum_split_to_latest"].to_numpy(dtype=np.float64)
            has_split = bool(np.any(s_fac != 1.0))
            if has_split:
                self._ticker_has_splits.add(str(tkr))

            self._ticker_arrays[str(tkr)] = {
                "dates_ns": dates_ns,
                "raw_close": grp["raw_close"].to_numpy(dtype=np.float64),
                "split_adj_close": grp["split_adj_close"].to_numpy(dtype=np.float64),
                "tr_close": tr_series,
                "split_factor": s_fac,
                "cum_split_to_latest": f_fut,
                "raw_volume": grp["raw_volume"].to_numpy(dtype=np.float64),
                "dollar_volume": grp["dollar_volume"].to_numpy(dtype=np.float64),
                "adv_20d_usd": grp["adv_20d_usd"].to_numpy(dtype=np.float64),
                "ret_total": grp["ret_total"].to_numpy(dtype=np.float64),
                "trading_days_history": grp["trading_days_history"].to_numpy(dtype=np.int64),
            }

        mp["tr_close"] = tr_vals
        self.market_prices = mp

        # Fast date -> ticker slice lookup for market_prices
        self._prices_by_date: dict[pd.Timestamp, pd.DataFrame] = {
            pd.Timestamp(dt): grp.set_index("yahoo_ticker")
            for dt, grp in mp.groupby("date", sort=False)
        }

        # Prepare SPY table with strictly forward-only total-return adjusted close
        spy = spy_prices.copy()
        spy["date"] = pd.to_datetime(spy["date"]).dt.normalize()
        spy = (
            spy.sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        if not spy.empty:
            spy_r = spy["ret_total"].fillna(0.0).to_numpy(dtype=np.float64, copy=True)
            spy_r[0] = 0.0
            spy["spy_adj_close"] = float(spy["raw_close"].iloc[0]) * np.cumprod(1.0 + spy_r)
            spy["spy_sma60"] = (
                spy["raw_close"]
                .rolling(window=DEFAULT_THRESHOLD_SPY_SMA_DAYS, min_periods=1)
                .mean()
            )
        else:
            spy["spy_adj_close"] = np.nan
            spy["spy_sma60"] = np.nan
        self.spy_prices = spy
        self._spy_dates_ns = spy["date"].to_numpy(dtype="datetime64[ns]")
        self._spy_sma60_arr = spy["spy_sma60"].to_numpy(dtype=np.float64)
        self._spy_by_date: dict[pd.Timestamp, dict[str, float]] = {
            pd.Timestamp(r["date"]): {
                "spy_raw_close": float(r["raw_close"]),
                "spy_adj_close": float(r["spy_adj_close"]),
                "spy_ret_total": float(r["ret_total"]) if pd.notna(r["ret_total"]) else np.nan,
            }
            for r in spy.to_dict(orient="records")
        }

        # Prepare PIT shares outstanding indexed by yahoo_ticker
        so = shares_out.copy()
        if not so.empty:
            so["yahoo_ticker"] = so["yahoo_ticker"].astype(str).str.strip().str.upper()
            so["filed_date"] = pd.to_datetime(so["filed_date"]).dt.normalize()
            so["end_date"] = pd.to_datetime(so["end_date"]).dt.normalize()
            so = so.dropna(subset=["filed_date", "end_date", "shares_outstanding"]).copy()
            # Scrub SEC XBRL future-dated typos (e.g., AAL end_date=2027-07-17 filed 2026-07-23)
            so = so[(so["shares_outstanding"] > 0) & (so["end_date"] <= so["filed_date"])].copy()
            # Scrub SEC XBRL unit-scale typos (1e3x/1e6x/1e9x or 1e-3x) strictly PIT at filed_date
            so = self._scrub_pit_xbrl_scale_outliers(so)
            so = so.sort_values(
                ["yahoo_ticker", "filed_date", "end_date", "tag_priority"],
                ascending=[True, True, True, False],
            ).reset_index(drop=True)
        self.shares_out = so

        self._shares_by_ticker: dict[str, pd.DataFrame] = (
            {
                str(tkr): grp.reset_index(drop=True)
                for tkr, grp in so.groupby("yahoo_ticker", sort=False)
            }
            if not so.empty
            else {}
        )

        # Dual-class issuer map: issuer_cik -> sorted list of yahoo_tickers sharing that CIK
        self._cik_to_tickers: dict[str, list[str]] = {}
        self._ticker_to_cik: dict[str, str] = {}
        if not so.empty and "issuer_cik" in so.columns:
            pairs = so[["yahoo_ticker", "issuer_cik"]].drop_duplicates()
            for r in pairs.to_dict(orient="records"):
                tkr = str(r["yahoo_ticker"])
                cik = str(r["issuer_cik"])
                self._ticker_to_cik[tkr] = cik
                self._cik_to_tickers.setdefault(cik, []).append(tkr)
            for cik in self._cik_to_tickers:
                self._cik_to_tickers[cik] = sorted(set(self._cik_to_tickers[cik]))

    @classmethod
    def from_parquet(
        cls,
        *,
        market_prices_path: Path = DEFAULT_MARKET_PRICES_PARQUET,
        spy_path: Path = DEFAULT_SPY_PARQUET,
        shares_path: Path = DEFAULT_SHARES_PARQUET,
        cusip_map_path: Path = DEFAULT_CUSIP_MAP_PARQUET,
    ) -> PitMarketStore:
        """Load all market, SPY, XBRL shares, and CUSIP mapping tables from Parquet."""
        return cls(
            market_prices=pd.read_parquet(market_prices_path),
            spy_prices=pd.read_parquet(spy_path),
            shares_out=pd.read_parquet(shares_path),
            cusip_map=pd.read_parquet(cusip_map_path),
        )

    def get_forward_split_factor(
        self,
        ticker: str,
        from_date: pd.Timestamp | np.datetime64,
        to_date: pd.Timestamp | np.datetime64,
    ) -> float:
        """Return exact cumulative split factor strictly in `(from_date, to_date]`.

        Zero-lookahead guarantee:
            Uses `F_fut(from_date) / F_fut(to_date)` where `F_fut(d) = prod_{tau > d} s_tau`.
            All splits strictly after `to_date` appear identically in numerator and denominator
            and cancel out completely.
        """
        tkr = str(ticker).strip().upper()
        if tkr not in self._ticker_has_splits:
            return 1.0

        arr = self._ticker_arrays.get(tkr)
        if arr is None or len(arr["dates_ns"]) == 0:
            return 1.0

        f_ns = np.datetime64(pd.Timestamp(from_date).normalize(), "ns")
        t_ns = np.datetime64(pd.Timestamp(to_date).normalize(), "ns")
        if f_ns >= t_ns:
            return 1.0

        dates_ns = arr["dates_ns"]
        f_fut = arr["cum_split_to_latest"]
        s_fac = arr["split_factor"]

        idx_from = int(np.searchsorted(dates_ns, f_ns, side="right")) - 1
        idx_to = int(np.searchsorted(dates_ns, t_ns, side="right")) - 1

        if idx_to < 0:
            return 1.0

        fut_from = float(f_fut[idx_from]) if idx_from >= 0 else float(f_fut[0] * s_fac[0])
        fut_to = float(f_fut[idx_to])
        if fut_to <= 0.0:
            return 1.0
        return fut_from / fut_to

    def get_dynamic_thresholds(
        self,
        query_date: str | pd.Timestamp,
        *,
        anchor_date: str | pd.Timestamp = DEFAULT_THRESHOLD_ANCHOR_DATE,
        anchor_mcap_usd: float = DEFAULT_ANCHOR_MCAP_USD,
        anchor_adv_usd: float = DEFAULT_ANCHOR_ADV_USD,
    ) -> dict[str, float]:
        """Compute strictly PIT dynamic Market-Cap and ADV floors scaled forward by `SPY`.

        Zero-lookahead guarantee:
          - Anchor level is fixed strictly BEFORE the IS window (`anchor_date = 2014-12-31`).
          - Current `SPY` level uses the 60-day trailing SMA of `SPY raw_close` (price index,
            excluding reinvested dividends) ending strictly on `t - 1` (`date < query_date`),
            so `query_date` thresholds never depend on `query_date` or future market moves.
        """
        if len(self._spy_dates_ns) == 0:
            return {
                "spy_threshold_scale": 1.0,
                "min_mcap_threshold_usd": float(anchor_mcap_usd),
                "min_adv_threshold_usd": float(anchor_adv_usd),
            }

        a_ns = np.datetime64(pd.Timestamp(anchor_date).normalize(), "ns")
        q_ns = np.datetime64(pd.Timestamp(query_date).normalize(), "ns")

        # Anchor SMA on last trading day <= anchor_date (fallback to index 0 on synthetic spans)
        a_idx = int(np.searchsorted(self._spy_dates_ns, a_ns, side="right")) - 1
        a_idx = max(0, a_idx)
        anchor_sma = float(self._spy_sma60_arr[a_idx])

        # Current SMA strictly BEFORE query_date (t - 1) so zero same-day or future lookahead
        q_idx = int(np.searchsorted(self._spy_dates_ns, q_ns, side="left")) - 1
        q_idx = max(0, q_idx)
        cur_sma = float(self._spy_sma60_arr[q_idx])

        scale = (cur_sma / anchor_sma) if (anchor_sma > 0 and cur_sma > 0) else 1.0
        return {
            "spy_threshold_scale": scale,
            "min_mcap_threshold_usd": float(anchor_mcap_usd) * scale,
            "min_adv_threshold_usd": float(anchor_adv_usd) * scale,
        }

    def _scrub_pit_xbrl_scale_outliers(self, so: pd.DataFrame) -> pd.DataFrame:
        """Remove SEC XBRL inline unit-scale typos strictly PIT using market data <= `filed_date`.

        Catches three physical impossibilities visible on `filed_date` with zero lookahead:
          1. Upward mega-scale typos (`1e3x`/`1e6x`/`1e9x`, e.g., EEFT, YUM, AJG, EIX, STNE):
             `shares * raw_close(<= filed_date) > $4T * spy_scale(filed_date - 1)`.
          2. Upward mid-cap `1e3x` typos (e.g., SHOE, CPK, WWD, TXNM):
             `implied_mcap > $50B` with liquid US trading (`adv_20d >= $5M`) yet sustained
             20-day daily turnover `adv_20d / implied_mcap < 2.5e-5` (`0.0025%/day`).
          3. Downward `1e-3x` typos (e.g., ARW 2017-11-02, AAPL 2014-04-24):
             sustained 20-day daily turnover `adv_20d / implied_mcap > 2.5` (`250%/day`).

        Because bad rows are removed before `(filed_date, end_date, tag_priority)` selection, a
        bad cover-page `dei` tag seamlessly falls back to a clean `us-gaap` tag in the same
        filing (or the prior clean quarter's filing), preserving exact parity between single-date
        snapshots and vectorized daily panel builds.
        """
        if so.empty or not self._ticker_arrays:
            return so

        # Precompute strictly PIT SPY scale per unique filed_date (uses SPY SMA60 ending t-1)
        unique_fdts = so["filed_date"].dropna().unique()
        spy_scale_map: dict[pd.Timestamp, float] = {
            pd.Timestamp(dt): self.get_dynamic_thresholds(dt)["spy_threshold_scale"]
            for dt in unique_fdts
        }

        keep_mask = np.ones(len(so), dtype=bool)
        so_reset = so.reset_index(drop=True)

        for tkr, grp in so_reset.groupby("yahoo_ticker", sort=False):
            arr = self._ticker_arrays.get(str(tkr))
            if arr is None or len(arr["dates_ns"]) == 0:
                continue

            dates_ns = arr["dates_ns"]
            raw_close = arr["raw_close"]
            adv_arr = arr["adv_20d_usd"]

            f_ns = grp["filed_date"].to_numpy(dtype="datetime64[ns]")
            pos = np.searchsorted(dates_ns, f_ns, side="right") - 1
            valid_pos = pos >= 0
            if not np.any(valid_pos):
                continue

            idxs = grp.index.to_numpy()[valid_pos]
            pos_v = pos[valid_pos]
            sh_v = grp["shares_outstanding"].to_numpy(dtype=np.float64)[valid_pos]
            px_v = raw_close[pos_v]
            adv_v = adv_arr[pos_v]
            scales_v = np.fromiter(
                (spy_scale_map[pd.Timestamp(d)] for d in grp["filed_date"].iloc[valid_pos]),
                dtype=np.float64,
                count=len(idxs),
            )

            px_ok = np.isfinite(px_v) & (px_v > 0.0)
            imp_mcap = np.where(px_ok, sh_v * px_v, np.nan)
            adv_ok = px_ok & np.isfinite(adv_v) & (adv_v > 0.0) & (imp_mcap > 0.0)
            turn = np.where(adv_ok, adv_v / imp_mcap, np.nan)

            bad_mega = px_ok & (imp_mcap > DEFAULT_XBRL_MAX_MCAP_ANCHOR_USD * scales_v)
            bad_up_turn = (
                adv_ok
                & (imp_mcap > DEFAULT_XBRL_UPWARD_TYPO_MIN_MCAP_USD)
                & (adv_v >= DEFAULT_XBRL_UPWARD_TYPO_MIN_ADV_USD)
                & (turn < DEFAULT_XBRL_UPWARD_TYPO_MAX_TURNOVER)
            )
            bad_down_turn = adv_ok & (turn > DEFAULT_XBRL_DOWNWARD_TYPO_MIN_TURNOVER)

            bad = bad_mega | bad_up_turn | bad_down_turn
            if np.any(bad):
                keep_mask[idxs[bad]] = False

        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            logger.info(
                "Scrubbed %d SEC XBRL unit-scale outlier rows (%.4f%% of %d) strictly PIT.",
                n_dropped,
                100.0 * n_dropped / len(so_reset),
                len(so_reset),
            )
        return so_reset.loc[keep_mask].reset_index(drop=True)

    def get_spy_on_date(self, query_date: str | pd.Timestamp) -> dict[str, float]:
        """Return `spy_raw_close`, `spy_adj_close` (forward-only TR), and `spy_ret_total`."""
        qdt = pd.Timestamp(query_date).normalize()
        rec = self._spy_by_date.get(qdt)
        if rec is not None:
            return rec
        # If query_date is a non-trading day, fall back strictly to latest prior trading day
        prior = self.spy_prices[self.spy_prices["date"] <= qdt]
        if prior.empty:
            return {"spy_raw_close": np.nan, "spy_adj_close": np.nan, "spy_ret_total": np.nan}
        last = prior.iloc[-1]
        return {
            "spy_raw_close": float(last["raw_close"]),
            "spy_adj_close": float(last["spy_adj_close"]),
            "spy_ret_total": float(last["ret_total"]) if pd.notna(last["ret_total"]) else np.nan,
        }

    def get_price_row_on_date(
        self,
        ticker: str,
        query_date: pd.Timestamp,
    ) -> dict[str, Any] | None:
        """Return exact trading-day price row for `ticker` on `query_date` (None if not traded)."""
        tkr = str(ticker).strip().upper()
        day_df = self._prices_by_date.get(query_date)
        if day_df is None or tkr not in day_df.index:
            return None
        row = day_df.loc[tkr]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return row.to_dict()

    def get_pit_shares_outstanding(
        self,
        ticker: str,
        query_date: str | pd.Timestamp,
        *,
        max_shares_age_days: int | None = DEFAULT_MAX_SHARES_AGE_DAYS,
    ) -> dict[str, Any]:
        """Return strictly PIT shares outstanding for `ticker` as of `query_date`.

        Point-in-time & corporate-action guarantees:
          1. Strictly filters `filed_date < query_date` (never same-day filing lookahead).
          2. Selects the row with max `(end_date, filed_date, -tag_priority)` so comparative
             prior-year rows inside a 10-Q/10-K are never chosen over current-period rows.
          3. Forward-adjusts `shares_outstanding` strictly for splits in `(filed_date, query_date]`
             (under GAAP ASC 260, splits occurring in `(end_date, filed_date]` are already
             reflected in the reported share count on `filed_date`).
          4. Flags stale share reports exceeding `max_shares_age_days`.
        """
        qdt = pd.Timestamp(query_date).normalize()
        tkr = str(ticker).strip().upper()
        empty = {
            "issuer_cik": self._ticker_to_cik.get(tkr, ""),
            "shares_out_raw": np.nan,
            "shares_out_split_factor": np.nan,
            "shares_out_pit": np.nan,
            "shares_end_date": pd.NaT,
            "shares_filed_date": pd.NaT,
            "shares_form": "",
            "shares_tag_source": "",
            "shares_out_age_days": np.nan,
            "is_shares_out_stale": False,
        }
        sub = self._shares_by_ticker.get(tkr)
        if sub is None or sub.empty:
            return empty

        visible = sub[sub["filed_date"] < qdt]
        if visible.empty:
            return empty

        # Pick max end_date -> max filed_date -> lowest tag_priority
        best = visible.sort_values(
            ["end_date", "filed_date", "tag_priority"],
            ascending=[True, True, False],
        ).iloc[-1]

        filed_dt = pd.Timestamp(best["filed_date"])
        end_dt = pd.Timestamp(best["end_date"])
        raw_sh = float(best["shares_outstanding"])
        # Forward-adjust strictly for splits in (filed_date, query_date]
        split_fac = self.get_forward_split_factor(tkr, filed_dt, qdt)
        adj_sh = raw_sh * split_fac
        age_days = int((qdt - end_dt).days)
        is_stale = bool(max_shares_age_days is not None and age_days > max_shares_age_days)

        return {
            "issuer_cik": str(best.get("issuer_cik", self._ticker_to_cik.get(tkr, ""))),
            "shares_out_raw": raw_sh,
            "shares_out_split_factor": split_fac,
            "shares_out_pit": np.nan if is_stale else adj_sh,
            "shares_end_date": end_dt,
            "shares_filed_date": filed_dt,
            "shares_form": str(best.get("form", "")),
            "shares_tag_source": str(best.get("tag_source", "")),
            "shares_out_age_days": age_days,
            "is_shares_out_stale": is_stale,
        }

    def get_dual_class_flags(
        self,
        ticker: str,
        query_date: pd.Timestamp,
        *,
        unequal_price_ratio: float = DEFAULT_UNEQUAL_CLASS_PRICE_RATIO,
    ) -> tuple[bool, bool, str]:
        """Return `(is_dual_class_issuer, has_unequal_class_ratio, sibling_tickers_csv)`.

        Strictly PIT: detects unequal economic class ratios (such as `BRK-A` vs `BRK-B` at
        1,500:1) solely from same-day `query_date` raw closes of sibling share classes.
        """
        tkr = str(ticker).strip().upper()
        cik = self._ticker_to_cik.get(tkr, "")
        siblings = self._cik_to_tickers.get(cik, [])
        if len(siblings) <= 1:
            return False, False, ""

        other_siblings = [s for s in siblings if s != tkr]
        day_df = self._prices_by_date.get(query_date)
        has_unequal = False
        if day_df is not None and tkr in day_df.index:
            my_px = float(day_df.loc[tkr, "raw_close"])
            if my_px > 0:
                for sib in other_siblings:
                    if sib in day_df.index:
                        sib_px = float(day_df.loc[sib, "raw_close"])
                        if sib_px > 0:
                            ratio = max(my_px, sib_px) / min(my_px, sib_px)
                            if ratio >= unequal_price_ratio:
                                has_unequal = True
                                break
        return True, has_unequal, ",".join(other_siblings)


def _prepare_cusip_map(cusip_map: pd.DataFrame) -> pd.DataFrame:
    """Normalize CUSIP mapping columns for deterministic security joins."""
    cm = cusip_map.copy()
    cm["cusip"] = cm["cusip"].astype(str).str.strip().str.upper()
    cm["security_id"] = cm["security_id"].fillna("").astype(str).str.strip()
    cm["yahoo_ticker"] = cm["yahoo_ticker"].fillna("").astype(str).str.strip().str.upper()
    cm["figi_name"] = cm.get("figi_name", "").fillna("").astype(str)
    cm["is_us_common_stock"] = cm["is_us_common_stock"].fillna(False).astype(bool)
    # Fall back to CUSIP when security_id is blank (unmapped rows if us_common_only=False)
    cm["security_id"] = np.where(cm["security_id"] != "", cm["security_id"], cm["cusip"])
    return cm.drop_duplicates(subset=["cusip"], keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 1: Security Mapping & Aggregation (Pure 13F + CUSIP Map)
# ---------------------------------------------------------------------------


def map_holdings_to_securities(
    pit_holdings: pd.DataFrame,
    cusip_map: pd.DataFrame,
    *,
    us_common_only: bool = True,
) -> pd.DataFrame:
    """Map filer-level PIT 13F holdings from `cusip` to canonical `security_id`.

    Collapses any rare same-filer, same-security multi-CUSIP rows (for example during a
    CUSIP-change transition where a filer reports both old and new CUSIPs in the same
    filing) so `(filer_id, security_id)` is guaranteed unique per `query_date`.
    """
    if pit_holdings.empty:
        return pd.DataFrame(
            columns=list(pit_holdings.columns)
            + ["security_id", "yahoo_ticker", "figi_name", "is_us_common_stock"]
        )

    cm = _prepare_cusip_map(cusip_map)[
        ["cusip", "security_id", "yahoo_ticker", "figi_name", "is_us_common_stock"]
    ]
    df = pit_holdings.copy()
    df["cusip"] = df["cusip"].astype(str).str.strip().str.upper()
    merged = df.merge(cm, on="cusip", how="left")
    merged["is_us_common_stock"] = merged["is_us_common_stock"].fillna(False).astype(bool)
    merged["security_id"] = merged["security_id"].fillna(merged["cusip"])
    merged["yahoo_ticker"] = merged["yahoo_ticker"].fillna("")
    merged["figi_name"] = merged["figi_name"].fillna(merged.get("name_of_issuer", ""))

    if us_common_only:
        merged = merged[merged["is_us_common_stock"]].copy()

    if merged.empty:
        return merged.reset_index(drop=True)

    # Collapse any same-(filer_id, security_id) multi-CUSIP rows cleanly
    if merged.duplicated(subset=["filer_id", "security_id"]).any():
        merged = (
            merged.sort_values(["filer_id", "security_id", "cusip"])
            .groupby(["filer_id", "security_id"], as_index=False)
            .agg(
                query_date=("query_date", "first"),
                canonical_name=("canonical_name", "first"),
                style=("style", "first"),
                period_of_report=("period_of_report", "max"),
                filing_date=("filing_date", "max"),
                acceptance_date=("acceptance_date", "max"),
                effective_date=("effective_date", "max"),
                accession_number=("accession_number", "last"),
                source_form_type=("source_form_type", "last"),
                amendment_type=("amendment_type", "last"),
                is_ctr_reveal=("is_ctr_reveal", "any"),
                cusip=("cusip", lambda s: ",".join(sorted(set(s)))),
                name_of_issuer=("name_of_issuer", "first"),
                title_of_class=("title_of_class", "first"),
                shares=("shares", "sum"),
                reported_value_usd=("reported_value_usd", "sum"),
                portfolio_weight_reported=("portfolio_weight_reported", "sum"),
                staleness_days=("staleness_days", "max"),
                yahoo_ticker=("yahoo_ticker", "first"),
                figi_name=("figi_name", "first"),
                is_us_common_stock=("is_us_common_stock", "all"),
            )
        )

    return merged.sort_values(["security_id", "filer_id"]).reset_index(drop=True)


def aggregate_positions_by_security(
    pit_holdings: pd.DataFrame,
    cusip_map: pd.DataFrame,
    *,
    us_common_only: bool = True,
    verify: bool = True,
) -> pd.DataFrame:
    """Stage 1 API: Aggregate filer-level PIT holdings to one row per security.

    Produces the pure 13F security universe for `query_date` BEFORE attaching market prices
    or split adjustments. Note that `total_shares_raw` sums unadjusted reported shares across
    filers; if filers are on different `period_of_report` dates straddling a stock split,
    `has_mixed_report_periods` is True and Stage 2 (`attach_pit_market_data`) computes the
    exact split-aligned `total_shares_owned_pit`.
    """
    output_cols = [
        "query_date",
        "security_id",
        "yahoo_ticker",
        "figi_name",
        "cusips",
        "is_us_common_stock",
        "n_filers",
        "total_shares_raw",
        "total_reported_value_usd",
        "mean_filer_weight_reported",
        "max_filer_weight_reported",
        "holder_hhi_reported",
        "min_period_of_report",
        "max_period_of_report",
        "has_mixed_report_periods",
        "latest_filing_date",
        "latest_acceptance_date",
        "latest_effective_date",
        "has_ctr_reveal",
    ]
    if pit_holdings.empty:
        return pd.DataFrame(columns=output_cols)

    qdt = pd.Timestamp(pit_holdings["query_date"].iloc[0]).normalize()
    if verify:
        verify_no_lookahead(pit_holdings, qdt)

    mapped = map_holdings_to_securities(
        pit_holdings,
        cusip_map,
        us_common_only=us_common_only,
    )
    if mapped.empty:
        return pd.DataFrame(columns=output_cols)

    rows: list[dict[str, Any]] = []
    for sec_id, grp in mapped.groupby("security_id", sort=True):
        val_arr = grp["reported_value_usd"].to_numpy(dtype=np.float64)
        tot_val = float(val_arr.sum())
        if tot_val > 0:
            holder_shares = val_arr / tot_val
            hhi = float(np.sum(holder_shares**2))
        else:
            hhi = np.nan

        cusip_list = sorted({c for item in grp["cusip"].astype(str) for c in item.split(",") if c})
        min_rep = pd.Timestamp(grp["period_of_report"].min())
        max_rep = pd.Timestamp(grp["period_of_report"].max())

        rows.append(
            {
                "query_date": qdt,
                "security_id": str(sec_id),
                "yahoo_ticker": str(grp["yahoo_ticker"].iloc[0]),
                "figi_name": str(grp["figi_name"].iloc[0]),
                "cusips": ",".join(cusip_list),
                "is_us_common_stock": bool(grp["is_us_common_stock"].all()),
                "n_filers": int(grp["filer_id"].nunique()),
                "total_shares_raw": float(grp["shares"].sum()),
                "total_reported_value_usd": tot_val,
                "mean_filer_weight_reported": float(grp["portfolio_weight_reported"].mean()),
                "max_filer_weight_reported": float(grp["portfolio_weight_reported"].max()),
                "holder_hhi_reported": hhi,
                "min_period_of_report": min_rep,
                "max_period_of_report": max_rep,
                "has_mixed_report_periods": bool(min_rep != max_rep),
                "latest_filing_date": pd.Timestamp(grp["filing_date"].max()),
                "latest_acceptance_date": pd.Timestamp(grp["acceptance_date"].max()),
                "latest_effective_date": pd.Timestamp(grp["effective_date"].max()),
                "has_ctr_reveal": bool(grp["is_ctr_reveal"].any()),
            }
        )

    agg_df = (
        pd.DataFrame(rows)
        .sort_values(["total_reported_value_usd", "security_id"], ascending=[False, True])
        .reset_index(drop=True)
    )

    if verify:
        verify_stage1_security_table(agg_df, qdt)

    return agg_df[output_cols]


def verify_stage1_security_table(df: pd.DataFrame, query_date: pd.Timestamp) -> None:
    """QA checks for Stage 1 security-aggregated PIT 13F table."""
    qdt = pd.Timestamp(query_date).normalize()
    if df.empty:
        return
    if df.duplicated(subset=["security_id"]).any():
        raise PitLookaheadError(f"Stage 1 duplicate security_id rows on {qdt.date()}.")
    if not (pd.to_datetime(df["latest_filing_date"]) < qdt).all():
        raise PitLookaheadError(f"Stage 1 lookahead: latest_filing_date >= {qdt.date()}.")
    if not (pd.to_datetime(df["latest_acceptance_date"]) < qdt).all():
        raise PitLookaheadError(f"Stage 1 lookahead: latest_acceptance_date >= {qdt.date()}.")
    if not (pd.to_datetime(df["latest_effective_date"]) <= qdt).all():
        raise PitLookaheadError(f"Stage 1 lookahead: latest_effective_date > {qdt.date()}.")
    if not (df["n_filers"] > 0).all() or not (df["total_shares_raw"] > 0).all():
        raise PitLookaheadError(f"Stage 1 invalid non-positive counts on {qdt.date()}.")


# ---------------------------------------------------------------------------
# Stage 2: Market Data Join, Split Alignment, SPY Close & Universe Gates
# ---------------------------------------------------------------------------


def attach_pit_market_data(
    stage1_df: pd.DataFrame,
    mapped_holdings_df: pd.DataFrame,
    market_store: PitMarketStore,
    *,
    min_seasoning_days: int = DEFAULT_MIN_SEASONING_DAYS,
    anchor_date: str | pd.Timestamp = DEFAULT_THRESHOLD_ANCHOR_DATE,
    anchor_mcap_usd: float = DEFAULT_ANCHOR_MCAP_USD,
    anchor_adv_usd: float = DEFAULT_ANCHOR_ADV_USD,
    min_mcap_usd: float | None = None,
    min_adv_usd: float | None = None,
    max_shares_age_days: int | None = DEFAULT_MAX_SHARES_AGE_DAYS,
    verify: bool = True,
) -> pd.DataFrame:
    """Stage 2 API: Attach strictly PIT market data, split alignments, SPY close & universe gates.

    For every security in `stage1_df` on `query_date`:
      1. Forward-adjusts EVERY individual filer's 13F share count strictly from its own
         `period_of_report` to `query_date` (`splits in (period_of_report, query_date]`)
         and sums them into `total_shares_owned_pit` — guaranteeing exact apples-to-apples
         summation even when filers straddle a stock split across different report periods.
      2. Attaches `raw_close`, `split_adj_close`, forward-only `tr_close`, `split_factor`,
         `cum_split_to_latest`, `raw_volume`, `dollar_volume`, `adv_20d_usd`, and
         `trading_days_history` (strictly `< query_date`).
      3. Attaches `spy_raw_close` and forward-only dividend-adjusted `spy_adj_close`.
      4. Joins strictly PIT XBRL `shares_outstanding` (`filed_date < query_date`), forward-
         adjusted strictly for splits in `(filed_date, query_date]` (per GAAP ASC 260).
      5. Computes `total_position_usd_pit = total_shares_owned_pit * raw_close` and
         `pit_mcap_usd = raw_close * shares_out_pit` (nulled if XBRL is stale or unequal
         multi-class share ratio such as BRK-B vs BRK-A is detected).
      6. Evaluates explicit PIT universe eligibility gates (`has_market_price`, `is_seasoned`,
         `passes_mcap_gate`, `passes_adv_gate`, `is_investable`) using dynamic `SPY`-scaled
         floors anchored strictly pre-IS (`2014-12-31`: `$1.0B` MCap / `$5.0M` ADV) unless
         fixed overrides `min_mcap_usd` / `min_adv_usd` are explicitly passed.
    """
    if stage1_df.empty:
        return stage1_df.copy()

    qdt = pd.Timestamp(stage1_df["query_date"].iloc[0]).normalize()
    spy_info = market_store.get_spy_on_date(qdt)
    dyn_thresh = market_store.get_dynamic_thresholds(
        qdt,
        anchor_date=anchor_date,
        anchor_mcap_usd=anchor_mcap_usd,
        anchor_adv_usd=anchor_adv_usd,
    )
    eff_mcap_floor = (
        float(min_mcap_usd)
        if min_mcap_usd is not None
        else dyn_thresh["min_mcap_threshold_usd"]
    )
    eff_adv_floor = (
        float(min_adv_usd)
        if min_adv_usd is not None
        else dyn_thresh["min_adv_threshold_usd"]
    )
    spy_scale = (
        1.0
        if (min_mcap_usd is not None and min_adv_usd is not None)
        else dyn_thresh["spy_threshold_scale"]
    )

    # Pre-group mapped filer holdings by security_id for exact per-filer split alignment
    holdings_by_sec: dict[str, pd.DataFrame] = (
        {str(sid): grp for sid, grp in mapped_holdings_df.groupby("security_id", sort=False)}
        if not mapped_holdings_df.empty
        else {}
    )

    records: list[dict[str, Any]] = []
    for base in stage1_df.to_dict(orient="records"):
        sec_id = str(base["security_id"])
        tkr = str(base["yahoo_ticker"]).strip().upper()
        px_row = market_store.get_price_row_on_date(tkr, qdt) if tkr else None

        # 1. Exact per-filer forward split adjustment from each filer's period_of_report -> qdt
        sec_h = holdings_by_sec.get(sec_id)
        if sec_h is not None and not sec_h.empty:
            adj_shares_list: list[float] = []
            split_factors_used: list[float] = []
            for h_row in sec_h.to_dict(orient="records"):
                rep_dt = pd.Timestamp(h_row["period_of_report"])
                k_hold = (
                    market_store.get_forward_split_factor(tkr, rep_dt, qdt)
                    if (tkr and px_row is not None)
                    else 1.0
                )
                adj_shares_list.append(float(h_row["shares"]) * k_hold)
                split_factors_used.append(k_hold)
            total_shares_pit = float(sum(adj_shares_list))
            n_filers_split_adj = int(sum(1 for k in split_factors_used if abs(k - 1.0) > 1e-9))
            min_k = float(min(split_factors_used))
            max_k = float(max(split_factors_used))
            has_split_straddle = bool(abs(max_k - min_k) > 1e-9)
        else:
            total_shares_pit = float(base["total_shares_raw"])
            n_filers_split_adj = 0
            min_k = 1.0
            max_k = 1.0
            has_split_straddle = False

        # 2. Market close prices, volume, ADV, and seasoning
        has_price = px_row is not None and pd.notna(px_row["raw_close"]) and px_row["raw_close"] > 0
        raw_close = float(px_row["raw_close"]) if has_price else np.nan
        split_adj_close = float(px_row["split_adj_close"]) if has_price else np.nan
        tr_close = float(px_row["tr_close"]) if has_price else np.nan
        split_factor_today = float(px_row["split_factor"]) if has_price else np.nan
        cum_split_to_latest = float(px_row["cum_split_to_latest"]) if has_price else np.nan
        raw_volume = float(px_row["raw_volume"]) if has_price else np.nan
        dollar_volume = float(px_row["dollar_volume"]) if has_price else np.nan
        adv_20d_usd = float(px_row["adv_20d_usd"]) if has_price else np.nan
        trading_days_hist = int(px_row["trading_days_history"]) if has_price else 0
        is_seasoned = bool(has_price and trading_days_hist >= min_seasoning_days)

        total_pos_usd_pit = (total_shares_pit * raw_close) if has_price else np.nan

        # 3. PIT XBRL shares outstanding + dual-class flags
        so_info = (
            market_store.get_pit_shares_outstanding(
                tkr, qdt, max_shares_age_days=max_shares_age_days
            )
            if tkr
            else market_store.get_pit_shares_outstanding("", qdt)
        )
        is_dual_class, has_unequal_class, siblings_csv = (
            market_store.get_dual_class_flags(tkr, qdt) if tkr else (False, False, "")
        )

        shares_out_pit = so_info["shares_out_pit"]
        if has_price and pd.notna(shares_out_pit) and shares_out_pit > 0 and not has_unequal_class:
            pit_mcap_usd = raw_close * float(shares_out_pit)
        else:
            pit_mcap_usd = np.nan

        passes_mcap = bool(pd.notna(pit_mcap_usd) and pit_mcap_usd >= eff_mcap_floor)
        passes_adv = bool(pd.notna(adv_20d_usd) and adv_20d_usd >= eff_adv_floor)
        is_investable = bool(has_price and is_seasoned and passes_mcap and passes_adv)

        rec = dict(base)
        rec.update(
            {
                # Split-aligned 13F positions at query_date close
                "total_shares_owned_pit": total_shares_pit,
                "total_position_usd_pit": total_pos_usd_pit,
                "n_filers_split_adjusted": n_filers_split_adj,
                "min_filer_split_factor": min_k,
                "max_filer_split_factor": max_k,
                "has_split_straddle": has_split_straddle,
                # Close prices & corporate action state
                "has_market_price": has_price,
                "raw_close": raw_close,
                "split_adj_close": split_adj_close,
                "tr_close": tr_close,
                "split_factor_today": split_factor_today,
                "cum_split_to_latest": cum_split_to_latest,
                "raw_volume": raw_volume,
                "dollar_volume": dollar_volume,
                "adv_20d_usd": adv_20d_usd,
                "trading_days_history": trading_days_hist,
                # SPY benchmark close (raw + forward-only adjusted)
                "spy_raw_close": spy_info["spy_raw_close"],
                "spy_adj_close": spy_info["spy_adj_close"],
                # PIT XBRL shares outstanding & market cap
                "issuer_cik": so_info["issuer_cik"],
                "shares_out_raw": so_info["shares_out_raw"],
                "shares_out_split_factor": so_info["shares_out_split_factor"],
                "shares_out_pit": shares_out_pit,
                "shares_end_date": so_info["shares_end_date"],
                "shares_filed_date": so_info["shares_filed_date"],
                "shares_form": so_info["shares_form"],
                "shares_tag_source": so_info["shares_tag_source"],
                "shares_out_age_days": so_info["shares_out_age_days"],
                "is_shares_out_stale": so_info["is_shares_out_stale"],
                "is_dual_class_issuer": is_dual_class,
                "has_unequal_class_ratio": has_unequal_class,
                "dual_class_siblings": siblings_csv,
                "pit_mcap_usd": pit_mcap_usd,
                # PIT dynamic universe thresholds & eligibility gates
                "spy_threshold_scale": spy_scale,
                "min_mcap_threshold_usd": eff_mcap_floor,
                "min_adv_threshold_usd": eff_adv_floor,
                "is_seasoned": is_seasoned,
                "passes_mcap_gate": passes_mcap,
                "passes_adv_gate": passes_adv,
                "is_investable": is_investable,
            }
        )
        records.append(rec)

    out = pd.DataFrame(records)
    if verify:
        verify_stage2_pit_snapshot(out, qdt)
    return out


def build_pit_security_snapshot(
    query_date: str | pd.Timestamp,
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
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
    """End-to-end convenience wrapper executing Stage 1 -> Stage 2 for a single `query_date`."""
    qdt = pd.Timestamp(query_date).normalize()
    pit_h = ledger.get_latest_holdings(
        qdt,
        max_staleness_days=max_staleness_days,
        verify=verify,
    )
    mapped_h = map_holdings_to_securities(
        pit_h,
        market_store.cusip_map,
        us_common_only=us_common_only,
    )
    stage1 = aggregate_positions_by_security(
        pit_h,
        market_store.cusip_map,
        us_common_only=us_common_only,
        verify=verify,
    )
    return attach_pit_market_data(
        stage1,
        mapped_h,
        market_store,
        min_seasoning_days=min_seasoning_days,
        anchor_date=anchor_date,
        anchor_mcap_usd=anchor_mcap_usd,
        anchor_adv_usd=anchor_adv_usd,
        min_mcap_usd=min_mcap_usd,
        min_adv_usd=min_adv_usd,
        max_shares_age_days=max_shares_age_days,
        verify=verify,
    )


def verify_stage2_pit_snapshot(df: pd.DataFrame, query_date: pd.Timestamp) -> None:
    """Strict runtime zero-lookahead and mathematical consistency checks for Stage 2."""
    qdt = pd.Timestamp(query_date).normalize()
    if df.empty:
        return

    verify_stage1_security_table(df, qdt)

    # 1. XBRL shares_outstanding must strictly satisfy filed_date < query_date
    has_so = df["shares_filed_date"].notna()
    if has_so.any():
        so_filed = pd.to_datetime(df.loc[has_so, "shares_filed_date"])
        if not (so_filed < qdt).all():
            bad = df.loc[has_so & (pd.to_datetime(df["shares_filed_date"]) >= qdt)][
                ["yahoo_ticker", "shares_filed_date", "shares_end_date"]
            ].head(5)
            raise PitLookaheadError(
                f"Stage 2 XBRL LOOKAHEAD on {qdt.date()}:\n{bad.to_string(index=False)}"
            )
        so_end = pd.to_datetime(df.loc[has_so, "shares_end_date"])
        if not (so_end <= so_filed).all():
            raise PitLookaheadError("Stage 2 XBRL integrity: shares_end_date > shares_filed_date.")

    # 2. Investable rows MUST satisfy every gate strictly
    inv = df[df["is_investable"]]
    if not inv.empty:
        if not inv["has_market_price"].all():
            raise PitLookaheadError("Stage 2 integrity: is_investable row lacks market price.")
        if not inv["is_seasoned"].all():
            raise PitLookaheadError("Stage 2 integrity: is_investable row is not seasoned.")
        if not inv["pit_mcap_usd"].notna().all():
            raise PitLookaheadError("Stage 2 integrity: is_investable row has NaN pit_mcap_usd.")
        if inv["has_unequal_class_ratio"].any():
            raise PitLookaheadError(
                "Stage 2 integrity: unequal dual-class ratio stock marked investable."
            )

    # 3. Positive prices and split-adjusted shares where present
    priced = df[df["has_market_price"]]
    if not priced.empty:
        if not (priced["raw_close"] > 0).all():
            raise PitLookaheadError("Stage 2 integrity: non-positive raw_close.")
        if not (priced["total_shares_owned_pit"] > 0).all():
            raise PitLookaheadError("Stage 2 integrity: non-positive total_shares_owned_pit.")


# ---------------------------------------------------------------------------
# Comprehensive Stage 1 + Stage 2 Empirical QA Audit
# ---------------------------------------------------------------------------


def run_stage2_qa_audit(
    ledger: PitHoldingsLedger,
    market_store: PitMarketStore,
) -> dict[str, Any]:
    """Run empirical QA across real splits, straddling dates, and historical cross-sections."""
    # 1. Verify split-ex-date continuity on flagship real stock splits
    split_cases = [
        ("AAPL", "2020-08-28", "2020-08-31", 4.0),
        ("AMZN", "2022-06-03", "2022-06-06", 20.0),
        ("GOOGL", "2022-07-15", "2022-07-18", 20.0),
        ("NVDA", "2024-06-07", "2024-06-10", 10.0),
    ]
    split_audit_rows: list[dict[str, Any]] = []
    for tkr, d_pre, d_ex, expected_s in split_cases:
        s_pre = build_pit_security_snapshot(d_pre, ledger, market_store, verify=True)
        s_ex = build_pit_security_snapshot(d_ex, ledger, market_store, verify=True)
        r_pre = s_pre[s_pre["yahoo_ticker"] == tkr]
        r_ex = s_ex[s_ex["yahoo_ticker"] == tkr]
        if r_pre.empty or r_ex.empty:
            continue
        p0 = r_pre.iloc[0]
        p1 = r_ex.iloc[0]
        # True 1-day return from market_prices on d_ex
        mp_ex = market_store.get_price_row_on_date(tkr, pd.Timestamp(d_ex))
        true_ret = float(mp_ex["ret_total"]) if mp_ex is not None else np.nan
        mcap_ret = (p1["pit_mcap_usd"] / p0["pit_mcap_usd"] - 1.0) if (
            pd.notna(p0["pit_mcap_usd"]) and pd.notna(p1["pit_mcap_usd"])
        ) else np.nan
        pos_ret = p1["total_position_usd_pit"] / p0["total_position_usd_pit"] - 1.0
        pct0 = p0["total_shares_owned_pit"] / p0["shares_out_pit"]
        pct1 = p1["total_shares_owned_pit"] / p1["shares_out_pit"]
        pct_diff = abs(pct1 - pct0)
        # Allow tiny diff if a 13F or 10-Q happened to arrive on d_ex itself
        same_filings = (
            p0["latest_filing_date"] == p1["latest_filing_date"]
            and p0["shares_filed_date"] == p1["shares_filed_date"]
        )
        mcap_err = abs(mcap_ret - true_ret) if (same_filings and pd.notna(mcap_ret)) else 0.0
        if same_filings and mcap_err > 0.01:
            raise PitLookaheadError(
                f"Split continuity FAILED for {tkr} ({d_pre}->{d_ex}): "
                f"mcap_ret={mcap_ret:.4f} vs true_ret={true_ret:.4f}"
            )
        if same_filings and pct_diff > 1e-6:
            raise PitLookaheadError(
                f"Ownership ratio continuity FAILED for {tkr} ({d_pre}->{d_ex}): "
                f"pct0={pct0:.6f} vs pct1={pct1:.6f}"
            )
        split_audit_rows.append(
            {
                "ticker": tkr,
                "pre_date": d_pre,
                "ex_date": d_ex,
                "split_factor": expected_s,
                "raw_close_pre": round(float(p0["raw_close"]), 2),
                "raw_close_ex": round(float(p1["raw_close"]), 2),
                "mcap_B_pre": round(float(p0["pit_mcap_usd"]) / 1e9, 1),
                "mcap_B_ex": round(float(p1["pit_mcap_usd"]) / 1e9, 1),
                "mcap_ret": round(float(mcap_ret), 4),
                "true_ret": round(float(true_ret), 4),
                "pos_ret": round(float(pos_ret), 4),
                "pct_owned_pre": round(float(pct0) * 100, 4),
                "pct_owned_ex": round(float(pct1) * 100, 4),
                "same_filings": same_filings,
            }
        )

    # 2. Sample 24 semi-annual + split-straddling trading dates across 2015-2026
    cal = pd.DatetimeIndex(market_store.spy_prices["date"])
    target_dates = [
        "2015-03-02", "2015-08-17", "2016-05-17", "2017-05-16", "2018-05-16",
        "2019-05-16", "2020-05-18", "2020-11-10", "2021-05-18", "2022-05-17",
        "2022-08-16", "2022-11-10", "2023-05-16", "2024-05-16", "2024-08-12",
        "2024-08-16", "2025-05-16", "2026-05-18",
    ]
    sample_stats: list[dict[str, Any]] = []
    anomalies_pct_gt_50: list[dict[str, Any]] = []
    straddle_examples: list[dict[str, Any]] = []
    stale_xbrl_tickers: set[str] = set()
    dual_class_tickers: set[str] = set()
    unequal_dual_tickers: set[str] = set()

    for d_str in target_dates:
        dt = pd.Timestamp(d_str)
        # Snap to nearest trading day on or after dt
        pos = int(cal.searchsorted(dt, side="left"))
        if pos >= len(cal):
            continue
        qdt = pd.Timestamp(cal[pos])
        snap = build_pit_security_snapshot(qdt, ledger, market_store, verify=True)

        valid_ratio = snap[snap["shares_out_pit"].notna() & (snap["shares_out_pit"] > 0)].copy()
        valid_ratio["pct_owned"] = (
            valid_ratio["total_shares_owned_pit"] / valid_ratio["shares_out_pit"]
        )

        high = valid_ratio[valid_ratio["pct_owned"] > 0.50]
        for r in high.to_dict(orient="records"):
            anomalies_pct_gt_50.append(
                {
                    "date": str(qdt.date()),
                    "ticker": r["yahoo_ticker"],
                    "n_filers": r["n_filers"],
                    "pct_owned": round(float(r["pct_owned"]) * 100, 2),
                    "mcap_B": round(float(r["pit_mcap_usd"]) / 1e9, 2)
                    if pd.notna(r["pit_mcap_usd"])
                    else None,
                    "shares_tag": r["shares_tag_source"],
                    "shares_age_d": r["shares_out_age_days"],
                    "is_dual": r["is_dual_class_issuer"],
                }
            )

        strad = snap[snap["has_split_straddle"]]
        for r in strad.to_dict(orient="records"):
            straddle_examples.append(
                {
                    "date": str(qdt.date()),
                    "ticker": r["yahoo_ticker"],
                    "n_filers": r["n_filers"],
                    "min_k": round(float(r["min_filer_split_factor"]), 4),
                    "max_k": round(float(r["max_filer_split_factor"]), 4),
                    "raw_sum_M": round(float(r["total_shares_raw"]) / 1e6, 3),
                    "pit_sum_M": round(float(r["total_shares_owned_pit"]) / 1e6, 3),
                }
            )

        stale_xbrl_tickers.update(snap.loc[snap["is_shares_out_stale"], "yahoo_ticker"].tolist())
        dual_class_tickers.update(snap.loc[snap["is_dual_class_issuer"], "yahoo_ticker"].tolist())
        unequal_dual_tickers.update(
            snap.loc[snap["has_unequal_class_ratio"], "yahoo_ticker"].tolist()
        )

        sample_stats.append(
            {
                "date": str(qdt.date()),
                "spy_mult": round(float(snap["spy_threshold_scale"].iloc[0]), 2)
                if not snap.empty
                else np.nan,
                "mcap_floor_B": round(float(snap["min_mcap_threshold_usd"].iloc[0]) / 1e9, 2)
                if not snap.empty
                else np.nan,
                "adv_floor_M": round(float(snap["min_adv_threshold_usd"].iloc[0]) / 1e6, 1)
                if not snap.empty
                else np.nan,
                "n_sec": len(snap),
                "n_priced": int(snap["has_market_price"].sum()),
                "n_seasoned": int(snap["is_seasoned"].sum()),
                "n_mcap": int(snap["passes_mcap_gate"].sum()),
                "n_adv": int(snap["passes_adv_gate"].sum()),
                "n_investable": int(snap["is_investable"].sum()),
                "n_split_straddle": int(snap["has_split_straddle"].sum()),
                "n_stale_xbrl": int(snap["is_shares_out_stale"].sum()),
                "n_missing_xbrl": int(
                    (snap["has_market_price"] & snap["shares_out_pit"].isna()).sum()
                ),
            }
        )

    return {
        "split_continuity": pd.DataFrame(split_audit_rows),
        "sample_dates": pd.DataFrame(sample_stats),
        "anomalies_pct_gt_50": pd.DataFrame(anomalies_pct_gt_50),
        "split_straddle_examples": pd.DataFrame(straddle_examples),
        "stale_xbrl_tickers": sorted(t for t in stale_xbrl_tickers if t),
        "dual_class_tickers": sorted(t for t in dual_class_tickers if t),
        "unequal_dual_tickers": sorted(t for t in unequal_dual_tickers if t),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and QA-audit Stage 1 (security aggregation) and Stage 2 (PIT market)."
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
        help="Run full Stage 1 & Stage 2 QA audit across real splits and historical dates.",
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

    snap = build_pit_security_snapshot(args.query_date, ledger, store, verify=True)
    print(
        f"Stage 1+2 Snapshot ({args.query_date}): "
        f"{len(snap):,} US Common Securities | "
        f"priced={snap['has_market_price'].sum():,} | "
        f"seasoned(>=252d)={snap['is_seasoned'].sum():,} | "
        f"investable={snap['is_investable'].sum():,}"
    )
    if not snap.empty:
        cols = [
            "yahoo_ticker",
            "n_filers",
            "raw_close",
            "spy_adj_close",
            "total_shares_owned_pit",
            "shares_out_pit",
            "pit_mcap_usd",
            "trading_days_history",
            "is_investable",
        ]
        print(snap[cols].head(10).to_string(index=False))

    if args.audit:
        print("\n=== RUNNING STAGE 1 & STAGE 2 QA AUDIT ===")
        qa = run_stage2_qa_audit(ledger, store)
        print("\n1. Real Stock Split Ex-Date Continuity (T-1 vs T):")
        print(qa["split_continuity"].to_string(index=False))
        print("\n2. Cross-Sectional Universe Funnel Across 2015-2026:")
        print(qa["sample_dates"].to_string(index=False))
        print(f"\n3. Real Split-Straddle Dates Caught ({len(qa['split_straddle_examples'])}):")
        if not qa["split_straddle_examples"].empty:
            print(qa["split_straddle_examples"].head(15).to_string(index=False))
        print(f"\n4. Ownership Ratio > 50% Anomalies ({len(qa['anomalies_pct_gt_50'])}):")
        if not qa["anomalies_pct_gt_50"].empty:
            print(qa["anomalies_pct_gt_50"].head(20).to_string(index=False))
        print(
            f"\n5. Stale XBRL Tickers Flagged & Nulled ({len(qa['stale_xbrl_tickers'])}): "
            f"{qa['stale_xbrl_tickers']}"
        )
        print(
            f"6. Dual-Class Issuer Tickers ({len(qa['dual_class_tickers'])}): "
            f"{qa['dual_class_tickers']}"
        )
        print(
            f"7. Unequal Economic-Ratio Dual-Class Tickers ({len(qa['unequal_dual_tickers'])}): "
            f"{qa['unequal_dual_tickers']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
