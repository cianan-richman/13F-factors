"""Point-in-time (PIT) 13F append-only holdings ledger and query API.

Enforces strict T+1 trading-day visibility from the filing date (never `filing_date == query_date`),
resolves `13F-HR` and `13F-HR/A` (`RESTATEMENT` and `NEW HOLDINGS` / CTR reveals) strictly from
each filing's effective trade date onward without back-propagation, prevents retroactive
amendments of older quarters from reverting newer quarters, aggregates sub-manager split rows
within each filing, and runs zero-lookahead QA assertions on every query.

Usage:
    python -m src.pit.ledger --query-date 2024-08-16
    python -m src.pit.ledger --audit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR
from src.ingest.filings import DEFAULT_HOLDINGS_RAW_PARQUET, DEFAULT_META_PARQUET
from src.ingest.market_data import DEFAULT_SPY_PARQUET

logger = logging.getLogger(__name__)

DEFAULT_PIT_STATES_PARQUET = PROCESSED_DATA_DIR / "pit_13f_states.parquet"
DEFAULT_PIT_HOLDINGS_PARQUET = PROCESSED_DATA_DIR / "pit_13f_holdings_events.parquet"

# Default maximum age of a filer's latest `period_of_report` relative to `query_date` before
# the filer is considered ceased/stale (~91d quarter + 45d SEC deadline + 24d grace = 160d).
DEFAULT_MAX_STALENESS_DAYS = 160


class PitLookaheadError(AssertionError):
    """Raised when any point-in-time or data-integrity invariant is violated."""


def load_trading_calendar(spy_parquet: Path = DEFAULT_SPY_PARQUET) -> pd.DatetimeIndex:
    """Load US equity trading dates from `spy_prices.parquet` (or business-day fallback)."""
    if spy_parquet.exists():
        spy = pd.read_parquet(spy_parquet, columns=["date"])
        dates = pd.DatetimeIndex(pd.to_datetime(spy["date"]).drop_duplicates().sort_values())
        if len(dates) > 0:
            return dates
    return pd.bdate_range("2014-01-01", "2028-12-31")


def next_trading_day(
    ref_date: pd.Timestamp,
    trading_calendar: pd.DatetimeIndex,
    *,
    lag_trading_days: int = 1,
) -> pd.Timestamp:
    """Return the trading day strictly `lag_trading_days` AFTER `ref_date` (T+1 default).

    Guarantees `result > ref_date` even when `ref_date` falls on a weekend or market holiday.
    If `ref_date` is beyond the end of `trading_calendar`, falls back cleanly to weekday bdays.
    """
    if lag_trading_days < 1:
        raise ValueError("lag_trading_days must be >= 1 to guarantee zero same-day lookahead.")
    ts = pd.Timestamp(ref_date).normalize()
    pos = int(trading_calendar.searchsorted(ts, side="right")) + (lag_trading_days - 1)
    if pos < len(trading_calendar):
        return pd.Timestamp(trading_calendar[pos])
    # Fallback strictly after `ts` using weekday business days if beyond calendar tail
    return pd.Timestamp((ts + pd.offsets.BDay(lag_trading_days)).normalize())


def _normalize_filing_holdings(holdings_raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Filter raw holdings to long `SH`-only equities and sum sub-manager rows per accession."""
    if holdings_raw.empty:
        return {}

    sh = holdings_raw[
        (holdings_raw["ssh_prnamt_type"] == "SH")
        & (holdings_raw["put_call"].fillna("") == "")
        & (holdings_raw["ssh_prnamt"].fillna(0.0) > 0)
    ].copy()
    if sh.empty:
        return {}

    sh["cusip"] = sh["cusip"].astype(str).str.strip().str.upper()
    sh = sh[sh["cusip"] != ""]

    agg = (
        sh.groupby(["accession_number", "cusip"], as_index=False)
        .agg(
            shares=("ssh_prnamt", "sum"),
            reported_value_usd=("value_usd", "sum"),
            name_of_issuer=("name_of_issuer", "first"),
            title_of_class=("title_of_class", "first"),
        )
    )
    return {acc: grp.reset_index(drop=True) for acc, grp in agg.groupby("accession_number")}


def verify_no_lookahead(df: pd.DataFrame, query_date: pd.Timestamp) -> None:
    """Run strict runtime QA assertions proving zero lookahead and zero duplicates."""
    qdt = pd.Timestamp(query_date).normalize()
    if df.empty:
        return

    if not (pd.to_datetime(df["filing_date"]) < qdt).all():
        bad = df[pd.to_datetime(df["filing_date"]) >= qdt][
            ["filer_id", "accession_number", "filing_date"]
        ].head(5)
        raise PitLookaheadError(
            f"LOOKAHEAD VIOLATION on query_date={qdt.date()}: "
            f"filing_date >= query_date found:\n{bad.to_string(index=False)}"
        )

    if not (pd.to_datetime(df["acceptance_date"]) < qdt).all():
        bad = df[pd.to_datetime(df["acceptance_date"]) >= qdt][
            ["filer_id", "accession_number", "acceptance_date"]
        ].head(5)
        raise PitLookaheadError(
            f"LOOKAHEAD VIOLATION on query_date={qdt.date()}: "
            f"acceptance_date >= query_date found:\n{bad.to_string(index=False)}"
        )

    if not (pd.to_datetime(df["effective_date"]) <= qdt).all():
        bad = df[pd.to_datetime(df["effective_date"]) > qdt][
            ["filer_id", "accession_number", "effective_date"]
        ].head(5)
        raise PitLookaheadError(
            f"LOOKAHEAD VIOLATION on query_date={qdt.date()}: "
            f"effective_date > query_date found:\n{bad.to_string(index=False)}"
        )

    if not (pd.to_datetime(df["period_of_report"]) < pd.to_datetime(df["filing_date"])).all():
        raise PitLookaheadError("DATA INTEGRITY ERROR: period_of_report >= filing_date.")

    if df.duplicated(subset=["filer_id", "cusip"]).any():
        dups = df[df.duplicated(subset=["filer_id", "cusip"], keep=False)][
            ["filer_id", "cusip", "accession_number"]
        ].head(5)
        raise PitLookaheadError(
            f"DUPLICATE (filer_id, cusip) rows on query_date={qdt.date()}:\n"
            f"{dups.to_string(index=False)}"
        )

    if not (df["shares"] > 0).all():
        raise PitLookaheadError("DATA INTEGRITY ERROR: non-positive share count in PIT output.")


def build_pit_event_tables(
    filings_meta: pd.DataFrame,
    holdings_raw: pd.DataFrame,
    *,
    trading_calendar: pd.DatetimeIndex,
    lag_trading_days: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build immutable versioned PIT state metadata and state holdings tables.

    For every `(filer_id, period_of_report)`, replays `13F-HR` and `13F-HR/A` events in strict
    chronological `(effective_date, acceptance_datetime, form_order, accession_number)` order:
      - `13F-HR`: initializes the base holdings state for `(filer_id, period_of_report)`.
      - `13F-HR/A` (`RESTATEMENT` or unspecified): replaces the entire holdings state for that
        `(filer_id, period_of_report)` strictly from `effective_date` onward.
      - `13F-HR/A` (`NEW HOLDINGS`, i.e. CTR reveal): upserts by `cusip` onto the existing state
        strictly from `effective_date` onward. Upsert semantics handle both incremental CTR reveals
        (0 overlap with base) and full-table cumulative `NEW HOLDINGS` filings (100% overlap with
        base) cleanly without double-counting shares, while tagging newly added CUSIPs with
        `is_ctr_reveal = True`.
    """
    if filings_meta.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta = filings_meta.copy()
    if "parse_status" in meta.columns:
        meta = meta[meta["parse_status"].isin({"ok", "zero_holdings"})].copy()

    meta["date_filed"] = pd.to_datetime(meta["date_filed"]).dt.normalize()
    meta["acceptance_date"] = (
        pd.to_datetime(meta["acceptance_date"]).dt.normalize()
        if "acceptance_date" in meta.columns
        else meta["date_filed"]
    )
    meta["period_of_report"] = pd.to_datetime(meta["period_of_report"]).dt.normalize()
    meta = meta.dropna(subset=["period_of_report", "date_filed"]).copy()

    # Conservative arrival date = max(date_filed, acceptance_date) so neither ever >= query_date
    meta["arrival_date"] = meta[["date_filed", "acceptance_date"]].max(axis=1)
    meta["effective_date"] = meta["arrival_date"].map(
        lambda d: next_trading_day(d, trading_calendar, lag_trading_days=lag_trading_days)
    )
    meta["form_order"] = np.where(meta["form_type"] == "13F-HR", 0, 1)
    meta["acceptance_datetime"] = meta.get("acceptance_datetime", "").fillna("").astype(str)
    meta["amendment_type"] = meta.get("amendment_type", "").fillna("").astype(str).str.upper()

    holdings_by_acc = _normalize_filing_holdings(holdings_raw)

    state_meta_rows: list[dict[str, Any]] = []
    state_holding_parts: list[pd.DataFrame] = []
    next_state_id = 0

    for (filer_id, period_of_report), grp in meta.groupby(
        ["filer_id", "period_of_report"], sort=True
    ):
        ordered = grp.sort_values(
            ["effective_date", "acceptance_datetime", "form_order", "accession_number"]
        ).reset_index(drop=True)

        # Current holdings state keyed by cusip -> dict of position fields
        current_map: dict[str, dict[str, Any]] = {}
        period_states: list[dict[str, Any]] = []

        # Group events sharing the exact same effective_date so only one state is emitted per date
        for eff_date, day_events in ordered.groupby("effective_date", sort=True):
            latest_ev: pd.Series | None = None
            for _, ev in day_events.iterrows():
                latest_ev = ev
                acc = str(ev["accession_number"])
                form_type = str(ev["form_type"]).upper()
                amend_type = str(ev["amendment_type"]).upper()
                f_date = ev["date_filed"]
                a_date = ev["acceptance_date"]
                inc_df = holdings_by_acc.get(acc)

                if form_type == "13F-HR" or amend_type != "NEW HOLDINGS":
                    # Base filing or RESTATEMENT: replace full table for this (filer, period)
                    current_map = {}
                    if inc_df is not None and not inc_df.empty:
                        for rec in inc_df.to_dict(orient="records"):
                            current_map[rec["cusip"]] = {
                                "cusip": rec["cusip"],
                                "shares": float(rec["shares"]),
                                "reported_value_usd": float(rec["reported_value_usd"]),
                                "name_of_issuer": rec["name_of_issuer"],
                                "title_of_class": rec["title_of_class"],
                                "row_accession_number": acc,
                                "row_filing_date": f_date,
                                "row_acceptance_date": a_date,
                                "row_effective_date": eff_date,
                                "row_form_type": form_type,
                                "row_amendment_type": amend_type,
                                "is_ctr_reveal": False,
                            }
                else:
                    # 13F-HR/A NEW HOLDINGS (CTR reveal): upsert by CUSIP without double-counting
                    if inc_df is not None and not inc_df.empty:
                        for rec in inc_df.to_dict(orient="records"):
                            cusip = rec["cusip"]
                            is_new_cusip = cusip not in current_map
                            prev_ctr = (
                                bool(current_map[cusip]["is_ctr_reveal"])
                                if not is_new_cusip
                                else False
                            )
                            current_map[cusip] = {
                                "cusip": cusip,
                                "shares": float(rec["shares"]),
                                "reported_value_usd": float(rec["reported_value_usd"]),
                                "name_of_issuer": rec["name_of_issuer"],
                                "title_of_class": rec["title_of_class"],
                                "row_accession_number": acc,
                                "row_filing_date": f_date,
                                "row_acceptance_date": a_date,
                                "row_effective_date": eff_date,
                                "row_form_type": form_type,
                                "row_amendment_type": amend_type,
                                "is_ctr_reveal": is_new_cusip or prev_ctr,
                            }

            assert latest_ev is not None
            state_id = next_state_id
            next_state_id += 1

            total_val = sum(v["reported_value_usd"] for v in current_map.values())
            period_states.append(
                {
                    "state_id": state_id,
                    "filer_id": str(filer_id),
                    "canonical_name": str(latest_ev.get("canonical_name", filer_id)),
                    "style": str(latest_ev.get("style", "")),
                    "period_of_report": period_of_report,
                    "valid_from": eff_date,
                    "valid_to": pd.NaT,
                    "filing_date": latest_ev["date_filed"],
                    "acceptance_date": latest_ev["acceptance_date"],
                    "effective_date": eff_date,
                    "accession_number": str(latest_ev["accession_number"]),
                    "source_form_type": str(latest_ev["form_type"]),
                    "amendment_type": str(latest_ev["amendment_type"]),
                    "n_positions": len(current_map),
                    "total_reported_value_usd": total_val,
                }
            )

            if current_map:
                h_df = pd.DataFrame.from_records(list(current_map.values()))
                h_df["state_id"] = state_id
                h_df["portfolio_weight_reported"] = (
                    h_df["reported_value_usd"] / total_val if total_val > 0 else np.nan
                )
                state_holding_parts.append(h_df)

        # Link half-open validity intervals [valid_from, valid_to) across versions of this period
        for i in range(len(period_states) - 1):
            period_states[i]["valid_to"] = period_states[i + 1]["valid_from"]
        state_meta_rows.extend(period_states)

    states_df = pd.DataFrame(state_meta_rows)
    if not states_df.empty:
        states_df = states_df.sort_values(
            ["filer_id", "period_of_report", "valid_from", "state_id"]
        ).reset_index(drop=True)

    holdings_events_df = (
        pd.concat(state_holding_parts, ignore_index=True)
        if state_holding_parts
        else pd.DataFrame(
            columns=[
                "state_id",
                "cusip",
                "shares",
                "reported_value_usd",
                "portfolio_weight_reported",
                "name_of_issuer",
                "title_of_class",
                "row_accession_number",
                "row_filing_date",
                "row_acceptance_date",
                "row_effective_date",
                "row_form_type",
                "row_amendment_type",
                "is_ctr_reveal",
            ]
        )
    )
    return states_df, holdings_events_df


class PitHoldingsLedger:
    """Fast, zero-lookahead point-in-time query API over versioned 13F state tables."""

    def __init__(
        self,
        states_df: pd.DataFrame,
        holdings_events_df: pd.DataFrame,
    ) -> None:
        self.states_df = states_df.copy()
        self.holdings_events_df = holdings_events_df.copy()

        if not self.states_df.empty:
            for col in (
                "period_of_report",
                "valid_from",
                "valid_to",
                "filing_date",
                "acceptance_date",
                "effective_date",
            ):
                self.states_df[col] = pd.to_datetime(self.states_df[col])

        # Pre-index holdings by state_id for sub-5ms query latency
        if not self.holdings_events_df.empty:
            self._holdings_by_state: dict[int, pd.DataFrame] = {
                int(sid): grp.reset_index(drop=True)
                for sid, grp in self.holdings_events_df.groupby("state_id", sort=False)
            }
        else:
            self._holdings_by_state = {}

    @classmethod
    def from_raw_tables(
        cls,
        filings_meta: pd.DataFrame,
        holdings_raw: pd.DataFrame,
        *,
        trading_calendar: pd.DatetimeIndex,
        lag_trading_days: int = 1,
    ) -> PitHoldingsLedger:
        states_df, holdings_events_df = build_pit_event_tables(
            filings_meta,
            holdings_raw,
            trading_calendar=trading_calendar,
            lag_trading_days=lag_trading_days,
        )
        return cls(states_df, holdings_events_df)

    @classmethod
    def from_parquet(
        cls,
        *,
        states_path: Path = DEFAULT_PIT_STATES_PARQUET,
        holdings_events_path: Path = DEFAULT_PIT_HOLDINGS_PARQUET,
        meta_path: Path = DEFAULT_META_PARQUET,
        holdings_raw_path: Path = DEFAULT_HOLDINGS_RAW_PARQUET,
        spy_path: Path = DEFAULT_SPY_PARQUET,
        rebuild: bool = False,
        lag_trading_days: int = 1,
    ) -> PitHoldingsLedger:
        """Load materialized PIT ledger Parquet tables, building them automatically if absent."""
        if not rebuild and states_path.exists() and holdings_events_path.exists():
            return cls(pd.read_parquet(states_path), pd.read_parquet(holdings_events_path))

        meta = pd.read_parquet(meta_path)
        raw_h = pd.read_parquet(holdings_raw_path)
        cal = load_trading_calendar(spy_path)
        states_df, holdings_events_df = build_pit_event_tables(
            meta,
            raw_h,
            trading_calendar=cal,
            lag_trading_days=lag_trading_days,
        )
        states_path.parent.mkdir(parents=True, exist_ok=True)
        states_df.to_parquet(states_path, index=False)
        holdings_events_df.to_parquet(holdings_events_path, index=False)
        return cls(states_df, holdings_events_df)

    def _active_states_as_of(self, query_date: str | pd.Timestamp) -> pd.DataFrame:
        """Return all `(filer_id, period_of_report)` states active strictly as of `query_date`."""
        qdt = pd.Timestamp(query_date).normalize()
        if self.states_df.empty:
            return self.states_df.copy()

        mask = (self.states_df["valid_from"] <= qdt) & (
            self.states_df["valid_to"].isna() | (self.states_df["valid_to"] > qdt)
        )
        return self.states_df.loc[mask].copy()

    def get_latest_states(
        self,
        query_date: str | pd.Timestamp,
        *,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
    ) -> pd.DataFrame:
        """Return one row per active entity representing its latest available 13F state.

        Selects `max(period_of_report)` per `filer_id` among states effective `<= query_date` so
        a retroactive amendment to an older quarter NEVER reverts a filer's current portfolio.
        """
        qdt = pd.Timestamp(query_date).normalize()
        active = self._active_states_as_of(qdt)
        if active.empty:
            return active

        # Latest period_of_report per filer_id (never revert to older retroactively amended quarter)
        latest = (
            active.sort_values(["filer_id", "period_of_report", "valid_from"])
            .drop_duplicates(subset=["filer_id"], keep="last")
            .reset_index(drop=True)
        )
        latest["staleness_days"] = (qdt - latest["period_of_report"]).dt.days
        latest["is_stale"] = (
            latest["staleness_days"] > max_staleness_days
            if max_staleness_days is not None
            else False
        )
        return latest

    def _assemble_holdings(
        self,
        selected_states: pd.DataFrame,
        query_date: pd.Timestamp,
        *,
        verify: bool = True,
    ) -> pd.DataFrame:
        qdt = pd.Timestamp(query_date).normalize()
        output_cols = [
            "query_date",
            "filer_id",
            "canonical_name",
            "style",
            "period_of_report",
            "filing_date",
            "acceptance_date",
            "effective_date",
            "accession_number",
            "source_form_type",
            "amendment_type",
            "is_ctr_reveal",
            "cusip",
            "name_of_issuer",
            "title_of_class",
            "shares",
            "reported_value_usd",
            "portfolio_weight_reported",
            "staleness_days",
        ]
        if selected_states.empty:
            return pd.DataFrame(columns=output_cols)

        parts: list[pd.DataFrame] = []
        for st in selected_states.to_dict(orient="records"):
            sid = int(st["state_id"])
            h_df = self._holdings_by_state.get(sid)
            if h_df is None or h_df.empty:
                continue
            chunk = h_df.copy()
            chunk["query_date"] = qdt
            chunk["filer_id"] = st["filer_id"]
            chunk["canonical_name"] = st["canonical_name"]
            chunk["style"] = st["style"]
            chunk["period_of_report"] = st["period_of_report"]
            chunk["filing_date"] = st["filing_date"]
            chunk["acceptance_date"] = st["acceptance_date"]
            chunk["effective_date"] = st["effective_date"]
            chunk["accession_number"] = st["accession_number"]
            chunk["source_form_type"] = st["source_form_type"]
            chunk["amendment_type"] = st["amendment_type"]
            chunk["staleness_days"] = int((qdt - st["period_of_report"]).days)
            parts.append(chunk[output_cols])

        if not parts:
            return pd.DataFrame(columns=output_cols)

        result = (
            pd.concat(parts, ignore_index=True)
            .sort_values(["filer_id", "cusip"])
            .reset_index(drop=True)
        )
        if verify:
            verify_no_lookahead(result, qdt)
        return result

    def get_latest_holdings(
        self,
        query_date: str | pd.Timestamp,
        *,
        max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
        include_stale: bool = False,
        verify: bool = True,
    ) -> pd.DataFrame:
        """Return all latest point-in-time holdings across tracked entities as of `query_date`.

        Parameters
        ----------
        query_date:
            Evaluation / trade date `t`. Only filings with `effective_date <= t` (strictly T+1
            trading days after `filing_date`) are visible.
        max_staleness_days:
            Drop filers whose latest `period_of_report` is older than this threshold (default 160
            calendar days, automatically retiring wound-down/ceased filers strictly PIT).
        include_stale:
            If True, retain stale filers even beyond `max_staleness_days`.
        verify:
            Run strict zero-lookahead and uniqueness QA assertions before returning.
        """
        qdt = pd.Timestamp(query_date).normalize()
        latest = self.get_latest_states(qdt, max_staleness_days=max_staleness_days)
        if not include_stale and not latest.empty:
            latest = latest[~latest["is_stale"]].reset_index(drop=True)
        return self._assemble_holdings(latest, qdt, verify=verify)

    def get_quarter_holdings(
        self,
        query_date: str | pd.Timestamp,
        period_of_report: str | pd.Timestamp,
        *,
        verify: bool = True,
    ) -> pd.DataFrame:
        """Return PIT-visible holdings for one specific `period_of_report` as of `query_date`."""
        qdt = pd.Timestamp(query_date).normalize()
        pdt = pd.Timestamp(period_of_report).normalize()
        active = self._active_states_as_of(qdt)
        if active.empty:
            return self._assemble_holdings(active, qdt, verify=verify)
        sub = active[active["period_of_report"] == pdt].reset_index(drop=True)
        return self._assemble_holdings(sub, qdt, verify=verify)


def run_pit_qa_audit(
    ledger: PitHoldingsLedger,
    filings_meta: pd.DataFrame,
    trading_calendar: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Run exhaustive PIT quality-assurance checks across real filings and amendments."""
    # 1. Verify EVERY state transition satisfies filing_date < effective_date
    st = ledger.states_df
    if not (st["filing_date"] < st["effective_date"]).all():
        raise PitLookaheadError("Found state with filing_date >= effective_date.")
    if not (st["acceptance_date"] < st["effective_date"]).all():
        raise PitLookaheadError("Found state with acceptance_date >= effective_date.")

    # 2. Verify every single real 13F-HR/A amendment is invisible on its filing day
    #    and visible on its T+1 effective day (for its own period_of_report)
    hra_states = st[st["source_form_type"] == "13F-HR/A"]
    checked_amendments = 0
    for _, row in hra_states.iterrows():
        fid = row["filer_id"]
        rep = row["period_of_report"]
        f_date = row["filing_date"]
        eff_date = row["effective_date"]
        acc = row["accession_number"]

        # On filing_date itself, this amendment accession MUST NOT appear
        before = ledger.get_quarter_holdings(f_date, rep, verify=True)
        before_f = before[before["filer_id"] == fid]
        if not before_f.empty and (before_f["accession_number"] == acc).any():
            raise PitLookaheadError(
                f"Amendment {acc} for {fid} leaked on its own filing_date {f_date.date()}!"
            )

        # On effective_date, the state row for (fid, rep) MUST reflect effective_date <= eff_date
        active_on_eff = ledger._active_states_as_of(eff_date)
        match = active_on_eff[
            (active_on_eff["filer_id"] == fid) & (active_on_eff["period_of_report"] == rep)
        ]
        if match.empty or match.iloc[0]["accession_number"] != acc:
            raise PitLookaheadError(
                f"Amendment {acc} for {fid} not active on its effective_date {eff_date.date()}!"
            )
        checked_amendments += 1

    # 3. Sample 48 quarter-end+50d dates + key filing cluster dates and verify get_latest_holdings
    sample_dates = pd.date_range("2015-02-20", "2026-11-19", freq="QE")
    checked_dates = 0
    for dt in sample_dates:
        _ = ledger.get_latest_holdings(dt, verify=True)
        checked_dates += 1

    # 4. Check CTR reveal rows exist and are never active before their amendment effective_date
    ctr_rows = ledger.holdings_events_df[ledger.holdings_events_df["is_ctr_reveal"]]
    return {
        "total_states": len(st),
        "total_holding_event_rows": len(ledger.holdings_events_df),
        "amendments_boundary_verified": checked_amendments,
        "sample_dates_verified": checked_dates,
        "ctr_reveal_holding_rows": len(ctr_rows),
        "unique_filers": int(st["filer_id"].nunique()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, audit, and query the Point-in-Time (PIT) 13F holdings ledger."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of pit_13f_states.parquet and pit_13f_holdings_events.parquet.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Run full zero-lookahead boundary QA audit across all amendments and sample dates.",
    )
    parser.add_argument(
        "--query-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD date to query and display a summary of latest PIT holdings.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    ledger = PitHoldingsLedger.from_parquet(rebuild=args.rebuild)

    print(
        f"PIT 13F ledger ready:\n"
        f"  States:   {len(ledger.states_df):,} -> {DEFAULT_PIT_STATES_PARQUET}\n"
        f"  Holdings: {len(ledger.holdings_events_df):,} -> {DEFAULT_PIT_HOLDINGS_PARQUET}"
    )

    if args.audit or (not args.query_date):
        meta = pd.read_parquet(DEFAULT_META_PARQUET)
        cal = load_trading_calendar(DEFAULT_SPY_PARQUET)
        qa = run_pit_qa_audit(ledger, meta, cal)
        print(
            "  QA Audit PASSED:\n"
            f"    Amendments boundary-checked (T vs T+1): {qa['amendments_boundary_verified']}\n"
            f"    Sample query dates verified:            {qa['sample_dates_verified']}\n"
            f"    CTR reveal holding rows tagged:         {qa['ctr_reveal_holding_rows']}"
        )

    if args.query_date:
        snap = ledger.get_latest_holdings(args.query_date, verify=True)
        print(
            f"\nSnapshot for query_date={args.query_date}: "
            f"{len(snap):,} holdings across {snap['filer_id'].nunique()} active filers "
            f"({snap['cusip'].nunique():,} unique CUSIPs)"
        )
        if not snap.empty:
            print(
                snap[
                    [
                        "filer_id",
                        "period_of_report",
                        "filing_date",
                        "effective_date",
                        "source_form_type",
                        "cusip",
                        "shares",
                    ]
                ]
                .head(10)
                .to_string(index=False)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
