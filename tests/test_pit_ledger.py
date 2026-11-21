"""Unit tests proving strict zero-lookahead PIT discipline in `src/pit/ledger.py`."""

from __future__ import annotations

import pandas as pd
import pytest

from src.pit.ledger import (
    PitHoldingsLedger,
    PitLookaheadError,
    next_trading_day,
    verify_no_lookahead,
)


def _calendar_2022() -> pd.DatetimeIndex:
    return pd.bdate_range("2022-01-03", "2023-12-29")


def test_exact_t_plus_1_and_restatement_amendment_example() -> None:
    """User's exact QA scenario:
    - Report date 2022-06-30 filed on 2022-08-14 -> available ONLY from 2022-08-15 onward.
    - Amended filing (RESTATEMENT) filed on 2022-08-25 -> available ONLY from 2022-08-26 onward.
    """
    cal = _calendar_2022()
    meta = pd.DataFrame(
        [
            {
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "growth_tmt_ls",
                "accession_number": "acc-base",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-08-14"),
                "acceptance_date": pd.Timestamp("2022-08-14"),
                "acceptance_datetime": "2022-08-14T16:15:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            {
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "growth_tmt_ls",
                "accession_number": "acc-amend",
                "form_type": "13F-HR/A",
                "date_filed": pd.Timestamp("2022-08-25"),
                "acceptance_date": pd.Timestamp("2022-08-25"),
                "acceptance_datetime": "2022-08-25T16:30:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "RESTATEMENT",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "filer_id": "fund_a",
                "accession_number": "acc-base",
                "cusip": "594918104",
                "name_of_issuer": "MICROSOFT CORP",
                "title_of_class": "COM",
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 25_000_000.0,
            },
            # Sub-manager split row for same CUSIP in base filing (must be summed, not duplicated)
            {
                "filer_id": "fund_a",
                "accession_number": "acc-base",
                "cusip": "594918104",
                "name_of_issuer": "MICROSOFT CORP",
                "title_of_class": "COM",
                "ssh_prnamt": 50_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 12_500_000.0,
            },
            # Put option row (must be excluded)
            {
                "filer_id": "fund_a",
                "accession_number": "acc-base",
                "cusip": "594918104",
                "name_of_issuer": "MICROSOFT CORP",
                "title_of_class": "COM",
                "ssh_prnamt": 999_999.0,
                "ssh_prnamt_type": "SH",
                "put_call": "PUT",
                "value_usd": 250_000_000.0,
            },
            # Restatement updates MSFT share count to 200,000
            {
                "filer_id": "fund_a",
                "accession_number": "acc-amend",
                "cusip": "594918104",
                "name_of_issuer": "MICROSOFT CORP",
                "title_of_class": "COM",
                "ssh_prnamt": 200_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 50_000_000.0,
            },
        ]
    )

    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    # On filing_date 2022-08-14 -> MUST NOT be visible (zero same-day lookahead!)
    snap_aug14 = ledger.get_latest_holdings("2022-08-14")
    assert snap_aug14.empty

    # On 2022-08-15 (T+1) -> base filing visible with summed sub-manager shares = 150,000
    snap_aug15 = ledger.get_latest_holdings("2022-08-15")
    assert len(snap_aug15) == 1
    assert snap_aug15.iloc[0]["accession_number"] == "acc-base"
    assert snap_aug15.iloc[0]["shares"] == 150_000.0

    # On amendment filing_date 2022-08-25 -> STILL sees base filing (150,000), NOT amendment!
    snap_aug25 = ledger.get_latest_holdings("2022-08-25")
    assert len(snap_aug25) == 1
    assert snap_aug25.iloc[0]["accession_number"] == "acc-base"
    assert snap_aug25.iloc[0]["shares"] == 150_000.0

    # On 2022-08-26 (T+1 after amendment) -> sees restated 200,000 shares!
    snap_aug26 = ledger.get_latest_holdings("2022-08-26")
    assert len(snap_aug26) == 1
    assert snap_aug26.iloc[0]["accession_number"] == "acc-amend"
    assert snap_aug26.iloc[0]["source_form_type"] == "13F-HR/A"
    assert snap_aug26.iloc[0]["shares"] == 200_000.0


def test_new_holdings_ctr_upsert_handles_both_incremental_and_cumulative_without_doubling() -> None:
    cal = _calendar_2022()
    meta = pd.DataFrame(
        [
            # Fund 1: incremental NEW HOLDINGS (1 new CUSIP only, 0 overlap with base)
            {
                "filer_id": "fund_inc",
                "canonical_name": "Fund Inc",
                "style": "value_contrarian_ls",
                "accession_number": "inc-base",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-08-15"),
                "acceptance_date": pd.Timestamp("2022-08-15"),
                "acceptance_datetime": "2022-08-15T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            {
                "filer_id": "fund_inc",
                "canonical_name": "Fund Inc",
                "style": "value_contrarian_ls",
                "accession_number": "inc-ctr",
                "form_type": "13F-HR/A",
                "date_filed": pd.Timestamp("2022-09-01"),
                "acceptance_date": pd.Timestamp("2022-09-01"),
                "acceptance_datetime": "2022-09-01T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "NEW HOLDINGS",
                "parse_status": "ok",
            },
            # Fund 2: cumulative NEW HOLDINGS (re-lists base CUSIP + 1 new CUSIP — must NOT double!)
            {
                "filer_id": "fund_cum",
                "canonical_name": "Fund Cum",
                "style": "quality_generalist",
                "accession_number": "cum-base",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-08-15"),
                "acceptance_date": pd.Timestamp("2022-08-15"),
                "acceptance_datetime": "2022-08-15T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            {
                "filer_id": "fund_cum",
                "canonical_name": "Fund Cum",
                "style": "quality_generalist",
                "accession_number": "cum-ctr",
                "form_type": "13F-HR/A",
                "date_filed": pd.Timestamp("2022-09-01"),
                "acceptance_date": pd.Timestamp("2022-09-01"),
                "acceptance_datetime": "2022-09-01T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "NEW HOLDINGS",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            # fund_inc base: MSFT
            {
                "filer_id": "fund_inc",
                "accession_number": "inc-base",
                "cusip": "594918104",
                "name_of_issuer": "MSFT",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000.0,
            },
            # fund_inc CTR reveal: AAPL only
            {
                "filer_id": "fund_inc",
                "accession_number": "inc-ctr",
                "cusip": "037833100",
                "name_of_issuer": "AAPL",
                "title_of_class": "COM",
                "ssh_prnamt": 250.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 35_000.0,
            },
            # fund_cum base: MSFT
            {
                "filer_id": "fund_cum",
                "accession_number": "cum-base",
                "cusip": "594918104",
                "name_of_issuer": "MSFT",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000.0,
            },
            # fund_cum cumulative NEW HOLDINGS: MSFT (repeated!) + AAPL (new!)
            {
                "filer_id": "fund_cum",
                "accession_number": "cum-ctr",
                "cusip": "594918104",
                "name_of_issuer": "MSFT",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000.0,
            },
            {
                "filer_id": "fund_cum",
                "accession_number": "cum-ctr",
                "cusip": "037833100",
                "name_of_issuer": "AAPL",
                "title_of_class": "COM",
                "ssh_prnamt": 250.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 35_000.0,
            },
        ]
    )

    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    # Before CTR reveal (2022-09-01): neither fund shows AAPL (no back-propagation!)
    before = ledger.get_latest_holdings("2022-09-01")
    assert set(before["cusip"]) == {"594918104"}

    # After CTR reveal (2022-09-02): BOTH funds have exact MSFT=100 (NOT 200!) and AAPL=250 (CTR!)
    after = ledger.get_latest_holdings("2022-09-02")
    for fid in ("fund_inc", "fund_cum"):
        sub = after[after["filer_id"] == fid].set_index("cusip")
        assert sub.loc["594918104", "shares"] == 100.0
        assert bool(sub.loc["594918104", "is_ctr_reveal"]) is False
        assert sub.loc["037833100", "shares"] == 250.0
        assert bool(sub.loc["037833100", "is_ctr_reveal"]) is True


def test_retroactive_amendment_of_older_quarter_never_reverts_latest_quarter() -> None:
    cal = _calendar_2022()
    meta = pd.DataFrame(
        [
            {
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality_generalist",
                "accession_number": "q1-base",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-05-16"),
                "acceptance_date": pd.Timestamp("2022-05-16"),
                "acceptance_datetime": "2022-05-16T16:00:00",
                "period_of_report": pd.Timestamp("2022-03-31"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            {
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality_generalist",
                "accession_number": "q2-base",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-08-15"),
                "acceptance_date": pd.Timestamp("2022-08-15"),
                "acceptance_datetime": "2022-08-15T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            # Retroactive restatement of Q1 2022 filed in October 2022 (AFTER Q2 was filed!)
            {
                "filer_id": "fund_a",
                "canonical_name": "Fund A",
                "style": "quality_generalist",
                "accession_number": "q1-retro-amend",
                "form_type": "13F-HR/A",
                "date_filed": pd.Timestamp("2022-10-03"),
                "acceptance_date": pd.Timestamp("2022-10-03"),
                "acceptance_datetime": "2022-10-03T16:00:00",
                "period_of_report": pd.Timestamp("2022-03-31"),
                "amendment_type": "RESTATEMENT",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "filer_id": "fund_a",
                "accession_number": "q1-base",
                "cusip": "111111111",
                "name_of_issuer": "OLD CO",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 1000.0,
            },
            {
                "filer_id": "fund_a",
                "accession_number": "q2-base",
                "cusip": "222222222",
                "name_of_issuer": "NEW CO",
                "title_of_class": "COM",
                "ssh_prnamt": 500.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 5000.0,
            },
            {
                "filer_id": "fund_a",
                "accession_number": "q1-retro-amend",
                "cusip": "111111111",
                "name_of_issuer": "OLD CO",
                "title_of_class": "COM",
                "ssh_prnamt": 120.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 1200.0,
            },
        ]
    )

    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    # On 2022-10-04, latest holdings MUST STILL be Q2 (2022-06-30, NEW CO), NOT reverted to Q1!
    latest = ledger.get_latest_holdings("2022-10-04")
    assert len(latest) == 1
    assert latest.iloc[0]["period_of_report"] == pd.Timestamp("2022-06-30")
    assert latest.iloc[0]["cusip"] == "222222222"

    # Querying Q1 specifically as of 2022-10-04 reflects the retroactive restatement (120)
    q1_after = ledger.get_quarter_holdings("2022-10-04", "2022-03-31")
    assert q1_after.iloc[0]["shares"] == 120.0
    # And querying Q1 as of 2022-10-03 still sees the original 100 (zero back-propagation!)
    q1_before = ledger.get_quarter_holdings("2022-10-03", "2022-03-31")
    assert q1_before.iloc[0]["shares"] == 100.0


def test_zero_holdings_winddown_and_staleness_retirement() -> None:
    cal = _calendar_2022()
    meta = pd.DataFrame(
        [
            {
                "filer_id": "fund_winddown",
                "canonical_name": "Fund Winddown",
                "style": "closed_peer",
                "accession_number": "wd-q1",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-05-16"),
                "acceptance_date": pd.Timestamp("2022-05-16"),
                "acceptance_datetime": "2022-05-16T16:00:00",
                "period_of_report": pd.Timestamp("2022-03-31"),
                "amendment_type": "",
                "parse_status": "ok",
            },
            {
                "filer_id": "fund_winddown",
                "canonical_name": "Fund Winddown",
                "style": "closed_peer",
                "accession_number": "wd-q2-empty",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-08-15"),
                "acceptance_date": pd.Timestamp("2022-08-15"),
                "acceptance_datetime": "2022-08-15T16:00:00",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "amendment_type": "",
                "parse_status": "zero_holdings",
            },
            {
                "filer_id": "fund_ceases",
                "canonical_name": "Fund Ceases",
                "style": "closed_peer",
                "accession_number": "cs-q1",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-05-16"),
                "acceptance_date": pd.Timestamp("2022-05-16"),
                "acceptance_datetime": "2022-05-16T16:00:00",
                "period_of_report": pd.Timestamp("2022-03-31"),
                "amendment_type": "",
                "parse_status": "ok",
            },
        ]
    )
    holdings = pd.DataFrame(
        [
            {
                "filer_id": "fund_winddown",
                "accession_number": "wd-q1",
                "cusip": "594918104",
                "name_of_issuer": "MSFT",
                "title_of_class": "COM",
                "ssh_prnamt": 100.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 10_000.0,
            },
            {
                "filer_id": "fund_ceases",
                "accession_number": "cs-q1",
                "cusip": "037833100",
                "name_of_issuer": "AAPL",
                "title_of_class": "COM",
                "ssh_prnamt": 200.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
                "value_usd": 20_000.0,
            },
        ]
    )

    ledger = PitHoldingsLedger.from_raw_tables(meta, holdings, trading_calendar=cal)

    # On 2022-08-16: fund_winddown's zero_holdings filing takes effect (0 rows),
    # while fund_ceases (138d old <= 160d) is still active.
    aug16 = ledger.get_latest_holdings("2022-08-16", max_staleness_days=160)
    assert list(aug16["filer_id"]) == ["fund_ceases"]

    # On 2022-10-01 (184d after 2022-03-31 > 160d): fund_ceases is retired as stale!
    oct01 = ledger.get_latest_holdings("2022-10-01", max_staleness_days=160)
    assert oct01.empty


def test_verify_no_lookahead_rejects_same_day_filing_and_friday_rolls_to_monday() -> None:
    cal = _calendar_2022()
    # Friday 2022-08-12 -> next trading day is Monday 2022-08-15
    assert next_trading_day(pd.Timestamp("2022-08-12"), cal) == pd.Timestamp("2022-08-15")

    bad_df = pd.DataFrame(
        [
            {
                "filer_id": "fund_a",
                "accession_number": "acc-1",
                "period_of_report": pd.Timestamp("2022-06-30"),
                "filing_date": pd.Timestamp("2022-08-15"),
                "acceptance_date": pd.Timestamp("2022-08-15"),
                "effective_date": pd.Timestamp("2022-08-15"),
                "cusip": "594918104",
                "shares": 100.0,
            }
        ]
    )
    with pytest.raises(PitLookaheadError, match="LOOKAHEAD VIOLATION"):
        verify_no_lookahead(bad_df, pd.Timestamp("2022-08-15"))
