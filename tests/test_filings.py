"""Unit tests for SEC EDGAR 13F-HR / 13F-HR/A complete submission downloader & XML parser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from src.ingest.edgar_client import EdgarBlockedError, EdgarClient
from src.ingest.filings import (
    check_cross_filer_other_managers,
    fetch_filing_submission,
    ingest_universe_filings,
    parse_filing_submission,
    raw_filing_path,
)

SAMPLE_SUBMISSION_PRE_2023 = """<SEC-DOCUMENT>0001167483-22-000001.txt : 20220516
<SEC-HEADER>0001167483-22-000001.hdr.sgml : 20220516
<ACCEPTANCE-DATETIME>20220516161245
CONFORMED SUBMISSION TYPE:	13F-HR
CONFORMED PERIOD OF REPORT:	20220331
FILED AS OF DATE:		20220516
</SEC-HEADER>
<DOCUMENT>
<TYPE>13F-HR
<SEQUENCE>1
<FILENAME>primary_doc.xml
<TEXT>
<XML>
<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
  <headerData>
    <submissionType>13F-HR</submissionType>
  </headerData>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>03-31-2022</reportCalendarOrQuarter>
      <isAmendment>false</isAmendment>
      <reportType>13F HOLDINGS REPORT</reportType>
      <isConfidentialOmitted>true</isConfidentialOmitted>
    </coverPage>
    <summaryPage>
      <tableEntryTotal>3</tableEntryTotal>
      <tableValueTotal>350000</tableValueTotal>
      <otherManagers2Info>
        <otherManager2>
          <sequenceNumber>1</sequenceNumber>
          <otherManager>
            <cik>0001999999</cik>
            <name>SUB ADVISOR LLC</name>
          </otherManager>
        </otherManager2>
      </otherManagers2Info>
    </summaryPage>
  </formData>
</edgarSubmission>
</XML>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>INFORMATION TABLE
<SEQUENCE>2
<FILENAME>infotable.xml
<TEXT>
<XML>
<?xml version="1.0" encoding="UTF-8"?>
<ns1:informationTable xmlns:ns1="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <ns1:infoTable>
    <ns1:nameOfIssuer>MICROSOFT CORP</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>594918104</ns1:cusip>
    <ns1:value>300000</ns1:value>
    <ns1:shrsOrPrnAmt>
      <ns1:sshPrnamt>1000000</ns1:sshPrnamt>
      <ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
    </ns1:shrsOrPrnAmt>
    <ns1:investmentDiscretion>SOLE</ns1:investmentDiscretion>
    <ns1:votingAuthority>
      <ns1:Sole>1000000</ns1:Sole>
      <ns1:Shared>0</ns1:Shared>
      <ns1:None>0</ns1:None>
    </ns1:votingAuthority>
  </ns1:infoTable>
  <ns1:infoTable>
    <ns1:nameOfIssuer>MICROSOFT CORP</ns1:nameOfIssuer>
    <ns1:titleOfClass>COM</ns1:titleOfClass>
    <ns1:cusip>594918104</ns1:cusip>
    <ns1:value>50000</ns1:value>
    <ns1:shrsOrPrnAmt>
      <ns1:sshPrnamt>166667</ns1:sshPrnamt>
      <ns1:sshPrnamtType>SH</ns1:sshPrnamtType>
    </ns1:shrsOrPrnAmt>
    <ns1:putCall>Put</ns1:putCall>
    <ns1:investmentDiscretion>DFND</ns1:investmentDiscretion>
    <ns1:otherManager>1</ns1:otherManager>
    <ns1:votingAuthority>
      <ns1:Sole>0</ns1:Sole>
      <ns1:Shared>166667</ns1:Shared>
      <ns1:None>0</ns1:None>
    </ns1:votingAuthority>
  </ns1:infoTable>
</ns1:informationTable>
</XML>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""

SAMPLE_AMENDMENT_POST_2023 = """<SEC-DOCUMENT>0001167483-24-000009.txt : 20240610
<SEC-HEADER>0001167483-24-000009.hdr.sgml : 20240610
<ACCEPTANCE-DATETIME>20240610170501
CONFORMED SUBMISSION TYPE:	13F-HR/A
CONFORMED PERIOD OF REPORT:	20231231
FILED AS OF DATE:		20240610
</SEC-HEADER>
<DOCUMENT>
<TYPE>13F-HR/A
<SEQUENCE>1
<TEXT>
<XML>
<edgarSubmission>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>12-31-2023</reportCalendarOrQuarter>
      <isAmendment>true</isAmendment>
      <amendmentNo>1</amendmentNo>
      <amendmentInfo>
        <amendmentType>NEW HOLDINGS</amendmentType>
      </amendmentInfo>
      <reportType>13F HOLDINGS REPORT</reportType>
    </coverPage>
    <summaryPage>
      <tableEntryTotal>1</tableEntryTotal>
      <tableValueTotal>42000000</tableValueTotal>
    </summaryPage>
  </formData>
</edgarSubmission>
</XML>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>INFORMATION TABLE
<SEQUENCE>2
<TEXT>
<XML>
<informationTable>
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>42000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>210000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>SOLE</investmentDiscretion>
    <votingAuthority>
      <Sole>210000</Sole>
      <Shared>0</Shared>
      <None>0</None>
    </votingAuthority>
  </infoTable>
</informationTable>
</XML>
</TEXT>
</DOCUMENT>
</SEC-DOCUMENT>
"""


def test_parse_pre_2023_submission_multiplies_value_by_1000_and_extracts_metadata() -> None:
    manifest_row = {
        "filer_id": "tiger_global",
        "canonical_name": "Tiger Global Management",
        "style": "growth_tmt_ls",
        "cik": "0001167483",
        "source_role": "primary",
        "accession_number": "0001167483-22-000001",
        "form_type": "13F-HR",
        "date_filed": "2022-05-16",
    }
    meta, rows = parse_filing_submission(SAMPLE_SUBMISSION_PRE_2023, manifest_row)

    assert meta["period_of_report"] == "2022-03-31"
    assert meta["acceptance_datetime"] == "2022-05-16T16:12:45"
    assert meta["is_amendment"] is False
    assert meta["is_confidential_omitted"] is True
    assert meta["value_multiplier"] == 1000
    assert meta["parsed_row_count"] == 2
    assert meta["other_managers_ciks"] == "0001999999"
    assert meta["value_unit_suspect"] is False
    assert abs(meta["median_implied_price"] - 300.0) < 1e-6

    assert len(rows) == 2
    assert rows[0]["cusip"] == "594918104"
    assert rows[0]["value_raw"] == 300000.0
    assert rows[0]["value_usd"] == 300_000_000.0
    assert rows[0]["put_call"] == ""
    assert rows[1]["put_call"] == "PUT"
    assert rows[1]["other_manager"] == "1"


def test_parse_post_2023_amendment_ctr_reveal_uses_exact_dollars() -> None:
    manifest_row = {
        "filer_id": "tiger_global",
        "canonical_name": "Tiger Global Management",
        "style": "growth_tmt_ls",
        "cik": "0001167483",
        "source_role": "primary",
        "accession_number": "0001167483-24-000009",
        "form_type": "13F-HR/A",
        "date_filed": "2024-06-10",
    }
    meta, rows = parse_filing_submission(SAMPLE_AMENDMENT_POST_2023, manifest_row)

    assert meta["period_of_report"] == "2023-12-31"
    assert meta["is_amendment"] is True
    assert meta["amendment_no"] == 1
    assert meta["amendment_type"] == "NEW HOLDINGS"
    assert meta["value_multiplier"] == 1
    assert meta["value_unit_suspect"] is False
    assert abs(meta["median_implied_price"] - 200.0) < 1e-6
    assert len(rows) == 1
    assert rows[0]["value_usd"] == 42_000_000.0


def test_ingest_universe_filings_caches_locally_and_checks_other_managers(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "filer_id": "tiger_global",
                "canonical_name": "Tiger Global Management",
                "style": "growth_tmt_ls",
                "cik": "0001167483",
                "source_role": "primary",
                "accession_number": "0001167483-22-000001",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-05-16"),
                "filename": "edgar/data/1167483/0001167483-22-000001.txt",
            }
        ]
    )
    mock_client = MagicMock(spec=EdgarClient)
    resp = MagicMock()
    resp.content = SAMPLE_SUBMISSION_PRE_2023.encode("utf-8")
    mock_client.get.return_value = resp

    raw_dir = tmp_path / "raw"
    meta_out = tmp_path / "meta.parquet"
    holdings_out = tmp_path / "holdings.parquet"

    meta_df, holdings_df = ingest_universe_filings(
        manifest,
        client=mock_client,
        raw_dir=raw_dir,
        meta_output_path=meta_out,
        holdings_output_path=holdings_out,
        show_progress=False,
    )

    assert len(meta_df) == 1
    assert len(holdings_df) == 2
    assert raw_filing_path("0001167483", "0001167483-22-000001", raw_dir).exists()
    assert mock_client.get.call_count == 1
    assert check_cross_filer_other_managers(meta_df) == []

    # Re-parse offline using cache only
    meta_df2, holdings_df2 = ingest_universe_filings(
        manifest,
        client=None,
        raw_dir=raw_dir,
        meta_output_path=meta_out,
        holdings_output_path=holdings_out,
        parse_only=True,
        show_progress=False,
    )
    assert len(meta_df2) == 1
    assert len(holdings_df2) == 2
    assert mock_client.get.call_count == 1


def test_fetch_submission_falls_back_to_index_json_on_502(tmp_path: Path) -> None:
    mock_client = MagicMock(spec=EdgarClient)
    err_resp = MagicMock()
    err_resp.status_code = 502

    idx_resp = MagicMock()
    idx_resp.status_code = 200
    idx_resp.json.return_value = {
        "directory": {
            "item": [
                {"name": "0001592643-22-000002.hdr.sgml"},
                {"name": "primary_doc.xml"},
                {"name": "infotable.xml"},
            ]
        }
    }
    hdr_resp = MagicMock()
    hdr_resp.status_code = 200
    hdr_resp.text = "<SEC-HEADER>\n<ACCEPTANCE-DATETIME>20220214160500\n</SEC-HEADER>"

    prim_resp = MagicMock()
    prim_resp.status_code = 200
    prim_resp.text = (
        "<edgarSubmission><formData><coverPage>"
        "<reportCalendarOrQuarter>12-31-2021</reportCalendarOrQuarter>"
        "<isAmendment>false</isAmendment>"
        "</coverPage></formData></edgarSubmission>"
    )

    info_resp = MagicMock()
    info_resp.status_code = 200
    info_resp.text = (
        "<informationTable><infoTable>"
        "<nameOfIssuer>APPLE INC</nameOfIssuer><titleOfClass>COM</titleOfClass>"
        "<cusip>037833100</cusip><value>17757</value>"
        "<shrsOrPrnAmt><sshPrnamt>100000</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>"
        "</infoTable></informationTable>"
    )

    def _side_effect(url: str, **_: object) -> MagicMock:
        if url.endswith(".txt"):
            raise requests.HTTPError("502 Bad Gateway", response=err_resp)
        if url.endswith("index.json"):
            return idx_resp
        if url.endswith(".hdr.sgml"):
            return hdr_resp
        if url.endswith("primary_doc.xml"):
            return prim_resp
        return info_resp

    mock_client.get.side_effect = _side_effect

    cached = fetch_filing_submission(
        mock_client,
        filename="edgar/data/1592643/0001592643-22-000002.txt",
        cik="0001592643",
        accession_number="0001592643-22-000002",
        raw_dir=tmp_path,
    )
    assert cached.exists()
    meta, rows = parse_filing_submission(
        __import__("gzip").decompress(cached.read_bytes()).decode("utf-8"),
        {
            "filer_id": "select_equity",
            "canonical_name": "Select Equity Group",
            "style": "quality_generalist",
            "cik": "0001592643",
            "source_role": "primary",
            "accession_number": "0001592643-22-000002",
            "form_type": "13F-HR",
            "date_filed": "2022-02-14",
        },
    )
    assert meta["parse_status"] == "ok"
    assert meta["period_of_report"] == "2021-12-31"
    assert meta["acceptance_datetime"] == "2022-02-14T16:05:00"
    assert len(rows) == 1
    assert rows[0]["cusip"] == "037833100"


def test_auto_corrects_filer_1000x_unit_errors_and_drops_zero_placeholders() -> None:
    # Post-2023 filing erroneously reported in $1,000s (like Baupost 2023-2026)
    post_2023_thousands = """<SEC-DOCUMENT>test.txt
<SEC-HEADER><ACCEPTANCE-DATETIME>20240213160000</SEC-HEADER>
<DOCUMENT><TYPE>13F-HR<TEXT><XML>
<edgarSubmission><formData><coverPage>
<reportCalendarOrQuarter>12-31-2023</reportCalendarOrQuarter>
</coverPage></formData></edgarSubmission></XML></TEXT></DOCUMENT>
<DOCUMENT><TYPE>INFORMATION TABLE<TEXT><XML>
<informationTable>
  <infoTable>
    <nameOfIssuer>ALPHABET INC</nameOfIssuer><titleOfClass>COM</titleOfClass>
    <cusip>02079K107</cusip><value>415678</value>
    <shrsOrPrnAmt><sshPrnamt>2949538</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>No securities</nameOfIssuer><titleOfClass>NONE</titleOfClass>
    <cusip>000000000</cusip><value>0</value>
    <shrsOrPrnAmt><sshPrnamt>0</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></shrsOrPrnAmt>
  </infoTable>
</informationTable></XML></TEXT></DOCUMENT></SEC-DOCUMENT>"""

    meta, rows = parse_filing_submission(
        post_2023_thousands,
        {
            "filer_id": "baupost",
            "canonical_name": "The Baupost Group",
            "style": "value_contrarian_ls",
            "cik": "0001061768",
            "accession_number": "0001061768-24-000001",
            "form_type": "13F-HR",
            "date_filed": "2024-02-13",
        },
    )
    assert meta["statutory_multiplier"] == 1
    assert meta["value_multiplier"] == 1000
    assert meta["value_unit_corrected"] is True
    assert meta["value_unit_reason"] == "post_2023_reported_in_thousands"
    assert meta["value_unit_suspect"] is False
    assert 140.0 < meta["median_implied_price"] < 142.0
    assert len(rows) == 1  # Dummy 000000000 zero-share row stripped
    assert rows[0]["value_usd"] == 415_678_000.0


def test_edgar_client_raises_blocked_error_on_403() -> None:
    mock_session = MagicMock(spec=requests.Session)
    mock_session.headers = {}
    resp = MagicMock()
    resp.status_code = 403
    mock_session.get.return_value = resp

    client = EdgarClient(
        user_agent="Test User test@example.com",
        max_retries=2,
        session=mock_session,
    )
    with pytest.raises(EdgarBlockedError):
        client.get("https://www.sec.gov/test")
    assert mock_session.get.call_count == 1  # Must fail fast without hammering


def test_corrupt_cache_self_heals_and_fault_isolation_preserves_batch(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    # Pre-seed a corrupt .txt.gz for filing #1
    corrupt_path = raw_filing_path("0001167483", "0001167483-22-000001", raw_dir)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not-valid-gzip-bytes")

    manifest = pd.DataFrame(
        [
            {
                "filer_id": "tiger_global",
                "canonical_name": "Tiger Global Management",
                "style": "growth_tmt_ls",
                "cik": "0001167483",
                "source_role": "primary",
                "accession_number": "0001167483-22-000001",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-05-16"),
                "filename": "edgar/data/1167483/0001167483-22-000001.txt",
            },
            {
                "filer_id": "tiger_global",
                "canonical_name": "Tiger Global Management",
                "style": "growth_tmt_ls",
                "cik": "0001167483",
                "source_role": "primary",
                "accession_number": "0001167483-24-000099",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2024-05-15"),
                "filename": "edgar/data/1167483/0001167483-24-000099.txt",
            },
        ]
    )

    mock_client = MagicMock(spec=EdgarClient)
    ok_resp = MagicMock()
    ok_resp.content = SAMPLE_SUBMISSION_PRE_2023.encode("utf-8")

    def _side_effect(url: str, **_: object) -> MagicMock:
        if "0001167483-22-000001" in url:
            return ok_resp
        raise requests.ConnectionError("persistent network failure")

    mock_client.get.side_effect = _side_effect

    meta_out = tmp_path / "meta.parquet"
    holdings_out = tmp_path / "holdings.parquet"
    failures_out = tmp_path / "failures.csv"

    meta_df, holdings_df = ingest_universe_filings(
        manifest,
        client=mock_client,
        raw_dir=raw_dir,
        meta_output_path=meta_out,
        holdings_output_path=holdings_out,
        failures_output_path=failures_out,
        max_workers=2,
        show_progress=False,
    )

    assert len(meta_df) == 2
    assert set(meta_df["parse_status"]) == {"ok", "fetch_error"}
    assert len(holdings_df) == 2  # Healthy filing still parsed & written after self-healing cache
    assert failures_out.exists()
    failures_df = pd.read_csv(failures_out)
    assert list(failures_df["accession_number"]) == ["0001167483-24-000099"]



