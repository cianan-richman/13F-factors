"""Unit tests for SEC EDGAR HTTP client and quarterly master.idx ingestion."""

from __future__ import annotations

import gzip
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.ingest.edgar_client import EdgarClient, EdgarConfigError
from src.ingest.master_index import (
    build_master_index,
    iter_quarters,
    parse_master_index_text,
    raw_quarter_path,
    search_filers,
)

SAMPLE_MASTER_IDX = """Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    March 31, 2024
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/
Cloud HTTP:            https://www.sec.gov/Archives/

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
1000045|NICHOLAS FINANCIAL INC|10-K|2024-03-15|edgar/data/1000045/0000950170-24-078912.txt
1167483|TIGER GLOBAL MANAGEMENT LLC|13F-HR|2024-02-14|edgar/data/1167483/0000919574-24-001123.txt
1167483|TIGER GLOBAL MANAGEMENT LLC|13F-HR/A|2024-03-01|edgar/data/1167483/0000919574-24-002456.txt
1061768|BAUPOST GROUP LLC/MA|13F-HR|2024-02-14|edgar/data/1061768/0001061768-24-000002.txt
1061768|BAUPOST GROUP LLC/MA|13F-NT|2024-02-14|edgar/data/1061768/0001061768-24-000003.txt
"""


def test_edgar_client_rejects_placeholder_user_agent() -> None:
    with pytest.raises(EdgarConfigError):
        EdgarClient(user_agent="")
    with pytest.raises(EdgarConfigError):
        EdgarClient(user_agent="Your Name your.email@example.com")
    with pytest.raises(EdgarConfigError):
        EdgarClient(user_agent="NoEmailProvided")


def test_iter_quarters_inclusive() -> None:
    assert list(iter_quarters(2015, 3, 2016, 2)) == [
        (2015, 3),
        (2015, 4),
        (2016, 1),
        (2016, 2),
    ]
    with pytest.raises(ValueError):
        list(iter_quarters(2016, 2, 2015, 1))


def test_parse_master_index_filters_13f_and_extracts_accession() -> None:
    df = parse_master_index_text(
        SAMPLE_MASTER_IDX,
        form_types=("13F-HR", "13F-HR/A"),
        index_year=2024,
        index_qtr=1,
    )
    assert len(df) == 3
    assert set(df["form_type"]) == {"13F-HR", "13F-HR/A"}

    tiger = df[(df["cik"] == "0001167483") & (df["form_type"] == "13F-HR")].iloc[0]
    assert tiger["cik_int"] == 1167483
    assert tiger["accession_number"] == "0000919574-24-001123"
    assert tiger["accession_nodash"] == "000091957424001123"
    assert (
        tiger["filing_dir_url"]
        == "https://www.sec.gov/Archives/edgar/data/1167483/000091957424001123/"
    )
    assert tiger["date_filed"] == pd.Timestamp("2024-02-14")


def test_search_filers_aggregates_counts() -> None:
    df = parse_master_index_text(SAMPLE_MASTER_IDX)
    res = search_filers(df, "tiger global")
    assert len(res) == 1
    assert res.iloc[0]["cik"] == "0001167483"
    assert res.iloc[0]["filing_count"] == 2


def test_build_master_index_caches_and_writes_parquet(tmp_path: Path) -> None:
    mock_client = MagicMock(spec=EdgarClient)
    gz_payload = gzip.compress(SAMPLE_MASTER_IDX.encode("latin-1"))

    resp = MagicMock()
    resp.status_code = 200
    resp.content = gz_payload
    mock_client.get.return_value = resp

    raw_dir = tmp_path / "raw"
    out_parquet = tmp_path / "processed" / "index.parquet"

    df = build_master_index(
        mock_client,
        start_year=2024,
        start_qtr=1,
        end_year=2024,
        end_qtr=1,
        raw_dir=raw_dir,
        output_path=out_parquet,
        refresh_current=False,
        show_progress=False,
    )

    assert len(df) == 3
    assert raw_quarter_path(2024, 1, raw_dir).exists()
    assert out_parquet.exists()
    assert mock_client.get.call_count == 1

    # Second run should hit local cache without calling EDGAR again
    df_cached = build_master_index(
        mock_client,
        start_year=2024,
        start_qtr=1,
        end_year=2024,
        end_qtr=1,
        raw_dir=raw_dir,
        output_path=out_parquet,
        refresh_current=False,
        show_progress=False,
    )
    assert len(df_cached) == 3
    assert mock_client.get.call_count == 1


def test_build_master_index_enforces_inclusive_end_date_cutoff(tmp_path: Path) -> None:
    """Pinned end_date strictly filters date_filed <= end_date inside the final quarter."""
    mock_client = MagicMock(spec=EdgarClient)
    gz_payload = gzip.compress(SAMPLE_MASTER_IDX.encode("latin-1"))

    resp = MagicMock()
    resp.status_code = 200
    resp.content = gz_payload
    mock_client.get.return_value = resp

    df = build_master_index(
        mock_client,
        start_year=2024,
        start_qtr=1,
        end_date="2024-02-20",
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "processed" / "index_pinned.parquet",
        refresh_current=False,
        show_progress=False,
    )
    # SAMPLE_MASTER_IDX has two 2024-02-14 filings and one 2024-03-01 filing -> drop 2024-03-01
    assert len(df) == 2
    assert df["date_filed"].max() == pd.Timestamp("2024-02-14")
