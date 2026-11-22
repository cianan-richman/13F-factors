"""Unit tests for OpenFIGI CUSIP mapping, US Common Stock classification, and caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

from src.ingest.cusip_mapper import (
    OpenFigiClient,
    build_cusip_map,
    is_common_stock_candidate,
    normalize_yahoo_ticker,
    openfigi_id_type,
    select_best_figi_candidate,
)


def test_normalize_yahoo_ticker_and_id_type() -> None:
    assert normalize_yahoo_ticker("BRK/B") == "BRK-B"
    assert normalize_yahoo_ticker("BF.B") == "BF-B"
    assert normalize_yahoo_ticker("  aapl ") == "AAPL"
    assert normalize_yahoo_ticker("") == ""
    assert openfigi_id_type("037833100") == "ID_CUSIP"
    assert openfigi_id_type("H84989104") == "ID_CINS"
    assert openfigi_id_type("G1151C101") == "ID_CINS"


def test_openfigi_client_keyed_vs_keyless_fallback() -> None:
    keyless = OpenFigiClient(api_key=None)
    assert keyless.batch_size == 10
    assert keyless.default_workers == 1
    assert "X-OPENFIGI-APIKEY" not in keyless._session.headers
    keyless.close()

    keyed = OpenFigiClient(api_key="dummy-key")
    assert keyed.batch_size == 100
    assert keyed.default_workers == 4
    assert keyed._session.headers["X-OPENFIGI-APIKEY"] == "dummy-key"
    keyed.close()


def test_select_best_figi_candidate_prefers_us_composite_common_stock() -> None:
    candidates = [
        {
            "figi": "BBG000B9XRY4_DE",
            "shareClassFIGI": "BBG001S5N8V8",
            "ticker": "APC",
            "exchCode": "GR",
            "marketSector": "Equity",
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
        },
        {
            "figi": "BBG000B9XRY4",
            "compositeFIGI": "BBG000B9XRY4",
            "shareClassFIGI": "BBG001S5N8V8",
            "ticker": "AAPL",
            "exchCode": "US",
            "marketSector": "Equity",
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
        },
    ]
    best = select_best_figi_candidate(candidates)
    assert best is not None
    assert best["exchCode"] == "US"
    assert best["ticker"] == "AAPL"
    assert is_common_stock_candidate(best) is True


def test_is_common_stock_excludes_reit_etf_adr_warrant() -> None:
    reit = {
        "exchCode": "US",
        "marketSector": "Equity",
        "securityType": "REIT",
        "securityType2": "REIT",
    }
    etf = {
        "exchCode": "US",
        "marketSector": "Equity",
        "securityType": "ETP",
        "securityType2": "Mutual Fund",
    }
    adr = {
        "exchCode": "US",
        "marketSector": "Equity",
        "securityType": "ADR",
        "securityType2": "Depositary Receipt",
    }
    warrant_disguised = {
        "exchCode": "US",
        "marketSector": "Equity",
        "securityType": "Equity WRT",
        "securityType2": "Common Stock",
    }
    assert is_common_stock_candidate(reit) is False
    assert is_common_stock_candidate(etf) is False
    assert is_common_stock_candidate(adr) is False
    # Warrant caught via securityType keyword check if containing WARRANT
    warrant_explicit = {**warrant_disguised, "securityType": "Equity Warrant"}
    assert is_common_stock_candidate(warrant_explicit) is False


def test_is_common_stock_excludes_otc_venues_and_foreign_fyq_tickers() -> None:
    """OTC venues (UD/UO/PQ) and 5-letter FINRA OTC F/Y/Q suffixes are rejected."""
    for venue in ("UD", "UO", "PQ"):
        rec = {
            "exchCode": venue,
            "ticker": "ABCD",
            "marketSector": "Equity",
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
        }
        assert is_common_stock_candidate(rec) is False

    for otc_ticker in ("BUDFF", "ASMLF", "NSRGY", "GPLDQ"):
        rec = {
            "exchCode": "US",
            "ticker": otc_ticker,
            "marketSector": "Equity",
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
        }
        assert is_common_stock_candidate(rec) is False

    # Normal 4-letter and 5-letter non-[FYQ] listed tickers (e.g., GOOGL, LBRDK) remain valid
    for listed_ticker in ("AAPL", "GOOGL", "LBRDK", "BRK/B"):
        rec = {
            "exchCode": "US",
            "ticker": listed_ticker,
            "marketSector": "Equity",
            "securityType": "Common Stock",
            "securityType2": "Common Stock",
        }
        assert is_common_stock_candidate(rec) is True


def test_build_cusip_map_caches_incrementally_and_applies_overrides(tmp_path: Path) -> None:
    holdings = pd.DataFrame(
        [
            {
                "filer_id": "tiger_global",
                "accession_number": "acc-1",
                "cusip": "037833100",
                "name_of_issuer": "APPLE INC",
                "title_of_class": "COM",
                "value_usd": 500_000_000.0,
                "ssh_prnamt": 2_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
            },
            {
                "filer_id": "tiger_global",
                "accession_number": "acc-1",
                "cusip": "999999999",
                "name_of_issuer": "MANUAL CORP",
                "title_of_class": "COM",
                "value_usd": 100_000_000.0,
                "ssh_prnamt": 1_000_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
            },
            # Option row should be ignored
            {
                "filer_id": "tiger_global",
                "accession_number": "acc-1",
                "cusip": "111111111",
                "name_of_issuer": "PUT ONLY CORP",
                "title_of_class": "COM",
                "value_usd": 50_000_000.0,
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "PUT",
            },
        ]
    )

    mock_client = MagicMock(spec=OpenFigiClient)
    mock_client.batch_size = 10
    mock_client.map_cusip_batch.return_value = [
        {
            "data": [
                {
                    "figi": "BBG000B9XRY4",
                    "compositeFIGI": "BBG000B9XRY4",
                    "shareClassFIGI": "BBG001S5N8V8",
                    "ticker": "AAPL",
                    "name": "APPLE INC",
                    "exchCode": "US",
                    "marketSector": "Equity",
                    "securityType": "Common Stock",
                    "securityType2": "Common Stock",
                }
            ]
        }
    ]

    cache_path = tmp_path / "cache.jsonl"
    out_parquet = tmp_path / "map.parquet"
    summary_csv = tmp_path / "summary.csv"
    overrides = {
        "999999999": {
            "ticker": "MAN/A",
            "share_class_figi": "BBGMANUAL001",
            "security_type2": "Common Stock",
        }
    }

    res = build_cusip_map(
        holdings,
        client=mock_client,
        overrides=overrides,
        cache_path=cache_path,
        output_path=out_parquet,
        summary_csv_path=summary_csv,
        show_progress=False,
    )

    assert len(res) == 2
    assert mock_client.map_cusip_batch.call_count == 1
    assert mock_client.map_cusip_batch.call_args[0][0] == ["037833100"]

    aapl = res[res["cusip"] == "037833100"].iloc[0]
    assert aapl["share_class_figi"] == "BBG001S5N8V8"
    assert aapl["yahoo_ticker"] == "AAPL"
    assert bool(aapl["is_us_common_stock"]) is True

    man = res[res["cusip"] == "999999999"].iloc[0]
    assert man["mapping_source"] == "manual_override"
    assert man["yahoo_ticker"] == "MAN-A"

    # Second run hits JSONL cache with 0 API calls
    res2 = build_cusip_map(
        holdings,
        client=mock_client,
        overrides=overrides,
        cache_path=cache_path,
        output_path=out_parquet,
        summary_csv_path=summary_csv,
        show_progress=False,
    )
    assert len(res2) == 2
    assert mock_client.map_cusip_batch.call_count == 1


def test_retired_foreign_cusip_auto_stitches_via_shared_share_class_figi(tmp_path: Path) -> None:
    holdings = pd.DataFrame(
        [
            {
                "filer_id": "tiger_global",
                "accession_number": "acc-1",
                "cusip": "741503403",
                "name_of_issuer": "PRICELINE GRP INC",
                "title_of_class": "COM",
                "value_usd": 200_000_000.0,
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
            },
            {
                "filer_id": "tiger_global",
                "accession_number": "acc-2",
                "cusip": "09857L108",
                "name_of_issuer": "BOOKING HOLDINGS INC",
                "title_of_class": "COM",
                "value_usd": 400_000_000.0,
                "ssh_prnamt": 100_000.0,
                "ssh_prnamt_type": "SH",
                "put_call": "",
            },
        ]
    )
    mock_client = MagicMock(spec=OpenFigiClient)
    mock_client.batch_size = 10
    mock_client.map_cusip_batch.return_value = [
        {
            "data": [
                {
                    "figi": "BBG000BLNNH6",
                    "shareClassFIGI": "BBG001S89N72",
                    "ticker": "BKNG",
                    "name": "BOOKING HOLDINGS INC",
                    "exchCode": "US",
                    "marketSector": "Equity",
                    "securityType": "Common Stock",
                    "securityType2": "Common Stock",
                }
            ]
        },
        {
            "data": [
                {
                    "figi": "BBG000OLDES1",
                    "shareClassFIGI": "BBG001S89N72",
                    "ticker": "PCLN",
                    "name": "PRICELINE GRP INC",
                    "exchCode": "ES",
                    "marketSector": "Equity",
                    "securityType": "Common Stock",
                    "securityType2": "Common Stock",
                }
            ]
        },
    ]

    res = build_cusip_map(
        holdings,
        client=mock_client,
        cache_path=tmp_path / "cache.jsonl",
        output_path=tmp_path / "map.parquet",
        summary_csv_path=tmp_path / "summary.csv",
        show_progress=False,
    )
    old_row = res[res["cusip"] == "741503403"].iloc[0]
    assert bool(old_row["is_us_common_stock"]) is True
    assert old_row["figi_ticker"] == "BKNG"
    assert old_row["mapping_source"] == "openfigi_figi_stitch"

