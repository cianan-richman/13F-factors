"""Unit tests for config/filers.yaml schema, CIK migration handoffs, and entity auditing."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import CONFIG_DIR, load_config
from src.ingest.entity_audit import (
    FilerConfigError,
    audit_edgar_index,
    resolve_universe_filings,
    validate_filers_config,
)


def _sample_index() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Predecessor filing cleanly before cutoff
            {
                "cik": "0000728014",
                "cik_int": 728014,
                "company_name": "RUANE, CUNNIFF & GOLDFARB INC",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2018-02-14"),
                "filename": "edgar/data/728014/0000919574-18-001254.txt",
                "accession_number": "0000919574-18-001254",
                "accession_nodash": "000091957418001254",
                "filing_dir_url": "https://www.sec.gov/Archives/edgar/data/728014/000091957418001254/",
                "index_year": 2018,
                "index_qtr": 1,
            },
            # Predecessor overlap filing on 2018-05-15 (must be dropped by max_filing_date)
            {
                "cik": "0000728014",
                "cik_int": 728014,
                "company_name": "RUANE, CUNNIFF & GOLDFARB INC",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2018-05-15"),
                "filename": "edgar/data/728014/0000919574-18-003521.txt",
                "accession_number": "0000919574-18-003521",
                "accession_nodash": "000091957418003521",
                "filing_dir_url": "https://www.sec.gov/Archives/edgar/data/728014/000091957418003521/",
                "index_year": 2018,
                "index_qtr": 2,
            },
            # Successor filing on 2018-05-15 (must be kept)
            {
                "cik": "0001720792",
                "cik_int": 1720792,
                "company_name": "Ruane, Cunniff & Goldfarb L.P.",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2018-05-15"),
                "filename": "edgar/data/1720792/0000919574-18-003599.txt",
                "accession_number": "0000919574-18-003599",
                "accession_nodash": "000091957418003599",
                "filing_dir_url": "https://www.sec.gov/Archives/edgar/data/1720792/000091957418003599/",
                "index_year": 2018,
                "index_qtr": 2,
            },
            # Excluded subsidiary filing (must never enter universe)
            {
                "cik": "0001872297",
                "cik_int": 1872297,
                "company_name": "Steadfast Capital Management (UK) Ltd.",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2022-02-14"),
                "filename": "edgar/data/1872297/0001872297-22-000001.txt",
                "accession_number": "0001872297-22-000001",
                "accession_nodash": "000187229722000001",
                "filing_dir_url": "https://www.sec.gov/Archives/edgar/data/1872297/000187229722000001/",
                "index_year": 2022,
                "index_qtr": 1,
            },
            # Unexpected new affiliate matching search alias
            {
                "cik": "0009999999",
                "cik_int": 9999999,
                "company_name": "RUANE, CUNNIFF EUROPE LLP",
                "form_type": "13F-HR",
                "date_filed": pd.Timestamp("2025-02-14"),
                "filename": "edgar/data/9999999/0009999999-25-000001.txt",
                "accession_number": "0009999999-25-000001",
                "accession_nodash": "000999999925000001",
                "filing_dir_url": "https://www.sec.gov/Archives/edgar/data/9999999/000999999925000001/",
                "index_year": 2025,
                "index_qtr": 1,
            },
        ]
    )


def test_real_filers_yaml_passes_validation() -> None:
    cfg = load_config(CONFIG_DIR)
    assert len(cfg.filers) == 100
    validate_filers_config(cfg.filers)


def test_validate_filers_rejects_subsidiary_overlap() -> None:
    bad_filers = [
        {
            "filer_id": "fund_a",
            "canonical_name": "Fund A",
            "style": "growth_tmt_ls",
            "status": "active",
            "cik": "0001111111",
            "predecessor_ciks": [],
            "subsidiary_ciks": ["0002222222"],
            "unrelated_ciks": [],
        },
        {
            "filer_id": "fund_b",
            "canonical_name": "Fund B",
            "style": "growth_tmt_ls",
            "status": "active",
            "cik": "0002222222",
            "predecessor_ciks": [],
            "subsidiary_ciks": [],
            "unrelated_ciks": [],
        },
    ]
    with pytest.raises(FilerConfigError, match="Overlap"):
        validate_filers_config(bad_filers)


def test_predecessor_cutoff_eliminates_duplicate_overlap_and_flags_unmapped() -> None:
    filers = [
        {
            "filer_id": "ruane_cunniff",
            "canonical_name": "Ruane, Cunniff & Goldfarb",
            "style": "quality_generalist",
            "status": "active",
            "cik": "0001720792",
            "predecessor_ciks": [
                {
                    "cik": "0000728014",
                    "name": "RUANE, CUNNIFF & GOLDFARB INC",
                    "max_filing_date": "2018-05-14",
                }
            ],
            "subsidiary_ciks": ["0001872297"],
            "unrelated_ciks": [],
            "search_aliases": ["RUANE, CUNNIFF"],
        }
    ]
    idx = _sample_index()
    resolved = resolve_universe_filings(idx, filers)

    assert len(resolved) == 2
    assert list(resolved["accession_number"]) == [
        "0000919574-18-001254",
        "0000919574-18-003599",
    ]
    assert list(resolved["source_role"]) == ["predecessor", "primary"]

    audit = audit_edgar_index(idx, filers)
    assert audit.duplicate_quarter_rows.empty
    assert len(audit.unmapped_candidates) == 1
    assert audit.unmapped_candidates.iloc[0]["cik"] == "0009999999"
