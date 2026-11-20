"""Unit tests for SEC EDGAR companyfacts strictly PIT shares outstanding extraction."""

from __future__ import annotations

from src.ingest.shares_outstanding import extract_share_facts_payload, parse_share_facts_rows


def test_parse_share_facts_handles_dual_class_and_tag_priority() -> None:
    raw_cf = {
        "entityName": "Alphabet Inc.",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            # Dual-class cover-page rows on same (filed, end, accn) -> sum classes
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-25",
                                "val": 5_900_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000094",
                            },
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-25",
                                "val": 870_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000094",
                            },
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-25",
                                "val": 5_730_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000094",
                            },
                            # Duplicate identical repeat must not be double-counted
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-25",
                                "val": 5_900_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000094",
                            },
                            # Non-periodic form (8-K) ignored
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-20",
                                "val": 100,
                                "form": "8-K",
                                "accn": "0001652044-23-000080",
                            },
                        ]
                    }
                }
            },
            "us-gaap": {
                "CommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            # Same (filed, end) as DEI -> DEI priority 1 wins
                            {
                                "end": "2023-10-17",
                                "filed": "2023-10-25",
                                "val": 12_000_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000094",
                            },
                            # Distinct earlier period only in us-gaap -> kept
                            {
                                "end": "2023-06-30",
                                "filed": "2023-07-26",
                                "val": 12_600_000_000,
                                "form": "10-Q",
                                "accn": "0001652044-23-000070",
                            },
                        ]
                    }
                }
            },
        },
    }

    payload = extract_share_facts_payload(raw_cf, "0001652044")
    rows = parse_share_facts_rows(payload)
    assert len(rows) == 2

    q2 = [r for r in rows if r["end_date"] == "2023-06-30"][0]
    assert q2["tag_source"] == "us-gaap:CommonStockSharesOutstanding"
    assert q2["shares_outstanding"] == 12_600_000_000.0

    q3 = [r for r in rows if r["end_date"] == "2023-10-17"][0]
    assert q3["tag_source"] == "dei:EntityCommonStockSharesOutstanding"
    assert q3["n_classes_reported"] == 3
    assert q3["shares_outstanding"] == 12_500_000_000.0
