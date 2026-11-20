"""Entity resolution and governance audit for the SEC EDGAR 13F master index.

Maps raw SEC EDGAR CIKs in `edgar_13f_index.parquet` to canonical fund complexes using the
declarative mapping & override registry in `config/filers.yaml`, enforces deduplication
invariants, detects missing quarters / legal renames / unmapped affiliate CIKs, and writes
the clean universe filing manifest consumed by downstream XML ingestion.

Usage:
    python -m src.ingest.entity_audit
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, load_config
from src.ingest.master_index import DEFAULT_OUTPUT_PARQUET, iter_quarters, load_13f_index

logger = logging.getLogger(__name__)

VALID_STYLES = {"growth_tmt_ls", "value_contrarian_ls", "quality_generalist", "closed_peer"}
VALID_STATUSES = {"active", "closed"}
_CIK_RE = re.compile(r"^\d{10}$")

DEFAULT_UNIVERSE_PARQUET = PROCESSED_DATA_DIR / "universe_13f_filings.parquet"
DEFAULT_AUDIT_CSV = OUTPUTS_DIR / "filer_audit_report.csv"


class FilerConfigError(ValueError):
    """Raised when `config/filers.yaml` violates schema or deduplication invariants."""


@dataclass(frozen=True)
class AuditResult:
    """Outputs of auditing the EDGAR master index against `config/filers.yaml`."""

    universe_filings: pd.DataFrame
    filer_report: pd.DataFrame
    unmapped_candidates: pd.DataFrame
    duplicate_quarter_rows: pd.DataFrame


def _validate_cik(cik: Any, context: str) -> str:
    if not isinstance(cik, str) or not _CIK_RE.match(cik):
        raise FilerConfigError(f"{context}: CIK must be a 10-digit zero-padded string, got {cik!r}")
    return cik


def validate_filers_config(filers: list[dict[str, Any]]) -> None:
    """Validate structural schema and cross-filer deduplication invariants."""
    if not filers:
        raise FilerConfigError("config/filers.yaml contains no filers.")

    seen_filer_ids: set[str] = set()
    active_ciks: dict[str, str] = {}
    excluded_ciks: dict[str, str] = {}

    for idx, entry in enumerate(filers):
        ctx = f"filers[{idx}]"
        filer_id = entry.get("filer_id")
        if not isinstance(filer_id, str) or not filer_id.strip():
            raise FilerConfigError(f"{ctx}: missing non-empty filer_id")
        if filer_id in seen_filer_ids:
            raise FilerConfigError(f"Duplicate filer_id: {filer_id!r}")
        seen_filer_ids.add(filer_id)

        style = entry.get("style")
        if style not in VALID_STYLES:
            raise FilerConfigError(f"{filer_id}: invalid style {style!r}; expected {VALID_STYLES}")

        status = entry.get("status")
        if status not in VALID_STATUSES:
            raise FilerConfigError(
                f"{filer_id}: invalid status {status!r}; expected {VALID_STATUSES}"
            )

        primary_cik = _validate_cik(entry.get("cik"), f"{filer_id}.cik")
        if primary_cik in active_ciks:
            raise FilerConfigError(
                f"CIK {primary_cik} assigned to both {active_ciks[primary_cik]!r} and {filer_id!r}"
            )
        active_ciks[primary_cik] = filer_id

        for p_idx, pred in enumerate(entry.get("predecessor_ciks") or []):
            pred_cik = _validate_cik(pred.get("cik"), f"{filer_id}.predecessor_ciks[{p_idx}]")
            if pred_cik in active_ciks:
                raise FilerConfigError(
                    f"Predecessor CIK {pred_cik} collides with {active_ciks[pred_cik]!r}"
                )
            max_date = pred.get("max_filing_date")
            if not max_date:
                raise FilerConfigError(
                    f"{filer_id}.predecessor_ciks[{p_idx}]: max_filing_date is required"
                )
            try:
                pd.Timestamp(max_date)
            except ValueError as exc:
                raise FilerConfigError(
                    f"{filer_id}.predecessor_ciks[{p_idx}]: invalid max_filing_date {max_date!r}"
                ) from exc
            min_date = pred.get("min_filing_date")
            if min_date is not None and pd.Timestamp(min_date) > pd.Timestamp(max_date):
                raise FilerConfigError(
                    f"{filer_id}.predecessor_ciks[{p_idx}]: min_filing_date > max_filing_date"
                )
            active_ciks[pred_cik] = filer_id

        for sub_cik in entry.get("subsidiary_ciks") or []:
            validated = _validate_cik(sub_cik, f"{filer_id}.subsidiary_ciks")
            excluded_ciks[validated] = filer_id

        for unrel_cik in entry.get("unrelated_ciks") or []:
            _validate_cik(unrel_cik, f"{filer_id}.unrelated_ciks")

    overlap = set(active_ciks).intersection(excluded_ciks)
    if overlap:
        details = [
            f"{cik} (active={active_ciks[cik]}, subsidiary_of={excluded_ciks[cik]})"
            for cik in sorted(overlap)
        ]
        raise FilerConfigError(
            "Overlap between canonical/predecessor CIKs and subsidiary_ciks: " + ", ".join(details)
        )


def resolve_universe_filings(
    index_df: pd.DataFrame,
    filers: list[dict[str, Any]],
) -> pd.DataFrame:
    """Resolve raw EDGAR index rows to canonical filers using `config/filers.yaml` overrides.

    Applies primary CIK mappings and date-bounded `predecessor_ciks` handoffs while excluding
    all `subsidiary_ciks`. Returns a deduplicated DataFrame ready for XML ingestion.
    """
    validate_filers_config(filers)

    if index_df.empty:
        return index_df.copy()

    df = index_df.copy()
    df["date_filed"] = pd.to_datetime(df["date_filed"])

    resolved_parts: list[pd.DataFrame] = []
    for entry in filers:
        filer_id = entry["filer_id"]
        canonical_name = entry["canonical_name"]
        style = entry["style"]
        status = entry["status"]
        primary_cik = entry["cik"]

        primary_rows = df[df["cik"] == primary_cik].copy()
        if not primary_rows.empty:
            primary_rows["filer_id"] = filer_id
            primary_rows["canonical_name"] = canonical_name
            primary_rows["style"] = style
            primary_rows["filer_status"] = status
            primary_rows["source_role"] = "primary"
            resolved_parts.append(primary_rows)

        for pred in entry.get("predecessor_ciks") or []:
            pred_cik = pred["cik"]
            max_dt = pd.Timestamp(pred["max_filing_date"])
            min_dt = pd.Timestamp(pred["min_filing_date"]) if pred.get("min_filing_date") else None

            mask = (df["cik"] == pred_cik) & (df["date_filed"] <= max_dt)
            if min_dt is not None:
                mask &= df["date_filed"] >= min_dt

            pred_rows = df.loc[mask].copy()
            if not pred_rows.empty:
                pred_rows["filer_id"] = filer_id
                pred_rows["canonical_name"] = canonical_name
                pred_rows["style"] = style
                pred_rows["filer_status"] = status
                pred_rows["source_role"] = "predecessor"
                resolved_parts.append(pred_rows)

    if not resolved_parts:
        return pd.DataFrame(
            columns=[
                "filer_id",
                "canonical_name",
                "style",
                "filer_status",
                "source_role",
                *df.columns.tolist(),
            ]
        )

    universe = (
        pd.concat(resolved_parts, ignore_index=True)
        .drop_duplicates(subset=["filer_id", "accession_number", "form_type"])
        .sort_values(["filer_id", "date_filed", "accession_number"])
        .reset_index(drop=True)
    )
    front_cols = ["filer_id", "canonical_name", "style", "filer_status", "source_role"]
    other_cols = [c for c in universe.columns if c not in front_cols]
    return universe[front_cols + other_cols]


def audit_edgar_index(
    index_df: pd.DataFrame,
    filers: list[dict[str, Any]],
) -> AuditResult:
    """Run full entity-resolution and data-governance audit against `edgar_13f_index.parquet`."""
    universe = resolve_universe_filings(index_df, filers)

    report_rows: list[dict[str, Any]] = []
    dup_parts: list[pd.DataFrame] = []

    for entry in filers:
        filer_id = entry["filer_id"]
        sub = universe[universe["filer_id"] == filer_id].sort_values("date_filed")
        hr = sub[sub["form_type"] == "13F-HR"]
        hra = sub[sub["form_type"] == "13F-HR/A"]

        # Duplicate 13F-HR check within same index (year, quarter)
        dup_qtrs: list[str] = []
        if not hr.empty:
            q_counts = hr.groupby(["index_year", "index_qtr"]).size()
            dup_pairs = q_counts[q_counts > 1].index.tolist()
            if dup_pairs:
                dup_qtrs = [f"{y}Q{q}" for y, q in dup_pairs]
                dup_mask = hr.set_index(["index_year", "index_qtr"]).index.isin(dup_pairs)
                dup_parts.append(hr.loc[dup_mask].copy())

        # Quarter continuity check within filer's active span
        missing_qtrs: list[str] = []
        first_filed = ""
        last_filed = ""
        if not hr.empty:
            first_filed = hr["date_filed"].min().strftime("%Y-%m-%d")
            last_filed = hr["date_filed"].max().strftime("%Y-%m-%d")
            observed_qtrs = sorted(
                {(int(y), int(q)) for y, q in zip(hr["index_year"], hr["index_qtr"], strict=False)}
            )
            first_q, last_q = observed_qtrs[0], observed_qtrs[-1]
            expected_qtrs = list(iter_quarters(first_q[0], first_q[1], last_q[0], last_q[1]))
            missing_qtrs = [f"{y}Q{q}" for y, q in sorted(set(expected_qtrs) - set(observed_qtrs))]

        observed_names = sub["company_name"].drop_duplicates().tolist() if not sub.empty else []
        used_ciks = sub["cik"].drop_duplicates().tolist() if not sub.empty else []

        report_rows.append(
            {
                "filer_id": filer_id,
                "canonical_name": entry["canonical_name"],
                "style": entry["style"],
                "status": entry["status"],
                "primary_cik": entry["cik"],
                "used_ciks": ";".join(used_ciks),
                "hr_count": len(hr),
                "hra_count": len(hra),
                "first_filed": first_filed,
                "last_filed": last_filed,
                "missing_qtr_count": len(missing_qtrs),
                "missing_qtrs": ";".join(missing_qtrs),
                "dup_hr_qtrs": ";".join(dup_qtrs),
                "distinct_names": "; ".join(observed_names),
                "has_rename": len(observed_names) > 1,
            }
        )

    filer_report = pd.DataFrame(report_rows)
    duplicate_rows = (
        pd.concat(dup_parts, ignore_index=True)
        if dup_parts
        else pd.DataFrame(columns=universe.columns)
    )

    # Scan broader index for unmapped affiliate/successor CIKs matching search_aliases
    known_ciks: set[str] = set()
    for entry in filers:
        known_ciks.add(entry["cik"])
        for pred in entry.get("predecessor_ciks") or []:
            known_ciks.add(pred["cik"])
        for sub_cik in entry.get("subsidiary_ciks") or []:
            known_ciks.add(sub_cik)
        for unrel_cik in entry.get("unrelated_ciks") or []:
            known_ciks.add(unrel_cik)

    unmapped_records: list[dict[str, Any]] = []
    if not index_df.empty:
        for entry in filers:
            aliases = entry.get("search_aliases") or []
            if not aliases:
                continue
            mask = pd.Series(False, index=index_df.index)
            for alias in aliases:
                mask |= index_df["company_name"].str.contains(
                    alias, case=False, na=False, regex=False
                )
            hits = index_df.loc[mask & (~index_df["cik"].isin(known_ciks))]
            if hits.empty:
                continue
            for cik, grp in hits.groupby("cik"):
                unmapped_records.append(
                    {
                        "matched_filer_id": entry["filer_id"],
                        "cik": cik,
                        "company_name": grp.sort_values("date_filed")["company_name"].iloc[-1],
                        "hr_count": int((grp["form_type"] == "13F-HR").sum()),
                        "hra_count": int((grp["form_type"] == "13F-HR/A").sum()),
                        "first_filed": pd.to_datetime(grp["date_filed"]).min().strftime("%Y-%m-%d"),
                        "last_filed": pd.to_datetime(grp["date_filed"]).max().strftime("%Y-%m-%d"),
                    }
                )

    unmapped_df = pd.DataFrame(
        unmapped_records,
        columns=[
            "matched_filer_id",
            "cik",
            "company_name",
            "hr_count",
            "hra_count",
            "first_filed",
            "last_filed",
        ],
    )

    return AuditResult(
        universe_filings=universe,
        filer_report=filer_report,
        unmapped_candidates=unmapped_df,
        duplicate_quarter_rows=duplicate_rows,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit EDGAR 13F master index against config/filers.yaml and write manifest."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Input master index Parquet path (default: {DEFAULT_OUTPUT_PARQUET}).",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=DEFAULT_UNIVERSE_PARQUET,
        help=f"Output resolved universe filings Parquet (default: {DEFAULT_UNIVERSE_PARQUET}).",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=DEFAULT_AUDIT_CSV,
        help=f"Output CSV filer audit report (default: {DEFAULT_AUDIT_CSV}).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if duplicate 13F-HR quarters or unmapped alias CIKs are detected.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config()
    index_df = load_13f_index(args.index_path)

    result = audit_edgar_index(index_df, cfg.filers)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result.universe_filings.to_parquet(args.output_manifest, index=False)

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    result.filer_report.to_csv(args.output_report, index=False)

    hr_total = int((result.universe_filings["form_type"] == "13F-HR").sum())
    hra_total = int((result.universe_filings["form_type"] == "13F-HR/A").sum())
    print(
        f"Resolved {len(result.universe_filings):,} filings "
        f"({hr_total:,} 13F-HR, {hra_total:,} 13F-HR/A) "
        f"across {result.filer_report['filer_id'].nunique()} canonical filers "
        f"-> {args.output_manifest}"
    )
    print(f"Audit report written -> {args.output_report}")

    renames = result.filer_report[result.filer_report["has_rename"]]
    if not renames.empty:
        print(f"\nLegal name transitions tracked ({len(renames)}):")
        for _, r in renames.iterrows():
            print(f"  - {r['filer_id']}: {r['distinct_names']}")

    gaps = result.filer_report[result.filer_report["missing_qtr_count"] > 0]
    if not gaps.empty:
        print(f"\nFilers with internal quarter gaps ({len(gaps)}):")
        for _, r in gaps.iterrows():
            print(
                f"  - {r['filer_id']}: {r['missing_qtr_count']} missing quarters "
                f"({r['missing_qtrs']})"
            )

    if not result.duplicate_quarter_rows.empty:
        print(
            f"\n[WARN] Duplicate 13F-HR filings detected in same quarter: "
            f"{len(result.duplicate_quarter_rows)} rows"
        )

    if not result.unmapped_candidates.empty:
        print(
            f"\n[WARN] Unmapped CIKs matching search_aliases ({len(result.unmapped_candidates)}):"
        )
        print(result.unmapped_candidates.to_string(index=False))

    if args.strict and (
        not result.duplicate_quarter_rows.empty or not result.unmapped_candidates.empty
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
