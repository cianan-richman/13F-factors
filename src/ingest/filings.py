"""Download, cache, and parse SEC EDGAR `13F-HR` and `13F-HR/A` complete submissions.

Reads the deduplicated manifest produced by `src/ingest/entity_audit.py`
(`data/processed/universe_13f_filings.parquet`), downloads each filing's complete SGML
submission `.txt` in a single SEC request (cached locally as `.txt.gz`), parses both the
cover-page XML (`primary_doc.xml`) and holdings XML (`informationTable`), enforces the
Jan 3, 2023 SEC value-unit rule ($ thousands before 2023-01-03 -> exact dollars from
2023-01-03 onward), and writes:

  - `data/processed/edgar_13f_filings_meta.parquet`
  - `data/processed/edgar_13f_holdings_raw.parquet`

Usage:
    python -m src.ingest.filings
    python -m src.ingest.filings --limit 20
    python -m src.ingest.filings --parse-only
"""

from __future__ import annotations

import argparse
import gzip
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

from src.config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, load_config
from src.ingest.edgar_client import EdgarBlockedError, EdgarClient
from src.ingest.entity_audit import DEFAULT_UNIVERSE_PARQUET
from src.ingest.master_index import ARCHIVES_BASE_URL

logger = logging.getLogger(__name__)

RAW_FILINGS_DIR = RAW_DATA_DIR / "edgar_filings"
DEFAULT_META_PARQUET = PROCESSED_DATA_DIR / "edgar_13f_filings_meta.parquet"
DEFAULT_HOLDINGS_RAW_PARQUET = PROCESSED_DATA_DIR / "edgar_13f_holdings_raw.parquet"
DEFAULT_FAILURES_CSV = OUTPUTS_DIR / "edgar_ingest_failures.csv"
OK_PARSE_STATUSES = {"ok", "zero_holdings"}

# SEC Form 13F rule change: filings submitted on/after 2023-01-03 report <value> in exact dollars;
# filings before 2023-01-03 reported <value> in thousands of dollars.
SEC_VALUE_DOLLARS_CUTOFF = pd.Timestamp("2023-01-03")

_DOC_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL | re.IGNORECASE)
_TYPE_RE = re.compile(r"<TYPE>\s*([^\r\n<]+)", re.IGNORECASE)
_XML_BLOCK_RE = re.compile(r"<XML>(.*?)</XML>", re.DOTALL | re.IGNORECASE)

_HEADER_PATTERNS = {
    "acceptance_datetime_raw": re.compile(
        r"<ACCEPTANCE-DATETIME>\s*(\d{14})", re.IGNORECASE
    ),
    "header_period_of_report": re.compile(
        r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", re.IGNORECASE
    ),
    "header_filed_date": re.compile(
        r"FILED AS OF DATE:\s*(\d{8})", re.IGNORECASE
    ),
    "header_form_type": re.compile(
        r"CONFORMED SUBMISSION TYPE:\s*([^\r\n]+)", re.IGNORECASE
    ),
}


def raw_filing_path(cik: str, accession_number: str, raw_dir: Path = RAW_FILINGS_DIR) -> Path:
    """Return local gzip cache path for a complete SEC submission `.txt`."""
    return raw_dir / cik / f"{accession_number}.txt.gz"


def _fetch_submission_via_directory_fallback(
    client: EdgarClient,
    *,
    cik: str,
    accession_number: str,
) -> bytes:
    """Fallback loader using `index.json` + individual XML/header files when `.txt` fails."""
    cik_int = int(cik)
    acc_nodash = accession_number.replace("-", "")
    dir_url = f"{ARCHIVES_BASE_URL}/edgar/data/{cik_int}/{acc_nodash}"
    idx_resp = client.get(f"{dir_url}/index.json")
    items = idx_resp.json().get("directory", {}).get("item", [])
    names = [str(it.get("name", "")) for it in items]

    parts: list[str] = [f"<SEC-DOCUMENT>{accession_number}.txt"]

    # Fetch SGML header if available (preserves exact ACCEPTANCE-DATETIME)
    hdr_name = next((n for n in names if n.lower().endswith(".hdr.sgml")), None)
    if hdr_name:
        hdr_resp = client.get(f"{dir_url}/{hdr_name}", allow_not_found=True)
        if hdr_resp.status_code == 200:
            parts.append(hdr_resp.text)

    # Fetch top-level XML documents (excluding XSLT rendered views)
    xml_files = [
        n
        for n in names
        if n.lower().endswith(".xml")
        and not n.lower().startswith("xsl")
        and "index" not in n.lower()
    ]
    if not xml_files:
        raise RuntimeError(
            f"Directory fallback found no XML files in {dir_url}/index.json (items={names[:10]})"
        )
    for seq, xml_name in enumerate(xml_files, start=1):
        x_resp = client.get(f"{dir_url}/{xml_name}")
        doc_type = "PRIMARY_XML" if "primary" in xml_name.lower() else "XML_DOC"
        parts.append(
            f"<DOCUMENT>\n<TYPE>{doc_type}\n<SEQUENCE>{seq}\n"
            f"<FILENAME>{xml_name}\n<TEXT>\n<XML>\n{x_resp.text}\n</XML>\n</TEXT>\n</DOCUMENT>"
        )

    parts.append("</SEC-DOCUMENT>")
    return "\n".join(parts).encode("utf-8")


def _read_gzip_text(path: Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace")


def fetch_filing_submission(
    client: EdgarClient,
    *,
    filename: str,
    cik: str,
    accession_number: str,
    raw_dir: Path = RAW_FILINGS_DIR,
    force_refresh: bool = False,
) -> Path:
    """Download and cache a complete SEC EDGAR submission `.txt` as gzipped bytes.

    Validates cached `.txt.gz` readability (self-healing corrupt caches by re-fetching) and
    falls back automatically to `index.json` + individual XML files on HTTP 5xx/404.
    """
    dest = raw_filing_path(cik, accession_number, raw_dir)
    if dest.exists() and not force_refresh:
        try:
            _read_gzip_text(dest)
            return dest
        except (gzip.BadGzipFile, EOFError, OSError):
            logger.warning("Corrupt cached filing at %s; deleting and re-fetching.", dest)
            dest.unlink(missing_ok=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{ARCHIVES_BASE_URL}/{filename.lstrip('/')}"
    try:
        payload = client.get(url).content
    except EdgarBlockedError:
        raise
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status in {404, 500, 502, 503, 504}:
            logger.warning(
                "Monolithic .txt failed with HTTP %s for %s; falling back to directory index.json",
                status,
                accession_number,
            )
            payload = _fetch_submission_via_directory_fallback(
                client,
                cik=cik,
                accession_number=accession_number,
            )
        else:
            raise

    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(payload))
    tmp.replace(dest)
    return dest


def _localname(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _find_child_text(elem: etree._Element, target_localname: str) -> str:
    target = target_localname.lower()
    for child in elem:
        if _localname(child.tag).lower() == target:
            return (child.text or "").strip()
    return ""


def _find_descendant_text(elem: etree._Element, target_localname: str) -> str:
    target = target_localname.lower()
    for node in elem.iter():
        if _localname(node.tag).lower() == target:
            text = (node.text or "").strip()
            if text:
                return text
    return ""


def _parse_xml_root(xml_text: str) -> etree._Element | None:
    cleaned = xml_text.strip()
    if not cleaned:
        return None
    # Strip leading junk before first '<' if present
    lt = cleaned.find("<")
    if lt > 0:
        cleaned = cleaned[lt:]
    parser = etree.XMLParser(recover=True, huge_tree=True, remove_comments=True)
    try:
        return etree.fromstring(cleaned.encode("utf-8", errors="replace"), parser=parser)
    except etree.XMLSyntaxError:
        return None


def extract_sec_header_fields(raw_text: str) -> dict[str, str]:
    """Extract key timestamps and dates from the SGML `<SEC-HEADER>` block."""
    doc_start = raw_text.find("<DOCUMENT>")
    header_slice = raw_text[:doc_start] if doc_start != -1 else raw_text[:8192]

    out: dict[str, str] = {}
    for key, pattern in _HEADER_PATTERNS.items():
        m = pattern.search(header_slice)
        out[key] = m.group(1).strip() if m else ""
    return out


def extract_document_xmls(raw_text: str) -> tuple[str | None, str | None]:
    """Extract `(primary_doc_xml, information_table_xml)` from a complete SEC `.txt` submission."""
    primary_xml: str | None = None
    infotable_xml: str | None = None

    for match in _DOC_RE.finditer(raw_text):
        doc_block = match.group(1)
        type_match = _TYPE_RE.search(doc_block)
        doc_type = type_match.group(1).strip().upper() if type_match else ""

        xml_match = _XML_BLOCK_RE.search(doc_block)
        if not xml_match:
            continue
        xml_payload = xml_match.group(1).strip()
        if not xml_payload:
            continue

        if doc_type in {"13F-HR", "13F-HR/A"} and primary_xml is None:
            primary_xml = xml_payload
        elif "INFORMATION TABLE" in doc_type and infotable_xml is None:
            infotable_xml = xml_payload
        else:
            # Fallback detection via root XML tag localname in case <TYPE> is non-standard
            root = _parse_xml_root(xml_payload[:2048] if len(xml_payload) > 200000 else xml_payload)
            if root is not None:
                root_name = _localname(root.tag).lower()
                if root_name == "edgarsubmission" and primary_xml is None:
                    primary_xml = xml_payload
                elif root_name == "informationtable" and infotable_xml is None:
                    infotable_xml = xml_payload

    return primary_xml, infotable_xml


def _normalize_date_str(val: str) -> str:
    """Convert MM-DD-YYYY, YYYYMMDD, or YYYY-MM-DD strings into ISO `YYYY-MM-DD`."""
    cleaned = (val or "").strip()
    if not cleaned:
        return ""
    for fmt in ("%m-%d-%Y", "%Y-%m-%d", "%Y%m%d", "%m/%d/%Y"):
        try:
            return pd.to_datetime(cleaned, format=fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    ts = pd.to_datetime(cleaned, errors="coerce")
    return "" if pd.isna(ts) else ts.strftime("%Y-%m-%d")


def _parse_bool(val: str) -> bool:
    return (val or "").strip().lower() in {"true", "1", "y", "yes"}


def _parse_int(val: str) -> int | None:
    cleaned = (val or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return None


def _parse_float(val: str) -> float | None:
    cleaned = (val or "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_primary_xml(xml_text: str) -> dict[str, Any]:
    """Parse cover-page metadata from a 13F `primary_doc.xml` payload."""
    root = _parse_xml_root(xml_text)
    if root is None:
        return {}

    period_raw = _find_descendant_text(root, "periodOfReport")
    if not period_raw:
        period_raw = _find_descendant_text(root, "reportCalendarOrQuarter")

    is_amendment = _parse_bool(_find_descendant_text(root, "isAmendment"))
    amendment_no = _parse_int(_find_descendant_text(root, "amendmentNo"))
    amendment_type = _find_descendant_text(root, "amendmentType").upper()
    report_type = _find_descendant_text(root, "reportType").upper()
    is_conf_omitted = _parse_bool(
        _find_descendant_text(root, "isConfidentialOmitted")
        or _find_descendant_text(root, "confidentialOmitted")
    )
    table_entry_total = _parse_int(_find_descendant_text(root, "tableEntryTotal"))
    table_value_total_raw = _parse_float(_find_descendant_text(root, "tableValueTotal"))

    # Collect CIKs declared under <otherManagersInfo> / <otherManagers2Info> for dedup auditing
    other_ciks: list[str] = []
    for node in root.iter():
        if _localname(node.tag).lower() in {"othermanager", "othermanager2"}:
            cik_txt = _find_descendant_text(node, "cik")
            if cik_txt.isdigit():
                padded = f"{int(cik_txt):010d}"
                if padded not in other_ciks:
                    other_ciks.append(padded)

    return {
        "period_of_report": _normalize_date_str(period_raw),
        "is_amendment": is_amendment,
        "amendment_no": amendment_no,
        "amendment_type": amendment_type,
        "report_type": report_type,
        "is_confidential_omitted": is_conf_omitted,
        "table_entry_total": table_entry_total,
        "table_value_total_raw": table_value_total_raw,
        "other_managers_ciks": ";".join(other_ciks),
    }


def parse_infotable_xml(xml_text: str, *, value_multiplier: int) -> list[dict[str, Any]]:
    """Parse every `<infoTable>` holding row from a 13F information table XML."""
    root = _parse_xml_root(xml_text)
    if root is None:
        return []

    rows: list[dict[str, Any]] = []
    row_idx = 0
    for node in root.iter():
        if _localname(node.tag).lower() != "infotable":
            continue

        cusip = re.sub(r"\s+", "", _find_child_text(node, "cusip")).upper()
        name_of_issuer = _find_child_text(node, "nameOfIssuer")
        title_of_class = _find_child_text(node, "titleOfClass")
        value_raw = _parse_float(_find_child_text(node, "value"))
        value_usd = value_raw * value_multiplier if value_raw is not None else None

        ssh_prnamt: float | None = None
        ssh_prnamt_type = ""
        voting_sole: int | None = None
        voting_shared: int | None = None
        voting_none: int | None = None

        for child in node:
            cname = _localname(child.tag).lower()
            if cname == "shrsorprnamt":
                ssh_prnamt = _parse_float(_find_child_text(child, "sshPrnamt"))
                ssh_prnamt_type = _find_child_text(child, "sshPrnamtType").upper()
            elif cname == "votingauthority":
                voting_sole = _parse_int(_find_child_text(child, "Sole"))
                voting_shared = _parse_int(_find_child_text(child, "Shared"))
                voting_none = _parse_int(_find_child_text(child, "None"))

        # Skip dummy wind-down placeholder rows (e.g., cusip=000000000, "No securities", 0 shares)
        if cusip == "000000000" and (ssh_prnamt or 0.0) == 0.0 and (value_raw or 0.0) == 0.0:
            continue

        put_call = _find_child_text(node, "putCall").upper()
        discretion = _find_child_text(node, "investmentDiscretion").upper()
        other_manager = _find_child_text(node, "otherManager")

        rows.append(
            {
                "row_idx": row_idx,
                "cusip": cusip,
                "name_of_issuer": name_of_issuer,
                "title_of_class": title_of_class,
                "value_raw": value_raw,
                "value_usd": value_usd,
                "ssh_prnamt": ssh_prnamt,
                "ssh_prnamt_type": ssh_prnamt_type,
                "put_call": put_call,
                "investment_discretion": discretion,
                "other_manager": other_manager,
                "voting_sole": voting_sole,
                "voting_shared": voting_shared,
                "voting_none": voting_none,
            }
        )
        row_idx += 1

    return rows


def value_multiplier_for_filing(filing_date: pd.Timestamp) -> int:
    """Return SEC `<value>` dollar multiplier (1000 before 2023-01-03, 1 on/after)."""
    return 1 if filing_date >= SEC_VALUE_DOLLARS_CUTOFF else 1000


def parse_filing_submission(
    raw_text: str,
    manifest_row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Parse a complete submission `.txt` into `(filing_meta_dict, holdings_row_list)`."""
    header = extract_sec_header_fields(raw_text)
    primary_xml, infotable_xml = extract_document_xmls(raw_text)

    date_filed = pd.Timestamp(manifest_row["date_filed"])
    acc_raw = header.get("acceptance_datetime_raw", "")
    acceptance_dt_iso = ""
    acceptance_date = date_filed.strftime("%Y-%m-%d")
    if len(acc_raw) == 14:
        try:
            acc_ts = pd.to_datetime(acc_raw, format="%Y%m%d%H%M%S")
            acceptance_dt_iso = acc_ts.strftime("%Y-%m-%dT%H:%M:%S")
            acceptance_date = acc_ts.strftime("%Y-%m-%d")
        except ValueError:
            pass

    primary_meta = parse_primary_xml(primary_xml) if primary_xml else {}
    period_of_report = primary_meta.get("period_of_report") or _normalize_date_str(
        header.get("header_period_of_report", "")
    )

    # Use acceptance_date (fallback date_filed) for statutory Jan 3, 2023 SEC unit switch,
    # then apply strictly point-in-time self-check via the filing's own implied share price
    # to catch filers that reported in $1,000s post-2023 (e.g. Baupost) or exact $ pre-2023
    # (e.g. Select Equity, Lone Pine Q4 2014).
    effective_filed_ts = pd.Timestamp(acceptance_date)
    statutory_multiplier = value_multiplier_for_filing(effective_filed_ts)

    holdings_rows: list[dict[str, Any]] = []
    if infotable_xml:
        holdings_rows = parse_infotable_xml(infotable_xml, value_multiplier=statutory_multiplier)

    raw_ratios = [
        r["value_raw"] / r["ssh_prnamt"]
        for r in holdings_rows
        if r["ssh_prnamt_type"] == "SH"
        and r["put_call"] == ""
        and r["value_raw"] is not None
        and r["ssh_prnamt"] is not None
        and r["ssh_prnamt"] > 0
        and r["value_raw"] > 0
    ]
    applied_multiplier = statutory_multiplier
    value_unit_corrected = False
    value_unit_reason = ""

    if raw_ratios:
        raw_med = float(np.median(raw_ratios))
        stat_med = raw_med * statutory_multiplier
        if statutory_multiplier == 1 and stat_med < 1.0 and 1.0 <= raw_med * 1000.0 <= 5_000.0:
            applied_multiplier = 1000
            value_unit_corrected = True
            value_unit_reason = "post_2023_reported_in_thousands"
        elif statutory_multiplier == 1000 and stat_med > 5_000.0 and 1.0 <= raw_med <= 5_000.0:
            applied_multiplier = 1
            value_unit_corrected = True
            value_unit_reason = "pre_2023_reported_in_dollars"

        if value_unit_corrected:
            for r in holdings_rows:
                if r["value_raw"] is not None:
                    r["value_usd"] = r["value_raw"] * applied_multiplier

        median_implied_price = raw_med * applied_multiplier
        value_unit_suspect = bool(median_implied_price < 1.0 or median_implied_price > 5_000.0)
    else:
        median_implied_price = np.nan
        value_unit_suspect = False

    if primary_xml is None and infotable_xml is None:
        parse_status = "no_xml_found"
    elif infotable_xml is None:
        parse_status = "no_info_table"
    elif not holdings_rows:
        parse_status = "zero_holdings"
    else:
        parse_status = "ok"

    # Attach filing keys to every holding row
    filer_id = manifest_row["filer_id"]
    accession_number = manifest_row["accession_number"]
    form_type = manifest_row["form_type"]
    cik = manifest_row["cik"]

    for row in holdings_rows:
        row["filer_id"] = filer_id
        row["cik"] = cik
        row["accession_number"] = accession_number
        row["form_type"] = form_type
        row["date_filed"] = date_filed.strftime("%Y-%m-%d")
        row["acceptance_date"] = acceptance_date
        row["acceptance_datetime"] = acceptance_dt_iso
        row["period_of_report"] = period_of_report

    filing_meta = {
        "filer_id": filer_id,
        "canonical_name": manifest_row["canonical_name"],
        "style": manifest_row["style"],
        "cik": cik,
        "source_role": manifest_row.get("source_role", "primary"),
        "accession_number": accession_number,
        "form_type": form_type,
        "date_filed": date_filed.strftime("%Y-%m-%d"),
        "acceptance_date": acceptance_date,
        "acceptance_datetime": acceptance_dt_iso,
        "period_of_report": period_of_report,
        "is_amendment": primary_meta.get("is_amendment", form_type == "13F-HR/A"),
        "amendment_no": primary_meta.get("amendment_no"),
        "amendment_type": primary_meta.get("amendment_type", ""),
        "report_type": primary_meta.get("report_type", ""),
        "is_confidential_omitted": primary_meta.get("is_confidential_omitted", False),
        "table_entry_total": primary_meta.get("table_entry_total"),
        "table_value_total_raw": primary_meta.get("table_value_total_raw"),
        "statutory_multiplier": statutory_multiplier,
        "value_multiplier": applied_multiplier,
        "value_unit_corrected": value_unit_corrected,
        "value_unit_reason": value_unit_reason,
        "parsed_row_count": len(holdings_rows),
        "median_implied_price": median_implied_price,
        "value_unit_suspect": value_unit_suspect,
        "other_managers_ciks": primary_meta.get("other_managers_ciks", ""),
        "parse_status": parse_status,
        "error_detail": "",
    }
    return filing_meta, holdings_rows


def _build_error_meta(manifest_row: dict[str, Any], status: str, detail: str) -> dict[str, Any]:
    date_filed = pd.Timestamp(manifest_row["date_filed"]).strftime("%Y-%m-%d")
    return {
        "filer_id": manifest_row["filer_id"],
        "canonical_name": manifest_row["canonical_name"],
        "style": manifest_row["style"],
        "cik": str(manifest_row["cik"]),
        "source_role": manifest_row.get("source_role", "primary"),
        "accession_number": str(manifest_row["accession_number"]),
        "form_type": str(manifest_row["form_type"]),
        "date_filed": date_filed,
        "acceptance_date": date_filed,
        "acceptance_datetime": "",
        "period_of_report": "",
        "is_amendment": str(manifest_row["form_type"]) == "13F-HR/A",
        "amendment_no": None,
        "amendment_type": "",
        "report_type": "",
        "is_confidential_omitted": False,
        "table_entry_total": None,
        "table_value_total_raw": None,
        "statutory_multiplier": value_multiplier_for_filing(pd.Timestamp(date_filed)),
        "value_multiplier": value_multiplier_for_filing(pd.Timestamp(date_filed)),
        "value_unit_corrected": False,
        "value_unit_reason": "",
        "parsed_row_count": 0,
        "median_implied_price": np.nan,
        "value_unit_suspect": False,
        "other_managers_ciks": "",
        "parse_status": status,
        "error_detail": detail,
    }


def ingest_universe_filings(
    manifest_df: pd.DataFrame,
    client: EdgarClient | None = None,
    *,
    raw_dir: Path = RAW_FILINGS_DIR,
    meta_output_path: Path = DEFAULT_META_PARQUET,
    holdings_output_path: Path = DEFAULT_HOLDINGS_RAW_PARQUET,
    failures_output_path: Path = DEFAULT_FAILURES_CSV,
    parse_only: bool = False,
    force_refresh: bool = False,
    max_workers: int = 5,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download/cache and parse all filings listed in `manifest_df` with fault isolation.

    Downloads run concurrently up to `max_workers` while respecting `EdgarClient`'s global
    thread-safe rate limit. Single-filing failures are captured in `meta_df` (`parse_status =
    "fetch_error"` / `"parse_error"`) and written to `failures_output_path` rather than aborting
    the entire batch. HTTP 403 (`EdgarBlockedError`) still fails fast to avoid hammering SEC.
    """
    if not parse_only and client is None:
        raise ValueError("EdgarClient is required unless parse_only=True.")

    records = manifest_df.to_dict(orient="records")
    workers = 1 if parse_only else max(1, min(max_workers, max(1, len(records))))

    def _process_one(row: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        cik = str(row["cik"])
        accession = str(row["accession_number"])
        filename = str(row["filename"])

        try:
            if parse_only:
                path = raw_filing_path(cik, accession, raw_dir)
                if not path.exists():
                    return None, []
            else:
                assert client is not None
                path = fetch_filing_submission(
                    client,
                    filename=filename,
                    cik=cik,
                    accession_number=accession,
                    raw_dir=raw_dir,
                    force_refresh=force_refresh,
                )
            raw_text = _read_gzip_text(path)
        except EdgarBlockedError:
            raise
        except Exception as exc:
            logger.error("Failed to fetch/read %s (%s): %s", accession, cik, exc)
            return _build_error_meta(row, "fetch_error", f"{type(exc).__name__}: {exc}"), []

        try:
            return parse_filing_submission(raw_text, row)
        except Exception as exc:
            logger.error("Failed to parse %s (%s): %s", accession, cik, exc)
            return _build_error_meta(row, "parse_error", f"{type(exc).__name__}: {exc}"), []

    meta_rows: list[dict[str, Any]] = []
    all_holdings: list[dict[str, Any]] = []
    missing_cache = 0

    if workers == 1:
        iterator = tqdm(records, desc="13F filings", unit="filing") if show_progress else records
        for row in iterator:
            meta, holdings = _process_one(row)
            if meta is None:
                missing_cache += 1
                continue
            meta_rows.append(meta)
            all_holdings.extend(holdings)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_one, row): row for row in records}
            pbar = (
                tqdm(total=len(records), desc="13F filings", unit="filing")
                if show_progress
                else None
            )
            try:
                for fut in as_completed(futures):
                    meta, holdings = fut.result()
                    if meta is None:
                        missing_cache += 1
                    else:
                        meta_rows.append(meta)
                        all_holdings.extend(holdings)
                    if pbar is not None:
                        pbar.update(1)
            finally:
                if pbar is not None:
                    pbar.close()

    if missing_cache:
        logger.warning("Skipped %d uncached filings under --parse-only.", missing_cache)

    meta_df = pd.DataFrame(meta_rows)
    holdings_df = pd.DataFrame(all_holdings)

    if not meta_df.empty:
        meta_df["date_filed"] = pd.to_datetime(meta_df["date_filed"])
        meta_df["acceptance_date"] = pd.to_datetime(meta_df["acceptance_date"])
        meta_df["period_of_report"] = pd.to_datetime(meta_df["period_of_report"], errors="coerce")
        meta_df = meta_df.sort_values(
            ["filer_id", "period_of_report", "acceptance_datetime", "accession_number"]
        ).reset_index(drop=True)

    if not holdings_df.empty:
        holdings_df["date_filed"] = pd.to_datetime(holdings_df["date_filed"])
        holdings_df["acceptance_date"] = pd.to_datetime(holdings_df["acceptance_date"])
        holdings_df["period_of_report"] = pd.to_datetime(
            holdings_df["period_of_report"], errors="coerce"
        )
        front_cols = [
            "filer_id",
            "cik",
            "accession_number",
            "form_type",
            "period_of_report",
            "date_filed",
            "acceptance_date",
            "acceptance_datetime",
            "row_idx",
            "cusip",
            "name_of_issuer",
            "title_of_class",
            "value_raw",
            "value_usd",
            "ssh_prnamt",
            "ssh_prnamt_type",
            "put_call",
            "investment_discretion",
            "other_manager",
            "voting_sole",
            "voting_shared",
            "voting_none",
        ]
        holdings_df = (
            holdings_df[front_cols]
            .sort_values(["filer_id", "date_filed", "accession_number", "row_idx"])
            .reset_index(drop=True)
        )

    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    holdings_output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(meta_output_path, index=False)
    holdings_df.to_parquet(holdings_output_path, index=False)

    # Write or clear failures manifest
    if not meta_df.empty:
        failed = meta_df[~meta_df["parse_status"].isin(OK_PARSE_STATUSES)]
        if not failed.empty:
            failures_output_path.parent.mkdir(parents=True, exist_ok=True)
            failed[
                [
                    "filer_id",
                    "cik",
                    "accession_number",
                    "form_type",
                    "date_filed",
                    "parse_status",
                    "error_detail",
                ]
            ].to_csv(failures_output_path, index=False)
        elif failures_output_path.exists():
            failures_output_path.unlink()

    return meta_df, holdings_df


def check_cross_filer_other_managers(meta_df: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return any `(filer_id, accession_number, overlap_cik)` where a filing lists another
    tracked canonical CIK inside `<otherManagersInfo>` (excluding self-references).
    """
    if meta_df.empty:
        return []
    cik_to_filer = dict(zip(meta_df["cik"], meta_df["filer_id"], strict=False))
    overlaps: list[tuple[str, str, str]] = []
    for _, row in meta_df.iterrows():
        raw_others = str(row.get("other_managers_ciks") or "")
        if not raw_others:
            continue
        for other_cik in raw_others.split(";"):
            other_cik = other_cik.strip()
            if other_cik in cik_to_filer and cik_to_filer[other_cik] != row["filer_id"]:
                overlaps.append((row["filer_id"], row["accession_number"], other_cik))
    return overlaps


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and parse 13F-HR / 13F-HR/A XML filings for the curated universe."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_UNIVERSE_PARQUET,
        help=f"Input resolved universe filings manifest (default: {DEFAULT_UNIVERSE_PARQUET}).",
    )
    parser.add_argument(
        "--filer-id",
        type=str,
        default=None,
        help="Optional single filer_id to ingest/parse (useful for debugging).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of filings to process (for smoke testing).",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Skip network requests and re-parse already cached .txt.gz filings only.",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-download submissions even if already cached locally.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Concurrent download workers (default: config edgar.max_workers).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any failed/malformed filing, unit anomaly, or cross-filer overlap.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config()
    edgar_cfg = cfg.settings["edgar"]

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Universe manifest not found at {args.manifest}. "
            "Run `python -m src.ingest.entity_audit --strict` first."
        )

    manifest = pd.read_parquet(args.manifest)
    if args.filer_id:
        manifest = manifest[manifest["filer_id"] == args.filer_id].reset_index(drop=True)
    if args.limit is not None:
        manifest = manifest.head(args.limit).reset_index(drop=True)

    workers = args.workers if args.workers is not None else int(edgar_cfg.get("max_workers", 5))

    if args.parse_only:
        meta_df, holdings_df = ingest_universe_filings(
            manifest,
            client=None,
            parse_only=True,
        )
    else:
        with EdgarClient(
            user_agent=cfg.sec_user_agent,
            max_requests_per_sec=float(edgar_cfg.get("max_requests_per_sec", 8.0)),
            max_retries=int(edgar_cfg.get("max_retries", 5)),
            backoff_base_sec=float(edgar_cfg.get("backoff_base_sec", 1.0)),
            pool_maxsize=max(workers + 2, 10),
        ) as client:
            meta_df, holdings_df = ingest_universe_filings(
                manifest,
                client=client,
                force_refresh=args.force_refresh,
                max_workers=workers,
            )

    overlaps = check_cross_filer_other_managers(meta_df)
    suspect_units = int(meta_df["value_unit_suspect"].sum()) if not meta_df.empty else 0
    corrected_units = int(meta_df["value_unit_corrected"].sum()) if not meta_df.empty else 0
    status_counts = meta_df["parse_status"].value_counts().to_dict() if not meta_df.empty else {}
    failed_count = (
        int((~meta_df["parse_status"].isin(OK_PARSE_STATUSES)).sum()) if not meta_df.empty else 0
    )

    print(
        f"Parsed {len(meta_df):,} filings -> {len(holdings_df):,} raw holding rows\n"
        f"  Meta:     {DEFAULT_META_PARQUET}\n"
        f"  Holdings: {DEFAULT_HOLDINGS_RAW_PARQUET}\n"
        f"  Status:   {status_counts}\n"
        f"  Unit auto-corrected: {corrected_units} | Remaining unit suspect: {suspect_units} | "
        f"Cross-filer <otherManagers> overlaps: {len(overlaps)}"
    )
    if failed_count:
        print(f"  [WARN] {failed_count} failed/incomplete filings -> {DEFAULT_FAILURES_CSV}")
    if not meta_df.empty:
        hra = meta_df[meta_df["form_type"] == "13F-HR/A"]
        if not hra.empty:
            print(
                "  13F-HR/A amendmentType breakdown:",
                hra["amendment_type"].replace("", "(unspecified)").value_counts().to_dict(),
            )
    if args.strict and (failed_count > 0 or suspect_units > 0 or len(overlaps) > 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
