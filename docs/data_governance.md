# Data Governance: EDGAR Index Auditing & Entity Mapping

This document records the institutional universe selection rules, manual CIK mapping/deduplication overrides, and the reproducible audit workflow executed after every SEC EDGAR `master.idx` refresh.

---

## 1. Universe Selection & Scope Cuts

We track an **audited 100-fund universe of concentrated active institutional US equity managers** (`84` active + `16` closed 2015-era peers, expanded from our initial `36`-fund baseline; see [research_log.md](research_log.md) Section 2 for the full 13,075 → 100 screening funnel and breadth upgrade), pooled identically across all positioning factors and governed by two explicit quantitative gates verified directly from SEC filings:
- **Concentration Gate:** Median quarterly long-`SH` holdings **`10–100` positions**.
- **Institutional Scale Gate:** Median quarterly 13F long-equity book **`>= $2.0B`** (tracked median book across the 100-entity cohort: **`\$850.1B`**; 2026 active total: **`\$829.3B`**).

- **Cohort Composition (100 audited entities, pooled identically):**
  - 30 Concentrated Growth / TMT L/S & Growth Managers
  - 27 Concentrated Fundamental Value & Contrarian L/S Managers
  - 27 Long-Horizon Quality Compounders & Fundamental Generalists
  - 16 Wound-Down 2015-Era Fundamental Peers (`Blue Ridge`, `Melvin`, `Highfields`, `Tourbillon`, `Eton Park`, `SPO Advisory`, `Omega`, `Hitchwood`, `Ivory`, `IVA`, `Vontobel`, `Northern Cross`, `Trilogy`, `Holland`, `Santa Barbara`, `ThomasPartners`) to mitigate **filer-level survivorship bias** across the `2015–2023` in-sample window
- **Deliberately Excluded:**
  - *Passive indexers & mega-asset gatherers* (Vanguard, BlackRock, State Street, Fidelity) — dilute active crowding and concentration shifts with passive index weights.
  - *Quant / stat-arb / market makers / multi-manager pod shops* (RenTech, Citadel, Jane Street, Two Sigma, D.E. Shaw, Millennium, Point72) — high turnover, delta-one option hedges, and internal pod netting make 45-day-lagged 13Fs stale within days.
  - *Over-diversified / quant-sleeve / multi-sleeve platforms failing `<= 100` median positions* (e.g., empirically removed `Maverick Capital` [238.5 pos; quant sleeve], `Chilton Investment` [145.5 pos; wealth arm], `Select Equity` [138.0 pos; multi-sleeve]).
  - *Sub-\$2B small-AUM managers failing `>= \$2.0B` median equity book* (empirically removed `Light Street` [\$0.71B], `Abdiel Capital` [\$1.45B], `Hound Partners` [\$1.57B]).
  - *Dedicated activists & merger-arb funds* (Elliott, Starboard, Icahn, Pershing Square) — deferred to future work because Schedule 13D / media leaks front-run Day-45 13Fs and M&A targets trade like cash deal spreads.
  - *Berkshire Hathaway* — \$300B+ insurance conglomerate whose size and single-stock concentration distort aggregate dollar-flow metrics.

---

## 2. Manual Entity Mapping & Deduplication Registry (`config/filers.yaml`)

Rather than maintaining ad-hoc scripts, [config/filers.yaml](../config/filers.yaml) acts as the single declarative override table consumed by `src/ingest/entity_audit.py` and downstream XML ingestion (`src/ingest/filings.py`). Each canonical filer defines:

- `filer_id` & `cik`: Stable canonical identifier and current 10-digit zero-padded parent CIK.
- `predecessor_ciks`: Explicit date-bounded CIK migration overrides (`max_filing_date`, inclusive) to stitch legal-entity restructurings seamlessly with zero duplicate quarters.
- `subsidiary_ciks`: Explicitly excluded overlapping subsidiary/affiliate CIKs (`13F-HR` duplicates and `13F-NT` sub-funds).
- `unrelated_ciks`: Known false-positive CIKs sharing a substring with a fund's name (e.g., `TCI Wealth Advisors`, `Chilton Capital Management`, `Abrams Bison`, `Glenview State Bank`).
- `search_aliases`: Name substrings scanned on every audit run so *new* successor or affiliate CIKs are flagged automatically when `master.idx` is refreshed.

---

## 3. Verified Audit Log (`2015 Q1 → 2026 Q4`)

Auditing the full 308,891-filing SEC EDGAR index (`2015-01-02` → `2026-11-19`) uncovered and resolved the following real-world entity edge cases:

### A. CIK Migrations (Stitched via `predecessor_ciks`)
| Canonical Filer | Predecessor CIK | Successor CIK | Cutoff Rule | Reason |
| :--- | :--- | :--- | :--- | :--- |
| **Greenlight / DME Capital** | `0001079114` (`GREENLIGHT CAPITAL INC`) | `0001489933` (`DME Capital Management, LP`) | `<= 2024-02-29` | Adviser entity transitioned in Q1 2024 (right at our OOS boundary). Yields unbroken 48/48 quarters. |
| **Appaloosa Management** | `0001006438` (`APPALOOSA MANAGEMENT LP`) | `0001656456` (`Appaloosa LP`) | `<= 2016-03-31` | Legal entity restructuring in Q1 2016. Yields unbroken 48/48 quarters. |
| **Ruane, Cunniff & Goldfarb** | `0000728014` (`RUANE, CUNNIFF & GOLDFARB INC`) | `0001720792` (`Ruane, Cunniff & Goldfarb L.P.`) | `<= 2018-05-14` | Inc. → L.P. conversion; **both CIKs filed on `2018-05-15`**. Capping predecessor at `2018-05-14` removes the 1-quarter duplicate cleanly (48/48). |
| **TCI Fund Management** | `0001362598` (`Childrens Investment Fund Management (UK) LLP`) | `0001647251` (`TCI Fund Management Ltd`) | `<= 2015-06-30` | `TCI Fund Management Ltd` began filing Q2 2015 (`2015-08-17`) while UK LLP overlapped through `2016-11-14`. Capping UK LLP at `2015-06-30` gives unbroken 48/48 with zero overlap. |

### B. Excluded Overlapping Subsidiary `13F-HR` & `13F-NT` Filers (`subsidiary_ciks`)
- **Steadfast Capital (`0001214822`):** Excluded overlapping `13F-HR` affiliates `0001872297` (`Steadfast Capital Management (UK) Ltd.`, 4 filings `2021–2022`) and `0001213310` (`STEADFAST ADVISORS LP`, 9 filings `2015–2017`).
- **Fundsmith LLP (`0001569205`):** Excluded 3 overlapping affiliate/fund CIKs: `0001868537` (`FUNDSMITH INVESTMENT SERVICES LTD.`, 22 filings `2021–2026`), `0001520023` (`Fundsmith Equity Fund, L.P.`), and `0001813932` (`Fundsmith Long/Short Master Fund`).
- **Sands Capital Management (`0001020066`):** Excluded overlapping venture/alternatives affiliate `0001846444` (`Sands Capital Ventures / Alternatives, LLC`, 21 filings `2021–2026`).
- **Confirmed `13F-NT` Notice-Filing Complexes:** Verified consolidated parent reporting for Coatue (39 sub-funds), Egerton (`Egerton Capital (US), LP`), Cantillon (`Cantillon Capital Management LLP`), and Senvest (`Senvest International LLC`).

### C. Legal Renames Under the Same CIK (Tracked Automatically)
- `0001569064`: `SUVRETTA CAPITAL MANAGEMENT, LLC` → `MONTANOVA CAPITAL, LLC` (Q3 2026; 48/48 continuous)
- `0001328785`: `RIMA SENVEST MANAGEMENT, L.L.C.` → `Senvest Management, LLC` (48/48 continuous)
- `0001592476`: `Tourbillon Capital Partners LP` → `Tourbillon Capital Partners, L.P.`

### D. PIT Entry, Exit & Internal Gap Notes
- **Clean PIT Late Entrant:** `D1 Capital Partners` (`2019 Q1`, 32 qtrs) enters strictly on its first real filing date with zero backfilling.
- **Anti-Survivorship Exits & Gaps:** `Melvin Capital` (36 qtrs through `2023 Q4`), `Highfields` (24 qtrs through `2020 Q4`), `Tourbillon` (17 qtrs through `2019 Q1`), and `Blue Ridge Capital` (16 qtrs `2015 Q1–2018 Q4`, 8-quarter hiatus `2019–2020`, then 4 filings in `2021`). `Cantillon Capital` has 47 continuous quarters through `2026 Q3` (Q3 2026 not filed as of `2026-11-19`).
- **Downstream PIT Rule:** The PIT ledger must **never** compute quarter-over-quarter share deltas across a multi-quarter filing gap (>1 skipped quarter) or across a terminal wind-down liquidation (>75% single-quarter book drop).

---

## 4. Re-Running the Audit Workflow

Whenever new quarters are added to SEC EDGAR, run:

```bash
# 1. Refresh quarterly master.idx archives into data/processed/edgar_13f_index.parquet
.venv/bin/python -m src.ingest.master_index

# 2. Validate config/filers.yaml, audit for duplicates/gaps/unmapped affiliates,
#    and regenerate data/processed/universe_13f_filings.parquet + outputs/filer_audit_report.csv
.venv/bin/python -m src.ingest.entity_audit --strict
```

If `--strict` exits non-zero:
1. **Unmapped CIK warning:** Inspect the reported CIK in `master.idx`. If it is a new subsidiary, add it to `subsidiary_ciks`; if it is a new successor entity (CIK migration), add the old CIK to `predecessor_ciks` with an explicit `max_filing_date` and update `cik`; if it is an unrelated firm sharing a name token, add it to `unrelated_ciks`.
2. **Duplicate 13F-HR quarter warning:** Inspect both accession numbers and tighten the `max_filing_date` cutoff on the overlapping entity.
3. Re-run `python -m src.ingest.entity_audit --strict` and `pytest` until clean.

---

## 5. Downstream Security-Level Overrides (`config/cusip_overrides.yaml`)

Analogous to `config/filers.yaml` for institutional entities, [config/cusip_overrides.yaml](../config/cusip_overrides.yaml) stores manual `CUSIP → (ticker, share_class_figi)` overrides for corporate restructurings or legacy CUSIPs not cleanly resolved by OpenFIGI `v3/mapping`.
