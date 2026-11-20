# 13F Positioning Factor & Backtester

Point-in-time (PIT) research pipeline that ingests SEC EDGAR `13F-HR` and `13F-HR/A` filings for a curated 36-fund universe of concentrated institutional managers, constructs positioning factors (**Crowding**, **Concentration Shifts**, and **Flow-vs-Price Divergence**), and evaluates them via daily cross-sectional Fama-MacBeth beta-adjusted factor regressions paired with a drifted, transaction-costed T+1 MOC backtester (`2015–2023` IS / `2024–present` OOS).

## Setup (Clean Clone)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` to configure your credentials:
- **`SEC_USER_AGENT` (required by SEC EDGAR fair-access policy):**
  ```bash
  SEC_USER_AGENT="FirstName LastName your.email@domain.com"
  ```
- **`OPENFIGI_API_KEY` (optional, recommended for faster CUSIP mapping):**
  1. Create a free account (no credit card) at https://www.openfigi.com/api.
  2. Verify your email, log in, and copy your key from **Account → API Key**.
  3. Add `OPENFIGI_API_KEY="your-key-here"` to `.env`.
  - **Automatic keyless fallback:** If `OPENFIGI_API_KEY` is omitted or left blank, `src.ingest.cusip_mapper` automatically falls back to OpenFIGI's free unauthenticated tier (`10 CUSIPs/batch`, ~23 req/min, ~15 minutes vs. ~90 seconds with a key). Progress is checkpointed incrementally per batch to `data/raw/openfigi/cusip_cache.jsonl` so interrupted runs resume seamlessly.

## Usage

### 1. Ingest & Audit SEC EDGAR Master Index
```bash
# Download & cache quarterly EDGAR master.gz archives (2015 Q1 -> current quarter)
python -m src.ingest.master_index

# Audit master index against config/filers.yaml overrides & build resolved universe manifest
python -m src.ingest.entity_audit --strict
```

See [docs/data_governance.md](docs/data_governance.md) for full documentation of universe selection rules, CIK migration overrides, subsidiary deduplication, and the repeatable audit workflow, and [docs/research_log.md](docs/research_log.md) for the running log of scope decisions, research rationale, and completed milestones.

### 2. Download & Parse 13F-HR / 13F-HR/A Filings
```bash
# Download complete submissions concurrently (<= 8 req/s, .txt.gz cache) and enforce strict checks
python -m src.ingest.filings --strict

# Re-parse offline from cache without network requests (~3 seconds)
python -m src.ingest.filings --parse-only --strict
```

### 3. Map 13F CUSIPs to `shareClassFIGI` & Tickers (OpenFIGI)
```bash
# Batch-map all long-equity CUSIPs/CINS via OpenFIGI (incrementally cached to JSONL)
python -m src.ingest.cusip_mapper

# Re-resolve offline from JSONL cache + config/cusip_overrides.yaml (~0.5 seconds)
python -m src.ingest.cusip_mapper --resolve-only
```

### 4. Full Pipeline & Tests
```bash
python -m src.run_pipeline --stage all
pytest
```

## Project Structure

```text
├── config/
│   ├── filers.yaml           # Curated 36-fund entity mapping & deduplication override registry
│   ├── settings.yaml         # IS/OOS windows, universe filters, execution rules, cost grids
│   └── cusip_overrides.yaml  # Manual corporate-action / CUSIP mapping overrides
├── docs/
│   └── data_governance.md    # EDGAR index audit workflow & entity resolution provenance
├── data/
│   ├── raw/                  # Cached raw EDGAR master.idx.gz, XMLs & market data (gitignored)
│   └── processed/            # Parquet index, universe manifest, PIT ledgers & panels (gitignored)
├── src/
│   ├── config.py             # Config & path loader
│   ├── run_pipeline.py       # Single CLI entrypoint
│   ├── ingest/               # EDGAR client, master.idx backfill, entity audit, XML & market data
│   ├── pit/                  # Append-only 13F ledger, 13F-HR/A & CTR resolver, split adjuster
│   ├── factors/              # Crowding, Concentration Shifts, Flow-vs-Price Divergence
│   └── backtest/             # T+1 MOC engine, cost model, FF5+Mom attribution
├── tests/                    # Schema, entity-dedup, and PIT discipline unit tests
├── outputs/                  # Audit reports, charts, summary tables, diagnostics
├── notebooks/                # Optional interactive inspection notebook
└── memo/                     # 3–5 page research memo source & memo.pdf
```
