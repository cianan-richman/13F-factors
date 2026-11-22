# SEC 13F Institutional Positioning Factor & Backtester

Point-in-time (PIT) quantitative research pipeline that ingests SEC EDGAR `13F-HR` and `13F-HR/A` filings for an audited **100-manager institutional universe** (`84` active + `16` closed 2015-era peers), evaluates four positioning factor hypotheses strictly In-Sample (`2015–2023`), and backtests our locked **2-Pillar Crowding** strategy via a corporate-action-clean, rack-based `T+1` MOC backtester across **Train (`2015–2023`)** and a single frozen **Out-of-Sample (`2024–present`)** holdout.

---

## Key Deliverables (Start Here)

- **Research Memo (5-Page PDF):** [memo/research_memo.pdf](memo/research_memo.pdf) *(GitHub Markdown version: [memo/research_memo.md](memo/research_memo.md))* — executive summary, factor hypotheses & alternatives tested, deliberate scope cuts, PIT data-engineering care, Train vs. Frozen OOS results, honest attribution, and prioritized next steps.
- **Self-Contained Interactive Notebook:** [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb) — 29 pre-executed cells containing all native `statsmodels` OLS summaries, decile & macro-regime tables, GMV-annotated backtest tearsheets, and all 12 saved charts ([outputs/charts/](outputs/charts/)).
- **Detailed Engineering & Audit Trail:** [docs/research_log.md](docs/research_log.md) (chronological technical log of every data issue encountered and resolved) and [docs/data_governance.md](docs/data_governance.md) (100-manager universe governance and CIK resolution).

---

## Cold-Clone Setup & Deployment

**Prerequisites:** Python `>= 3.11`, `git`, and internet access to SEC EDGAR, OpenFIGI, and Yahoo Finance.

```bash
# 1. Clone the repository
git clone https://github.com/cianan-richman/13F-factors.git
cd 13F-factors

# 2. Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package with all runtime & dev/notebook dependencies
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Create your local environment file from the template
cp .env.example .env
```

**5. Configure `.env` before running data ingestion:**
- **`SEC_USER_AGENT` (required by SEC EDGAR fair-access policy):**
  ```bash
  SEC_USER_AGENT="FirstName LastName your.email@domain.com"
  ```
- **`OPENFIGI_API_KEY` (optional, recommended for faster CUSIP mapping):**
  1. Create a free account at https://www.openfigi.com/api and copy your API key.
  2. Add `OPENFIGI_API_KEY="your-key-here"` to `.env`.
  3. **Automatic keyless fallback:** If left blank, `src.ingest.cusip_mapper` automatically falls back to OpenFIGI's free unauthenticated tier (`10 CUSIPs/batch`, incrementally checkpointed to `data/raw/openfigi/cusip_cache.jsonl` so interrupted runs resume cleanly).

---

## Running the Pipeline

```bash
# 1. Full end-to-end clean-clone run (ingest -> pit -> factors -> backtest)
python -m src.run_pipeline --stage all

# 2. Or execute individual stages independently:
python -m src.run_pipeline --stage ingest                 # Phase 1: EDGAR, OpenFIGI, yfinance, XBRL
python -m src.run_pipeline --stage pit --verify-parity    # Phase 2: PIT ledger & daily security panel
python -m src.run_pipeline --stage factors                # Phase 3: IS validation of all 4 factor hypotheses
python -m src.run_pipeline --stage backtest               # Phase 4: Train (IS) + Frozen OOS (2024+) backtests

# 3. Fast ~60-second live ingestion smoke check (recent filings & top tickers)
python -m src.run_pipeline --stage ingest --smoke

# 4. Run full automated QA & unit test suite (65 tests) + linter
pytest
ruff check .
```

> **Note:** To explore or re-run the pre-rendered results interactively after building the panel, open [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb) in Jupyter / VS Code using the `.venv` kernel.

**Expected Cold-Clone Runtime:**
- `--stage ingest`: ~10–15 minutes with a free `OPENFIGI_API_KEY` (~25 minutes keyless); `< 30 seconds` when re-run from cached raw files.
- `--stage pit`: ~3 minutes (materializes `3.86M` daily rows across `2,983` US trading days and verifies bit-for-bit parity `<= 6.5e-13`).
- `--stage factors`: ~45 seconds (strictly In-Sample `2015–2023`, zero OOS touch).
- `--stage backtest`: ~15 seconds (Train horizon & regime studies + frozen OOS `2024+` evaluation).

---

## Key Research & Backtest Summary

For full empirical tables, regression diagnostics (`R²`, clustered SEs, deciles, `10-BD SPY` regimes), and GMV-annotated equity curves, see [memo/research_memo.pdf](memo/research_memo.pdf), [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb), and [docs/research_log.md](docs/research_log.md).

### Phase 3 — Factor Validation (Strictly In-Sample `2015–2023`, Zero OOS Touch)

| Signal Hypothesis | Primary Metric (Size-Neutral) | Slope (`bps/d`) | `z` / `t` | `p`-value | Decision |
|---|---|---:|---:|---:|---|
| **1. Crowding (2-Pillar)** | `crowding_z` (`0.5*pct_own + 0.5*days_to_exit`) | **`-0.535`** *(D10−Rest: `-1.744`)* | **`-2.38`** *(`-2.74`)* | **`0.0175`** *(`0.0061`)* | **VALIDATED → Backtest** |
| **2. Concentration Shifts** | `delta_pct_own` (PIT `13F-HR/A`-safe QoQ Δ) | `-0.055` | `-0.23` | `0.8166` | **REJECTED** *(stale 45d flow, `ρ_63d = 0.165`)* |
| **3a. Smart Money (Dollars)** | `dollars_rpt` / `log_dollars_rpt_sn` *(report-date)* | `+0.004` / `-0.246` | `+0.01` / `-0.77` | `0.990` / `0.442` | **REJECTED** *(size-confounded / dominated)* |
| **3b. Smart Money (Skill)** | `skill_conv_sn` *(8Q static `IR_badj` manager rank)* | `-0.044` | `-0.06` | `0.9509` | **REJECTED** *(weak persistence `IC=+0.034`)* |

### Phase 4 — Frozen Strategy Backtest (`Buffered 25 bps`, `5 bps` One-Way = `10 bps` RT, `5% ADV` Cap)

| Frozen Strategy | Window | Days | Avg GMV | `pnl_raw` | `pnl` (Net) | Ann Ret | Sharpe (Raw/Net) | Max DD | Turn (`/yr`) | `SPY β` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Short `> D7` vs. `SPY`** | Train (`2015–2023`) | 2,232 | `$6.89M` | `+$2.57M` | `+$2.43M` | `+3.99%` | `+0.51 / +0.49` | `-23.1%` | `3.71×` | `-0.031` |
| **Short `> D7` vs. `SPY`** | **Frozen OOS (`2024–2026`)** | **726** | `$7.75M` | `+$1.45M` | **`+$1.39M`** | **`+6.20%`** | **`+0.71 / +0.68`** | **`-10.5%`** | `4.40×` | `+0.066` |
| **Short `> D7` vs. `SPY`** | **Full (`2015–2026`)** | **2,958** | `$7.11M` | `+$4.06M` | **`+$3.87M`** | **`+4.64%`** | **`+0.57 / +0.55`** | `-22.3%` | `3.90×` | `-0.010` |
| Short `D10` / Long `D1–D9` | Train (`2015–2023`) | 2,232 | `$10.07M` | `+$1.52M` | `+$1.30M` | `+1.46%` | `+0.65 / +0.55` | `-4.7%` | `4.69×` | `-0.009` |
| Short `D10` / Long `D1–D9` | **Frozen OOS (`2024–2026`)** | **726** | `$11.32M` | `+$48.5k` | **`-$42.2k`** | `-0.13%` | `+0.05 / -0.04` | `-5.0%` | `4.95×` | `+0.023` |
| Short `D10` / Long `D1–D9` | Full (`2015–2026`) | 2,958 | `$10.81M` | `+$1.74M` | `+$1.43M` | `+1.13%` | `+0.51 / +0.42` | `-5.0%` | `4.59×` | `-0.003` |

---

## Project Structure

```text
├── config/
│   ├── filers.yaml           # Audited 100-manager fundamental registry (84 active + 16 closed peers)
│   ├── settings.yaml         # IS/OOS boundaries, dynamic SPY-scaled universe floors, cost/rebal grids
│   └── cusip_overrides.yaml  # Manual corporate-action / CUSIP mapping overrides
├── src/
│   ├── config.py             # Config & .env credential loader
│   ├── run_pipeline.py       # Single CLI entrypoint (--stage ingest|pit|factors|backtest|all)
│   ├── ingest/               # EDGAR client, master.idx, entity audit, XML parser, OpenFIGI, yfinance, XBRL
│   │   └── filer_screener.py # Optional Phase-1 governance utility used to screen the 100-manager cohort
│   ├── pit/                  # PIT 13F-HR/A & CTR ledger, split-aligned market store, XBRL scrubber, daily panel
│   ├── factors/              # ownership (core helpers), crowding (+ SPY regime), concentration, smart_money, manager_skill
│   └── backtest/             # Rack-based corporate-action-clean T+1 MOC engine (horizons, D10-vs-Rest, OOS)
├── tests/                    # 65 unit & adversarial PIT/lookahead/amendment/backtest invariant tests
├── notebooks/
│   └── factor_research.ipynb # Self-contained pre-executed research & backtest notebook (29 cells)
├── outputs/                  # Generated audit reports, factor CSVs, backtest parquets, and 12 charts
├── docs/
│   ├── data_governance.md    # Universe selection & entity-resolution provenance
│   └── research_log.md       # Chronological technical research log & living backlog
└── memo/
    ├── research_memo.pdf     # 5-page submission Research Memo (PDF)
    └── research_memo.md      # 5-page submission Research Memo (Markdown)
```
