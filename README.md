# SEC 13F Institutional Positioning Factor & Backtester

Point-in-time (PIT) quantitative research pipeline that ingests SEC EDGAR `13F-HR` and `13F-HR/A` filings for an audited **100-manager institutional universe** (`84` active + `16` closed 2015-era peers), evaluates four positioning factor hypotheses strictly In-Sample (`2015–2023`) on a domestic listed US Common Stock universe, and backtests our locked **2-Pillar Crowding** strategy via a corporate-action-clean, rack-based `T+1` MOC backtester across **Train (`2015–2023`)** and a single frozen **Out-of-Sample (`2024–Nov 2026`, pinned `end_date: "2026-11-20"`)** holdout.

---

## Key Deliverables (Start Here)

- **Research Memo (5-Page PDF):** [memo/research_memo.pdf](memo/research_memo.pdf) *(GitHub Markdown version: [memo/research_memo.md](memo/research_memo.md))* — executive summary, factor hypotheses & alternatives tested, deliberate scope cuts, PIT data-engineering care, Train vs. Frozen OOS results, honest attribution, and prioritized next steps.
- **Self-Contained Interactive Notebook:** [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb) — 29 pre-executed cells containing all native `statsmodels` OLS summaries, decile & macro-regime tables, Total-GMV-annotated backtest tearsheets, and all 12 saved charts ([outputs/charts/](outputs/charts/)).
- **Detailed Engineering & Audit Trail:** [docs/research_log.md](docs/research_log.md) (chronological technical log of every data issue encountered and resolved) and [docs/data_governance.md](docs/data_governance.md) (100-manager universe governance, CIK resolution, and CTR handling).

---

## Cold-Clone Setup & Deployment

**Prerequisites:** Python `>= 3.11`, `git`, and internet access to SEC EDGAR, OpenFIGI, and Yahoo Finance.

```bash
# 1. Clone the repository
git clone https://github.com/ciananrichman/13F-factors.git
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

# 4. Run full automated QA & unit test suite (73 tests) + linter
pytest
ruff check .
```

> **Note:** To explore or re-run the pre-rendered results interactively after building the panel, open [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb) in Jupyter / VS Code using the `.venv` kernel.

**Expected Cold-Clone Runtime:**
- `--stage ingest`: ~10–15 minutes with a free `OPENFIGI_API_KEY` (~25 minutes keyless); `< 30 seconds` when re-run from cached raw files.
- `--stage pit`: ~3 minutes (materializes `3.71M` daily rows across `2,981` US trading days, avg `683.6` eligible domestic US stocks/day, and verifies bit-for-bit parity `<= 6.54e-13`).
- `--stage factors`: ~45 seconds (strictly In-Sample `2015–2023`, zero OOS touch).
- `--stage backtest`: ~15 seconds (Train horizon & regime studies + frozen OOS `2024+` evaluation across all 3 locked expressions).

---

## Key Research & Backtest Summary

For full empirical tables, regression diagnostics (`R²`, clustered SEs, deciles, `10-BD SPY` regimes), and Total-GMV-annotated equity curves, see [memo/research_memo.pdf](memo/research_memo.pdf), [notebooks/factor_research.ipynb](notebooks/factor_research.ipynb), and [docs/research_log.md](docs/research_log.md).

### Phase 3 — Factor Validation (Strictly In-Sample `2015–2023`, Zero OOS Touch)

| Signal Hypothesis | Primary Metric (Size-Neutral) | Slope (`bps/d`) | `z` / `t` | `p`-value | Decision |
|---|---|---:|---:|---:|---|
| **1. Crowding (2-Pillar)** | `crowding_z` (`0.5*pct_own + 0.5*days_to_exit`) | **`-0.458`** *(D10−Rest: `-1.393`)* | **`-2.00`** *(`-2.15`)* | **`0.0452`** *(`0.0313`)* | **VALIDATED → Backtest** |
| **2. Concentration Shifts** | `delta_pct_own` (PIT `13F-HR/A`-safe, ex-CTR) | `-0.035` | `-0.15` | `0.8820` | **REJECTED** *(stale 45d flow, `ρ_63d = 0.169`)* |
| **3a. Smart Money (Dollars)** | `dollars_rpt` / `log_dollars_rpt_sn` *(report-date)* | `+0.059` / `-0.255` | `+0.17` / `-0.79` | `0.868` / `0.432` | **REJECTED** *(size-confounded / dominated)* |
| **3b. Smart Money (Skill)** | `skill_conv_sn` *(8Q static `IR_badj` manager rank)* | `+0.016` | `+0.02` | `0.9820` | **REJECTED** *(weak persistence `IC=+0.031`)* |

### Phase 4 — Frozen Strategy Backtest (`Buffered ±20% / ±2%`, Tiered `5/12 bps` One-Way, `Total GMV` Basis, `5% ADV` Cap)

| Frozen Strategy | Window | Days | Avg Total GMV | Eff. Cost | `pnl_raw` | `pnl` (Net) | Ann Ret | Sharpe (Raw/Net) | Max DD | 1W Turn (`/yr`) | `SPY β` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Continuous Crowding L/S** | Train (`2015–2023`) | 2,232 | `$11.62M` | `10.0 bps` | `+$1.34M` | `+$1.01M` | `+0.98%` | `+0.59 / +0.44` | `-4.4%` | `1.96×` | `-0.001` |
| **Continuous Crowding L/S** | **Frozen OOS (`2024–2026`)** | **726** | `$12.60M` | `9.5 bps` | `+$900k` | **`+$759k`** | **`+2.09%`** | **`+0.90 / +0.76`** | **`-3.2%`** | `2.42×` | `+0.013` |
| **Continuous Crowding L/S** | **Full (`2015–2026`)** | **2,958** | `$11.87M` | `9.8 bps` | `+$2.26M` | **`+$1.80M`** | **`+1.29%`** | **`+0.68 / +0.54`** | **`-4.9%`** | `2.09×` | `+0.002` |
| Short `D10` / Long `D1–D9` | Train (`2015–2023`) | 2,232 | `$7.27M` | `9.2 bps` | `+$1.06M` | `+$740k` | `+1.15%` | `+0.64 / +0.45` | `-5.3%` | `3.16×` | `-0.006` |
| Short `D10` / Long `D1–D9` | **Frozen OOS (`2024–2026`)** | **726** | `$8.51M` | `8.9 bps` | `+$272k` | **`+$152k`** | `+0.62%` | `+0.36 / +0.20` | **`-3.8%`** | `3.17×` | `+0.022` |
| Short `D10` / Long `D1–D9` | Full (`2015–2026`) | 2,958 | `$7.58M` | `9.1 bps` | `+$1.34M` | `+$909k` | `+1.02%` | `+0.55 / +0.37` | `-5.1%` | `3.16×` | `-0.000` |
| **Short `> D7` vs. `SPY`** | Train (`2015–2023`) | 2,232 | `$11.64M` | `9.6 bps` | `+$1.73M` | `+$1.55M` | `+1.50%` | `+0.43 / +0.39` | `-12.1%` | `1.39×` | `-0.014` |
| **Short `> D7` vs. `SPY`** | **Frozen OOS (`2024–2026`)** | **726** | `$11.97M` | `9.4 bps` | `+$1.32M` | **`+$1.24M`** | **`+3.61%`** | **`+0.81 / +0.76`** | **`-5.6%`** | `1.76×` | `+0.030` |
| **Short `> D7` vs. `SPY`** | **Full (`2015–2026`)** | **2,958** | `$11.73M` | `9.6 bps` | `+$3.08M` | **`+$2.82M`** | **`+2.05%`** | **`+0.54 / +0.50`** | `-12.0%` | `1.48×` | `-0.005` |

---

## Project Structure

```text
├── config/
│   ├── filers.yaml           # Audited 100-manager fundamental registry (84 active + 16 closed peers)
│   ├── settings.yaml         # Pinned IS/OOS boundaries, dynamic SPY-scaled floors, tiered 5/12 bps costs
│   └── cusip_overrides.yaml  # Manual corporate-action / CUSIP mapping overrides
├── src/
│   ├── config.py             # Config & .env credential loader
│   ├── run_pipeline.py       # Single CLI entrypoint (--stage ingest|pit|factors|backtest|all)
│   ├── ingest/               # EDGAR client, master.idx, entity audit, XML parser, OpenFIGI, yfinance, XBRL
│   │   └── filer_screener.py # Optional Phase-1 governance utility used to screen the 100-manager cohort
│   ├── pit/                  # PIT 13F-HR/A & CTR ledger, split-aligned market store, XBRL scrubber, daily panel
│   ├── factors/              # ownership (core helpers), crowding (+ SPY regime), concentration, smart_money, manager_skill
│   └── backtest/             # Rack-based corporate-action-clean T+1 MOC engine (horizons, D10-vs-Rest, OOS)
├── tests/                    # 73 unit & adversarial PIT/lookahead/amendment/data-hygiene/backtest tests
├── notebooks/
│   └── factor_research.ipynb # Self-contained pre-executed research & backtest notebook (29 cells)
├── outputs/                  # Generated audit reports, factor CSVs, backtest parquets, and 12 charts
├── docs/
│   ├── data_governance.md    # Universe selection, entity-resolution provenance & CTR governance
│   └── research_log.md       # Chronological technical research log & living backlog
└── memo/
    ├── research_memo.pdf     # 5-page submission Research Memo (PDF)
    └── research_memo.md      # 5-page submission Research Memo (Markdown)
```
