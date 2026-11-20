# Research & Engineering Log

Concise running log of scope decisions, research rationale, and implementation milestones for the **13F Positioning Factor & Backtester** case study. Maintained continuously as a reference for the final 3–5 page memo and 60-minute technical defense.

---

## 1. Research Framing & Core Scope Decisions

- **Primary Objective:** Construct economically grounded institutional positioning factors directly from SEC 13F filings and evaluate them via a **daily cross-sectional Fama-MacBeth beta-adjusted factor return framework** paired with a **tradable, drifted, transaction-costed backtest**. External multi-factor attribution (Kenneth French FF5 + Momentum) is deferred for now to keep the pipeline focused on pure market-beta-adjusted positioning factor returns.
- **Train / Test Split (`~80/20`):**
  - **In-Sample (IS):** `2015-01-01` → `2023-12-31` (9 years / 36 quarters; includes Q4 2014 baseline for Q1 2015 deltas). Covers ZIRP/low-vol growth (`2015–2019`), COVID shock (`2020`), retail/meme de-grossing (`2021`), rate-hike & hedge-fund crowding unwind (`2022`), and early AI concentration (`2023`).
  - **Out-of-Sample (OOS):** `2024-01-01` → present (~2.9 years / ~11–12 quarters; model frozen after `2023-12-31`).
  - *Why not `2026 → present` only:* For a quarterly signal, <1 year (~3 cycles) has virtually no statistical power ($\text{SE}(\widehat{\text{SR}}) \approx 1.2$). `2024–present` improves power (~2× tighter SE) and directly tests modern crowding/alpha decay, while we explicitly acknowledge in the memo that ~3 years in a largely bull/dispersion regime still carries wide Sharpe confidence intervals.
- **Three Positioning Factors Selected (all computed over the identical pooled filer universe):**
  1. **Crowding** *(Level / Fragility):* Liquidity- and size-normalized peer concentration (e.g., Days-to-Liquidate = total tracked \$ / ADV, tracked % of market cap, holder breadth) capturing forced-liquidation left-tail risk.
  2. **Concentration Shifts** *(Conviction Delta):* Active portfolio-weight shifts ($\Delta w^{\text{active}}$, stripping out passive price drift), top-conviction entries/exits, and holder HHI shifts ($\Delta \text{HHI}$).
  3. **Flow-vs-Price Divergence** *(Contrarian Active Flow):* Split-adjusted active institutional buying into quarterly drawdowns vs. selling into rallies—directly attacking the Price Momentum (`UMD`) confounder inherent in naive ownership momentum.
- **Deliberately Deferred to "What We Would Build Next":**
  - **Activist-focused signals:** Strong economic story (multi-year catalysts, sub-5% large-cap stakes), but deferred because Schedule 13D filings and press leaks frequently front-run Day-45 13Fs, and M&A targets trade like cash deal-spreads that are hard to scrub cleanly on free market data without an institutional event/merger database.
  - **Scaling to all ~500–800 fundamental hedge funds:** Requires Form ADV / FactSet investor-type classification rather than manual curation.

---

## 2. Filer Universe Design (`36` Curated Fundamental Managers)

- **Why 36 curated fundamental funds (NOT all ~7,000 SEC filers):**
  - Over 80% of 13F filers are passive indexers (Vanguard, BlackRock, State Street), private wealth RIAs, bank custody arms, or market makers/stat-arb/pod-shops (Citadel, Jane Street, RenTech, Millennium, Point72) whose 13Fs dilute active positioning with index weights, options delta hedges, or fast-turnover pod netting.
  - We also explicitly exclude **Berkshire Hathaway** (a \$300B+ insurance conglomerate whose size and 40%+ single-stock weights distort aggregate dollar-flow metrics) and pure activist/merger-arb shops.
  - All 36 funds are **pooled together for every factor** (style tags are metadata for diagnostics only) to maximize stock breadth and guarantee apples-to-apples comparisons across factors. Staying at 36 (vs. expanding to 45–50+) preserves high auditability and respects the 48-hour clock.
- **Style Balance (avoids single-style bias):**
  - `12` Tiger Cubs & Fundamental Growth / TMT L/S (core crowding & concentration engine)
  - `10` Concentrated Fundamental Value & Contrarian L/S (essential so Flow-vs-Price Divergence captures genuine contrarian buying into drawdowns)
  - `10` Long-Horizon Quality Compounders & Fundamental Generalists (sticky-hands conviction shifts)
  - `4` Wound-Down 2015-Era Fundamental Peers (`Blue Ridge`, `Melvin`, `Highfields`, `Tourbillon`)
- **Why include 4 closed funds (Anti-Survivorship Rigor):**
  - Hand-picking only funds thriving in 2026 injects **filer-level survivorship bias** into `2015–2023`. Including prominent 2015-era funds that later wound down mitigates that bias. Their `status: "closed"` field is strictly documentation metadata (never lookahead); in the PIT engine, late entrants (`D1 Capital`, `Abdiel`) enter on first filing date and closures exit naturally when filings cease (with a wind-down guard so terminal liquidations aren't misread as bearish stock calls).

---

## 3. Implementation & Data Engineering Log

### Milestone 1 — Workspace, Packaging & Config (`config/`, `src/`, `tests/`)
- Initialized git repository on `main`, editable Python 3.12 package ([pyproject.toml](../pyproject.toml)), `.gitignore`, `.env` (gitignored, SEC `User-Agent` configured), and locked IS/OOS boundaries + execution/cost grids in [settings.yaml](../config/settings.yaml).

### Milestone 2 — SEC EDGAR Quarterly `master.idx` Backfill ([edgar_client.py](../src/ingest/edgar_client.py), [master_index.py](../src/ingest/master_index.py))
- **Why `master.gz` before per-CIK APIs:** Gives an authoritative, flat catalog of every `13F-HR` and `13F-HR/A` filed with the SEC without `submissions/CIK.json` pagination overflow (>1,000 filings), and enables millisecond offline discovery of CIK migrations, renames, and subsidiary filers.
- **Result:** Downloaded & cached all **48/48 quarters** (`2015 Q1` → `2026 Q4`, `2015-01-02` → `2026-11-19`) in 55 seconds (`159 MB` raw `.idx.gz` cache → `15 MB` `data/processed/edgar_13f_index.parquet`): **308,891 filings** (`294,365 13F-HR`, `14,526 13F-HR/A`) across `13,075` unique CIKs (`0` missing accession numbers).

### Milestone 3 — Entity Mapping Override Registry & Automated Audit ([filers.yaml](../config/filers.yaml), [entity_audit.py](../src/ingest/entity_audit.py), [data_governance.md](data_governance.md))
- Built a declarative entity-resolution table (`config/filers.yaml`) + reproducible `--strict` audit CLI (`python -m src.ingest.entity_audit --strict`).
- **Real-world edge cases caught & resolved:**
  - *4 CIK migrations stitched via date-bounded `predecessor_ciks`:* `Greenlight → DME Capital` (right at our 2024 OOS boundary!), `Appaloosa Management LP → Appaloosa LP` (2016), `Ruane, Cunniff Inc. → L.P.` (capped at `2018-05-14` to eliminate a 1-day duplicate filing on `2018-05-15`), and `TCI UK LLP → TCI Fund Management Ltd` (capped at `2015-06-30` to avoid 2015–2016 overlap).
  - *Overlapping `13F-HR` subsidiaries excluded:* `Steadfast UK` (`0001872297`) and `Steadfast Advisors LP` (`0001213310`); plus local `13F-NT` scan confirming consolidated parent reporting for Coatue (39 sub-funds), Hound (4), Egerton, Cantillon, and Senvest.
  - *4 same-CIK legal renames tracked:* Notably `Suvretta Capital Management, LLC → Montanova Capital, LLC` in Q3 2026 (prevented a false exit).
  - *1 internal filing gap surfaced:* `Blue Ridge Capital` (`2019–2020` 8-quarter hiatus before 4 filings in 2021) — flagged so the PIT engine never computes QoQ deltas across skipped quarters.
- **Result:** Clean manifest `data/processed/universe_13f_filings.parquet` containing **1,677 deduplicated filings** (`1,612 13F-HR`, `65 13F-HR/A`), `0` duplicate quarters, `0` unmapped alias CIKs, plus [filer_audit_report.csv](../outputs/filer_audit_report.csv).

### Milestone 4 — Complete-Submission 13F XML Loader & Full 1,677-Filing Onboard ([filings.py](../src/ingest/filings.py))
- **Design choices:**
  - Downloads each filing's complete SGML `.txt` submission in **1 HTTP request** (instead of 3) and caches it gzipped (`data/raw/edgar_filings/{cik}/{accession}.txt.gz`, `8.7 MB` total across all 1,677 filings), giving exact `<ACCEPTANCE-DATETIME>` from `SEC-HEADER` alongside `primary_doc.xml` and `informationTable` XML. Offline re-parsing of all 1,677 filings (`--parse-only`) completes in **3 seconds**.
  - **Resilient directory fallback:** Automatically falls back to `index.json` + `.hdr.sgml` + individual XMLs when SEC EDGAR's Akamai edge throws persistent HTTP `502`s on monolithic `.txt` blobs (rescued 2 real filings: `0001592643-22-000002` and `0001172661-25-002342`).
  - **Strictly PIT self-healing SEC `<value>` unit normalization (major data-engineering catch):** Statutory SEC rules switched `<value>` from `\$1,000s` to exact `\$1` on `2023-01-03`. Auditing `median(value_raw / ssh_prnamt)` strictly within each filing caught **47 real filer unit violations** that would have silently introduced 1,000x errors:
    - *Post-2023 filings still reported in `\$1,000s` (18 filings):* `Baupost` (all 16 quarters `2023–2026`), `Abdiel` (`2023-02-14`), `Altimeter` (`2023-02-14`).
    - *Pre-2023 filings reported in exact dollars (29 filings):* `Select Equity` (28 consecutive quarters `2015–2021`), `Lone Pine` (`2015-02-17`).
    - Auto-corrected all 47 strictly point-in-time (zero lookahead; logged transparently via `statutory_multiplier`, `value_multiplier`, `value_unit_corrected`, `value_unit_reason`) $\Rightarrow$ **`0` remaining unit-suspect filings**.
  - **Dummy wind-down placeholder filtering:** Stripped `cusip == 000000000` / `0`-share `"No securities"` placeholder rows (`18` zero-holding wind-down filings from closed peers cleanly tagged `parse_status == "zero_holdings"`).
- **Full Onboard Results (`data/processed/edgar_13f_filings_meta.parquet`, `edgar_13f_holdings_raw.parquet`):**
  - **1,677 / 1,677** filings parsed (`1,659 ok`, `18 zero_holdings`, `0` errors) $\rightarrow$ **103,622 raw holding rows** (`100,237` long `SH`-only equity rows across **4,747 unique CUSIPs**; `1,979 CALL`, `1,201 PUT`, and `205 PRN` rows preserved & tagged for exclusion).
  - **`13F-HR/A` amendments:** `50 RESTATEMENT` + **`15 NEW HOLDINGS` (real CTR reveals)** ready for our PIT append-only event ledger.
  - **Cross-filer deduplication confirmed:** **`0`** cross-filer `<otherManagersInfo>` overlaps across all 1,677 filings.
  - **Reporting lag verified:** Median `45 days` from quarter-end to `acceptance_date`, with **`210` `13F-HR` filings arriving >46 days** (up to 48 days)—validating exact acceptance-date PIT discipline over a naive Day-45 assumption.

### Milestone 4b — Ingestion Architecture Trade-Offs & Resilience Hardening ([edgar_client.py](../src/ingest/edgar_client.py), [master_index.py](../src/ingest/master_index.py), [filings.py](../src/ingest/filings.py))
- **Why `master.idx` + per-filing `.txt` XML instead of SEC Quarterly Bulk Form 13F Data Sets (`form13f.zip`):**
  - SEC publishes quarterly bulk TSV ZIPs containing all ~7,000 filers, which is the right backfill mechanism at 500+ fund scale—**but** bulk ZIPs publish with a multi-week lag (missing the live `2026 Q4` tail), omit exact second-level `<ACCEPTANCE-DATETIME>`, require downloading ~3–5 GB to discard 99.5% of filers, and lose raw XML auditability for diagnosing filer bugs (such as Baupost's `\$1,000s` reporting error). For a curated 36-fund universe, our two-stage XML pipeline is only `8.7 MB` cached, live through yesterday, and directly auditable.
- **Performance hardening (cold-clone speedup ~2.5×):**
  - Replaced sequential single-threaded HTTP fetching (which RTT-capped throughput at ~3.2 req/s) with bounded `ThreadPoolExecutor` concurrency (`max_workers=5`, configurable via `config/settings.yaml` and `--workers`) coordinated by a **thread-safe global rate limiter** (`threading.Lock`) and HTTP keep-alive connection pooling inside `EdgarClient`. Strictly saturates `<= 8 req/s` (safely under SEC's `10 req/s` cap), cutting clean-clone backfill time from ~8–9 min to ~3.5 min.
- **Resilience & failure-mode hardening:**
  - *Per-filing fault isolation:* Unrecoverable single-filing fetch/parse failures are captured as `fetch_error`/`parse_error` metadata rows and exported to `outputs/edgar_ingest_failures.csv` rather than aborting the batch before Parquet persistence.
  - *Fast-fail on HTTP `403`:* Raises `EdgarBlockedError` immediately so concurrent workers never hammer a blocked IP or invalid `User-Agent`, alongside separate `(connect=10s, read=30s)` timeouts.
  - *Self-healing cache:* Validates `.txt.gz` / `.idx.gz` readability on read; automatically deletes and re-fetches corrupted cache files during live runs, and raises if the `index.json` fallback yields zero XML files.
  - *Strict integrity gate (`--strict`):* Exits non-zero if any filing fails parsing, triggers an unresolved `value_unit_suspect` flag, or exhibits a cross-filer `<otherManagers>` overlap.

### Milestone 4c — Data-Driven Roster Refinement: Concentration & Scale Gates ([filers.yaml](../config/filers.yaml))
- **Why we refined the 36-fund roster after inspecting parsed holdings:**
  - Empirical audit of our initial 36 filers revealed **3 funds violating concentration (`> 100` median positions)**—`Maverick Capital` (`238.5` pos; runs a quant equity sleeve), `Chilton Investment` (`145.5` pos; wealth/multi-sleeve), and `Select Equity` (`138.0` pos; diversified multi-sleeve)—and **3 active funds violating institutional scale (`< \$2.0B` median 13F book)**—`Light Street` (`\$0.71B`), `Abdiel` (`\$1.45B`, `\$0.04B` in 2026), and `Hound Partners` (`\$1.57B`, `\$0.87B` in 2026)—whose small dollar footprints barely affect dollar-based Crowding or Active Flow features.
  - Formalized two objective inclusion gates for active managers (**`median positions <= 100`** AND **`median 13F equity AUM >= \$2.0B`**) and performed a **6-for-6 swap** using live-sampled `48/48` (`2015–2026`) institutional heavyweights:
    - Added: `Sands Capital Management` (`68` pos, `\$33.7B` med), `Fundsmith LLP` (`34.5` pos, `\$19.4B` med), `Independent Franchise Partners` (`24` pos, `\$11.0B` med), `Sustainable Growth Advisers` (`52` pos, `\$10.7B` med), `EdgePoint Investment Group` (`40` pos, `\$10.1B` med; contrarian value), and `Dragoneer Investment Group` (`23` pos, `\$3.0B` med / `\$6.0B` 2026).
    - Audited and excluded overlapping affiliate CIKs for `Sands Capital` (`0001846444` Ventures/Alternatives) and `Fundsmith` (`0001868537`, `0001520023`, `0001813932`).
- **Refined 36-Fund Universe Results (`entity_audit --strict` & `filings --strict` passed in 41s):**
  - **36/36 funds pass both gates:** max median positions = `83.0` (universe median `36.5`); min median AUM = `\$2.03B`.
  - **Tracked institutional equity footprint:** Median book jumped from `\$255.7B` → **`\$339.8B`** (latest 2026 active total **`\$407.4B`**).
  - **Cleaner security universe:** `1,686` filings (`1,668 ok`, `18 zero_holdings`, `0` errors) → **`84,049` raw holding rows** across **`3,732` unique long-`SH` CUSIPs** (down from `4,747`—cut over 1,000 noisy tail CUSIPs; **`1,413` CUSIPs ever held by `>= 2` funds simultaneously**).
  - **Data quality:** `21` unit-error filings auto-corrected (`Baupost` 16, `SGA` 2, `Altimeter` 1, `Lone Pine` 1, `Sands` 1), **`0` remaining unit-suspect**, **`0` cross-filer `<otherManagers>` overlaps**, `53 RESTATEMENT` + `17 NEW HOLDINGS` (CTR reveals).

### Equity Universe, Market Data & Cross-Sectional Factor Methodology (Locked Before Step 3)
- **Zero-key free market-data stack:** OpenFIGI `v3/mapping` (CUSIP $\rightarrow$ `shareClassFIGI` / ticker / `securityType2`, with `includeUnlistedEquities: true` to track delistings and stitch CUSIP changes) + `yfinance` (`auto_adjust=False, actions=True`, mathematically un-splitting `Close` using `Stock Splits` and cross-verifying against 13F implied prices) + **SEC EDGAR `companyfacts`** (strictly PIT unadjusted `shares_outstanding` by exact 10-Q/10-K `filed` date).
- **Equity scope:** **US Common Stocks only** (`securityType2 == "Common Stock"`, excluding REITs, ETFs, ADRs, SPAC units, warrants). **Explicitly rejected restricting to S&P 500** because today's S&P 500 introduces severe lookahead/survivorship bias and cuts out the mid-cap (`\$2B–\$25B`) pool where institutional positioning has the highest dispersion.
- **Cross-Sectional Fama-MacBeth Factor Return & Backtest Methodology:**
  1. **Daily PIT Universe & Features (day $t$):** Evaluate the tradeable US Common Stock universe available strictly as of Close of day $t$ (`PIT Market Cap >= \$2B` from `raw_close × PIT shares_outstanding`, `PIT ADV_20d >= \$10M`). Compute positioning features (e.g., Crowding = $\sum_f S^{\text{adj}}_{i,f,t} / \text{SharesOut}_{i,t}$ & Days-to-Liquidate, Concentration Shifts, Flow-vs-Price Divergence) using strictly 13F filings with **`acceptance_date <= t - 1`** (enforcing T+1 MOC execution with zero lookahead) and `companyfacts` shares outstanding with **`filed <= t - 1`**.
  2. **Cross-Sectional Rank-to-Gaussian Standardization:** Convert each raw daily feature $x_{i,t}$ into a cross-sectional percentile rank and map through $\Phi^{-1}(\cdot)$ to produce standardized factor exposures $z_{i,t}$ (eliminating extreme OLS outlier leverage).
  3. **Strictly PIT Rolling Beta (Step 6):** Estimate each stock's rolling equity beta $\hat{\beta}_{i,t}$ each day via OLS regression of stock $i$'s trailing **252 trading-day** (1-year) daily returns on `SPY` returns over `[t-251, t]` (never using Yahoo's static lookahead `info['beta']`).
  4. **Next-Day Beta-Adjusted Return:**
     $$r^{\text{resid}}_{i,t+1} = r_{i,t+1} - \hat{\beta}_{i,t}\, r_{\text{SPY}, t+1}$$
  5. **Daily Cross-Sectional Factor Regression:** Regress next-day beta-adjusted returns on day-$t$ standardized exposures:
     $$r^{\text{resid}}_{i,t+1} = \alpha_{t+1} + f_{t+1}\, z_{i,t} + \varepsilon_{i,t+1}$$
     where $\hat{f}_{t+1}$ is the realized daily **factor return** (algebraically equivalent to a dollar-neutral, beta-hedged factor-mimicking portfolio with weights $w_{i,t} \propto z_{i,t} - \bar{z}_t$), and $\bar{f} = \frac{1}{T}\sum \hat{f}_{t+1}$ (with Newey-West HAC $t$-stat and daily Rank IC) estimates the expected factor return. Run both **univariate** (each factor standalone) and **multivariate** (all 3 factors jointly).
  6. **Tradable Drifted & Costed Portfolio Companion:** Pair the frictionless daily Fama-MacBeth series with an implementable T+1 MOC portfolio using the same $z_{i,t}$-proportional beta-hedged weights that **drifts** between 13F arrivals (with a 25 bps turnover buffer) and evaluates net Sharpe across `0 / 5 / 10 / 20 bps` one-way costs plus event-driven vs. fixed quarterly rebalancing.

### Milestone 5 — OpenFIGI CUSIP/CINS Mapping & Corporate-Action Stitching ([cusip_mapper.py](../src/ingest/cusip_mapper.py), [cusip_overrides.yaml](../config/cusip_overrides.yaml))
- **Key data-engineering discoveries during mapping:**
  - *Unit-threshold tightening (`filings.py`):* Inspecting top dollar-weighted CUSIPs caught 1 additional pre-2023 exact-dollar filing (`edgepoint 2017-08-11`, implied price `\$45,240 → \$45.24`) that fell just below our initial `\$50,000` ceiling. Tightened the PIT implied-price band to `[\$1, \$5,000]` (all legitimate filings lie inside `[\$3.87, \$357.37]`) $\Rightarrow$ `22` unit-error filings auto-corrected, `0` remaining suspect.
  - *CINS routing (`ID_CINS`):* **557 / 3,733** (`14.9%`) of our 13F security codes are letter-prefixed international CINS identifiers (e.g., `Accenture`, `Linde`, `TE Connectivity`, `NXP`, `Medtronic`, `Spotify`). Routing letter-prefixed codes to `idType: "ID_CINS"` prevented massive silent mapping losses.
  - *Automatic `shareClassFIGI` stitching for retired CUSIPs:* When a US company renames or reverse-splits (`Priceline → BKNG`, `United Technologies → RTX`, `old GE → GE`, `Dr Pepper Snapple → KDP`, `Michael Kors → CPRI`, `Medtronic → MDT`, `Mondelez → MDLZ`, `Super Micro → SMCI`), the retired CUSIP drops off `exchCode=US` while stale foreign listings keep the exact same `shareClassFIGI`. A second-pass `shareClassFIGI` propagation stitch automatically rescued **43** retired US Common Stocks without manual hardcoding.
  - *Manual corporate-action overrides (`config/cusip_overrides.yaml`):* Added 12 verified major rename/merger predecessor CUSIPs (`Canadian Pacific → CP`, `Valeant → BHC`, `McGraw Hill → SPGI`, `Apollo LLC → APO`, `DowDuPont → DD`, `Quintiles → IQV`, `Colfax → ENOV`, `Hilton old → HLT`, `Restoration Hardware → RH`, `HealthSouth → EHC`, `Willis → WTW`, `Platform Specialty → ESI`) linked directly to each verified successor `shareClassFIGI`.
- **Final Mapping Coverage (`data/processed/cusip_figi_map.parquet`, `outputs/cusip_mapping_summary.csv`):**
  - **Mapped (any type):** `3,478 / 3,727` (`93.3%` of CUSIPs, **`99.42%` of all tracked 13F dollars**; only `0.58%` unmapped tail).
  - **US Common Stocks:** `2,640` CUSIPs $\rightarrow$ **`2,524` unique persistent `security_id`s** (**`90.73%`** of all tracked 13F dollars), cleanly excluding ADRs (`5.94%`), REITs (`1.18%`), ETFs/Mutual Funds (`0.47%`), Partnerships (`0.12%`), Units/Warrants/Rights (`0.07%`).
  - **Corporate-action continuity:** **`116` multi-CUSIP $\rightarrow$ single `shareClassFIGI` stitches** resolved automatically + via overrides.

### Milestone 6 — Market Data (`yfinance`) & Strictly PIT Unadjusted Shares Outstanding (`SEC companyfacts`) ([market_data.py](../src/ingest/market_data.py), [shares_outstanding.py](../src/ingest/shares_outstanding.py))
- **Yahoo Finance Market Data (`data/processed/market_prices.parquet`, `spy_prices.parquet`):**
  - Downloaded `2014-01-01` → `2026-11-20` (`2014` provides a full 252-trading-day burn-in so PIT rolling beta is populated on day 1 of the `2015` IS window) with `auto_adjust=False, actions=True`.
  - **Mathematical un-splitting verified to the penny against 13F implied prices (`value_usd / ssh_prnamt`):** Reconstructed `raw_close`, `raw_volume`, `raw_dividend`, `cum_split_to_latest`, forward-only total return $R_{i,t}$, split-invariant `adv_20d_usd`, and vectorized **252-day PIT rolling beta vs. `SPY`**. Verified exact matches across `NVDA` (`40×` split: `\$522.20` vs `\$522.20`), `AMZN` (`20×`: `\$3,256.94` vs `\$3,256.93`), `GOOGL` (`20×`: `\$1,752.64` vs `\$1,752.64`), and `NFLX` (`10×`: `\$540.73` vs `\$540.73`).
  - **Survivorship & coverage quantified (`outputs/market_data_survivorship_report.csv`):** **`1,168` tickers priced (`3.10M` daily rows, `91.05%` of ALL US Common Stock 13F dollars)**; `3.00%` are retired Bloomberg numeric IDs and `5.95%` are delisted/acquired tickers (`ATVI`, `TWTR`, `SPLK`, `PXD`, `AET`, `TWX`, `LNKD`, `CERN`, `WBA`, `ALXN`, etc.) or Yahoo rate-limited micro-cap tail (`0.10%` of dollars). Added `YFRateLimitError` cooldown/retry and guarded against caching empty rate-limited batches.
- **SEC EDGAR `companyfacts` PIT Shares Outstanding (`data/processed/shares_outstanding_pit.parquet`, `outputs/shares_outstanding_coverage.csv`):**
  - Mapped US Common Stock tickers to SEC issuer CIKs via `company_tickers.json` and fetched `1,593` issuers concurrently (`3m 23s` at `7.84 cik/s`, cached compactly at `10 MB` total).
  - Extracted **`228,327` strictly PIT share-count rows** across **`1,589` tickers** (`89.07%` of queryable US Common Stock 13F dollars), preserving both `filed_date` (public arrival) and `end_date` (balance-sheet/cover-page date) and summing multi-class cover-page DEI rows cleanly (`GOOGL`/`GOOG` verified at `12.29B` shares / `\$3.26T` mcap).
  - **Joint coverage:** **`1,132` tickers have BOTH daily `yfinance` prices + PIT SEC `companyfacts` shares (`86.31%` of ALL US Common Stock 13F dollars)**.
  - **Notable XBRL finding documented:** 14 multi-class filers—most notably **Visa (`V`, `3.41%` of US Common Stock \$)** and **Constellation Brands (`STZ`, `0.33%`)**—report common share counts exclusively inside dimensional XBRL contexts that SEC's flat `companyfacts` JSON strips. Documented transparently for the memo and "What We Would Build Next" (inline-XBRL / Compustat shares); these names remain eligible for ADV-normalized signals.
- **Pipeline-wide Concurrency & Clean-Clone UX (`src/run_pipeline.py`, `cusip_mapper.py`):**
  - Added thread-safe rate-limited 4-worker concurrency to `OpenFigiClient` when `OPENFIGI_API_KEY` is provided (`~95s → ~25s`; automatically stays 1-worker on the keyless tier).
  - Wired `run_pipeline.py --stage ingest` (plus `--smoke` for ~60s verification) to run `market_data.py` (Yahoo Finance, kept gently throttled at `threads=4` + 2s inter-batch pause) and `shares_outstanding.py` (SEC EDGAR, 5 workers `<= 8 req/s`) **in parallel** since they hit completely independent external APIs—saving ~2 minutes on cold clones without adding any Yahoo rate-limit pressure (~8–9 min cold clone keyed; <30s cached).
