"""Single CLI entrypoint for the point-in-time SEC 13F positioning research pipeline.

Usage:
    # Full cold-clone end-to-end run (ingest -> pit -> factors -> backtest)
    python -m src.run_pipeline --stage all

    # Individual stages
    python -m src.run_pipeline --stage ingest
    python -m src.run_pipeline --stage ingest --smoke
    python -m src.run_pipeline --stage pit --verify-parity
    python -m src.run_pipeline --stage factors
    python -m src.run_pipeline --stage backtest
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from src.backtest.engine import (
    FROZEN_IS_CROWDING_BETA_BPS,
    run_crowding_horizon_comparison,
    run_d10_vs_rest_regime_study,
    run_frozen_oos_evaluation,
)
from src.config import AppConfig, load_config
from src.factors.concentration import run_concentration_shifts_study
from src.factors.crowding import run_crowding_study
from src.factors.manager_skill import run_manager_skill_study
from src.factors.smart_money import run_smart_money_study
from src.ingest.cusip_mapper import OpenFigiClient, build_cusip_map
from src.ingest.edgar_client import EdgarClient
from src.ingest.entity_audit import (
    DEFAULT_AUDIT_CSV,
    DEFAULT_UNIVERSE_PARQUET,
    audit_edgar_index,
)
from src.ingest.filings import (
    OK_PARSE_STATUSES,
    check_cross_filer_other_managers,
    ingest_universe_filings,
)
from src.ingest.market_data import DEFAULT_SPY_PARQUET, ingest_market_data
from src.ingest.master_index import build_master_index
from src.ingest.shares_outstanding import ingest_shares_outstanding
from src.pit.compute import DEFAULT_BETA_MIN_PERIODS, DEFAULT_BETA_WINDOW, Stage3MarketMetrics
from src.pit.daily_panel import (
    DEFAULT_DAILY_PANEL_PARQUET,
    build_daily_pit_panel,
    verify_panel_parity_against_snapshot,
)
from src.pit.ledger import (
    DEFAULT_META_PARQUET,
    PitHoldingsLedger,
    load_trading_calendar,
    run_pit_qa_audit,
)
from src.pit.panel import PitMarketStore

logger = logging.getLogger(__name__)
STAGES = ("ingest", "pit", "factors", "backtest", "all")


def run_ingest_stage(config: AppConfig, *, smoke: bool = False) -> None:
    """Execute Phase 1 ingestion end-to-end with safe API-level concurrency.

    1. EDGAR master.idx backfill (concurrent <= 8 req/s)
    2. Entity resolution & strict 100-manager deduplication audit
    3. 13F-HR / 13F-HR/A XML download & parse (concurrent <= 8 req/s)
    4. OpenFIGI CUSIP/CINS mapping (4 workers when keyed, 1 worker automatic keyless fallback)
    5. Concurrent independent pulls:
       - Yahoo Finance market prices/splits/dividends + 252d PIT rolling beta (threads=4)
       - SEC EDGAR companyfacts PIT unadjusted shares outstanding (concurrent <= 8 req/s)
    """
    edgar_cfg = config.settings["edgar"]
    hedging_cfg = config.settings["hedging"]
    workers = int(edgar_cfg.get("max_workers", 5))

    with EdgarClient(
        user_agent=config.sec_user_agent,
        max_requests_per_sec=float(edgar_cfg.get("max_requests_per_sec", 8.0)),
        max_retries=int(edgar_cfg.get("max_retries", 5)),
        backoff_base_sec=float(edgar_cfg.get("backoff_base_sec", 1.0)),
        pool_maxsize=max(workers + 2, 10),
    ) as edgar:
        index_df = build_master_index(
            edgar,
            start_year=2024 if smoke else 2015,
            start_qtr=1,
            form_types=edgar_cfg.get("form_types", ["13F-HR", "13F-HR/A"]),
            max_workers=workers,
        )

        audit = audit_edgar_index(index_df, config.filers)
        if not audit.duplicate_quarter_rows.empty or not audit.unmapped_candidates.empty:
            raise RuntimeError(
                "Entity audit failed strict checks: "
                f"dup_qtrs={len(audit.duplicate_quarter_rows)}, "
                f"unmapped_aliases={len(audit.unmapped_candidates)}"
            )
        DEFAULT_UNIVERSE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        audit.universe_filings.to_parquet(DEFAULT_UNIVERSE_PARQUET, index=False)
        DEFAULT_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        audit.filer_report.to_csv(DEFAULT_AUDIT_CSV, index=False)

        manifest = audit.universe_filings.head(40) if smoke else audit.universe_filings
        meta_df, holdings_df = ingest_universe_filings(
            manifest,
            client=edgar,
            max_workers=workers,
        )
        bad = meta_df[~meta_df["parse_status"].isin(OK_PARSE_STATUSES)]
        if not bad.empty or int(meta_df["value_unit_suspect"].sum()) > 0:
            raise RuntimeError(
                f"Filing ingestion failed strict checks: bad={len(bad)}, "
                f"unit_suspect={int(meta_df['value_unit_suspect'].sum())}"
            )
        if check_cross_filer_other_managers(meta_df):
            raise RuntimeError("Cross-filer <otherManagers> overlap detected.")

        with OpenFigiClient(api_key=config.openfigi_api_key) as figi:
            cusip_map = build_cusip_map(
                holdings_df,
                client=figi,
                overrides=config.cusip_overrides,
                limit=100 if smoke else None,
            )

        ticker_limit = 50 if smoke else None
        with ThreadPoolExecutor(max_workers=2) as stage_pool:
            fut_mkt = stage_pool.submit(
                ingest_market_data,
                cusip_map,
                benchmark_ticker=str(hedging_cfg.get("benchmark_ticker", "SPY")),
                end_date=config.settings["window"].get("end_date"),
                beta_window=int(hedging_cfg.get("rolling_beta_window_days", 252)),
                limit=ticker_limit,
            )
            fut_sh = stage_pool.submit(
                ingest_shares_outstanding,
                cusip_map,
                edgar,
                max_workers=workers,
                limit=ticker_limit,
            )
            fut_mkt.result()
            fut_sh.result()


def run_pit_stage(config: AppConfig, *, verify_parity: bool = False) -> None:
    """Execute Phase 2: rebuild PIT 13F ledger, run QA audit, and materialize daily PIT panel."""
    window = config.settings["window"]

    ledger = PitHoldingsLedger.from_parquet(rebuild=True)
    meta = pd.read_parquet(DEFAULT_META_PARQUET)
    cal = load_trading_calendar(DEFAULT_SPY_PARQUET)
    qa = run_pit_qa_audit(ledger, meta, cal)
    print(
        f"[pit] Ledger ready: {len(ledger.states_df):,} states, "
        f"{len(ledger.holdings_events_df):,} holding rows | "
        f"Amendments T vs T+1 verified: {qa['amendments_boundary_verified']} | "
        f"CTR rows tagged: {qa['ctr_reveal_holding_rows']}"
    )

    store = PitMarketStore.from_parquet()
    s3_mkt = Stage3MarketMetrics(
        store,
        beta_window=DEFAULT_BETA_WINDOW,
        beta_min_periods=DEFAULT_BETA_MIN_PERIODS,
    )
    panel = build_daily_pit_panel(
        ledger,
        store,
        s3_mkt,
        start_date=str(window["start_date"]),
        end_date=window.get("end_date"),
    )
    elig = panel[panel["is_panel_eligible"]]
    print(
        f"[pit] Daily PIT panel -> {DEFAULT_DAILY_PANEL_PARQUET} | "
        f"{panel['query_date'].nunique():,} days "
        f"({panel['query_date'].min().date()} -> {panel['query_date'].max().date()}) | "
        f"eligible={len(elig):,} (avg {len(elig) / max(1, panel['query_date'].nunique()):.1f}/day)"
    )

    if verify_parity:
        check_dates = [
            "2015-05-18",
            "2020-08-31",
            "2022-11-10",
            "2024-06-10",
            "2024-08-12",
            "2024-08-16",
        ]
        parity = verify_panel_parity_against_snapshot(panel, ledger, store, s3_mkt, check_dates)
        print("[pit] Bit-for-bit parity check:\n" + parity.to_string(index=False))


def run_factors_stage() -> None:
    """Execute Phase 3: strictly In-Sample (2015-2023) validation of all 4 candidate signals."""
    cr = run_crowding_study(sample="in_sample")
    cs = run_concentration_shifts_study(sample="in_sample")
    sm = run_smart_money_study(sample="in_sample")
    ms = run_manager_skill_study(sample="in_sample")

    cr_row = cr["summary"][cr["summary"]["variable"] == "crowding_z"].iloc[0]
    cr_d10 = cr["deciles_summary"][cr["deciles_summary"]["bucket"] == "D10 - Rest (D1-D9)"].iloc[0]
    cs_row = cs["summary"][cs["summary"]["signal"] == "delta_pct_own_sn"].iloc[0]
    sm_row = sm["summary"][sm["summary"]["signal"] == "log_dollars_rpt_sn"].iloc[0]
    ms_row = ms["summary"][ms["summary"]["signal"] == "skill_conv_sn"].iloc[0]

    tbl = pd.DataFrame(
        [
            {
                "Signal": "1. Crowding (2-Pillar crowding_z)",
                "Slope (bps/d)": f"{cr_row['coef_bps']:+.3f}",
                "z-stat": f"{cr_row['z_stat']:+.2f}",
                "p-value": f"{cr_row['p_value']:.4f}",
                "Verdict": f"VALIDATED (D10-Rest: {cr_d10['badj_mean_bps']:+.2f} bps/d, "
                f"p={cr_d10['badj_p_value']:.4f})",
            },
            {
                "Signal": "2. Concentration Shifts (delta_pct_own)",
                "Slope (bps/d)": f"{cs_row['coef_bps']:+.3f}",
                "z-stat": f"{cs_row['z_stat']:+.2f}",
                "p-value": f"{cs_row['p_value']:.4f}",
                "Verdict": "REJECTED (Stale 45d flow)",
            },
            {
                "Signal": "3a. Smart Money (Report-Date Log Dollars)",
                "Slope (bps/d)": f"{sm_row['coef_bps']:+.3f}",
                "z-stat": f"{sm_row['z_stat']:+.2f}",
                "p-value": f"{sm_row['p_value']:.4f}",
                "Verdict": "REJECTED (Dominated by Crowding)",
            },
            {
                "Signal": "3b. Smart Money (8Q Static IR_badj Skill)",
                "Slope (bps/d)": f"{ms_row['coef_bps']:+.3f}",
                "z-stat": f"{ms_row['z_stat']:+.2f}",
                "p-value": f"{ms_row['p_value']:.4f}",
                "Verdict": "REJECTED (Weak persistence / style whipsaw)",
            },
        ]
    )
    print("\n=== [factors] PHASE 3 IN-SAMPLE SIGNAL VALIDATION SUMMARY (ZERO OOS) ===")
    print(tbl.to_string(index=False))


def run_backtest_stage(config: AppConfig) -> None:
    """Execute Phase 4: Train (IS) horizon & regime studies + Frozen OOS (2024+) evaluation."""
    exec_cfg = config.settings["execution"]
    costs_cfg = config.settings["costs"]
    buf_bps = float(exec_cfg.get("turnover_buffer_bps", 25.0))
    q_offset = int(exec_cfg.get("fixed_rebalance_offset_days", 46))
    one_way_bps = float(costs_cfg.get("large_cap_one_way_bps", 5.0))

    hz = run_crowding_horizon_comparison(
        sample="in_sample",
        strategy_mode="short_gt_d7",
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=one_way_bps,
        buffer_bps_of_gmv=buf_bps,
        quarterly_offset_days=q_offset,
    )
    d10_is = run_d10_vs_rest_regime_study(
        sample="in_sample",
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=one_way_bps,
        buffer_bps_of_gmv=buf_bps,
    )
    oos = run_frozen_oos_evaluation(
        notional_per_bps=50_000.0,
        max_adv_pct=0.05,
        one_way_cost_bps=one_way_bps,
        buffer_bps_of_gmv=buf_bps,
        frozen_slope_bps=FROZEN_IS_CROWDING_BETA_BPS,
    )

    cols = [
        "name",
        "split",
        "n_trading_days",
        "cum_pnl_raw",
        "cum_pnl",
        "ann_return",
        "ann_sdev",
        "sharpe_raw",
        "sharpe",
        "max_dd_pct",
        "ann_turnover_x",
        "realized_spy_beta",
    ]
    print("\n=== [backtest] TRAIN HORIZON COMPARISON (SHORT >D7, 5 bps ONE-WAY) ===")
    print(
        hz["summary"][["name", "cum_pnl", "ann_return", "sharpe", "ann_turnover_x"]].to_string(
            index=False
        )
    )
    print("\n=== [backtest] TRAIN DOLLAR-NEUTRAL D10 vs REST (4 ITERATIONS) ===")
    print(
        d10_is["summary"][
            ["name", "pct_active_days", "cum_pnl_raw", "cum_pnl", "sharpe_raw", "sharpe"]
        ].to_string(index=False)
    )
    print("\n=== [backtest] FROZEN TRAIN (IS) vs. TEST (OOS 2024+) vs. FULL ===")
    print(oos["summary"][cols].to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the point-in-time SEC 13F positioning factor pipeline."
    )
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast small-slice ingestion verification (~60s; applies to --stage ingest).",
    )
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help="Verify bit-for-bit parity against single-date snapshots during --stage pit.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    config = load_config()
    window = config.settings["window"]
    end_label = window["end_date"] or "present"
    print(
        f"[sec13f-positioning] stage={args.stage} smoke={args.smoke} "
        f"filers={len(config.filers)} "
        f"IS={window['start_date']}..{window['in_sample_end']} "
        f"OOS={window['oos_start']}..{end_label}"
    )

    if args.stage in ("ingest", "all"):
        run_ingest_stage(config, smoke=args.smoke)
    if args.stage in ("pit", "all"):
        run_pit_stage(config, verify_parity=args.verify_parity)
    if args.stage in ("factors", "all"):
        run_factors_stage()
    if args.stage in ("backtest", "all"):
        run_backtest_stage(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
