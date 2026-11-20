"""Single CLI entrypoint for the 13F positioning research pipeline.

Usage:
    python -m src.run_pipeline --stage all
    python -m src.run_pipeline --stage ingest
    python -m src.run_pipeline --stage ingest --smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

from src.config import AppConfig, load_config
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
from src.ingest.market_data import ingest_market_data
from src.ingest.master_index import build_master_index
from src.ingest.shares_outstanding import ingest_shares_outstanding

logger = logging.getLogger(__name__)
STAGES = ("ingest", "pit", "factors", "backtest", "all")


def run_ingest_stage(config: AppConfig, *, smoke: bool = False) -> None:
    """Execute Phase 1 ingestion end-to-end with safe API-level concurrency.

    1. EDGAR master.idx backfill (concurrent <= 8 req/s)
    2. Entity resolution & strict deduplication audit
    3. 13F-HR / 13F-HR/A XML download & parse (concurrent <= 8 req/s)
    4. OpenFIGI CUSIP/CINS mapping (4 workers when keyed, 1 worker keyless)
    5. Concurrent independent pulls:
       - Yahoo Finance market prices/splits/dividends + 252d PIT rolling beta (gentle threads=4)
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

        # Market data (Yahoo Finance) and PIT shares outstanding (SEC EDGAR) hit independent
        # external APIs, so run them concurrently without increasing load on Yahoo Finance.
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
        help="Run a fast small-slice verification (~60s) across recent filings and top tickers.",
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

    if args.stage == "ingest":
        run_ingest_stage(config, smoke=args.smoke)
        return 0

    raise NotImplementedError(
        f"Stage '{args.stage}' requires src.pit, src.factors, and src.backtest (in progress). "
        "Run `--stage ingest` to execute Phase 1 end-to-end."
    )


if __name__ == "__main__":
    sys.exit(main())
