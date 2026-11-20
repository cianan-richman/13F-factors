"""Single CLI entrypoint for the 13F positioning research pipeline.

Usage:
    python -m src.run_pipeline --stage all
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_config

STAGES = ("ingest", "pit", "factors", "backtest", "all")


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config()
    window = config.settings["window"]
    end_label = window["end_date"] or "present"
    print(
        f"[sec13f-positioning] stage={args.stage} "
        f"filers={len(config.filers)} "
        f"IS={window['start_date']}..{window['in_sample_end']} "
        f"OOS={window['oos_start']}..{end_label}"
    )
    raise NotImplementedError(
        "Pipeline stages are scaffolded but not yet implemented. "
        "Next step: implement src.ingest, src.pit, src.factors, and src.backtest."
    )


if __name__ == "__main__":
    sys.exit(main())
