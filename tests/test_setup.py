"""Workspace, configuration, and pipeline CLI smoke tests."""

import pytest

from src.config import CONFIG_DIR, load_config
from src.run_pipeline import STAGES, parse_args


def test_config_loads_cleanly() -> None:
    cfg = load_config(CONFIG_DIR)
    window = cfg.settings["window"]
    assert window["start_date"] == "2015-01-01"
    assert window["in_sample_end"] == "2023-12-31"
    assert window["oos_start"] == "2024-01-01"
    assert window["in_sample_end"] < window["oos_start"]
    assert window["end_date"] == "2026-11-20"
    assert window["oos_start"] < window["end_date"]
    assert cfg.settings["execution"]["lag_trading_days"] == 1
    assert cfg.settings["universe"]["exclude_put_call"] is True
    assert isinstance(cfg.filers, list)
    assert len(cfg.filers) == 100
    assert isinstance(cfg.cusip_overrides, dict)


def test_run_pipeline_cli_args() -> None:
    assert STAGES == ("ingest", "pit", "factors", "backtest", "all")

    default_args = parse_args([])
    assert default_args.stage == "all"
    assert default_args.smoke is False
    assert default_args.verify_parity is False

    smoke_args = parse_args(["--stage", "ingest", "--smoke"])
    assert smoke_args.stage == "ingest"
    assert smoke_args.smoke is True

    pit_args = parse_args(["--stage", "pit", "--verify-parity"])
    assert pit_args.stage == "pit"
    assert pit_args.verify_parity is True

    for stage in ("factors", "backtest", "all"):
        parsed = parse_args(["--stage", stage])
        assert parsed.stage == stage

    with pytest.raises(SystemExit):
        parse_args(["--stage", "invalid_stage"])
