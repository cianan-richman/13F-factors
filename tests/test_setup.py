"""Workspace and configuration smoke tests."""

from src.config import CONFIG_DIR, load_config


def test_config_loads_cleanly() -> None:
    cfg = load_config(CONFIG_DIR)
    window = cfg.settings["window"]
    assert window["start_date"] == "2015-01-01"
    assert window["in_sample_end"] == "2023-12-31"
    assert window["oos_start"] == "2024-01-01"
    assert window["in_sample_end"] < window["oos_start"]
    assert window["end_date"] is None
    assert cfg.settings["execution"]["lag_trading_days"] == 1
    assert cfg.settings["universe"]["exclude_put_call"] is True
    assert isinstance(cfg.filers, list)
    assert isinstance(cfg.cusip_overrides, dict)
