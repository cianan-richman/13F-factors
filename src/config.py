"""Configuration and workspace path utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class AppConfig:
    """Loaded configuration values and runtime environment credentials."""

    settings: dict[str, Any]
    filers: list[dict[str, Any]]
    cusip_overrides: dict[str, dict[str, Any]]
    sec_user_agent: str
    openfigi_api_key: str | None = None
    tiingo_api_key: str | None = None
    polygon_api_key: str | None = None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(config_dir: Path = CONFIG_DIR) -> AppConfig:
    """Load YAML configs and environment variables from `.env` if present."""
    load_dotenv(REPO_ROOT / ".env")

    settings = _load_yaml(config_dir / "settings.yaml")
    filers_doc = _load_yaml(config_dir / "filers.yaml")
    overrides_doc = _load_yaml(config_dir / "cusip_overrides.yaml")

    return AppConfig(
        settings=settings,
        filers=filers_doc.get("filers", []),
        cusip_overrides=overrides_doc.get("overrides", {}),
        sec_user_agent=os.getenv("SEC_USER_AGENT", "").strip(),
        openfigi_api_key=os.getenv("OPENFIGI_API_KEY", "").strip() or None,
        tiingo_api_key=os.getenv("TIINGO_API_KEY", "").strip() or None,
        polygon_api_key=os.getenv("POLYGON_API_KEY", "").strip() or None,
    )
