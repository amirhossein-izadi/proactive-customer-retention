"""Project path constants — single source of truth for all modules."""

from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
SPLITS = DATA / "splits"
ARTIFACTS = BASE / "artifacts"
OOF_DIR = ARTIFACTS / "oof_predictions"
METRICS_DIR = ARTIFACTS / "metrics"
RANKINGS_DIR = ARTIFACTS / "rankings"
MODELS = BASE / "models"
CHAMPION_DIR = MODELS / "champion"
REPORTS = BASE / "reports"
CONFIGS = BASE / "configs"
