"""Runtime settings — paths and YAML-backed configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from retention.utils.paths import CONFIGS


@dataclass(frozen=True)
class Settings:
    """Versioned, YAML-backed project settings."""

    data_version: str = "didar_retention_v1"
    feature_version: str = "2.2.0"
    model_version: str = "1.1.0"
    policy_version: str = "1.1.0"
    policy: dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> int:
        return int(self.policy.get("top_k", 50))

    @property
    def uncertainty_penalty(self) -> float:
        return float(self.policy.get("uncertainty_penalty", 1.0))

    @property
    def propensity_support(self) -> tuple[float, float]:
        abst = self.policy.get("abstention", {})
        lo = float(abst.get("min_common_support", 0.05))
        hi = 1.0 - float(abst.get("max_propensity_clip", 0.02))
        return lo, min(hi, 0.98)


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_settings(config_dir: str | Path = CONFIGS) -> Settings:
    config_dir = Path(config_dir)
    base = load_yaml(config_dir / "base.yaml")
    policy = load_yaml(config_dir / "policy.yaml")
    return Settings(
        data_version=base.get("data_version", "didar_retention_v1"),
        feature_version=base.get("feature_version", "2.2.0"),
        model_version=base.get("model_version", "1.1.0"),
        policy_version=policy.get("policy_version", "1.1.0"),
        policy=policy,
    )
