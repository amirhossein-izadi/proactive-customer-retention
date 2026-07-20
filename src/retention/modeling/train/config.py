"""Load challenger config from YAML, delegating to ``modeling/factories`` for the
canonical challenger list. This module is the bridge between versioned YAML
configuration and the in-memory challenger definitions used by the runner.
"""

from __future__ import annotations

from pathlib import Path

from retention.modeling.factories import build_challengers
from retention.utils.paths import CONFIGS

_YAML_PATH = CONFIGS / "models.yaml"


def load_challengers(config_path: str | Path | None = None) -> list[tuple]:
    """Load challengers returning tuples compatible with the runner."""
    path = config_path or _YAML_PATH
    dicts = build_challengers(str(path) if path.exists() else None)
    tuples = []
    for d in dicts:
        kind = d["kind"]
        if kind not in ("t", "s", "si", "dr", "x"):
            continue
        cls = d["cls"]
        if isinstance(cls, str):
            continue  # not a valid sklearn-like class
        tuples.append(
            (
                d["name"],
                cls,
                d["params"],
                d.get("seed_key"),
                f"{kind}_learner",
            )
        )
    return tuples
