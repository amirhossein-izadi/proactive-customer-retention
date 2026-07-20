"""Seed manifest — deterministic sub-seed derivation."""
import json
from pathlib import Path

import numpy as np


def generate_seeds(master: int = 42) -> dict:
    rng = np.random.default_rng(master)
    return {
        "master_seed": master,
        "holdout_seed": int(rng.integers(0, 2**31)),
        "outer_cv_seed": int(rng.integers(0, 2**31)),
        "inner_cv_seed": int(rng.integers(0, 2**31)),
        "model_seed": int(rng.integers(0, 2**31)),
        "bootstrap_seed": int(rng.integers(0, 2**31)),
    }


def save_seeds(seeds: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(seeds, indent=2))
