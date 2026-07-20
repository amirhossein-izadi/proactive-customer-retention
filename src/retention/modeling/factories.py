"""Model factory — builds challenger specs from YAML configuration or defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def _default_challengers() -> list[dict[str, Any]]:
    return [
        # ── P0: S-learners (use all data, lowest variance) ──
        {
            "name": "logistic_s_int",
            "kind": "si",
            "cls": LogisticRegression,
            "params": {"max_iter": 2000, "C": 1.0},
            "seed_key": "random_state",
        },
        {
            "name": "logistic_s",
            "kind": "s",
            "cls": LogisticRegression,
            "params": {"max_iter": 2000, "C": 1.0},
            "seed_key": "random_state",
        },
        {
            "name": "catboost_s",
            "kind": "s",
            "cls": CatBoostClassifier,
            "params": {
                "iterations": 500, "depth": 3, "learning_rate": 0.03,
                "l2_leaf_reg": 30, "verbose": False, "allow_writing_files": False,
            },
            "seed_key": "random_seed",
        },
        # ── P1: T-learners (stable baselines) ──
        {
            "name": "logistic_t",
            "kind": "t",
            "cls": LogisticRegression,
            "params": {"max_iter": 3000, "C": 1.0},
            "seed_key": "random_state",
        },
        {
            "name": "catboost_t",
            "kind": "t",
            "cls": CatBoostClassifier,
            "params": {
                "iterations": 500, "depth": 4, "learning_rate": 0.025,
                "l2_leaf_reg": 25, "min_data_in_leaf": 30,
                "verbose": False, "allow_writing_files": False,
            },
            "seed_key": "random_seed",
        },
        # ── P2: DR-learners (causal, doubly robust) ──
        {
            "name": "logistic_dr",
            "kind": "dr",
            "cls": LogisticRegression,
            "params": {"n_folds": 5, "final_alpha": 0.1, "final_l1_ratio": 0.5},
            "seed_key": "random_state",
        },
        {
            "name": "catboost_dr",
            "kind": "dr",
            "cls": CatBoostClassifier,
            "params": {
                "iterations": 300, "depth": 3, "learning_rate": 0.03,
                "l2_leaf_reg": 30, "min_data_in_leaf": 30,
                "verbose": False, "allow_writing_files": False,
                "n_folds": 5, "final_alpha": 0.1, "final_l1_ratio": 0.5,
            },
            "seed_key": "random_seed",
        },
        # ── P3: X-learners (imputed effect models) ──
        {
            "name": "logistic_x",
            "kind": "x",
            "cls": LogisticRegression,
            "params": {"n_folds": 5, "effect_alpha": 0.5},
            "seed_key": "random_state",
        },
        {
            "name": "catboost_x",
            "kind": "x",
            "cls": CatBoostClassifier,
            "params": {
                "iterations": 300, "depth": 3, "learning_rate": 0.03,
                "l2_leaf_reg": 30, "min_data_in_leaf": 30,
                "verbose": False, "allow_writing_files": False,
                "n_folds": 5, "effect_alpha": 0.5,
            },
            "seed_key": "random_seed",
        },
        # ── P4: Boosting challengers (diversity, disabled by default) ──
        {
            "name": "lightgbm_t",
            "kind": "t",
            "cls": LGBMClassifier,
            "params": {
                "n_estimators": 300, "max_depth": 3, "num_leaves": 15,
                "min_child_samples": 50, "reg_lambda": 15, "verbose": -1,
            },
            "seed_key": "random_state",
        },
        {
            "name": "xgboost_t",
            "kind": "t",
            "cls": XGBClassifier,
            "params": {
                "n_estimators": 300, "max_depth": 3, "min_child_weight": 15,
                "reg_lambda": 15, "subsample": 0.8, "verbosity": 0,
            },
            "seed_key": "random_state",
        },
    ]


def build_challengers(config_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Return enabled challengers by merging YAML config with built-in defaults."""
    defaults = _default_challengers()
    if not config_path or not Path(config_path).exists():
        return defaults
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    user_cfgs = config.get("challengers", {})
    seen: dict[str, bool] = {}
    for d in defaults:
        seen[d["name"]] = True
        if d["name"] not in user_cfgs:
            continue
        ucfg = user_cfgs[d["name"]]
        if not ucfg.get("enabled", True):
            d["_disabled"] = True
            continue
        if "params" in ucfg:
            d["params"] = {**d["params"], **ucfg["params"]}
        if "type" in ucfg:
            d["kind"] = ucfg["type"]
        if "seed_key" in ucfg:
            d["seed_key"] = ucfg["seed_key"]
    for name, cfg in user_cfgs.items():
        if not isinstance(cfg, dict) or not cfg.get("enabled", True):
            continue
        if name not in seen:
            defaults.append(
                {
                    "name": name,
                    "kind": cfg.get("type", "t"),
                    "cls": cfg.get("cls", LogisticRegression),
                    "params": cfg.get("params", {}),
                    "seed_key": cfg.get("seed_key", "random_state"),
                }
            )
    return [c for c in defaults if not c.pop("_disabled", False)]
