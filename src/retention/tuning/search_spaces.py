"""Optuna parameter search spaces for all model families."""

from __future__ import annotations

from typing import Any

import optuna


def suggest_logistic(trial: optuna.Trial) -> dict[str, Any]:
    penalty = trial.suggest_categorical("penalty", ["l2", "elasticnet"])
    params: dict[str, Any] = {
        "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
        "penalty": penalty,
        "solver": "saga",
        "max_iter": 5000,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": 42,
        "n_jobs": 1,
    }
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params


def suggest_catboost(trial: optuna.Trial, random_seed: int) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=100),
        "depth": trial.suggest_int("depth", 3, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 120, step=10),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": 1,
    }


def suggest_lightgbm(trial: optuna.Trial, random_seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "num_leaves": trial.suggest_categorical("num_leaves", [7, 15, 31]),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 150, step=10),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        "random_state": random_seed,
        "n_jobs": 1,
        "verbosity": -1,
    }


def suggest_xgboost(trial: optuna.Trial, random_seed: int) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 5),
        "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 80.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "eval_metric": "logloss",
        "random_state": random_seed,
        "n_jobs": 1,
        "tree_method": "hist",
    }
