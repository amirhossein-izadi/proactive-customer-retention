"""Shared utilities for all training functions — single source of truth."""

from __future__ import annotations

import optuna
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

from retention.settings.config import get_settings

_META_KEYS = {"n_folds", "final_alpha", "final_l1_ratio", "effect_alpha"}


def capacity() -> int:
    return get_settings().top_k


def model_kw(model_cls, params: dict, seed_key: str | None, seed_value: int) -> dict:
    """Build keyword args for model_cls: strip meta-params, add seed."""
    kw = {k: v for k, v in params.items() if k not in _META_KEYS}
    if seed_key:
        kw[seed_key] = seed_value
    else:
        kw["random_seed"] = seed_value
    return kw


def tune_elasticnet(X: np.ndarray, psi: np.ndarray, n_folds: int = 3, seed: int = 42, n_trials: int = 20) -> dict:
    """Tune ElasticNet alpha/l1_ratio on pseudo-outcomes. Returns best params."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 10.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, va in skf.split(X, np.zeros(len(X))):
            m = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=seed)
            m.fit(X[tr], psi[tr])
            scores.append(mean_squared_error(psi[va], m.predict(X[va])))
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def tune_ridge(X: np.ndarray, psi: np.ndarray, n_folds: int = 3, seed: int = 42, n_trials: int = 15) -> dict:
    """Tune Ridge alpha. Returns best params."""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.001, 100.0, log=True)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        scores = []
        for tr, va in skf.split(X, np.zeros(len(X))):
            m = Ridge(alpha=alpha, random_state=seed)
            m.fit(X[tr], psi[tr])
            scores.append(mean_squared_error(psi[va], m.predict(X[va])))
        return float(np.mean(scores))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
