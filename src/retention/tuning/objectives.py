"""Optuna objectives — inner-CV scoring functions for nested tuning."""

from __future__ import annotations

import numpy as np
import optuna
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from retention.features.builder import build_preprocessor
from retention.tuning.search_spaces import (
    suggest_catboost,
    suggest_lightgbm,
    suggest_logistic,
    suggest_xgboost,
)

_SPACE_FNS = {
    "logistic": suggest_logistic,
    "catboost": suggest_catboost,
    "lightgbm": suggest_lightgbm,
    "xgboost": suggest_xgboost,
}


def _build_model(model_name: str, params: dict):
    if model_name == "logistic":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    if model_name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params)
    if model_name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model: {model_name}")


def factual_objective(
    trial: optuna.Trial,
    X_raw,
    T: np.ndarray,
    Y: np.ndarray,
    model_name: str,
    seed: int = 42,
    inner_folds: int = 3,
) -> float:
    """Tune a single factual outcome model on the full dataset (for S-learner/risk)."""
    suggest_fn = _SPACE_FNS[model_name]
    params = suggest_fn(trial) if model_name == "logistic" else suggest_fn(trial, seed)
    prep = build_preprocessor(include_pca=True, scale=True)
    X_all = prep.fit_transform(X_raw)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    fold_losses = []

    for tr, va in skf.split(X_all, Y):
        model = _build_model(model_name, params)
        model.fit(X_all[tr], Y[tr])
        proba = model.predict_proba(X_all[va])[:, 1]
        fold_losses.append(log_loss(Y[va], proba) + 0.5 * brier_score_loss(Y[va], proba))

    return float(np.mean(fold_losses))


def t_learner_objective(
    trial: optuna.Trial,
    X_raw,
    T: np.ndarray,
    Y: np.ndarray,
    model_name: str,
    seed: int = 42,
    inner_folds: int = 3,
) -> float:
    """Tune both arms jointly on factual prediction quality."""
    suggest_fn = _SPACE_FNS[model_name]
    params = suggest_fn(trial) if model_name == "logistic" else suggest_fn(trial, seed)
    prep = build_preprocessor(include_pca=True, scale=True)
    X_all = prep.fit_transform(X_raw)
    skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    factual = np.full(len(X_all), np.nan)

    for tr, va in skf.split(X_all, np.array([f"{t}_{y}" for t, y in zip(T, Y)])):
        X_tr, X_va = X_all[tr], X_all[va]
        T_tr, T_va = T[tr], T[va]
        Y_tr = Y[tr]
        m0 = _build_model(model_name, params)
        m1 = _build_model(model_name, {**params, "random_seed": seed + 1} if "random_seed" in params else params)
        m0.fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
        m1.fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
        factual[va] = np.where(T_va == 1, m1.predict_proba(X_va)[:, 1], m0.predict_proba(X_va)[:, 1])

    valid = np.isfinite(factual)
    return float(log_loss(Y[valid], factual[valid]) + 0.5 * brier_score_loss(Y[valid], factual[valid]))
