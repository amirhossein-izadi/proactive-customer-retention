"""Smart tuning runner — auto-detect top models, tune with appropriate trial budgets.

Usage:
    retention tune --top 3       # tune top 3 by Test DR@50
    retention tune --all          # tune all enabled models
    retention tune t_learner catboost 40   # specific
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold

from retention.data.joining import feature_frame, join_customers_usage
from retention.data.loading import load_all
from retention.data.splitting import generate_seed_manifest
from retention.features.builder import build_preprocessor
from retention.tuning.search_spaces import (
    suggest_catboost,
    suggest_lightgbm,
    suggest_logistic,
    suggest_xgboost,
)
from retention.utils.paths import ARTIFACTS, CONFIGS, REPORTS, SPLITS

OPTUNA_DIR = ARTIFACTS / "optuna"

_MODEL_SPACES = {
    "logistic": suggest_logistic,
    "catboost": suggest_catboost,
    "lightgbm": suggest_lightgbm,
    "xgboost": suggest_xgboost,
}
_MODEL_TRIALS = {"logistic": 25, "catboost": 40, "lightgbm": 40, "xgboost": 40}
_ALL_ARCHITECTURES = ["s_learner", "t_learner"]


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
    raise ValueError(f"Unknown: {model_name}")


def _load_dev_data():
    customers, usage = load_all()
    dev_ids = pd.read_csv(SPLITS / "development_set.csv")["customer_id"]
    merged = join_customers_usage(customers, usage, dev_ids)
    X = feature_frame(merged)
    T = merged["contacted"].values.astype(int)
    Y = merged["churned"].values.astype(int)
    return X, T, Y


def _get_top_models(k: int = 3) -> list[tuple[str, str]]:
    """Read model_comparison.csv and return top-k (architecture, model) tuples by test DR@50."""
    path = REPORTS / "model_comparison.csv"
    if not path.exists():
        print("No model_comparison.csv found. Run 'retention train' first.")
        return []
    df = pd.read_csv(path)
    valid = df[~df["model"].str.startswith("baseline")].dropna(subset=["test_dr50"])
    top = valid.nlargest(k, "test_dr50")
    pairs = []
    for _, row in top.iterrows():
        name = row["model"]
        # architecture from naming convention: catboost_s -> s_learner, logistic_t -> t_learner
        if "_s_int" in name:
            arch = "s_learner"
            model = name.replace("_s_int", "").replace("logistic_", "")
        elif "_dr" in name:
            arch = "t_learner"  # DR learners use T-learner outcome models for tuning
            model = name.replace("_dr", "")
        elif "_x" in name:
            arch = "t_learner"
            model = name.replace("_x", "")
        elif "_s" in name:
            arch = "s_learner"
            model = name.replace("_s", "")
        elif "_t" in name:
            arch = "t_learner"
            model = name.replace("_t", "")
        else:
            arch, model = "t_learner", name
        pairs.append((arch, model))
    print(f"Top {k} by test DR@50: {[(n, a) for a, n in pairs]}")
    return pairs


def _run_study(
    X_raw, T: np.ndarray, Y: np.ndarray,
    architecture: str, model_name: str, n_trials: int, seed: int = 42,
) -> dict:
    """Run one Optuna study with nested CV on dev data only."""
    OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    study_name = f"{model_name}_{architecture}"
    db_path = OPTUNA_DIR / f"{study_name}.db"

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
    study = optuna.create_study(
        study_name=study_name, direction="minimize",
        sampler=sampler, pruner=pruner,
        storage=f"sqlite:///{db_path}", load_if_exists=True,
    )

    suggest_fn = _MODEL_SPACES[model_name]

    def objective(trial):
        params = suggest_fn(trial) if model_name == "logistic" else suggest_fn(trial, seed)
        prep = build_preprocessor(include_pca=True, scale=True)
        X_all = prep.fit_transform(X_raw)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        factual = np.full(len(X_all), np.nan)

        if architecture == "s_learner":
            for tr, va in skf.split(X_all, Y):
                model = _build_model(model_name, params)
                model.fit(X_all[tr], Y[tr])
                factual[va] = model.predict_proba(X_all[va])[:, 1]
        else:  # t_learner
            stratum = np.array([f"{t}_{y}" for t, y in zip(T, Y)])
            for tr, va in skf.split(X_all, stratum):
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

    print(f"  {model_name}/{architecture}: {n_trials} trials ... ", end="", flush=True)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"best={study.best_value:.4f}")

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OPTUNA_DIR / f"{study_name}_trials.csv", index=False)
    best = dict(study.best_params)
    best["best_value"] = study.best_value
    best["architecture"] = architecture
    best["model"] = model_name
    best["n_trials"] = n_trials
    (OPTUNA_DIR / f"{study_name}_best.json").write_text(json.dumps(best, indent=2))
    return best


def smart_tune(top_k: int = 0, all_models: bool = False, arch: str = "", model: str = "", n_trials: int = 0) -> list[dict]:
    """Smart tuning: auto-detect or manually specify what to tune."""
    X, T, Y = _load_dev_data()
    seeds = generate_seed_manifest(42)
    seed = seeds["model_seed"]

    pairs = []
    if arch and model:
        pairs = [(arch, model)]
    elif top_k > 0:
        pairs = _get_top_models(top_k)
    elif all_models:
        pairs = [
            (a, m)
            for a in _ALL_ARCHITECTURES
            for m in _MODEL_SPACES.keys()
            # skip lightgbm/xgboost unless explicitly requested
            if m in ("logistic", "catboost")
        ]
    else:
        # default: top 3
        pairs = _get_top_models(3)

    if not pairs:
        print("Nothing to tune. Run 'retention train' first.")
        return []

    results = []
    for arch, mname in pairs:
        trials = n_trials or _MODEL_TRIALS.get(mname, 30)
        best = _run_study(X, T, Y, arch, mname, trials, seed)
        results.append(best)
        seed += 1

    # Save combined best params
    all_best = {f"{r['model']}_{r['architecture']}": r for r in results}
    (OPTUNA_DIR / "all_best.json").write_text(json.dumps(all_best, indent=2))
    print(f"\nSaved: {OPTUNA_DIR / 'all_best.json'}")

    return results
