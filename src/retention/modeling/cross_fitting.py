"""Cross-fitted OOF prediction generation — the statistical core.

All models are fitted inside each outer fold to guarantee no leakage.
Uses the canonical preprocessor from ``features.builder`` (scaled,
one-hot-encoded, chronological temporal features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from retention.evaluation.predictive import compute_metrics as factual_metrics
from retention.evaluation.policy_value import dr_policy_value
from retention.features.builder import build_preprocessor, get_categorical_indices
from retention.modeling.s_learner import SLearner


def _maybe_cat_features(model_cls, prep) -> list[int]:
    """Return cat_features indices if model_cls is CatBoost, else empty list."""
    try:
        from catboost import CatBoostClassifier as _CB

        if issubclass(model_cls, _CB):
            return get_categorical_indices(prep)
    except Exception:
        pass
    return []


def repeated_cross_fit(
    X_raw: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    model_cls,
    params: dict,
    seed_key: str | None,
    n_repeats: int = 3,
    n_folds: int = 5,
) -> dict[str, np.ndarray]:
    """Generate repeated cross-fitted OOF predictions for a T-learner.

    Returns dict with p0, p1, e — each (n_samples, n_repeats).
    """
    n = len(X_raw)
    stratum = np.array([f"{t}_{y}" for t, y in zip(T, Y)])
    p0_all = np.zeros((n, n_repeats))
    p1_all = np.zeros((n, n_repeats))
    e_all = np.zeros((n, n_repeats))

    for rep in range(n_repeats):
        rep_seed = 4200 + rep * 100
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
        for fold, (tr, va) in enumerate(skf.split(X_raw, stratum)):
            prep = build_preprocessor(include_pca=True, scale=True)
            X_tr = prep.fit_transform(X_raw.iloc[tr])
            X_va = prep.transform(X_raw.iloc[va])
            T_tr, Y_tr = T[tr], Y[tr]
            s = 42 + fold + rep * 10

            def _mk(cfg, add):
                kw = dict(cfg)
                if seed_key:
                    kw[seed_key] = s + add
                else:
                    kw["random_seed"] = s + add
                return kw

            kw0, kw1 = _mk(params, 0), _mk(params, 1)
            cat_idx = _maybe_cat_features(model_cls, prep)
            if cat_idx:
                kw0["cat_features"] = kw1["cat_features"] = cat_idx

            m0 = model_cls(**kw0).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            m1 = model_cls(**kw1).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            p0_all[va, rep] = m0.predict_proba(X_va)[:, 1]
            p1_all[va, rep] = m1.predict_proba(X_va)[:, 1]

            ep = LogisticRegression(max_iter=1000, random_state=s).fit(X_tr, T_tr)
            e_all[va, rep] = np.clip(ep.predict_proba(X_va)[:, 1], 0.02, 0.98)

    return {"p0": p0_all, "p1": p1_all, "e": e_all}


def repeated_cross_fit_slearner(
    X_raw: pd.DataFrame,
    T: np.ndarray,
    Y: np.ndarray,
    model_cls,
    params: dict,
    seed_key: str | None,
    n_repeats: int = 3,
    n_folds: int = 5,
) -> dict[str, np.ndarray]:
    """OOF predictions for an S-learner (single model with T as feature)."""
    n = len(X_raw)
    stratum = np.array([f"{t}_{y}" for t, y in zip(T, Y)])
    p0_all = np.zeros((n, n_repeats))
    p1_all = np.zeros((n, n_repeats))
    e_all = np.zeros((n, n_repeats))

    for rep in range(n_repeats):
        rep_seed = 4200 + rep * 100
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep_seed)
        for fold, (tr, va) in enumerate(skf.split(X_raw, stratum)):
            prep = build_preprocessor(include_pca=True, scale=True)
            X_tr = prep.fit_transform(X_raw.iloc[tr])
            X_va = prep.transform(X_raw.iloc[va])
            T_tr, Y_tr = T[tr], Y[tr]
            s = 42 + fold + rep * 10

            def _mk(cfg, add):
                kw = dict(cfg)
                if seed_key:
                    kw[seed_key] = s + add
                else:
                    kw["random_seed"] = s + add
                return kw

            kw_sl = _mk(params, 0)
            cat_idx = _maybe_cat_features(model_cls, prep)
            if cat_idx:
                kw_sl["cat_features"] = cat_idx

            sl = SLearner(model_cls(**kw_sl))
            sl.fit(pd.DataFrame(X_tr), T_tr, Y_tr)
            p0_all[va, rep], p1_all[va, rep] = sl.predict_potential_outcomes(pd.DataFrame(X_va))

            ep = LogisticRegression(max_iter=1000, random_state=s).fit(X_tr, T_tr)
            e_all[va, rep] = np.clip(ep.predict_proba(X_va)[:, 1], 0.02, 0.98)

    return {"p0": p0_all, "p1": p1_all, "e": e_all}


# Backward-compat: imported by train/t_learner.py and train/s_learner.py
dr_policy_value_50 = dr_policy_value
