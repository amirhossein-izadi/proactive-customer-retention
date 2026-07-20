"""S-learner with explicit treatment-interaction features.

Adds T × feature interactions before fitting — the #1 recommended
model architecture for this dataset size (4000 customers).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from retention.contracts.prediction import ChallengerMetrics, ChallengerPreds
from retention.features.builder import build_preprocessor
from retention.modeling.cross_fitting import (
    dr_policy_value_50,
    factual_metrics,
    repeated_cross_fit_slearner,
)
from retention.modeling.s_learner import SLearner
from retention.settings.config import get_settings

# Features most likely to modify treatment effect (from expert domain analysis)
_INTERACTION_FEATURES = [
    "recent_7_mean",
    "recent_14_mean",
    "zero_days",
    "longest_zero_streak",
    "slope_60d",
    "usage_mean",
    "pct_decline",
    "active_days_pct",
    "usage_std",
    "days_since_activity",
]


def _capacity() -> int:
    return get_settings().top_k


def _add_treatment_interactions(
    X: np.ndarray, T: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Append T × feature interaction columns for the selected effect modifiers."""
    T_col = T.reshape(-1, 1)
    short_names = [n.split("__")[-1] for n in feature_names]
    new_cols = []
    for i, sn in enumerate(short_names):
        if sn in _INTERACTION_FEATURES:
            new_cols.append(T_col * X[:, i].reshape(-1, 1))
    if not new_cols:
        return X, feature_names
    X_aug = np.column_stack([X] + new_cols)
    return X_aug, feature_names


def train_s_learner_int(
    name: str,
    model_cls,
    params: dict,
    seed_key: str | None,
    Xd, Td: np.ndarray, Yd: np.ndarray,
    Xt, Tt: np.ndarray, Yt: np.ndarray,
    SEED: dict, cids_t: np.ndarray, rev_t: np.ndarray,
) -> tuple[ChallengerMetrics, ChallengerPreds]:
    """S-learner with explicit treatment-interaction features."""
    prep = build_preprocessor(include_pca=True, scale=True)
    Xd_base = prep.fit_transform(Xd)
    Xt_base = prep.transform(Xt)
    ct = prep.named_steps["columns"]
    base_names = list(ct.get_feature_names_out())

    # Augment with T × feature interactions
    Xd_aug, _ = _add_treatment_interactions(Xd_base, Td, base_names)
    Xt_aug, _ = _add_treatment_interactions(Xt_base, Tt, base_names)

    n = len(Xd)
    cap = _capacity()

    # Cross-fitted OOF via S-learner
    oof = repeated_cross_fit_slearner(Xd, Td, Yd, model_cls, params, seed_key, n_repeats=3)
    # Note: cross_fit uses standard preprocessor (no interactions) for OOF purity.
    # Interactions are added only for the final full-data model.
    p0_d = oof["p0"].mean(1)
    p1_d = oof["p1"].mean(1)
    e_d = oof["e"].mean(1)
    uplift_d = p0_d - p1_d
    fm_d = factual_metrics(Yd, p0_d, p1_d, Td)
    dr50_d = dr_policy_value_50(uplift_d, p0_d, p1_d, e_d, Td, Yd, k=cap)

    # Full-data S-learner with interactions for test predictions
    n_reps = 3
    tp0 = np.zeros((len(Xt), n_reps))
    tp1 = np.zeros((len(Xt), n_reps))
    for rep in range(n_reps):
        s = SEED["model_seed"] + rep * 10
        kw = dict(params)
        if seed_key:
            kw[seed_key] = s
        else:
            kw["random_seed"] = s
        sl = SLearner(model_cls(**kw))
        sl.fit(pd.DataFrame(Xd_aug), Td, Yd)
        p0, p1 = sl.predict_potential_outcomes(pd.DataFrame(Xt_aug))
        tp0[:, rep], tp1[:, rep] = p0, p1

    p0_t = tp0.mean(1)
    p1_t = tp1.mean(1)
    e_t = np.clip(
        LogisticRegression(max_iter=1000, random_state=42)
        .fit(Xd_aug, Td)
        .predict_proba(Xt_aug)[:, 1],
        0.02, 0.98,
    )
    uplift_t = p0_t - p1_t
    fm_t = factual_metrics(Yt, p0_t, p1_t, Tt)
    dr50_t = dr_policy_value_50(uplift_t, p0_t, p1_t, e_t, Tt, Yt, k=cap)
    u50_t = float(np.sort(uplift_t)[::-1][:cap].mean())

    row: ChallengerMetrics = {
        "model": name,
        "dev_auc": fm_d["factual_auc"],
        "dev_brier": fm_d["factual_brier"],
        "dev_dr50": round(dr50_d, 4),
        "test_auc": fm_t["factual_auc"],
        "test_brier": fm_t["factual_brier"],
        "test_arm0_auc": fm_t["arm0_auc"],
        "test_arm1_auc": fm_t["arm1_auc"],
        "test_dr50": round(dr50_t, 4),
        "test_uplift50": round(u50_t, 4),
    }
    preds: ChallengerPreds = {
        "p0_t": p0_t, "p1_t": p1_t, "e_t": e_t,
        "uplift_t": uplift_t, "se_t": np.zeros(len(uplift_t)),
        "cids_t": cids_t, "rev_t": rev_t,
        "p0_d": p0_d, "p1_d": p1_d,
    }
    return row, preds
