"""S-learner training — OOF dev + repeat test predictions."""

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
from retention.modeling.train.shared import capacity


def train_s_learner(
    name: str,
    model_cls,
    params: dict,
    seed_key: str | None,
    Xd,
    Td: np.ndarray,
    Yd: np.ndarray,
    Xt,
    Tt: np.ndarray,
    Yt: np.ndarray,
    SEED: dict,
    cids_t: np.ndarray,
    rev_t: np.ndarray,
) -> tuple[ChallengerMetrics, ChallengerPreds]:
    """Cross-fitted S-learner. Returns (metrics, predictions)."""
    oof = repeated_cross_fit_slearner(Xd, Td, Yd, model_cls, params, seed_key, n_repeats=3)
    p0_d = oof["p0"].mean(1)
    p1_d = oof["p1"].mean(1)
    e_d = oof["e"].mean(1)
    uplift_d = p0_d - p1_d
    se_d = np.sqrt(np.var(oof["p0"] - oof["p1"], axis=1, ddof=1))
    fm_d = factual_metrics(Yd, p0_d, p1_d, Td)
    dr50_d = dr_policy_value_50(uplift_d, p0_d, p1_d, e_d, Td, Yd, k=capacity())

    prep = build_preprocessor(include_pca=True, scale=True)
    Xd_all = prep.fit_transform(Xd)
    Xt_all = prep.transform(Xt)
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
        sl.fit(pd.DataFrame(Xd_all), Td, Yd)
        p0, p1 = sl.predict_potential_outcomes(pd.DataFrame(Xt_all))
        tp0[:, rep], tp1[:, rep] = p0, p1

    p0_t = tp0.mean(1)
    p1_t = tp1.mean(1)
    e_t = np.clip(
        LogisticRegression(max_iter=1000, random_state=42)
        .fit(Xd_all, Td)
        .predict_proba(Xt_all)[:, 1],
        0.02,
        0.98,
    )
    uplift_t = p0_t - p1_t
    se_t = np.sqrt(np.var(tp0 - tp1, axis=1, ddof=1))
    fm_t = factual_metrics(Yt, p0_t, p1_t, Tt)
    cap = capacity()
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
        "p0_t": p0_t,
        "p1_t": p1_t,
        "e_t": e_t,
        "uplift_t": uplift_t,
        "se_t": se_t,
        "cids_t": cids_t,
        "rev_t": rev_t,
        "p0_d": p0_d,
        "p1_d": p1_d,
    }
    return row, preds
