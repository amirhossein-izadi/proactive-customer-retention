"""T-learner training — cross-fitted dev OOF + repeat test predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from retention.features.builder import build_preprocessor, get_categorical_indices
from retention.contracts.prediction import ChallengerMetrics, ChallengerPreds
from retention.modeling.cross_fitting import (
    _maybe_cat_features,
    dr_policy_value_50,
    factual_metrics,
    repeated_cross_fit,
)
from retention.settings.config import get_settings


def _capacity() -> int:
    return get_settings().top_k


def train_t_learner(
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
    """Cross-fitted T-learner. Returns (metrics, predictions)."""
    oof = repeated_cross_fit(Xd, Td, Yd, model_cls, params, seed_key, n_repeats=3)
    p0_d = oof["p0"].mean(1)
    p1_d = oof["p1"].mean(1)
    e_d = oof["e"].mean(1)
    uplift_d = p0_d - p1_d
    se_d = np.sqrt(np.var(oof["p0"] - oof["p1"], axis=1, ddof=1))
    fm_d = factual_metrics(Yd, p0_d, p1_d, Td)
    dr50_d = dr_policy_value_50(uplift_d, p0_d, p1_d, e_d, Td, Yd, k=_capacity())

    prep = build_preprocessor(include_pca=True, scale=True)
    Xd_all = prep.fit_transform(Xd)
    Xt_all = prep.transform(Xt)
    n_reps = 3
    tp0 = np.zeros((len(Xt), n_reps))
    tp1 = np.zeros((len(Xt), n_reps))
    te = np.zeros((len(Xt), n_reps))

    for rep in range(n_reps):
        s = SEED["model_seed"] + rep * 10
        kw0, kw1 = dict(params), dict(params)
        for kws in (kw0, kw1):
            if seed_key:
                kws[seed_key] = s
            else:
                kws["random_seed"] = s
        kw1[seed_key if seed_key else "random_seed"] = s + 1
        cat_idx = _maybe_cat_features(model_cls, prep)
        if cat_idx:
            kw0["cat_features"] = kw1["cat_features"] = cat_idx

        m0 = model_cls(**kw0).fit(Xd_all[Td == 0], Yd[Td == 0])
        m1 = model_cls(**kw1).fit(Xd_all[Td == 1], Yd[Td == 1])
        tp0[:, rep] = m0.predict_proba(Xt_all)[:, 1]
        tp1[:, rep] = m1.predict_proba(Xt_all)[:, 1]
        ep = LogisticRegression(max_iter=1000, random_state=s).fit(Xd_all, Td)
        te[:, rep] = np.clip(ep.predict_proba(Xt_all)[:, 1], 0.02, 0.98)

    p0_t = tp0.mean(1)
    p1_t = tp1.mean(1)
    e_t = te.mean(1)
    uplift_t = p0_t - p1_t
    se_t = np.sqrt(np.var(tp0 - tp1, axis=1, ddof=1))
    fm_t = factual_metrics(Yt, p0_t, p1_t, Tt)
    cap = _capacity()
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
