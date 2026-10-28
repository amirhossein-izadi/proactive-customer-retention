"""DR-learner training — cross-fitted nuisance models + tuned ElasticNet CATE."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from retention.contracts.prediction import ChallengerMetrics, ChallengerPreds
from retention.evaluation.policy_value import dr_pseudo_outcome, dr_policy_value
from retention.evaluation.predictive import compute_metrics as factual_metrics
from retention.features.builder import build_preprocessor
from retention.modeling.train.shared import capacity, model_kw, tune_elasticnet


def train_dr_learner(
    name: str,
    model_cls,
    params: dict,
    seed_key: str | None,
    Xd, Td: np.ndarray, Yd: np.ndarray,
    Xt, Tt: np.ndarray, Yt: np.ndarray,
    SEED: dict, cids_t: np.ndarray, rev_t: np.ndarray,
) -> tuple[ChallengerMetrics, ChallengerPreds]:
    prep = build_preprocessor(include_pca=True, scale=True)
    Xd_all = prep.fit_transform(Xd)
    Xt_all = prep.transform(Xt)
    n = len(Xd)
    cap = capacity()
    n_folds = params.get("n_folds", 5)

    stratum = np.array([f"{t}_{y}" for t, y in zip(Td, Yd)])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED["outer_cv_seed"])
    psi = np.zeros(n)
    p0_d = np.zeros(n)
    p1_d = np.zeros(n)
    e_d = np.zeros(n)
    s = SEED["model_seed"]

    for tr, va in skf.split(Xd_all, stratum):
        X_tr, X_va = Xd_all[tr], Xd_all[va]
        T_tr, T_va = Td[tr], Td[va]
        Y_tr, Y_va = Yd[tr], Yd[va]

        m0 = model_cls(**model_kw(model_cls, params, seed_key, s)).fit(
            X_tr[T_tr == 0], Y_tr[T_tr == 0])
        m1 = model_cls(**model_kw(model_cls, params, seed_key, s + 1)).fit(
            X_tr[T_tr == 1], Y_tr[T_tr == 1])
        e = LogisticRegression(max_iter=1000, random_state=s).fit(X_tr, T_tr)

        p0_d[va] = m0.predict_proba(X_va)[:, 1]
        p1_d[va] = m1.predict_proba(X_va)[:, 1]
        e_d[va] = np.clip(e.predict_proba(X_va)[:, 1], 0.02, 0.98)
        psi[va] = dr_pseudo_outcome(p0_d[va], p1_d[va], e_d[va], T_va, Y_va)

    # Tune ElasticNet CATE on the cross-fitted pseudo-outcomes
    best = tune_elasticnet(Xd_all, psi, seed=s)
    cate = ElasticNet(alpha=best["alpha"], l1_ratio=best["l1_ratio"], max_iter=5000, random_state=s)
    cate.fit(Xd_all, psi)

    m0_full = model_cls(**model_kw(model_cls, params, seed_key, s)).fit(
        Xd_all[Td == 0], Yd[Td == 0])
    m1_full = model_cls(**model_kw(model_cls, params, seed_key, s + 1)).fit(
        Xd_all[Td == 1], Yd[Td == 1])
    e_full = LogisticRegression(max_iter=1000, random_state=s).fit(Xd_all, Td)

    p0_t = m0_full.predict_proba(Xt_all)[:, 1]
    p1_t = m1_full.predict_proba(Xt_all)[:, 1]
    e_t = np.clip(e_full.predict_proba(Xt_all)[:, 1], 0.02, 0.98)
    uplift_t = cate.predict(Xt_all).ravel()

    uplift_d = p0_d - p1_d
    fm_d = factual_metrics(Yd, p0_d, p1_d, Td)
    fm_t = factual_metrics(Yt, p0_t, p1_t, Tt)
    dr50_d = dr_policy_value(uplift_d, p0_d, p1_d, e_d, Td, Yd, k=cap)
    dr50_t = dr_policy_value(uplift_t, p0_t, p1_t, e_t, Tt, Yt, k=cap)
    u50_t = float(np.sort(uplift_t)[::-1][:cap].mean())

    row: ChallengerMetrics = {
        "model": name, "dev_auc": fm_d["factual_auc"],
        "dev_brier": fm_d["factual_brier"], "dev_dr50": round(dr50_d, 4),
        "test_auc": fm_t["factual_auc"], "test_brier": fm_t["factual_brier"],
        "test_arm0_auc": fm_t["arm0_auc"], "test_arm1_auc": fm_t["arm1_auc"],
        "test_dr50": round(dr50_t, 4), "test_uplift50": round(u50_t, 4),
    }
    preds: ChallengerPreds = {
        "p0_t": p0_t, "p1_t": p1_t, "e_t": e_t,
        "uplift_t": uplift_t, "se_t": np.zeros(len(uplift_t)),
        "cids_t": cids_t, "rev_t": rev_t, "p0_d": p0_d, "p1_d": p1_d,
    }
    return row, preds
