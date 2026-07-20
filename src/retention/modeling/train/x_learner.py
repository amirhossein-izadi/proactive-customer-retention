"""X-learner training — cross-fitted pseudo-effects + two effect models.

Model-agnostic: uses whatever ``model_cls`` the runner passes for
outcome models. CatBoost gets native categorical features injected.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from retention.contracts.prediction import ChallengerMetrics, ChallengerPreds
from retention.evaluation.policy_value import dr_policy_value
from retention.evaluation.predictive import compute_metrics as factual_metrics
from retention.features.builder import build_preprocessor, get_categorical_indices
from retention.settings.config import get_settings

_META_KEYS = {"n_folds", "effect_alpha"}


def _capacity() -> int:
    return get_settings().top_k


def _model_kw(model_cls, params: dict, prep, seed_key: str | None, seed_value: int) -> dict:
    kw = {k: v for k, v in params.items() if k not in _META_KEYS}
    if seed_key:
        kw[seed_key] = seed_value
    else:
        kw["random_seed"] = seed_value
    try:
        from catboost import CatBoostClassifier as _CBC

        if issubclass(model_cls, _CBC):
            kw["cat_features"] = get_categorical_indices(prep)
    except Exception:
        pass
    return kw


def train_x_learner(
    name: str,
    model_cls,
    params: dict,
    seed_key: str | None,
    Xd, Td: np.ndarray, Yd: np.ndarray,
    Xt, Tt: np.ndarray, Yt: np.ndarray,
    SEED: dict, cids_t: np.ndarray, rev_t: np.ndarray,
) -> tuple[ChallengerMetrics, ChallengerPreds]:
    """Cross-fitted X-learner.

    Procedure:
    1. Cross-fit m0 (on controls) and m1 (on treated) via K-fold.
    2. Compute pseudo-effects:
       - D1 = m0(X) - Y  for treated (m0 overpredicts -> treatment helps)
       - D0 = Y - m1(X)  for controls (Y exceeds m1 -> treatment would help)
    3. Fit separate effect models tau1 on D1 (treated only), tau0 on D0 (controls only).
    4. Combine: tau = e * tau0 + (1 - e) * tau1.
    """
    prep = build_preprocessor(include_pca=True, scale=True)
    Xd_all = prep.fit_transform(Xd)
    Xt_all = prep.transform(Xt)
    n = len(Xd)
    cap = _capacity()
    n_folds = params.get("n_folds", 5)
    effect_alpha = params.get("effect_alpha", 1.0)

    stratum = np.array([f"{t}_{y}" for t, y in zip(Td, Yd)])
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED["outer_cv_seed"])
    s = SEED["model_seed"]

    p0_d = np.zeros(n)
    p1_d = np.zeros(n)
    e_d = np.zeros(n)
    D0_buf = []
    D1_buf = []
    X0_buf = []
    X1_buf = []

    for tr, va in skf.split(Xd_all, stratum):
        X_tr, X_va = Xd_all[tr], Xd_all[va]
        T_tr, T_va = Td[tr], Td[va]
        Y_tr, Y_va = Yd[tr], Yd[va]

        m0 = model_cls(**_model_kw(model_cls, params, prep, seed_key, s)).fit(
            X_tr[T_tr == 0], Y_tr[T_tr == 0]
        )
        m1 = model_cls(**_model_kw(model_cls, params, prep, seed_key, s + 1)).fit(
            X_tr[T_tr == 1], Y_tr[T_tr == 1]
        )
        e = LogisticRegression(max_iter=1000, random_state=s).fit(X_tr, T_tr)

        p0_va = m0.predict_proba(X_va)[:, 1]
        p1_va = m1.predict_proba(X_va)[:, 1]
        e_va = np.clip(e.predict_proba(X_va)[:, 1], 0.02, 0.98)
        p0_d[va] = p0_va
        p1_d[va] = p1_va
        e_d[va] = e_va

        ctrl_mask = T_va == 0
        trt_mask = T_va == 1
        if ctrl_mask.any():
            D0_buf.append(Y_va[ctrl_mask] - m1.predict_proba(X_va[ctrl_mask])[:, 1])
            X0_buf.append(X_va[ctrl_mask])
        if trt_mask.any():
            D1_buf.append(m0.predict_proba(X_va[trt_mask])[:, 1] - Y_va[trt_mask])
            X1_buf.append(X_va[trt_mask])

    # ── Fit effect models on pooled pseudo-effects ──
    D0_all = np.concatenate(D0_buf)
    D1_all = np.concatenate(D1_buf)
    X0_all = np.concatenate(X0_buf)
    X1_all = np.concatenate(X1_buf)

    tau0_model = Ridge(alpha=effect_alpha).fit(X0_all, D0_all)
    tau1_model = Ridge(alpha=effect_alpha).fit(X1_all, D1_all)

    # ── Full-data nuisance models for test predictions ──
    m0_full = model_cls(**_model_kw(model_cls, params, prep, seed_key, s)).fit(
        Xd_all[Td == 0], Yd[Td == 0]
    )
    m1_full = model_cls(**_model_kw(model_cls, params, prep, seed_key, s + 1)).fit(
        Xd_all[Td == 1], Yd[Td == 1]
    )
    e_full = LogisticRegression(max_iter=1000, random_state=s).fit(Xd_all, Td)

    p0_t = m0_full.predict_proba(Xt_all)[:, 1]
    p1_t = m1_full.predict_proba(Xt_all)[:, 1]
    e_t = np.clip(e_full.predict_proba(Xt_all)[:, 1], 0.02, 0.98)
    tau0_t = tau0_model.predict(Xt_all).ravel()
    tau1_t = tau1_model.predict(Xt_all).ravel()
    e_d_fd = np.clip(e_full.predict_proba(Xd_all)[:, 1], 0.02, 0.98)
    uplift_t = e_t * tau0_t + (1 - e_t) * tau1_t

    uplift_d = p0_d - p1_d
    fm_d = factual_metrics(Yd, p0_d, p1_d, Td)
    fm_t = factual_metrics(Yt, p0_t, p1_t, Tt)
    dr50_d = dr_policy_value(uplift_d, p0_d, p1_d, e_d, Td, Yd, k=cap)
    dr50_t = dr_policy_value(uplift_t, p0_t, p1_t, e_t, Tt, Yt, k=cap)
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
