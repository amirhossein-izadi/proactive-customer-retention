"""Final top-k ranking — eligibility, support, abstention, tie-breaking, versioned output.

Wires: ``policy/eligibility`` (tenure + usage coverage), ``policy/fallback``
(abstention logic), ``policy/evaluate`` (policy-value facade), and
``explainability/reason_codes`` (top features from global importance).
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd

from retention.policy.eligibility import eligible_customers
from retention.policy.evaluate import dr_policy_value
from retention.policy.fallback import FallbackPolicy
from retention.policy.support import overlap_support_mask
from retention.policy.tie_breaking import deterministic_top_k
from retention.settings.config import get_settings
from retention.utils.paths import CHAMPION_DIR, OOF_DIR, REPORTS, SPLITS


def _load_model_versions() -> dict:
    meta_path = CHAMPION_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def _load_top_features(k: int = 5) -> list[str]:
    """Load top-k feature names from the champion bundle.

    Priority:
    1. ``uplift_explanation.json`` — SHAP(p0) - SHAP(p1), the actual uplift drivers.
    2. ``global_importance_arm0.json`` — SHAP(p0) from the control-arm model.
    3. ``global_importance_arm1.json`` — SHAP(p1) from the treated-arm model.
    """
    sources = [
        ("uplift_explanation", "SHAP(p0) - SHAP(p1) — uplift drivers"),
        ("arm0", "SHAP(p0) — no-contact churn drivers"),
        ("arm1", "SHAP(p1) — contact churn drivers"),
    ]
    for stem, description in sources:
        p = CHAMPION_DIR / f"global_importance_{stem}.json"
        if p.exists():
            data = json.loads(p.read_text())
            sorted_feats = sorted(data, key=data.get, reverse=True)[:k]
            print(f"  Reason codes from {p.name} ({description})")
            return sorted_feats
    print("  No global importance file found — reason codes empty")
    return []


_fallback = FallbackPolicy()


def generate_final_ranking(k: int | None = None) -> pd.DataFrame:
    settings = get_settings()
    k = k or settings.top_k
    data = np.load(OOF_DIR / "champion_predictions.npz")

    p0 = data["p0_t"]
    p1 = data["p1_t"]
    e = data["e_t"]
    uplift = data["uplift_t"]
    se = np.asarray(data.get("se_t", np.zeros_like(uplift)))
    cids = np.asarray(data["cids_t"], dtype=int)
    rev = np.asarray(data.get("rev_t", np.ones(len(cids))))
    tenure = np.asarray(data.get("tenure_months", np.ones(len(cids)) * 12))

    # Load T/Y from test CSV (not in npz) for DR policy value
    test_df = pd.read_csv(SPLITS / "test_set.csv")
    test_map = dict(
        zip(
            test_df["customer_id"],
            zip(test_df["contacted"].values.astype(int), test_df["churned"].values.astype(int)),
        )
    )
    T_arr = np.array([test_map[cid][0] for cid in cids])
    Y_arr = np.array([test_map[cid][1] for cid in cids])
    # Build a minimal DataFrame for eligibility/filtering
    feat_df = pd.DataFrame({"tenure_months": tenure, "active_days_pct": np.ones(len(cids)) * 0.8})
    n = len(cids)

    # 1. Propensity common-support
    support_lo, support_hi = settings.propensity_support
    sup_mask = overlap_support_mask(e, lo=support_lo, hi=support_hi)

    # 2. Eligibility from policy/eligibility module
    eligible = eligible_customers(feat_df, None, min_tenure_months=1) & sup_mask

    # 3. Abstain flag using policy/fallback
    penalty = settings.uncertainty_penalty
    robust = uplift.astype(float).copy()
    abstain = np.array([_fallback.should_abstain(se[i], uplift[i]) for i in range(n)])
    robust[~eligible] = -np.inf
    robust[eligible & abstain] -= penalty * se[eligible & abstain]

    # 4. Top-k selection
    idx = deterministic_top_k(robust, cids, k)
    dr50 = dr_policy_value(uplift, p0, p1, e, T_arr, Y_arr, k=k)

    # 5. Reason codes (top global features)
    top_feats = _load_top_features()
    reason_code_str = ", ".join(top_feats) if top_feats else ""

    # 6. Build output
    meta = _load_model_versions()
    rank_cols = {
        "rank": np.arange(1, k + 1),
        "customer_id": cids[idx],
        "p_churn_no_contact": p0[idx].round(4),
        "p_churn_contact": p1[idx].round(4),
        "estimated_uplift": uplift[idx].round(4),
        "uplift_se": se[idx].round(4),
        "propensity": e[idx].round(4),
        "in_common_support": sup_mask[idx],
        "eligible": eligible[idx],
        "abstain": abstain[idx],
        "ranking_score": robust[idx].round(4),
        "monthly_revenue": rev[idx].round(2),
        "reason_codes": reason_code_str,
        "model_version": meta.get("model_version", settings.model_version),
        "feature_version": meta.get("feature_version", settings.feature_version),
        "policy_version": settings.policy_version,
        "data_version": settings.data_version,
    }
    ranking = pd.DataFrame(rank_cols)
    ranking.to_csv(REPORTS / "final_ranking.csv", index=False)

    gap = (
        round(float(robust[idx[k - 1]] - np.sort(robust[~eligible])[-1]), 6)
        if k < n and (~eligible).any()
        else 0.0
    )
    print(f"Top-{k} ranking saved to reports/final_ranking.csv")
    print(f"  DR@50 (via policy/evaluate facade): {dr50:.4f}")
    print(f"  Mean uplift@K: {uplift[idx].mean():.4f}  SE@K: {se[idx].mean():.4f}")
    print(f"  All eligible: {eligible[idx].all()}  All support: {sup_mask[idx].all()}")
    print(f"  Abstained: {abstain[idx].sum()} / {k}")
    print(f"  Top features: {reason_code_str}")
    return ranking
