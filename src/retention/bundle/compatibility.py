"""Bundle compatibility check — loads all pickles and runs a synthetic smoke prediction."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from retention.bundle.model_bundle import REQUIRED as SERVING_REQUIRED
from retention.bundle.model_bundle import RetentionPolicyBundle

# Analysis artifacts produced alongside the serving bundle (used by SHAP,
# reporting and golden tests — not required to serve predictions).
ANALYSIS_ARTIFACTS = ["m0.pkl", "m1.pkl"]

REQUIRED_ARTIFACTS = list(SERVING_REQUIRED) + ANALYSIS_ARTIFACTS

# champion name prefix → expected class-name fragment of the pickled arm model.
# Guards against train/serve skew: reporting/ranking describe the champion from
# `champion_spec.json`, so the bundle must ship that same estimator family.
CHAMPION_CLASS_HINTS = {
    "logistic": "LogisticRegression",
    "catboost": "CatBoost",
    "lightgbm": "LGBM",
    "xgboost": "XGB",
}


def _check_champion_consistency(root: Path, result: dict) -> None:
    """Fail when metadata's champion family disagrees with the pickled arm model."""
    meta_path = root / "metadata.json"
    m0_path = root / "m0.pkl"
    if not meta_path.exists() or not m0_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    champion_name = meta.get("champion_name", "")
    if not champion_name:
        result["checks"]["champion_consistency"] = "skipped (legacy metadata)"
        return
    family = champion_name.split("_")[0]
    expected = CHAMPION_CLASS_HINTS.get(family)
    if expected is None:
        result["checks"]["champion_consistency"] = f"skipped (unknown family {family})"
        return
    with open(m0_path, "rb") as f:
        arm_model = pickle.load(f)
    actual = type(arm_model).__name__
    if expected not in actual:
        result["passed"] = False
        result["errors"].append(
            f"Train/serve skew: metadata champion '{champion_name}' expects {expected}, "
            f"but m0.pkl is {actual}"
        )
        result["checks"]["champion_consistency"] = f"mismatch: {actual} != {expected}"
    else:
        result["checks"]["champion_consistency"] = f"ok ({actual})"


def check_bundle(dir_path: str | Path) -> dict:
    """Verify champion bundle integrity. Returns status dictionary.

    - Checks all required files exist.
    - Loads each pickle.
    - Runs a synthetic smoke inference (rand usage + static data).
    - Returns pass/fail and diagnostics.
    """
    root = Path(dir_path)
    result = {"path": str(root), "passed": True, "errors": [], "checks": {}}

    for name in REQUIRED_ARTIFACTS:
        p = root / name
        if not p.exists():
            result["passed"] = False
            result["errors"].append(f"Missing: {name}")
            result["checks"][name] = "missing"
            continue
        if name.endswith(".pkl"):
            # Verify it loads
            with open(p, "rb") as f:
                try:
                    pickle.load(f)
                    result["checks"][name] = "loadable"
                except Exception as e:
                    result["passed"] = False
                    result["checks"][name] = f"corrupt: {e}"
        else:
            result["checks"][name] = "present"
        if not result["passed"]:
            return result  # fail fast: no point continuing

    _check_champion_consistency(root, result)
    if not result["passed"]:
        return result

    # Smoke prediction on synthetic customer
    np.random.seed(42)
    synthetic_static = pd.DataFrame(
        {
            "company_size": [100],
            "monthly_revenue": [500.0],
            "tenure_months": [24],
            "subscription_type": ["Annual"],
            "email_engagement": [0.5],
            "support_tickets_30d": [2],
            "package": ["Pro"],
        }
    )
    wide = {f"usage_day_{d}": [float(np.random.poisson(10))] for d in range(-59, 1)}
    synthetic_wide = pd.concat([synthetic_static, pd.DataFrame(wide)], axis=1)
    try:
        b = RetentionPolicyBundle().load(str(root))
        scores = b.score(synthetic_wide)
        p0, p1 = scores["p_churn_no_contact"], scores["p_churn_contact"]
        result["checks"]["smoke_inference"] = (
            "pass" if all(0 <= float(s) <= 1 for s in np.r_[p0, p1]) else "bounds_violation"
        )
        result["checks"]["prediction_shape"] = f"p0={p0.shape}, p1={p1.shape}"
    except Exception as e:
        result["passed"] = False
        result["errors"].append(f"smoke_inference failed: {e}")
        result["checks"]["smoke_inference"] = f"error: {e}"
    result["passed"] = result["passed"] and not result["errors"]
    return result
