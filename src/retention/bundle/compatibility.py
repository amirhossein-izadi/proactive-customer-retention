"""Bundle compatibility check — loads all pickles and runs a synthetic smoke prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from retention.bundle.model_bundle import RetentionPolicyBundle

REQUIRED_ARTIFACTS = [
    "preprocessor.pkl",
    "m0.pkl",
    "m1.pkl",
    "calibrator_0.pkl",
    "calibrator_1.pkl",
    "propensity.pkl",
    "metadata.json",
]


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
                    import pickle

                    pickle.load(f)
                    result["checks"][name] = "loadable"
                except Exception as e:
                    result["passed"] = False
                    result["checks"][name] = f"corrupt: {e}"
        else:
            result["checks"][name] = "present"
        if not result["passed"]:
            return result  # fail fast: no point continuing

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
