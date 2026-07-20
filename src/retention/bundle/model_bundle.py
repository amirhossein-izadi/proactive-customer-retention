"""RetentionPolicyBundle — preprocessor + calibrated models + propensity + policy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from retention.policy.support import overlap_support_mask
from retention.policy.tie_breaking import deterministic_top_k
from retention.utils.serialization import load_pickle
from retention.utils.paths import CHAMPION_DIR

REQUIRED = [
    "preprocessor.pkl",
    "calibrator_0.pkl",
    "calibrator_1.pkl",
    "propensity.pkl",
    "metadata.json",
]


class RetentionPolicyBundle:
    def __init__(self):
        self.preprocessor = None
        self.cal0 = None
        self.cal1 = None
        self.propensity = None
        self.meta: dict = {}

    def load(self, path: str | Path = CHAMPION_DIR) -> "RetentionPolicyBundle":
        root = Path(path)
        for name in REQUIRED:
            p = root / name
            if not p.exists():
                raise FileNotFoundError(f"Bundle missing: {p}")
        self.preprocessor = load_pickle(root / "preprocessor.pkl")
        self.cal0 = load_pickle(root / "calibrator_0.pkl")
        self.cal1 = load_pickle(root / "calibrator_1.pkl")
        self.propensity = load_pickle(root / "propensity.pkl")
        self.meta = json.loads((root / "metadata.json").read_text())
        return self

    def score(self, X_raw: pd.DataFrame) -> dict:
        """Return per-customer scoring dict for a wide raw DataFrame."""
        Xp = self.preprocessor.transform(X_raw)
        p0 = self.cal0.predict_proba(Xp)[:, 1]
        p1 = self.cal1.predict_proba(Xp)[:, 1]
        e = np.clip(self.propensity.predict_proba(Xp)[:, 1], 0.02, 0.98)
        uplift = p0 - p1
        sup = overlap_support_mask(e)
        return {
            "p_churn_no_contact": p0,
            "p_churn_contact": p1,
            "estimated_uplift": uplift,
            "propensity": e,
            "in_common_support": sup,
        }

    def rank(self, X_raw: pd.DataFrame, k: int = 50) -> pd.DataFrame:
        """Return top-k ranking DataFrame for a batch of customers."""
        scores = self.score(X_raw)
        cids = np.arange(len(X_raw)) + 1  # fallback if no ID column
        if "customer_id" in X_raw.columns:
            cids = X_raw["customer_id"].values
        uplift = scores["estimated_uplift"]
        sup = scores["in_common_support"]
        robust = uplift.copy()
        robust[~sup] = -np.inf
        idx = deterministic_top_k(robust, cids, k)
        rank_df = pd.DataFrame(
            {
                "customer_id": cids[idx],
                "p_churn_no_contact": scores["p_churn_no_contact"][idx],
                "p_churn_contact": scores["p_churn_contact"][idx],
                "estimated_uplift": uplift[idx],
                "propensity": scores["propensity"][idx],
                "in_common_support": sup[idx],
            }
        )
        rank_df["rank"] = range(1, k + 1)
        return rank_df
