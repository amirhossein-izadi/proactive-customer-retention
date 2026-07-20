"""S-learner — single model with treatment as feature.

Uses ALL data (both arms) so it has lower variance than a T-learner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class SLearner:
    def __init__(self, estimator=None):
        self.estimator = estimator or LogisticRegression(max_iter=2000, C=1.0)
        self.fitted_ = None

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "SLearner":
        X_ = X.copy()
        if isinstance(X_, np.ndarray):
            X_ = pd.DataFrame(X_)
        X_.columns = [str(c) for c in X_.columns]
        X_["contacted"] = T
        self.fitted_ = self.estimator.fit(X_, Y)
        return self

    def predict_potential_outcomes(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X0, X1 = X.copy(), X.copy()
        if isinstance(X0, np.ndarray):
            X0, X1 = pd.DataFrame(X0), pd.DataFrame(X1)
        X0.columns = [str(c) for c in X0.columns]
        X1.columns = [str(c) for c in X1.columns]
        X0["contacted"], X1["contacted"] = 0, 1
        p0 = self.fitted_.predict_proba(X0)[:, 1]
        p1 = self.fitted_.predict_proba(X1)[:, 1]
        return p0, p1

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_potential_outcomes(X)
        return p0 - p1
