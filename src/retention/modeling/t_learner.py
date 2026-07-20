"""T-learner — two separate models (one per treatment arm).

The base estimator defaults to plain ``LogisticRegression`` with no
``class_weight`` — see the model diagnostics section of the methodology
for why unbalanced weighting distorts probability differences.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

# DR-learner is no longer duplicated here; import the single implementation.
from retention.modeling.dr_learner import DoublyRobustLearner  # noqa: F401


class TLearner:
    def __init__(self, estimator: BaseEstimator | None = None):
        self.estimator = estimator or LogisticRegression(max_iter=1000)
        self.model_0: BaseEstimator | None = None
        self.model_1: BaseEstimator | None = None

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "TLearner":
        self.model_0 = clone(self.estimator).fit(X[T == 0], Y[T == 0])
        self.model_1 = clone(self.estimator).fit(X[T == 1], Y[T == 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p0 = self.model_0.predict_proba(X)[:, 1]
        p1 = self.model_1.predict_proba(X)[:, 1]
        return p0, p1

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        p0, p1 = self.predict_proba(X)
        return p0 - p1
