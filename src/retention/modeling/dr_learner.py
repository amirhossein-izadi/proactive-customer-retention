"""Doubly-Robust (DR) learner for CATE estimation.

Approach:
1. Cross-fit nuisance models (m0, m1, e) using K-fold.
2. Compute the DR pseudo-outcome psi = m0-m1 + (1-T)(Y-m0)/(1-e) - T(Y-m1)/e.
3. Fit a final CATE model on psi ~ X.

This removes the bias from outcome-model subtraction while allowing
flexible CATE function approximation in the second stage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold

from retention.evaluation.policy_value import dr_pseudo_outcome


class DoublyRobustLearner:
    def __init__(
        self,
        outcome_model: BaseEstimator | None = None,
        propensity_model: BaseEstimator | None = None,
        final_model: BaseEstimator | None = None,
        n_folds: int = 5,
    ):
        self.outcome_model = outcome_model or LogisticRegression(max_iter=1000)
        self.propensity_model = propensity_model or LogisticRegression(max_iter=1000)
        self.final_model_ = final_model or Ridge(alpha=1.0)
        self.n_folds = n_folds

    def cross_fit_pseudo_outcomes(
        self,
        X: pd.DataFrame,
        T: np.ndarray,
        Y: np.ndarray,
        seed: int = 42,
    ) -> np.ndarray:
        """Cross-fitted DR pseudo-outcomes. Requires DataFrame input."""
        X = X.reset_index(drop=True) if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        n = len(X)
        stratum = np.array([f"{t}_{y}" for t, y in zip(T, Y)])
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=seed)
        pseudo = np.zeros(n)
        for tr, va in skf.split(X, stratum):
            X_tr, X_va = X.iloc[tr], X.iloc[va]
            T_tr, T_va = T[tr], T[va]
            Y_tr, Y_va = Y[tr], Y[va]
            e = clone(self.propensity_model).fit(X_tr, T_tr).predict_proba(X_va)[:, 1]
            m0 = clone(self.outcome_model).fit(X_tr[T_tr == 0], Y_tr[T_tr == 0])
            m1 = clone(self.outcome_model).fit(X_tr[T_tr == 1], Y_tr[T_tr == 1])
            p0 = m0.predict_proba(X_va)[:, 1] if hasattr(m0, "predict_proba") else m0.predict(X_va)
            p1 = m1.predict_proba(X_va)[:, 1] if hasattr(m1, "predict_proba") else m1.predict(X_va)
            pseudo[va] = dr_pseudo_outcome(p0, p1, e, T_va, Y_va)
        return pseudo

    def fit(self, X: pd.DataFrame, T: np.ndarray, Y: np.ndarray) -> "DoublyRobustLearner":
        pseudo = self.cross_fit_pseudo_outcomes(X, T, Y)
        self.final_model_.fit(X, pseudo)
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        return self.final_model_.predict(X).ravel()
