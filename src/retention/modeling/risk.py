"""Single churn-risk model — baseline for comparison against uplift methods."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from retention.features.builder import build_preprocessor


def fit_churn_risk_model(X_raw: pd.DataFrame, Y: np.ndarray) -> Pipeline:
    """Train a logistic regression to predict P(churn | X) without treatment.

    Used by ``baselines.py`` to compute a churn-risk-only ranking for
    comparison with the uplift-based champion.
    """
    prep = build_preprocessor(include_pca=True, scale=True)
    prep.fit(X_raw, Y)
    pipe = Pipeline(
        [("prep", prep), ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=42))]
    )
    pipe.fit(X_raw, Y)
    return pipe
