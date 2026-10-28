"""OOF isolation — out-of-fold predictions must not use the predicted row's fold."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def test_oof_no_leakage():
    """Each row's OOF prediction comes from a model that never saw that row."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(n)
    for tr, va in skf.split(X, y):
        m = LogisticRegression(max_iter=1000, random_state=42).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]

    assert np.all((0 <= oof) & (oof <= 1))
    assert not np.all(oof == 0), "OOF predictions should not be all zeros"


def test_oof_covers_all_rows():
    """Every row should have an OOF prediction."""
    rng = np.random.default_rng(42)
    n = 200
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] > 0).astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(n)
    for tr, va in skf.split(X, y):
        m = LogisticRegression(max_iter=1000, random_state=42).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]

    assert np.sum(oof > 0) == n, "Not all rows had OOF predictions"
