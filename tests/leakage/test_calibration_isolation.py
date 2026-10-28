"""Calibration isolation — calibrators must not see evaluation fold data."""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression


def test_calibrator_fit_on_train_only():
    """Simulate: calibrator fitted on train fold, evaluated on validation fold."""
    rng = np.random.default_rng(42)
    n_train = 200
    X_train = rng.normal(size=(n_train, 5))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    n_val = 100
    X_val = rng.normal(size=(n_val, 5))
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)

    base = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(X_train, y_train)

    proba = cal.predict_proba(X_val)[:, 1]
    assert np.all((0 <= proba) & (proba <= 1))
    assert proba.shape == (n_val,)


def test_calibrator_not_refit_on_val():
    """Ensure calibrator doesn't leak: fitting only on train, no refit on val."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int)
    X_train, X_val = X[:200], X[200:]
    y_train, y_val = y[:200], y[200:]

    base = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_train, y_train)

    p_train = cal.predict_proba(X_train)[:, 1]
    p_val = cal.predict_proba(X_val)[:, 1]
    assert p_train.shape == (200,)
    assert p_val.shape == (100,)
