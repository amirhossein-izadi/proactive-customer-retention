"""Integration test: bundle serialization roundtrip + prediction parity."""

from __future__ import annotations

import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

BASE = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE / "models" / "champion"


def test_bundle_roundtrip():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 3))
    T = np.array([0] * 15 + [1] * 15)
    Y = np.array([0] * 8 + [1] * 7 + [0] * 7 + [1] * 8)

    m0 = LogisticRegression(max_iter=1000).fit(X[T == 0], Y[T == 0])
    m1 = LogisticRegression(max_iter=1000).fit(X[T == 1], Y[T == 1])
    p0_pre = m0.predict_proba(X)[:, 1]
    p1_pre = m1.predict_proba(X)[:, 1]

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        pickle.dump(m0, open(Path(tmp) / "m0.pkl", "wb"))
        pickle.dump(m1, open(Path(tmp) / "m1.pkl", "wb"))
        cal0 = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="sigmoid", cv=3)
        cal1 = CalibratedClassifierCV(LogisticRegression(max_iter=1000), method="sigmoid", cv=3)
        cal0.fit(X[T == 0], Y[T == 0])
        cal1.fit(X[T == 1], Y[T == 1])
        pickle.dump(cal0, open(Path(tmp) / "calibrator_0.pkl", "wb"))
        pickle.dump(cal1, open(Path(tmp) / "calibrator_1.pkl", "wb"))
        m0_l = pickle.load(open(Path(tmp) / "m0.pkl", "rb"))
        m1_l = pickle.load(open(Path(tmp) / "m1.pkl", "rb"))

    p0_post = m0_l.predict_proba(X)[:, 1]
    p1_post = m1_l.predict_proba(X)[:, 1]
    assert np.allclose(p0_pre, p0_post), "m0 predictions changed"
    assert np.allclose(p1_pre, p1_post), "m1 predictions changed"


def test_bundle_exists():
    import pytest

    if not (MODELS_DIR / "m0.pkl").exists():
        pytest.skip("champion bundle not built yet")
    for f in [
        "m0.pkl",
        "m1.pkl",
        "calibrator_0.pkl",
        "calibrator_1.pkl",
        "propensity.pkl",
        "preprocessor.pkl",
        "metadata.json",
    ]:
        assert (MODELS_DIR / f).exists(), f"Missing: {f}"
