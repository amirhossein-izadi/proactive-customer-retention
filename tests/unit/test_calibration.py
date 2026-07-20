"""Items 256-258: Calibration diagnostics — sigmoid, reliability, slope/intercept."""
import numpy as np
from sklearn.calibration import calibration_curve

def test_reliability_diagram():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = np.clip(y_true * 0.9 + rng.normal(0, 0.1, 500), 0, 1)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    assert ece < 0.2, f"ECE {ece:.4f} too high"

def test_calibration_slope():
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    logit = np.log(0.3 / 0.7)
    y_prob = 1 / (1 + np.exp(-(logit + rng.normal(0, 0.5, 500))))
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=10).fit(y_prob.reshape(-1, 1), y_true)
    slope = lr.coef_[0][0]
    assert 0.5 < slope < 2.0, f"Calibration slope {slope:.4f} outside [0.5, 2.0]"
