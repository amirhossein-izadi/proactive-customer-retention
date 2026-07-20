"""Items 214-219: Propensity AUC, overlap, clipping sensitivity."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

def test_propensity_diagnostics():
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, 3))
    T = (0.3 + 0.2 * X[:, 0] + 0.1 * X[:, 1] > rng.uniform(0, 1, n)).astype(int)
    prop = LogisticRegression(max_iter=2000).fit(X, T)
    e = prop.predict_proba(X)[:, 1]
    auc = roc_auc_score(T, e)
    overlap = ((e >= 0.05) & (e <= 0.95)).mean() * 100
    assert auc > 0.6, f"Propensity AUC {auc:.4f} too low"
    assert overlap > 50, f"Common support {overlap:.1f}% too low"
