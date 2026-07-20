"""Item 230: Synthetic data test — known beneficial treatment produces positive uplift."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_known_treatment_effect():
    rng = np.random.default_rng(42)
    n = 500
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    T = rng.integers(0, 2, size=n)
    p_churn = 0.3 - 0.2 * T + 0.1 * X["x1"]  
    Y = (rng.random(n) < p_churn).astype(int)
    from retention.modeling.s_learner import SLearner
    sl = SLearner(LogisticRegression(max_iter=2000, C=1.0))
    sl.fit(X, T, Y)
    pul = sl.predict_uplift(X)
    assert pul.mean() > 0, f"S-learner: uplift mean {pul.mean():.4f} should be positive"
    from retention.modeling.t_learner import TLearner
    tl = TLearner(LogisticRegression(max_iter=2000, C=1.0))
    tl.fit(X, T, Y)
    pul2 = tl.predict_uplift(X)
    assert pul2.mean() > 0, f"T-learner: uplift mean {pul2.mean():.4f} should be positive"
