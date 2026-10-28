"""Training pipeline — verify trainer can be imported and basic structure is valid."""

import numpy as np
from retention.utils.seeds import generate_seeds
from retention.modeling.t_learner import TLearner
from retention.modeling.s_learner import SLearner
from sklearn.linear_model import LogisticRegression


def test_t_learner_instantiation():
    tl = TLearner(estimator=LogisticRegression(max_iter=1000))
    assert tl is not None


def test_t_learner_train_and_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    T = rng.integers(0, 2, size=100)
    Y = (X[:, 0] + X[:, 1] + T * 0.5 > 0).astype(int)

    tl = TLearner(estimator=LogisticRegression(max_iter=1000, random_state=42))
    tl.fit(X, T, Y)
    p0, p1 = tl.predict_proba(X)
    uplift = tl.predict_uplift(X)

    assert p0.shape == (100,)
    assert p1.shape == (100,)
    assert uplift.shape == (100,)
    assert np.all((0 <= p0) & (p0 <= 1))
    assert np.all((0 <= p1) & (p1 <= 1))


def test_s_learner_train_and_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 5))
    T = rng.integers(0, 2, size=100)
    Y = (X[:, 0] + X[:, 1] + T * 0.5 > 0).astype(int)

    import pandas as pd
    sl = SLearner(LogisticRegression(max_iter=1000, random_state=42))
    sl.fit(pd.DataFrame(X), T, Y)
    p0, p1 = sl.predict_potential_outcomes(pd.DataFrame(X))

    assert p0.shape == (100,)
    assert p1.shape == (100,)


def test_seeds_work_with_trainer():
    seeds = generate_seeds(42)
    assert "model_seed" in seeds
    assert isinstance(seeds["model_seed"], int)
