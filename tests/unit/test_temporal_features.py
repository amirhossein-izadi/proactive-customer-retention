"""Feature ordering invariance — shuffled columns produce identical features."""

import numpy as np
import pandas as pd
from retention.features.builder import TemporalEngineer


def test_chronological_ordering():
    rng = np.random.default_rng(42)
    data = {f"usage_day_{d}": rng.poisson(5, 10).astype(float) for d in range(-59, 1)}
    base = pd.DataFrame(data)
    shuffled = list(base.columns)
    rng.shuffle(shuffled)
    base_shuffled = base[shuffled]
    e = TemporalEngineer()
    f1 = e.fit_transform(base)
    f2 = e.fit_transform(base_shuffled)
    np.testing.assert_array_almost_equal(f1, f2)


def test_missing_not_zero():
    arr = np.array([[np.nan, 0.0, 5.0, np.nan, 0.0, 3.0, 1.0, 2.0, 0.0, 4.0]])
    df = pd.DataFrame([{f"usage_day_{i}": arr[0, i] for i in range(arr.shape[1])}])
    feats = TemporalEngineer().fit_transform(df)
    names = TemporalEngineer().feature_names()
    assert feats[0, names.index("zero_days")] == 3.0
    assert feats[0, names.index("missing_days")] == 2.0
