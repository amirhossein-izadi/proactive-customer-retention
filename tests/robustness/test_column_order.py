"""Column-order invariance — shuffled usage columns produce identical features."""

import numpy as np
import pandas as pd
from retention.features.builder import TemporalEngineer


def test_temporal_permutation_invariance():
    rng = np.random.default_rng(42)
    n_days = 10
    col_map = {f"usage_day_{d}": float(rng.poisson(5)) for d in range(n_days)}
    df = pd.DataFrame([col_map])
    e = TemporalEngineer()
    f1 = e.fit_transform(df)
    shuffled_cols = list(df.columns)
    rng.shuffle(shuffled_cols)
    df2 = df[shuffled_cols]
    f2 = e.fit_transform(df2)
    np.testing.assert_array_almost_equal(f1, f2, err_msg="Features differ after column shuffle")
