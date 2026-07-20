"""Global feature importance via SHAP."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import shap


def global_importance(
    model: Any,
    X: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> dict[str, float]:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    ma: np.ndarray = np.abs(sv).mean(0)
    fn: Sequence[str] = feature_names or [f"f{i}" for i in range(len(ma))]
    return dict(zip(fn, ma.tolist()))
