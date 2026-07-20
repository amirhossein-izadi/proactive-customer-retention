"""Uplift-specific explanations — SHAP(p0) - SHAP(p1)."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import shap


def uplift_explanations(
    m0: Any,
    m1: Any,
    X: np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> np.ndarray:
    e0 = shap.TreeExplainer(m0)
    e1 = shap.TreeExplainer(m1)
    sv0 = e0.shap_values(X)
    sv1 = e1.shap_values(X)
    if isinstance(sv0, list):
        sv0 = sv0[1]
    if isinstance(sv1, list):
        sv1 = sv1[1]
    return sv0 - sv1
