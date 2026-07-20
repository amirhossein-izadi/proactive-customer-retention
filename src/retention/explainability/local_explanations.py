"""Per-customer explanations from champion model."""

from typing import Any

import numpy as np

from retention.explainability.reason_codes import reason_codes

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]


def local_explanations(
    model: Any,
    X: np.ndarray,
    idx: int,
    feature_names: list[str],
) -> dict:
    if shap is None:
        return {"features": [], "shap_values": []}
    e = shap.TreeExplainer(model)
    sv = e.shap_values(X[[idx]])
    if isinstance(sv, list):
        sv = sv[1]
    return {"features": reason_codes(sv[0], feature_names), "shap_values": sv[0].tolist()}
