"""Business-readable reason codes from SHAP values."""

import numpy as np


def reason_codes(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_k: int = 4,
) -> list[str]:
    idx = np.argsort(np.abs(shap_values))[::-1][:top_k]
    return [feature_names[i] for i in idx]
