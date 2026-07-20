"""Feature registry — versioned feature specs for the model manifest."""

from __future__ import annotations

from retention.contracts.features import FeatureSpec
from retention.features.feature_names import (
    CATEGORICAL_FEATURES,
    PCA_FEATURES,
    STATIC_FEATURES,
    TEMPORAL_FEATURES,
)

FEATURE_VERSION = "2.2.0"

_FAMILIES = {
    "static": (STATIC_FEATURES, "customer attributes"),
    "temporal": (TEMPORAL_FEATURES, "60-day usage sequence"),
    "pca": (PCA_FEATURES, "PCA over 60-day usage sequence"),
    "categorical": (CATEGORICAL_FEATURES, "package one-hot"),
}


def feature_specs(include_pca: bool = True) -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []
    for family, (names, source) in _FAMILIES.items():
        if family == "pca" and not include_pca:
            continue
        specs.extend(
            FeatureSpec(name=n, dtype="float64", family=family, source=source) for n in names
        )
    return specs


def feature_manifest(include_pca: bool = True) -> dict:
    specs = feature_specs(include_pca)
    return {
        "feature_version": FEATURE_VERSION,
        "n_features": len(specs),
        "features": [s.to_dict() for s in specs],
    }
