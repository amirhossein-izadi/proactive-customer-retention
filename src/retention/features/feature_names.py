"""Engineered feature-name constants — imported by registry, bundle and tests."""

from __future__ import annotations

from retention.contracts.raw_data import usage_columns
from retention.features.builder import (
    PCAEmbedding,
    StaticEngineer,
    TemporalEngineer,
)

USAGE_COLS = usage_columns()
STATIC_FEATURES = StaticEngineer().feature_names()
TEMPORAL_FEATURES = TemporalEngineer().feature_names()
PCA_FEATURES = ["pca_usage_0", "pca_usage_1", "pca_usage_2", "pca_usage_3", "pca_usage_4"]
CATEGORICAL_FEATURES = [
    "package_Basic",
    "package_Enterprise",
    "package_Pro",
    "subscription_type_Annual",
    "subscription_type_Monthly",
]


def all_feature_names(include_pca: bool = True) -> list[str]:
    """Names in the exact column order produced by ``build_preprocessor``."""
    names = STATIC_FEATURES + TEMPORAL_FEATURES
    if include_pca:
        names += PCA_FEATURES
    return names + CATEGORICAL_FEATURES
