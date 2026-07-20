"""Feature specification contract used by the feature registry and manifests."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    dtype: str
    family: str  # static | temporal | pca | categorical | missingness
    source: str  # raw column(s) or derivation
    formula: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
