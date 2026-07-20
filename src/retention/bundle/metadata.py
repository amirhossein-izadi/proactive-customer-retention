"""Manifest writers — produces the metadata JSON files in the champion directory."""

from __future__ import annotations

import json
from pathlib import Path

from retention.features.registry import feature_manifest
from retention.utils.hashing import sha256_file
from retention.utils.serialization import save_json

METADATA_TEMPLATE = """# Model Card: Didar Retention Champion

- **Model family**: {model_family}
- **Model version**: {model_version}
- **Feature version**: {feature_version}
- **Data version**: {data_version}
- **Training samples**: {n_dev_samples}
- **Features**: {n_features}

## Limitations
- Observational (not experimental) — unmeasured confounding remains a risk.
- Calibration on dev only; test-set calibration not reported.
- Top-k ranking is deterministic only within the same bundle version.
"""

REQUIRED_ARTIFACTS = [
    "preprocessor.pkl",
    "m0.pkl",
    "m1.pkl",
    "propensity.pkl",
    "calibrator_0.pkl",
    "calibrator_1.pkl",
]


def write_champion_manifests(
    champion_dir: Path,
    *,
    model_family: str = "CatBoost T-learner",
    model_version: str = "1.1.0",
    feature_version: str = "2.2.0",
    data_version: str = "didar_retention_v1",
    n_features: int = 0,
    n_dev_samples: int = 0,
    extra_metadata: dict | None = None,
) -> None:
    """Write all manifest JSON and model card into the champion directory."""
    champ = Path(champion_dir)
    champ.mkdir(parents=True, exist_ok=True)

    # Metadata
    metadata = {
        "model_version": model_version,
        "feature_version": feature_version,
        "data_version": data_version,
        "model_family": model_family,
        "n_features": n_features,
        "n_dev_samples": n_dev_samples,
        **(extra_metadata or {}),
    }
    save_json(metadata, champ / "metadata.json")

    # Feature manifest
    save_json(feature_manifest(include_pca=True), champ / "feature_manifest.json")

    # Input / output schema (minimal — exact Pydantic schemas are in contracts/)
    save_json(
        {
            "schema_version": "1.0",
            "input": ["customer_statics", "usage_60days"],
            "output": ["p_churn_no_contact", "p_churn_contact", "estimated_uplift"],
        },
        champ / "input_schema.json",
    )
    save_json(
        {
            "schema_version": "1.0",
            "ranking_columns": [
                "rank",
                "customer_id",
                "p_churn_no_contact",
                "p_churn_contact",
                "estimated_uplift",
                "uplift_se",
                "propensity",
                "ranking_score",
            ],
        },
        champ / "output_schema.json",
    )

    # Checksums of all pickle artifacts
    checksums = {}
    for name in REQUIRED_ARTIFACTS:
        p = champ / name
        checksums[name] = sha256_file(p) if p.exists() else None
    save_json(checksums, champ / "checksums.json")

    # Model card
    card = METADATA_TEMPLATE.format(
        model_family=model_family,
        model_version=model_version,
        feature_version=feature_version,
        data_version=data_version,
        n_dev_samples=n_dev_samples,
        n_features=n_features,
    )
    (champ / "model_card.md").write_text(card)
