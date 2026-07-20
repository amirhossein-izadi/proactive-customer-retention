"""Dataset snapshot and lineage tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from retention.utils.hashing import sha256_file
from retention.utils.serialization import save_json


def create_dataset_manifest(
    customer_path: str | Path,
    usage_path: str | Path,
    feature_version: str = "0.0.0",
    schema_version: str = "1.0.0",
    git_commit: str = "",
) -> dict:
    return {
        "dataset_version": "didar_retention_v1",
        "customer_file_sha256": sha256_file(customer_path),
        "usage_file_sha256": sha256_file(usage_path),
        "customer_row_count": sum(1 for _ in open(customer_path)) - 1,
        "usage_row_count": sum(1 for _ in open(usage_path)) - 1,
        "schema_version": schema_version,
        "feature_version": feature_version,
        "usage_window": "[-59, 0]",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
    }


def save_dataset_manifest(manifest: dict, output_path: str | Path) -> None:
    save_json(manifest, output_path)
