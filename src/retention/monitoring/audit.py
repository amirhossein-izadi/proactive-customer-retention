"""Audit logging for model decisions — PII-safe structured events."""

from __future__ import annotations

from typing import Any

import structlog

from retention.monitoring.logging import hash_identifier

logger = structlog.get_logger("retention.audit")


def audit_event(event: str, customer_id: Any | None = None, **fields: Any) -> None:
    """Emit an audit event. Raw customer IDs are hashed, never logged in clear."""
    if customer_id is not None:
        fields["customer_ref"] = hash_identifier(str(customer_id))
    logger.info(event, **fields)
