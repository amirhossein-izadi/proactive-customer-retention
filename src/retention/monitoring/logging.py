"""Structured JSON logging with PII-safe field filtering.

Filters out customer_id, email, company_name from log records.
Uses structlog for structured output. Compatible with FastAPI middleware.
"""

from __future__ import annotations

import hashlib

import structlog
from structlog.processors import JSONRenderer, TimeStamper, add_log_level

PII_FIELDS = {"customer_id", "email", "company_name", "email_address", "full_name", "phone"}


def hash_identifier(value: str) -> str:
    """One-way pseudonym for identifiers that must appear in audit trails."""
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def filter_pii(logger, method_name, event_dict):
    """Remove PII fields from log output."""
    for field in PII_FIELDS:
        event_dict.pop(field, None)
    return event_dict


def configure_logging(service_name: str = "retention-ranking"):
    structlog.configure(
        processors=[
            add_log_level,
            TimeStamper(fmt="iso"),
            filter_pii,
            JSONRenderer(),
        ],
        context_class=dict,
        cache_logger_on_first_use=True,
        wrapper_class=structlog.BoundLogger,
    )
    return structlog.get_logger(service_name)
