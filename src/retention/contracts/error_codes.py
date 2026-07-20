"""Explicit API error taxonomy — structured error codes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ErrorCode:
    code: str
    message: str
    http_status: int
    retryable: bool = False


ERROR_CODES = {
    "USAGE_DUPLICATE_DAY": ErrorCode("USAGE_DUPLICATE_DAY", "Duplicate usage day found", 422),
    "USAGE_NEGATIVE_VALUE": ErrorCode("USAGE_NEGATIVE_VALUE", "Negative daily usage", 422),
    "ENGAGEMENT_OUT_OF_RANGE": ErrorCode(
        "ENGAGEMENT_OUT_OF_RANGE", "email_engagement must be [0, 1]", 422
    ),
    "DAY_OUT_OF_RANGE": ErrorCode("DAY_OUT_OF_RANGE", "day must be in [-59, 0]", 422),
    "UNKNOWN_PACKAGE": ErrorCode("UNKNOWN_PACKAGE", "Unrecognized package value", 400),
    "SCHEMA_VERSION_UNSUPPORTED": ErrorCode(
        "SCHEMA_VERSION_UNSUPPORTED", "Unsupported schema version", 400
    ),
    "BATCH_TOO_LARGE": ErrorCode("BATCH_TOO_LARGE", "Batch exceeds 5000 limit", 413),
    "INSUFFICIENT_USAGE_HISTORY": ErrorCode(
        "INSUFFICIENT_USAGE_HISTORY", "Fewer than 45 usage days", 422
    ),
    "DUPLICATE_CUSTOMER_ID": ErrorCode(
        "DUPLICATE_CUSTOMER_ID", "Customer ID repeated in batch", 409
    ),
    "FUTURE_USAGE_DAY": ErrorCode("FUTURE_USAGE_DAY", "Usage day > 0 (temporal leakage)", 422),
    "MODEL_NOT_LOADED": ErrorCode("MODEL_NOT_LOADED", "Model not yet loaded", 503, True),
    "INTERNAL_ERROR": ErrorCode("INTERNAL_ERROR", "Unexpected internal error", 500, True),
    "FEATURE_VERSION_MISMATCH": ErrorCode(
        "FEATURE_VERSION_MISMATCH", "Feature version mismatch with model", 503
    ),
}


def get_error(code: str) -> ErrorCode:
    return ERROR_CODES.get(code, ERROR_CODES["INTERNAL_ERROR"])
