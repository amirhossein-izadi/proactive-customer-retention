"""Prometheus metric definitions — imported by middleware and routes.

Label cardinality is intentionally low: no customer-level labels ever.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency", ["endpoint"])
INFERENCE_REQUESTS = Counter(
    "inference_requests_total", "Inference requests", ["endpoint", "status"]
)
INFERENCE_LATENCY = Histogram("inference_duration_seconds", "Model inference latency", ["endpoint"])
UPLIFT_SCORES = Histogram(
    "estimated_uplift",
    "Distribution of estimated uplift scores",
    buckets=[-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0],
)
MODEL_LOADED = Gauge("model_loaded", "Champion bundle loaded (1=yes, 0=no)")
SCHEMA_VALIDATION_FAILURES = Counter(
    "schema_validation_failures_total", "Schema validation failures (pandera)", ["error_type"]
)
FEATURE_PSI = Gauge("feature_psi", "Population Stability Index per feature", ["feature"])
ACTIVE_DAYS_PCT = Gauge("active_days_pct", "Active days coverage ratio")
ZERO_DAYS_RATIO = Gauge("zero_days_ratio", "Zero-usage day ratio in scoring request")
INFERENCE_COMMON_SUPPORT = Gauge(
    "inference_common_support_ratio", "Fraction of last scored batch inside propensity support"
)
DATA_QUALITY_MISSING_RATE = Gauge(
    "data_quality_missing_rate", "Missing-value rate per column", ["column"]
)
NEGATIVE_UPLIFT_RATE = Counter(
    "inference_negative_uplift_total", "Inferences where estimated uplift < 0"
)
