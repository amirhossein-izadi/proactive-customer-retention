"""Prometheus instrumentation middleware for FastAPI."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from retention.monitoring.metrics import HTTP_LATENCY, HTTP_REQUESTS


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path == "/metrics":
            return await call_next(request)
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        HTTP_REQUESTS.labels(
            method=request.method, endpoint=path, status=response.status_code
        ).inc()
        HTTP_LATENCY.labels(endpoint=path).observe(duration)
        return response
