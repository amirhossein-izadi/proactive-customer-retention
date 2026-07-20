"""FastAPI exception handlers — maps error codes to structured responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from retention.contracts.error_codes import get_error


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception):
        err = get_error("INTERNAL_ERROR")
        return JSONResponse(
            status_code=err.http_status,
            content={
                "request_id": request.headers.get("x-request-id", ""),
                "error": {"code": err.code, "message": err.message, "retryable": err.retryable},
            },
        )

    @app.exception_handler(ValueError)
    async def validation_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=422,
            content={
                "request_id": request.headers.get("x-request-id", ""),
                "error": {"code": "VALIDATION_ERROR", "message": str(exc)},
            },
        )
