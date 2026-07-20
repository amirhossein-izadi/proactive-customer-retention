"""FastAPI application — routes, middleware, metrics, and lifecycle."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest
from starlette.responses import Response

from retention.bundle.compatibility import check_bundle
from retention.monitoring.logging import configure_logging
from retention.monitoring.metrics import MODEL_LOADED
from retention.serving.dependencies import get_bundle
from retention.utils.paths import CHAMPION_DIR
from retention.serving.exception_handlers import register_handlers
from retention.serving.middleware import PrometheusMiddleware
from retention.serving.routes_health import router as health_router
from retention.serving.routes_score import router as score_router
from retention.serving.routes_rank import router as rank_router
from retention.serving.routes_model_info import router as model_info_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging("retention-api")
    b = get_bundle()
    if b is not None:
        compat = check_bundle(CHAMPION_DIR)
        print(
            f"Bundle compatibility: {'PASS' if compat['passed'] else 'FAIL'} ({len(compat.get('errors', []))} errors)"
        )
    else:
        print("No champion bundle found. API will return 503 on /v1/score.")
    MODEL_LOADED.set(1 if b is not None else 0)
    yield


app = FastAPI(
    title="Didar Retention Ranking API",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(PrometheusMiddleware)

register_handlers(app)

app.include_router(health_router)
app.include_router(score_router)
app.include_router(rank_router)
app.include_router(model_info_router)


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
