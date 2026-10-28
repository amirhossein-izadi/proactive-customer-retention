FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ src/

RUN uv sync --frozen --no-dev --group monitoring --group explainability

COPY configs/ configs/
COPY models/ models/

RUN addgroup --system --gid 1001 appgroup && \
    useradd --system --uid 1001 --gid 1001 --no-create-home appuser && \
    chown -R appuser:appgroup /app/src /app/configs /app/models

USER appuser

ENV PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "retention.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
