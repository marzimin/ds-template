"""FastAPI application factory.

Run it with::

    make api          # or: cd backend && uv run uvicorn src.api.app:app --reload

Interactive documentation is served at ``/docs``, which is the quickest way to
exercise the backend before any frontend exists.
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import ErrorResponse
from src.api.routers import health, predict, runs
from src.config import project_name, read_config
from src.ml.inference import ModelNotAvailableError, get_cached_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

#: Origins allowed to call this API from a browser. The Vite dev server runs on
#: 5173 by default. Override with a comma-separated CORS_ALLOW_ORIGINS.
_DEFAULT_ALLOWED_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def allowed_origins() -> list[str]:
    """Return the browser origins permitted to call this API."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", _DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the model cache at startup without requiring a model to exist.

    Loading eagerly means the first real request is fast. Tolerating failure
    means a fresh checkout still starts: prediction routes answer 503 with
    instructions, and every other route works normally.
    """
    try:
        model = get_cached_model(read_config())
        logger.info("Loaded model %s version %s", model.name, model.version)
    except ModelNotAvailableError as exc:
        logger.warning(
            "Starting without a model: %s Prediction endpoints will return 503 "
            "until the training pipeline has run.",
            exc,
        )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(
        title=f"{project_name()} API",
        description=(
            "Serves predictions from the latest registered MLflow model, plus "
            "run metrics and artifacts for a dashboard.\n\n"
            "Feature names are never hardcoded: call `GET /api/predict/schema` "
            "to discover what a prediction request needs."
        ),
        version="0.1.0",
        lifespan=lifespan,
        # Declared once for every route so the OpenAPI schema — and the
        # TypeScript types generated from it — describe failures, not just the
        # happy path.
        responses={
            422: {"model": ErrorResponse, "description": "Invalid request"},
            503: {"model": ErrorResponse, "description": "No model available"},
        },
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(predict.router, prefix="/api")
    app.include_router(runs.router, prefix="/api")

    return app


app = create_app()
