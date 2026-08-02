"""Shared dependencies for API routes.

FastAPI calls these before a handler runs and passes the results in. Keeping the
model lookup here means a route body never has to think about caching or about
what to do when nothing has been trained yet.
"""

import logging
from typing import Annotated, Any

import mlflow
from fastapi import Depends, HTTPException, status
from mlflow.tracking import MlflowClient

from src.config import read_config
from src.ml.inference import LoadedModel, ModelNotAvailableError, get_cached_model

logger = logging.getLogger(__name__)


def get_config() -> dict[str, Any]:
    """Return the parsed project configuration."""
    return read_config()


def tracking_uri() -> str:
    """Return the MLflow tracking URI the API is actually reading from.

    Asks MLflow rather than re-deriving it from the environment. The two used to
    be computed independently, so health could report the configured server
    while the client read a different store entirely — the report agreed with
    the documentation and disagreed with reality, which is the least useful
    combination. Startup calls ``configure_tracking_uri``; this reflects it.
    """
    return str(mlflow.get_tracking_uri())


ConfigDep = Annotated[dict[str, Any], Depends(get_config)]


def get_model(config: ConfigDep) -> LoadedModel:
    """Return the loaded model, or fail with 503 if there is not one yet.

    A missing model is an expected state on a fresh checkout, not a bug, so the
    response explains how to fix it instead of surfacing a traceback.

    Raises:
        HTTPException: 503 when no registered model can be loaded.
    """
    try:
        return get_cached_model(config)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            # The exception already explains what is missing and how to fix it;
            # only add that recovery needs no restart, since a failed load is
            # not cached and the next request retries.
            detail=f"{exc} The API does not need restarting.",
        ) from exc


ModelDep = Annotated[LoadedModel, Depends(get_model)]


def get_optional_model(config: ConfigDep) -> LoadedModel | None:
    """Return the loaded model, or None if there is not one yet.

    For routes that must answer whether a model exists rather than requiring
    one — health being the obvious case. Kept as a dependency, not a direct
    call, so it can be substituted in tests like any other.
    """
    try:
        return get_cached_model(config)
    except ModelNotAvailableError as exc:
        logger.debug("No model available: %s", exc)
        return None


OptionalModelDep = Annotated[LoadedModel | None, Depends(get_optional_model)]


def get_mlflow_client() -> MlflowClient:
    """Return an MLflow client for reading runs and artifacts."""
    return MlflowClient()


ClientDep = Annotated[MlflowClient, Depends(get_mlflow_client)]
