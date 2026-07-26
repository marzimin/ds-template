"""Liveness endpoint.

Deliberately does not require a model: this must answer even on a fresh
checkout, so that "is the API up?" and "has anything been trained?" are two
distinct questions with two distinct answers.
"""

from fastapi import APIRouter

from src.api.deps import OptionalModelDep, tracking_uri
from src.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health")
def health(model: OptionalModelDep) -> HealthResponse:
    """Report whether the API is serving and whether a model is loaded."""
    return HealthResponse(
        status="ok",
        mlflow_tracking_uri=tracking_uri(),
        model_available=model is not None,
        model_name=model.name if model else None,
        model_version=model.version if model else None,
    )
