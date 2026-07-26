"""Feature discovery and single-record prediction."""

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.contracts import (
    FeatureSpecResponse,
    ModelReloadResponse,
    PredictRequest,
    PredictResponse,
    PredictSchemaResponse,
)
from src.api.deps import ConfigDep, ModelDep, get_model
from src.ml.inference import FeatureValidationError, clear_model_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get(
    "/schema",
    response_model=PredictSchemaResponse,
    summary="Discover the features a prediction needs",
)
def predict_schema(model: ModelDep) -> PredictSchemaResponse:
    """Describe the model's inputs, read from its logged signature.

    A frontend calls this to build its prediction form, so that changing dataset
    changes the form without any code change.
    """
    return PredictSchemaResponse(
        model_name=model.name,
        model_version=model.version,
        features=[
            FeatureSpecResponse(
                name=spec.name,
                mlflow_type=spec.mlflow_type,
                kind=spec.kind,
                required=spec.required,
                example=spec.example,
            )
            for spec in model.features
        ],
    )


@router.post(
    "",
    response_model=PredictResponse,
    summary="Predict for a single record",
)
def predict(request: PredictRequest, model: ModelDep) -> PredictResponse:
    """Predict for one set of feature values.

    Raises:
        HTTPException: 422 if the submitted features do not match the model
            signature, listing exactly which names were missing or unknown.
    """
    try:
        result = model.predict(request.features)
    except FeatureValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc
    except (ValueError, TypeError, KeyError) as exc:
        # Wrong value types reach the estimator and fail there. Report it as a
        # client error rather than a 500: the request is at fault, not the API.
        logger.info("Prediction rejected for invalid input: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Could not predict from the submitted values: {exc}",
        ) from exc

    return PredictResponse(
        prediction=result.prediction,
        probabilities=result.probabilities,
        model_name=model.name,
        model_version=model.version,
    )


@router.post(
    "/reload",
    response_model=ModelReloadResponse,
    summary="Pick up a newly trained model version",
)
def reload_model(config: ConfigDep) -> ModelReloadResponse:
    """Drop the cached model so the newest registered version is loaded.

    A server holds one model in memory for the life of the process, so training
    a new version while the API runs would otherwise keep serving the old one
    indefinitely. This makes that recoverable without a restart.

    Raises:
        HTTPException: 503 if no model can be loaded after clearing the cache.
    """
    clear_model_cache()
    model = get_model(config)
    logger.info("Reloaded model %s version %s", model.name, model.version)
    return ModelReloadResponse(
        model_name=model.name,
        model_version=model.version,
    )
