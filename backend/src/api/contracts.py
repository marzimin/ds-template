"""Request and response shapes for the API.

Named ``contracts`` rather than the usual FastAPI ``models`` because in this
repository "model" already means a trained estimator, and ``schemas`` already
means the Pandera DataFrame contracts in :mod:`src.schemas`. Everything here is
an HTTP payload shape; nothing here is an ML model.

These are the API's contract. FastAPI validates against them, documents them at
``/docs``, and publishes them as an OpenAPI schema from which the frontend's
TypeScript types are generated — so a field renamed here surfaces as a compile
error in the frontend rather than a silently missing value.

Nothing here names a dataset feature. Prediction inputs are a free-form mapping
validated at request time against the loaded model's signature, which is what
keeps the template usable with any dataset.
"""

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """A failed request. FastAPI uses ``detail`` for its own errors too."""

    detail: str = Field(description="Human-readable explanation of the failure.")


class HealthResponse(BaseModel):
    """Liveness and readiness of the service and its dependencies."""

    status: str = Field(description="'ok' when the API itself is serving.")
    mlflow_tracking_uri: str = Field(description="Tracking server the API reads.")
    model_available: bool = Field(
        description="Whether a registered model could be loaded. False on a "
        "fresh checkout until the training pipeline has run."
    )
    model_name: str | None = Field(
        default=None, description="Registered model name, when one is loaded."
    )
    model_version: str | None = Field(
        default=None, description="Registered model version, when one is loaded."
    )


class FeatureSpecResponse(BaseModel):
    """One input the model expects, described so a UI can render a control."""

    name: str = Field(description="Feature name, as the model expects it.")
    mlflow_type: str = Field(description="MLflow schema type, e.g. 'double'.")
    kind: str = Field(
        description="Coarse type for widget selection: number, integer, "
        "boolean, string, or datetime."
    )
    required: bool = Field(description="Whether a prediction request must set it.")
    example: Any = Field(
        default=None,
        description="Representative value from the logged input example, "
        "suitable for pre-filling a form. Null if unavailable.",
    )


class PredictSchemaResponse(BaseModel):
    """The prediction contract, discovered from the model signature.

    A frontend reads this and builds its input form at runtime, so swapping
    datasets needs no frontend changes.
    """

    model_name: str
    model_version: str
    features: list[FeatureSpecResponse]


class PredictRequest(BaseModel):
    """Feature values for a single prediction.

    Keys are validated against the model signature when the request is handled,
    not statically, because the valid keys depend on the trained model.
    """

    features: dict[str, Any] = Field(
        description="Feature name to value. Call GET /api/predict/schema to "
        "discover the expected names and types."
    )


class PredictResponse(BaseModel):
    """The model's answer for one record."""

    prediction: Any = Field(description="The predicted label or value.")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Per-class scores, when the estimator exposes "
        "predict_proba. Null for regressors and estimators without it.",
    )
    model_name: str
    model_version: str


class ModelReloadResponse(BaseModel):
    """Which model version is in memory after a reload."""

    model_name: str
    model_version: str
    reloaded: bool = Field(
        default=True, description="Always true; a failure returns an error instead."
    )


class RunSummary(BaseModel):
    """One MLflow run, as listed in a dashboard."""

    run_id: str
    run_name: str | None = None
    status: str | None = None
    start_time: int | None = Field(
        default=None, description="Unix epoch milliseconds, as MLflow reports it."
    )
    end_time: int | None = None
    metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Final logged value per metric key. Keys depend on what the "
        "pipeline logged, so consumers should render whatever is present.",
    )


class RunDetail(RunSummary):
    """One MLflow run with its parameters and tags."""

    params: dict[str, str] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class ArtifactEntry(BaseModel):
    """One file or directory logged against a run."""

    path: str = Field(description="Path relative to the run's artifact root.")
    is_dir: bool
    file_size: int | None = None
