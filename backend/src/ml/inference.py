"""Loading a registered model and making predictions with it.

This module is the serving counterpart to :mod:`src.ml.train_model`. It knows
nothing about HTTP: it raises domain exceptions that the API layer translates
into status codes, which keeps prediction logic testable without a web server.

The template must work for any dataset, so nothing here names a feature. The
feature contract is read back from the signature MLflow logged at training time,
which is what lets a frontend build its input form at runtime.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.ml.tracking import registered_model_name

logger = logging.getLogger(__name__)

#: MLflow schema types mapped to a coarse kind a UI can pick a widget from.
_TYPE_KINDS = {
    "double": "number",
    "float": "number",
    "long": "integer",
    "integer": "integer",
    "boolean": "boolean",
    "string": "string",
    "datetime": "datetime",
}


class ModelNotAvailableError(RuntimeError):
    """Raised when no registered model can be loaded.

    Expected on a fresh checkout: nothing has been trained yet. The API turns
    this into a 503 with instructions rather than failing to start.
    """


class FeatureValidationError(ValueError):
    """Raised when submitted features do not match the model signature."""

    def __init__(
        self,
        message: str,
        missing: list[str] | None = None,
        unexpected: list[str] | None = None,
    ) -> None:
        """Record which feature names were missing or unrecognised."""
        super().__init__(message)
        self.message = message
        self.missing = missing or []
        self.unexpected = unexpected or []


@dataclass(frozen=True)
class FeatureSpec:
    """One input the model expects, as declared by its signature."""

    name: str
    mlflow_type: str
    kind: str
    required: bool
    example: Any = None


@dataclass(frozen=True)
class Prediction:
    """A single prediction and, where the estimator supports it, class scores."""

    prediction: Any
    probabilities: dict[str, float] | None = None


@dataclass
class LoadedModel:
    """A registered model held in memory together with its input contract."""

    name: str
    version: str
    run_id: str | None
    features: tuple[FeatureSpec, ...]
    #: The flavour-agnostic pyfunc wrapper; what predictions go through.
    pyfunc_model: Any = field(repr=False)
    #: The underlying estimator, when reachable. Only needed for predict_proba.
    estimator: Any = field(default=None, repr=False)

    @property
    def feature_names(self) -> list[str]:
        """Feature names in the order the model expects them."""
        return [spec.name for spec in self.features]

    def build_frame(self, values: Mapping[str, Any]) -> pd.DataFrame:
        """Validate submitted values and shape them into a one-row DataFrame.

        Args:
            values: Feature name to value, as submitted by a caller.

        Returns:
            A single-row DataFrame with columns in signature order.

        Raises:
            FeatureValidationError: If required features are missing or unknown
                names were supplied.
        """
        required = [spec.name for spec in self.features if spec.required]
        known = set(self.feature_names)

        missing = [name for name in required if name not in values]
        unexpected = [name for name in values if name not in known]
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing required features: {missing}")
            if unexpected:
                parts.append(f"unrecognised features: {unexpected}")
            raise FeatureValidationError(
                "Submitted features do not match the model signature — "
                + "; ".join(parts),
                missing=missing,
                unexpected=unexpected,
            )

        ordered = {name: values[name] for name in self.feature_names if name in values}
        return pd.DataFrame([ordered])

    def predict(self, values: Mapping[str, Any]) -> Prediction:
        """Predict for a single record of feature values.

        Args:
            values: Feature name to value.

        Returns:
            The prediction, with class probabilities when available.

        Raises:
            FeatureValidationError: If the values do not match the signature.
        """
        frame = self.build_frame(values)
        try:
            raw_prediction = self.pyfunc_model.predict(frame)
        except MlflowException as exc:
            # MLflow enforces the logged signature and rejects values it cannot
            # coerce — a string where a double is expected, for example. That is
            # bad input, not a server fault, so surface it as a validation error
            # rather than letting it become a 500.
            raise FeatureValidationError(
                f"Feature values do not satisfy the model schema: "
                f"{_concise_schema_error(exc)}"
            ) from exc
        prediction = _first_scalar(raw_prediction)
        return Prediction(
            prediction=prediction,
            probabilities=self._probabilities(frame),
        )

    def _probabilities(self, frame: pd.DataFrame) -> dict[str, float] | None:
        """Return per-class probabilities, or None if unsupported.

        Read from the underlying estimator rather than the pyfunc wrapper, since
        ``pyfunc.predict`` returns labels only. Regressors and estimators
        without ``predict_proba`` simply report no probabilities.
        """
        estimator = self.estimator
        if estimator is None or not hasattr(estimator, "predict_proba"):
            return None
        try:
            scores = estimator.predict_proba(frame)
        except Exception:  # noqa: BLE001 - never fail a prediction over scores
            logger.warning("predict_proba failed; returning prediction only.")
            return None

        classes = getattr(estimator, "classes_", None)
        row = scores[0]
        labels = (
            [str(c) for c in classes]
            if classes is not None
            else [str(i) for i in range(len(row))]
        )
        return {label: float(score) for label, score in zip(labels, row)}


def _concise_schema_error(exc: MlflowException) -> str:
    """Reduce an MLflow schema-enforcement message to its useful tail.

    MLflow embeds the offending DataFrame and the full signature in the message,
    which is far too much to return over HTTP. The actionable part follows the
    final ``Error:``.
    """
    message = str(exc)
    _, separator, tail = message.rpartition("Error: ")
    concise = tail if separator else message
    return concise[:300].strip()


def _first_scalar(predictions: Any) -> Any:
    """Extract the first prediction as a JSON-serialisable scalar."""
    if isinstance(predictions, pd.DataFrame):
        value = predictions.iloc[0, 0]
    elif isinstance(predictions, pd.Series):
        value = predictions.iloc[0]
    else:
        value = predictions[0]
    return value.item() if hasattr(value, "item") else value


def _describe_features(model: Any) -> tuple[FeatureSpec, ...]:
    """Read the model's input schema into feature specifications.

    The logged input example, when present, supplies representative values so a
    UI can pre-fill its form with a row that is known to predict successfully.
    """
    schema = model.metadata.get_input_schema()
    if schema is None:
        logger.warning(
            "Model has no input signature; feature discovery is unavailable. "
            "Retrain so the signature is logged."
        )
        return ()

    examples = _load_example_row(model)
    specs = []
    for input_field in schema.inputs:
        type_name = getattr(input_field.type, "name", str(input_field.type))
        specs.append(
            FeatureSpec(
                name=str(input_field.name),
                mlflow_type=type_name,
                kind=_TYPE_KINDS.get(type_name, "string"),
                required=bool(getattr(input_field, "required", True)),
                example=examples.get(str(input_field.name)),
            )
        )
    return tuple(specs)


def _load_example_row(model: Any) -> dict[str, Any]:
    """Return one row of the logged input example, or an empty mapping.

    Best effort only: this downloads artifacts, so it must never prevent a model
    from being served.
    """
    try:
        local_path = mlflow.artifacts.download_artifacts(
            model.metadata.get_model_info().model_uri
        )
        example = model.metadata.load_input_example(local_path)
    except Exception:  # noqa: BLE001 - purely a convenience for pre-filling
        logger.debug("No usable input example for this model.", exc_info=True)
        return {}

    if example is None or not isinstance(example, pd.DataFrame) or example.empty:
        return {}
    row = example.iloc[0]
    return {
        str(name): (value.item() if hasattr(value, "item") else value)
        for name, value in row.items()
    }


def _latest_version(client: MlflowClient, name: str) -> Any:
    """Return the highest-numbered version of a registered model.

    Raises:
        ModelNotAvailableError: If the model is unregistered or has no versions.
    """
    try:
        versions = client.search_model_versions(
            f"name='{name}'", order_by=["version_number DESC"], max_results=1
        )
    except MlflowException as exc:
        raise ModelNotAvailableError(
            f"Could not query the MLflow Model Registry for {name!r}: {exc}"
        ) from exc

    if not versions:
        raise ModelNotAvailableError(
            f"No versions registered for model {name!r}. Train one first — "
            "run `make pipeline`."
        )
    return versions[0]


def load_model(config: dict[str, Any]) -> LoadedModel:
    """Load the latest registered model and describe its inputs.

    Args:
        config: Parsed ``cfg/config.yaml`` contents.

    Returns:
        The loaded model with its feature contract.

    Raises:
        ModelNotAvailableError: If no model has been registered yet, or the
            registry is unreachable.
    """
    name = registered_model_name(config)
    client = MlflowClient()
    version = _latest_version(client, name)
    model_uri = f"models:/{name}/{version.version}"

    logger.info("Loading model %s", model_uri)
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except MlflowException as exc:
        raise ModelNotAvailableError(
            f"Model {model_uri!r} is registered but could not be loaded: {exc}"
        ) from exc

    try:
        raw_model = model.get_raw_model()
    except Exception:  # noqa: BLE001 - only needed for probabilities
        logger.debug("Underlying estimator unavailable.", exc_info=True)
        raw_model = None

    return LoadedModel(
        name=name,
        version=str(version.version),
        run_id=getattr(version, "run_id", None),
        features=_describe_features(model),
        pyfunc_model=model,
        estimator=raw_model,
    )


# A server loads a model once and reuses it for every request; loading takes
# seconds, which is fine at startup and not per request.
_cache_lock = threading.Lock()
_cached_model: LoadedModel | None = None


def get_cached_model(config: dict[str, Any]) -> LoadedModel:
    """Return the cached model, loading it on first use.

    Raises:
        ModelNotAvailableError: If no model could be loaded.
    """
    global _cached_model
    with _cache_lock:
        if _cached_model is None:
            _cached_model = load_model(config)
        return _cached_model


def clear_model_cache() -> None:
    """Drop the cached model so the next request reloads it.

    Used after training a new version, and by tests.
    """
    global _cached_model
    with _cache_lock:
        _cached_model = None
