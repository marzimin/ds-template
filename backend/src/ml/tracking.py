"""MLflow tracking setup and flavor-aware model logging.

The template lets ``cfg/config.yaml`` name any estimator via ``model_registry``,
so model logging cannot assume a single MLflow flavor. ``mlflow.sklearn`` in
particular serialises through skops, which refuses to persist non-sklearn types
such as ``xgboost.sklearn.XGBClassifier``. :func:`log_model` therefore picks the
flavor from the registry's import path, which keeps the registry honest: adding
``lightgbm.LGBMClassifier`` to the config is enough to make it work.

Models are logged with a signature and registered under a stable name so that
consumers — the API in particular — can load ``models:/<name>/<alias>`` without
knowing which run produced it, and can read the input schema to discover what
features a prediction request needs.
"""

import importlib
import logging
import os
from contextlib import AbstractContextManager, nullcontext
from types import ModuleType
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mlflow
from mlflow.exceptions import MlflowException, RestException
from mlflow.models import ModelSignature, infer_signature

from src.config import project_name, read_config

logger = logging.getLogger(__name__)

DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

#: Root module of a ``model_registry`` import path → MLflow flavor module.
#: Anything unlisted falls back to ``mlflow.sklearn``, which covers every
#: estimator that implements the scikit-learn API natively.
_FLAVOR_BY_ROOT_MODULE = {
    "xgboost": "mlflow.xgboost",
    "lightgbm": "mlflow.lightgbm",
    "catboost": "mlflow.catboost",
    "sklearn": "mlflow.sklearn",
    "statsmodels": "mlflow.statsmodels",
}
_DEFAULT_FLAVOR = "mlflow.sklearn"


def configure_tracking_uri() -> str:
    """Point MLflow at the configured tracking server and return that URI.

    Every entry point must call this before touching MLflow. Left unset, MLflow
    falls back to a store beside the current working directory, which is not the
    server the pipeline wrote to — and the failure is quiet rather than loud:
    the API finds the registered model but its artifacts are recorded as
    ``mlflow-artifacts:`` URIs that only the tracking server can resolve, so
    loading fails and every prediction answers 503 after a training run that
    reported success.

    Separate from :func:`setup_mlflow` because the API must not require a
    reachable server to start: it degrades to "no model yet" instead.

    Returns:
        The tracking URI now in effect.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)
    if mlflow.get_tracking_uri() != tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        logger.info("MLflow tracking URI set to: %s", tracking_uri)
    return tracking_uri


def setup_mlflow() -> str:
    """Configure and verify the MLflow tracking server.

    The template intentionally defaults to a local MLflow server. Start it with
    ``mlflow server`` before running the CLI, or set ``MLFLOW_TRACKING_URI`` to
    a reachable tracking server.

    Returns:
        The active MLflow tracking URI.

    Raises:
        RuntimeError: If the configured MLflow server cannot be reached, or if
            it is reachable but rejects the first API call.
    """
    tracking_uri = configure_tracking_uri()

    if not tracking_uri.startswith(("http://", "https://")):
        return tracking_uri

    health_url = tracking_uri.rstrip("/") + "/health"
    request = Request(health_url, method="GET")
    try:
        with urlopen(request, timeout=3):
            pass
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(
            "MLflow tracking server is required but is not reachable at "
            f"{tracking_uri!r}. Start it with `mlflow server --host 127.0.0.1 "
            "--port 5000`, or set MLFLOW_TRACKING_URI to a reachable server. "
            "The template defaults to a server-backed tracking URI so runs, "
            "metrics, models, and artifacts are captured consistently."
        ) from exc

    # Deliberately outside the reachability check above. A server that answers
    # /health can still reject the API call — MLflow 3 returns 403 "Invalid Host
    # header" for any host it was not told to allow, which is what happens the
    # first time it is addressed by a container name. Reporting that as "not
    # reachable" sends you looking at networking instead of at the response.
    try:
        # Only meaningful before a run starts; set_experiment does not move an
        # already-active run, so skip it when one is in flight.
        if mlflow.active_run() is None:
            mlflow.set_experiment(experiment_name(read_config()))
    except MlflowException as exc:
        raise RuntimeError(
            f"MLflow answered at {tracking_uri!r} but rejected the request: "
            f"{exc}. The server is running — check what it returned rather than "
            "whether it is up."
        ) from exc

    return tracking_uri


def active_or_new_run(run_name: str) -> AbstractContextManager[Any]:
    """Return a context manager that reuses the active run, or starts one.

    Pipeline steps are normally called inside a run the CLI already opened, but
    each is also usable on its own. Reusing the active run keeps every step's
    metrics and artifacts on a single run rather than scattering them.

    Args:
        run_name: Name for the run, used only when starting a new one.

    Returns:
        A no-op context manager if a run is already active, otherwise a new run.
    """
    if mlflow.active_run():
        return nullcontext()
    return mlflow.start_run(run_name=run_name)


def registered_model_name(config: dict[str, Any]) -> str:
    """Return the MLflow Model Registry name for this project.

    Derived from ``[project].name`` in ``backend/pyproject.toml`` so that
    renaming the project renames the registered model too, leaving no template
    placeholder to forget. Set ``tracking.registered_model_name`` in
    ``cfg/config.yaml`` to override.

    Args:
        config: Parsed ``cfg/config.yaml`` contents.

    Returns:
        The registered model name.
    """
    configured = (config.get("tracking") or {}).get("registered_model_name")
    return str(configured) if configured else project_name()


def experiment_name(config: dict[str, Any]) -> str:
    """Return the MLflow experiment name, defaulting to the project name."""
    configured = (config.get("tracking") or {}).get("experiment_name")
    return str(configured) if configured else project_name()


def resolve_flavor(class_path: str) -> ModuleType:
    """Return the MLflow flavor module that can serialise ``class_path``.

    Args:
        class_path: Fully qualified estimator path from ``model_registry``,
            e.g. ``"xgboost.XGBClassifier"``.

    Returns:
        The imported MLflow flavor module, e.g. ``mlflow.xgboost``.
    """
    root_module = class_path.split(".", 1)[0]
    flavor_path = _FLAVOR_BY_ROOT_MODULE.get(root_module, _DEFAULT_FLAVOR)
    try:
        return importlib.import_module(flavor_path)
    except ImportError:
        logger.warning(
            "MLflow flavor %r is unavailable for %r; falling back to %r.",
            flavor_path,
            class_path,
            _DEFAULT_FLAVOR,
        )
        return importlib.import_module(_DEFAULT_FLAVOR)


def build_signature(features: Any, predictions: Any) -> ModelSignature:
    """Infer an MLflow model signature from training features and predictions.

    The signature is what lets consumers discover the feature names and types a
    prediction request must supply, so it is always logged rather than left
    optional.
    """
    return infer_signature(features, predictions)


def log_model(
    model: Any,
    class_path: str,
    config: dict[str, Any],
    signature: ModelSignature | None = None,
    input_example: Any = None,
    artifact_name: str = "model",
) -> str:
    """Log ``model`` under the correct MLflow flavor and register it.

    Args:
        model: The fitted estimator.
        class_path: Its ``model_registry`` import path, used to pick the flavor.
        config: Parsed ``cfg/config.yaml`` contents.
        signature: Model signature describing inputs and outputs.
        input_example: A small sample of representative input rows.
        artifact_name: Artifact subdirectory within the run.

    Returns:
        The ``runs:/<run_id>/<artifact_name>`` URI of the logged model.

    Raises:
        RuntimeError: If there is no active MLflow run to log into.
    """
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("log_model requires an active MLflow run.")

    flavor = resolve_flavor(class_path)
    model_name = registered_model_name(config)

    logger.info(
        "Logging model %r with flavor %r, registering as %r.",
        class_path,
        flavor.__name__,
        model_name,
    )

    try:
        flavor.log_model(
            model,
            name=artifact_name,
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
        )
    except RestException:
        # Registration needs a database-backed tracking server. Logging the
        # model artifact is the part that matters for reproducibility, so fall
        # back rather than failing the whole training run.
        logger.warning(
            "Could not register model %r; logging without registration. "
            "Model Registry requires a database-backed tracking server.",
            model_name,
        )
        flavor.log_model(
            model,
            name=artifact_name,
            signature=signature,
            input_example=input_example,
        )

    return f"runs:/{active_run.info.run_id}/{artifact_name}"
