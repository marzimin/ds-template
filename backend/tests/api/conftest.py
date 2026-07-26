"""Fixtures for API tests.

The API is tested through FastAPI's TestClient with the model and MLflow client
dependencies overridden, so these tests need no running MLflow server. The
round-trip against real MLflow is covered by tests/test_tracking.py.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import deps
from src.api.app import create_app
from src.ml.inference import FeatureSpec, LoadedModel, ModelNotAvailableError

_FEATURES = (
    FeatureSpec(
        name="FEATURE1",
        mlflow_type="double",
        kind="number",
        required=True,
        example=1.5,
    ),
    FeatureSpec(
        name="FEATURE2",
        mlflow_type="double",
        kind="number",
        required=True,
        example=2.5,
    ),
)


class _StubPyfunc:
    """Stands in for an MLflow PyFuncModel."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return one label per row."""
        return np.array([1] * len(frame))


class _StubEstimator:
    """Stands in for the underlying estimator, exposing predict_proba."""

    classes_ = np.array([0, 1])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return fixed class scores per row."""
        return np.array([[0.25, 0.75]] * len(frame))


@pytest.fixture(name="model")
def model_fixture() -> LoadedModel:
    """A loaded model with two numeric features and probability support."""
    return LoadedModel(
        name="test-model",
        version="3",
        run_id="run-abc",
        features=_FEATURES,
        pyfunc_model=_StubPyfunc(),
        estimator=_StubEstimator(),
    )


@pytest.fixture(name="mlflow_client")
def mlflow_client_fixture() -> MagicMock:
    """A mock MlflowClient that route handlers read runs from."""
    return MagicMock()


@pytest.fixture(name="config")
def config_fixture() -> dict[str, Any]:
    """Minimal configuration with a fixed experiment name."""
    return {"tracking": {"experiment_name": "test-experiment"}}


@pytest.fixture(name="quiet_startup", autouse=True)
def quiet_startup_fixture():
    """Stop the startup hook from reaching a real MLflow server.

    Dependency overrides do not apply to the lifespan hook, so without this the
    app would try to load a model from whatever tracking store happens to be
    configured on the machine running the tests.
    """
    with patch(
        "src.api.app.get_cached_model",
        side_effect=ModelNotAvailableError("suppressed during tests"),
    ):
        yield


@pytest.fixture(name="client")
def client_fixture(model, mlflow_client, config):
    """A TestClient with the model, client, and config dependencies supplied."""
    app = create_app()
    app.dependency_overrides[deps.get_model] = lambda: model
    app.dependency_overrides[deps.get_optional_model] = lambda: model
    app.dependency_overrides[deps.get_mlflow_client] = lambda: mlflow_client
    app.dependency_overrides[deps.get_config] = lambda: config
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(name="client_without_model")
def client_without_model_fixture(mlflow_client, config):
    """A TestClient where no model has been trained yet.

    Only the config and MLflow client are overridden; the model dependencies run
    for real so the 503 and model_available=False paths are genuinely exercised.
    """
    app = create_app()
    app.dependency_overrides[deps.get_mlflow_client] = lambda: mlflow_client
    app.dependency_overrides[deps.get_config] = lambda: config
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
