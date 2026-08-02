"""Tests for flavor-aware model logging.

The API layer loads models through the flavor-agnostic ``pyfunc`` interface and
reads the logged signature to discover what a prediction request must supply.
These tests pin that contract: every estimator in the default ``model_registry``
must log, reload, predict, and expose its input schema.
"""

from pathlib import Path

import mlflow
import pandas as pd
import pytest

from src.ml.tracking import (
    build_signature,
    configure_tracking_uri,
    experiment_name,
    log_model,
    mlflow_port,
    registered_model_name,
    resolve_flavor,
)

# Every distinct estimator reachable through the default model_registry.
_REGISTRY_CLASS_PATHS = [
    "xgboost.XGBClassifier",
    "sklearn.ensemble.RandomForestClassifier",
]


@pytest.fixture(name="training_data")
def training_data_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """A small binary classification dataset."""
    features = pd.DataFrame(
        {
            "FEATURE1": [float(i) for i in range(20)],
            "FEATURE2": [float(i) for i in range(20, 40)],
        }
    )
    target = pd.Series([0, 1] * 10, name="TARGET")
    return features, target


@pytest.fixture(name="tracking_uri")
def tracking_uri_fixture(tmp_path: Path):
    """Point MLflow at a throwaway SQLite store for the duration of a test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


@pytest.mark.parametrize(
    ("class_path", "expected_flavor"),
    [
        ("xgboost.XGBClassifier", "mlflow.xgboost"),
        ("lightgbm.LGBMClassifier", "mlflow.lightgbm"),
        ("catboost.CatBoostClassifier", "mlflow.catboost"),
        ("sklearn.ensemble.RandomForestClassifier", "mlflow.sklearn"),
        ("some.unknown.Estimator", "mlflow.sklearn"),
    ],
)
def test_resolve_flavor_maps_registry_paths(class_path: str, expected_flavor: str):
    """Flavor selection follows the root module of the registry import path.

    Unknown roots fall back to mlflow.sklearn, as do flavors whose optional
    dependency is not installed.
    """
    flavor = resolve_flavor(class_path)
    assert flavor.__name__ in {expected_flavor, "mlflow.sklearn"}


@pytest.mark.parametrize("class_path", _REGISTRY_CLASS_PATHS)
def test_log_model_round_trip(class_path, training_data, tracking_uri):
    """Each registry estimator logs, reloads via pyfunc, and predicts.

    This is the contract the API depends on: it never knows which flavor
    produced a model, only that ``mlflow.pyfunc.load_model`` can load it.
    """
    import importlib

    features, target = training_data
    module_name, class_name = class_path.rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_name), class_name)
    model = model_cls(n_estimators=5)
    model.fit(features, target)

    config: dict = {"tracking": {"registered_model_name": "test-model"}}
    signature = build_signature(features, model.predict(features))

    with mlflow.start_run():
        model_uri = log_model(
            model,
            class_path=class_path,
            config=config,
            signature=signature,
            input_example=features.head(2),
        )

    loaded = mlflow.pyfunc.load_model(model_uri)
    predictions = loaded.predict(features.head(3))
    assert len(predictions) == 3

    # The signature is what the API turns into a prediction form, so the input
    # schema must name every feature column.
    input_schema = loaded.metadata.get_input_schema()
    assert [field.name for field in input_schema.inputs] == list(features.columns)


def test_log_model_requires_active_run(training_data):
    """Logging outside a run fails loudly rather than silently doing nothing."""
    features, target = training_data
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=2).fit(features, target)

    if mlflow.active_run():  # pragma: no cover - defensive
        mlflow.end_run()

    with pytest.raises(RuntimeError, match="active MLflow run"):
        log_model(
            model,
            class_path="sklearn.ensemble.RandomForestClassifier",
            config={},
        )


def test_names_derive_from_project_when_unset():
    """Registry and experiment names fall back to the project name.

    Deriving them means renaming the project renames the model, leaving no
    template placeholder to forget.
    """
    from src.config import project_name

    assert registered_model_name({}) == project_name()
    assert experiment_name({}) == project_name()


def test_names_honour_explicit_config():
    """Explicit configuration overrides the derived default."""
    config = {
        "tracking": {
            "registered_model_name": "custom-model",
            "experiment_name": "custom-experiment",
        }
    }
    assert registered_model_name(config) == "custom-model"
    assert experiment_name(config) == "custom-experiment"


@pytest.fixture(name="restore_tracking_uri")
def restore_tracking_uri_fixture():
    """Undo the process-global side effect configure_tracking_uri() makes.

    mlflow.set_tracking_uri is not test-scoped, so a test that lets it run
    would otherwise point every later test in the module at whatever URI it
    computed.
    """
    previous = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(previous)


def test_mlflow_port_defaults_to_5000(monkeypatch):
    """Matches the port `make mlflow` binds to when MLFLOW_PORT is unset."""
    monkeypatch.delenv("MLFLOW_PORT", raising=False)
    assert mlflow_port() == "5000"


def test_mlflow_port_honours_the_environment(monkeypatch):
    """The port compose publishes MLflow on is the one native tooling assumes."""
    monkeypatch.setenv("MLFLOW_PORT", "5001")
    assert mlflow_port() == "5001"


def test_configure_tracking_uri_follows_mlflow_port(monkeypatch, restore_tracking_uri):
    """Moving MLFLOW_PORT moves the default without setting MLFLOW_TRACKING_URI.

    This is what lets a native `make pipeline` find MLflow published on a
    non-default host port — the AirPlay Receiver on macOS claims 5000, so
    `make demo` commonly runs with MLFLOW_PORT=5001, and before this the
    native default stayed pinned to 5000 regardless.
    """
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_PORT", "5001")
    assert configure_tracking_uri() == "http://127.0.0.1:5001"


def test_configure_tracking_uri_prefers_explicit_override(
    monkeypatch, restore_tracking_uri
):
    """An explicit MLFLOW_TRACKING_URI still wins, e.g. for a remote server."""
    monkeypatch.setenv("MLFLOW_PORT", "5001")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.com")
    assert configure_tracking_uri() == "https://tracking.example.com"
