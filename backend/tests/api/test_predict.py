"""Tests for feature discovery, prediction, and the no-model path."""

from unittest.mock import patch

from mlflow.exceptions import MlflowException

from src.ml.inference import LoadedModel, ModelNotAvailableError


def test_health_reports_a_loaded_model(client):
    """Health reflects the loaded model without requiring one."""
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["model_name"] == "test-model"
    assert body["model_version"] == "3"


def test_schema_lists_features_from_the_signature(client):
    """The prediction contract is discovered, not hardcoded."""
    response = client.get("/api/predict/schema")
    assert response.status_code == 200
    body = response.json()

    assert body["model_name"] == "test-model"
    assert [f["name"] for f in body["features"]] == ["FEATURE1", "FEATURE2"]
    assert all(f["kind"] == "number" for f in body["features"])
    assert all(f["required"] for f in body["features"])
    # Example values let a UI pre-fill a form with a row known to work.
    assert [f["example"] for f in body["features"]] == [1.5, 2.5]


def test_predict_returns_label_and_probabilities(client):
    """A valid request yields a prediction with class scores."""
    response = client.post(
        "/api/predict", json={"features": {"FEATURE1": 1.0, "FEATURE2": 2.0}}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probabilities"] == {"0": 0.25, "1": 0.75}
    assert body["model_version"] == "3"


def test_predict_rejects_missing_features(client):
    """A missing feature is a client error naming the offending field."""
    response = client.post("/api/predict", json={"features": {"FEATURE1": 1.0}})
    assert response.status_code == 422
    assert "FEATURE2" in response.json()["detail"]


def test_predict_rejects_unknown_features(client):
    """An unrecognised feature is reported rather than silently ignored."""
    response = client.post(
        "/api/predict",
        json={"features": {"FEATURE1": 1.0, "FEATURE2": 2.0, "NOPE": 3.0}},
    )
    assert response.status_code == 422
    assert "NOPE" in response.json()["detail"]


def test_predict_requires_a_features_object(client):
    """A malformed body is rejected by FastAPI before the handler runs."""
    response = client.post("/api/predict", json={"wrong_key": {}})
    assert response.status_code == 422


def test_predict_rejects_values_failing_schema_enforcement(client, model):
    """A value the model schema cannot coerce is a 422, never a 500.

    MLflow enforces the logged signature and raises MlflowException, whose
    message embeds the whole DataFrame and signature. That must become a concise
    client error rather than a traceback.
    """

    class _RejectingPyfunc:
        def predict(self, frame):
            raise MlflowException(
                "Failed to enforce schema of data '<huge frame>' with schema "
                "'<huge signature>'. Error: Failed to convert column FEATURE1 "
                "from type object to DataType.double."
            )

    model.pyfunc_model = _RejectingPyfunc()

    response = client.post(
        "/api/predict",
        json={"features": {"FEATURE1": "not-a-number", "FEATURE2": 2.0}},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "FEATURE1" in detail
    # The giant schema dump must not be echoed back to the caller.
    assert "huge signature" not in detail
    assert len(detail) < 400


def test_omits_probabilities_without_predict_proba(client, model):
    """Estimators lacking predict_proba still predict, reporting no scores."""
    model.estimator = None
    response = client.post(
        "/api/predict", json={"features": {"FEATURE1": 1.0, "FEATURE2": 2.0}}
    )
    assert response.status_code == 200
    assert response.json()["probabilities"] is None


def test_reload_picks_up_a_new_version(client, model):
    """Reload clears the cache so a newly trained version is served.

    Without this, a server that cached version 1 keeps serving it forever even
    after training version 2.
    """
    newer = LoadedModel(
        name="test-model",
        version="4",
        run_id="run-def",
        features=model.features,
        pyfunc_model=model.pyfunc_model,
        estimator=model.estimator,
    )
    with (
        patch("src.api.routers.predict.clear_model_cache") as mock_clear,
        patch("src.api.routers.predict.get_model", return_value=newer),
    ):
        response = client.post("/api/predict/reload")

    assert response.status_code == 200
    assert response.json()["model_version"] == "4"
    mock_clear.assert_called_once()


class TestWithoutATrainedModel:
    """A fresh checkout has no registered model; that is expected, not broken."""

    def test_health_still_serves(self, client_without_model):
        """The service reports itself up and the model absent."""
        with patch(
            "src.api.deps.get_cached_model",
            side_effect=ModelNotAvailableError("nothing registered"),
        ):
            response = client_without_model.get("/api/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_available"] is False
        assert body["model_name"] is None

    def test_predict_returns_503_with_instructions(self, client_without_model):
        """Prediction fails with 503 and tells the user how to fix it.

        The remedy comes from the domain exception; the API only adds that no
        restart is needed, since a failed load is not cached.
        """
        with patch(
            "src.api.deps.get_cached_model",
            side_effect=ModelNotAvailableError(
                "No versions registered for model 'demo'. Train one first — "
                "run `make pipeline`."
            ),
        ):
            response = client_without_model.post(
                "/api/predict", json={"features": {"FEATURE1": 1.0}}
            )

        assert response.status_code == 503
        detail = response.json()["detail"]
        assert "make pipeline" in detail
        assert "does not need restarting" in detail

    def test_schema_returns_503(self, client_without_model):
        """Feature discovery also reports 503 rather than an empty contract."""
        with patch(
            "src.api.deps.get_cached_model",
            side_effect=ModelNotAvailableError("No versions registered."),
        ):
            response = client_without_model.get("/api/predict/schema")

        assert response.status_code == 503
