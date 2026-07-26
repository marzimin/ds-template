"""Tests for run listing, run detail, and artifact serving."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException


def _run(run_id="run-1", name="demo", metrics=None, params=None, tags=None):
    """Build a stand-in for an mlflow Run entity."""
    return SimpleNamespace(
        info=SimpleNamespace(
            run_id=run_id,
            run_name=name,
            status="FINISHED",
            start_time=1000,
            end_time=2000,
        ),
        data=SimpleNamespace(
            metrics=metrics or {"test_accuracy": 0.95},
            params=params or {"model_name": "xgboost"},
            tags=tags or {"mlflow.source.name": "pipeline"},
        ),
    )


def _artifact(path, is_dir=False, size=128):
    """Build a stand-in for an mlflow FileInfo entity."""
    return SimpleNamespace(path=path, is_dir=is_dir, file_size=size)


def test_list_runs_returns_summaries(client, mlflow_client):
    """Runs are listed with whatever metrics were logged."""
    mlflow_client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="7"
    )
    mlflow_client.search_runs.return_value = [_run(), _run(run_id="run-2")]

    response = client.get("/api/runs")
    assert response.status_code == 200
    body = response.json()
    assert [r["run_id"] for r in body] == ["run-1", "run-2"]
    assert body[0]["metrics"] == {"test_accuracy": 0.95}


def test_list_runs_empty_when_experiment_absent(client, mlflow_client):
    """No experiment yet yields an empty list, not an error.

    A dashboard should render an empty state rather than handle a failure.
    """
    mlflow_client.get_experiment_by_name.return_value = None

    response = client.get("/api/runs")
    assert response.status_code == 200
    assert response.json() == []
    mlflow_client.search_runs.assert_not_called()


def test_list_runs_rejects_out_of_range_limit(client):
    """The limit is bounded so a client cannot request unbounded results."""
    assert client.get("/api/runs", params={"limit": 0}).status_code == 422
    assert client.get("/api/runs", params={"limit": 10_000}).status_code == 422


def test_get_run_includes_params_and_tags(client, mlflow_client):
    """Run detail carries parameters and tags alongside metrics."""
    mlflow_client.get_run.return_value = _run()

    response = client.get("/api/runs/run-1")
    assert response.status_code == 200
    body = response.json()
    assert body["params"] == {"model_name": "xgboost"}
    assert body["tags"] == {"mlflow.source.name": "pipeline"}


def test_get_run_404_when_absent(client, mlflow_client):
    """An unknown run id is a 404, not a 500."""
    mlflow_client.get_run.side_effect = MlflowException("no such run")

    response = client.get("/api/runs/nope")
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_list_artifacts(client, mlflow_client):
    """Artifacts are listed with their directory flag and size."""
    mlflow_client.get_run.return_value = _run()
    mlflow_client.list_artifacts.return_value = [
        _artifact("eda", is_dir=True),
        _artifact("eda/hist_feature1.png"),
    ]

    response = client.get("/api/runs/run-1/artifacts")
    assert response.status_code == 200
    body = response.json()
    assert body[0]["is_dir"] is True
    assert body[1]["path"] == "eda/hist_feature1.png"


def test_artifact_file_served_when_listed(client, mlflow_client, tmp_path):
    """A listed artifact downloads with a guessed media type."""
    mlflow_client.get_run.return_value = _run()
    mlflow_client.list_artifacts.return_value = [_artifact("eda/plot.png")]

    local_file = tmp_path / "plot.png"
    local_file.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    with patch(
        "src.api.routers.runs.mlflow.artifacts.download_artifacts",
        return_value=str(local_file),
    ):
        response = client.get(
            "/api/runs/run-1/artifacts/file", params={"path": "eda/plot.png"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == b"\x89PNG\r\n\x1a\nfake"


def test_artifact_file_404_when_not_listed(client, mlflow_client):
    """A path MLflow does not report cannot be downloaded."""
    mlflow_client.get_run.return_value = _run()
    mlflow_client.list_artifacts.return_value = [_artifact("eda/plot.png")]

    response = client.get(
        "/api/runs/run-1/artifacts/file", params={"path": "eda/other.png"}
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "malicious_path",
    [
        "../../../../etc/passwd",
        "eda/../../../etc/passwd",
        "/etc/passwd",
        "..",
    ],
)
def test_artifact_file_rejects_path_traversal(client, mlflow_client, malicious_path):
    """Traversal attempts are refused before any filesystem access.

    The artifact path is attacker-controlled, so it is validated against the
    run's own listing and never joined onto a local path directly.
    """
    mlflow_client.get_run.return_value = _run()
    mlflow_client.list_artifacts.return_value = [_artifact("eda/plot.png")]

    with patch(
        "src.api.routers.runs.mlflow.artifacts.download_artifacts"
    ) as mock_download:
        response = client.get(
            "/api/runs/run-1/artifacts/file", params={"path": malicious_path}
        )

    assert response.status_code == 404
    mock_download.assert_not_called()


def test_artifact_file_404_for_directory(client, mlflow_client):
    """A directory is not downloadable as a file."""
    mlflow_client.get_run.return_value = _run()
    mlflow_client.list_artifacts.return_value = [_artifact("eda", is_dir=True)]

    response = client.get("/api/runs/run-1/artifacts/file", params={"path": "eda"})
    assert response.status_code == 404


def test_list_runs_502_when_mlflow_fails(client, mlflow_client):
    """An MLflow outage is reported as an upstream failure, not our bug."""
    mlflow_client.get_experiment_by_name.return_value = SimpleNamespace(
        experiment_id="7"
    )
    mlflow_client.search_runs.side_effect = MlflowException("connection refused")

    response = client.get("/api/runs")
    assert response.status_code == 502


def test_openapi_schema_is_generated(client):
    """The OpenAPI schema exists: it is the source for frontend TS types."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "/api/predict" in schema["paths"]
    assert "PredictResponse" in schema["components"]["schemas"]


def test_cors_allows_the_dev_server_origin(client):
    """The Vite dev server origin is permitted, or the browser blocks calls."""
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_route_is_404(client):
    """Unrouted paths 404 rather than erroring."""
    assert client.get("/api/nonexistent").status_code == 404


def test_mlflow_client_dependency_is_a_mock(mlflow_client):
    """Guard that the fixture really is isolated from a live MLflow."""
    assert isinstance(mlflow_client, MagicMock)
