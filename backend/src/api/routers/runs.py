"""Reading MLflow runs, their metrics, and their artifacts.

The dashboard is run-scoped and reads from MLflow rather than the local
``outputs/`` directory, so it keeps working when the API runs in a separate
container from whatever produced the plots.
"""

import logging
import mimetypes
from pathlib import Path

import mlflow
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from mlflow.entities import Run, ViewType
from mlflow.exceptions import MlflowException

from src.api.contracts import ArtifactEntry, RunDetail, RunSummary
from src.api.deps import ClientDep, ConfigDep
from src.ml.tracking import experiment_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


def _experiment_ids(client: ClientDep, config: ConfigDep) -> list[str]:
    """Return the configured experiment's id, or empty if it does not exist."""
    name = experiment_name(config)
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        logger.info("Experiment %r does not exist yet.", name)
        return []
    return [experiment.experiment_id]


@router.get("", response_model=list[RunSummary], summary="List runs")
def list_runs(
    client: ClientDep,
    config: ConfigDep,
    limit: int = Query(default=50, ge=1, le=500),
) -> list[RunSummary]:
    """List runs for the configured experiment, newest first.

    Returns an empty list rather than an error when nothing has been logged, so
    a dashboard can render an empty state instead of handling a failure.
    """
    experiment_ids = _experiment_ids(client, config)
    if not experiment_ids:
        return []

    try:
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["attributes.start_time DESC"],
            max_results=limit,
        )
    except MlflowException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not read runs from MLflow: {exc}",
        ) from exc

    return [_to_summary(run) for run in runs]


@router.get("/{run_id}", response_model=RunDetail, summary="Get one run")
def get_run(run_id: str, client: ClientDep) -> RunDetail:
    """Return a single run with its metrics, parameters, and tags.

    Raises:
        HTTPException: 404 if the run does not exist.
    """
    run = _fetch_run(run_id, client)
    summary = _to_summary(run)
    return RunDetail(
        **summary.model_dump(),
        params=dict(run.data.params),
        tags=dict(run.data.tags),
    )


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a run",
)
def delete_run(run_id: str, client: ClientDep) -> None:
    """Delete one run from MLflow.

    This is a soft delete on MLflow's side (the run moves to its deleted-run
    view rather than disappearing outright), which is enough to drop it out of
    the dashboard's listing without touching artifact storage here.

    Raises:
        HTTPException: 404 if the run does not exist.
    """
    _fetch_run(run_id, client)
    try:
        client.delete_run(run_id)
    except MlflowException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not delete run {run_id!r}: {exc}",
        ) from exc


@router.get(
    "/{run_id}/artifacts",
    response_model=list[ArtifactEntry],
    summary="List a run's artifacts",
)
def list_artifacts(
    run_id: str,
    client: ClientDep,
    path: str = Query(default="", description="Subdirectory to list."),
) -> list[ArtifactEntry]:
    """List artifacts logged against a run, optionally within a subdirectory."""
    _fetch_run(run_id, client)
    try:
        entries = client.list_artifacts(run_id, path or None)
    except MlflowException as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not list artifacts for run {run_id!r}: {exc}",
        ) from exc

    return [
        ArtifactEntry(
            path=entry.path, is_dir=bool(entry.is_dir), file_size=entry.file_size
        )
        for entry in entries
    ]


@router.get(
    "/{run_id}/artifacts/file",
    summary="Download one artifact file",
    response_class=FileResponse,
)
def get_artifact_file(
    run_id: str,
    client: ClientDep,
    path: str = Query(description="Artifact path, exactly as listed."),
) -> FileResponse:
    """Serve a single artifact file, such as an EDA plot.

    ``path`` comes from the client, so it is never joined onto the filesystem
    directly. It is checked against the run's own artifact listing first, and
    the resolved file must still sit inside the download directory. Without both
    checks this endpoint would be a path-traversal hole.

    Raises:
        HTTPException: 404 if the run or artifact does not exist, 400 if the
            path escapes the artifact tree.
    """
    _fetch_run(run_id, client)

    if not _artifact_exists(client, run_id, path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {path!r} is not listed for run {run_id!r}.",
        )

    try:
        local_path = Path(
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path)
        ).resolve()
    except MlflowException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not download artifact {path!r}: {exc}",
        ) from exc

    if not local_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact {path!r} is not a file.",
        )

    media_type, _ = mimetypes.guess_type(local_path.name)
    return FileResponse(
        local_path,
        media_type=media_type or "application/octet-stream",
        filename=local_path.name,
    )


def _artifact_exists(client: ClientDep, run_id: str, path: str) -> bool:
    """Check ``path`` against the run's artifact listing.

    Listing the parent directory and matching exactly means only paths MLflow
    itself reports can ever be downloaded.
    """
    if not path or path.startswith("/") or ".." in Path(path).parts:
        return False
    parent = str(Path(path).parent)
    prefix = "" if parent == "." else parent
    try:
        entries = client.list_artifacts(run_id, prefix or None)
    except MlflowException:
        return False
    return any(entry.path == path and not entry.is_dir for entry in entries)


def _fetch_run(run_id: str, client: ClientDep) -> Run:
    """Return a run by id.

    Raises:
        HTTPException: 404 if it does not exist.
    """
    try:
        return client.get_run(run_id)
    except MlflowException as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run {run_id!r} was not found.",
        ) from exc


def _to_summary(run: Run) -> RunSummary:
    """Convert an MLflow run entity into a response model."""
    return RunSummary(
        run_id=run.info.run_id,
        run_name=run.info.run_name,
        status=run.info.status,
        start_time=run.info.start_time,
        end_time=run.info.end_time,
        metrics={key: float(value) for key, value in run.data.metrics.items()},
    )
