# syntax=docker/dockerfile:1.7
#
# Build from the REPOSITORY ROOT, not from backend/, because the image needs
# cfg/ as well as backend/:
#   docker build -f backend/Dockerfile -t <image-name> .
#
# One image, three jobs. The default command trains; override it to serve:
#   docker run --rm <image>                                    # pipeline
#   docker run --rm -p 8000:8000 <image> \
#       uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
#   docker run --rm -p 5000:5000 <image> \
#       uv run mlflow server --host 0.0.0.0 --port 5000
#
# docker-compose.yml at the repository root wires all three together.
FROM python:3.14-slim

# Stamped by .github/workflows/build.yml with the tag it just pushed, so a
# running container can be traced back to a commit. Without the ARG the
# workflow's --build-arg is silently discarded.
ARG VERSION=dev
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/marzimin/ds-template"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/opt/app

# Mirrors the repository layout: /opt/app/backend holds the code, /opt/app holds
# cfg/, data/ and outputs/. DS_PROJECT_ROOT pins the root explicitly so path
# resolution never depends on this file's depth on disk.
ENV DS_PROJECT_ROOT=${APP_HOME} \
    UV_CACHE_DIR=/opt/app/.uv-cache \
    # `uv run` re-syncs the project before every command, and the sync it
    # performs includes the dev group — so a container built with --no-dev would
    # reinstall mypy, ruff and jupyterlab on each start, undoing that and adding
    # seconds to every run. The environment is already complete; use it as-is.
    UV_NO_SYNC=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create the runtime user and its directories *before* anything large is
# written, then build everything as that user. Ownership has to be settled up
# front: a `chown -R` after the dependency install rewrites every file in the
# 1.8 GB virtualenv into a new layer and doubles the image.
#
# /mlflow is where docker-compose.yml mounts the tracking server's volume. It
# has to exist here, owned by `app`: Docker seeds a named volume's ownership
# from the image, and a mount point absent from the image is created root-owned,
# which the tracking server then cannot write to.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p ${APP_HOME}/backend ${APP_HOME}/outputs ${APP_HOME}/data/processed \
                ${UV_CACHE_DIR} /mlflow \
    && chown -R app:app ${APP_HOME} /mlflow

USER app
WORKDIR ${APP_HOME}/backend

# Copy only dependency manifests first for caching
COPY --chown=app:app backend/pyproject.toml backend/uv.lock* ./

# Install dependencies only (no project yet). --no-dev drops the `dev`
# dependency group declared in pyproject.toml: pytest, mypy, jupyterlab and the
# rest have no business in a runtime image.
#
# Do not add `uv cache prune` here. uv hardlinks the virtualenv to its cache, so
# pruning breaks the links and the next `uv run` rewrites real copies into a new
# layer: measured at 30 MB saved here against 263 MB added below.
RUN uv sync --no-dev --no-install-project

# Copy the backend code and the project-root configuration it reads
COPY --chown=app:app backend/ ./
COPY --chown=app:app cfg/ ${APP_HOME}/cfg/

# Install the project package
RUN uv sync --no-dev

# Generate the demo datasets into ${APP_HOME}/data/raw. At runtime, pass
# MLFLOW_TRACKING_URI for a reachable MLflow server, for example
# http://host.docker.internal:5000.
#
# The virtualenv's interpreter directly, not `uv run`: uv revalidates and
# reinstalls the project on every invocation, which rewrites files hardlinked
# into the layer below and forces overlayfs to copy up the whole virtualenv.
# Measured at 263 MB for writing three small CSVs.
RUN .venv/bin/python scripts/generate_sample_data.py

ENV PYTHONPATH="${APP_HOME}/backend:${PYTHONPATH}"

# Documents the API port for `docker run -P` and for readers; publishing is
# still up to the caller. MLflow's 5000 is deliberately not listed — this image
# serves the API by default and the tracking server only as an override.
EXPOSE 8000

CMD ["uv", "run", "pipeline"]
