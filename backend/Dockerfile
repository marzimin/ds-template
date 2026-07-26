# syntax=docker/dockerfile:1.7
#
# Build from the REPOSITORY ROOT, not from backend/, because the image needs
# cfg/ as well as backend/:
#   docker build -f backend/Dockerfile -t <image-name> .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/opt/app

# Mirrors the repository layout: /opt/app/backend holds the code, /opt/app holds
# cfg/, data/ and outputs/. DS_PROJECT_ROOT pins the root explicitly so path
# resolution never depends on this file's depth on disk.
ENV DS_PROJECT_ROOT=${APP_HOME}

WORKDIR ${APP_HOME}/backend

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only dependency manifests first for caching
COPY backend/pyproject.toml backend/uv.lock* ./

# Install dependencies only (no project yet)
RUN uv sync --no-dev --no-install-project

# Copy the backend code and the project-root configuration it reads
COPY backend/ ./
COPY cfg/ ${APP_HOME}/cfg/

# Install the project package
RUN uv sync --no-dev

# Generate the demo datasets into ${APP_HOME}/data/raw. At runtime, pass
# MLFLOW_TRACKING_URI for a reachable MLflow server, for example
# http://host.docker.internal:5000.
RUN uv run python scripts/generate_sample_data.py

ENV PYTHONPATH="${APP_HOME}/backend:${PYTHONPATH}"

CMD ["uv", "run", "pipeline"]
