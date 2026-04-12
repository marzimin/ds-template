# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    APP_HOME=/opt/app \
    SM_MODEL_DIR=/opt/ml/model \
    SM_OUTPUT_DATA_DIR=/opt/ml/output/data \
    SM_CHANNEL_TRAINING=/opt/ml/input/data/training

WORKDIR ${APP_HOME}

# System deps (build-essential helps on slim if anything needs compiling)
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry globally and tell it NOT to create venvs
RUN python -m pip install --no-cache-dir "poetry>=1.8" && \
    poetry config virtualenvs.create false

# Copy only dependency manifests first for caching
COPY pyproject.toml poetry.lock* ./

# Install dependencies only (no project yet)
RUN poetry install --only main --no-root --no-ansi

# Copy the rest of the code
COPY . .

# Install your project package (ds_template + src) now that code exists
RUN poetry install --only main --no-ansi

# Ensure imports work (optional, but handy)
ENV PYTHONPATH="${APP_HOME}:${APP_HOME}/src:${PYTHONPATH}"

CMD ["poetry", "run", "pipeline"]
