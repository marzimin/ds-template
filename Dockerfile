# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/opt/app

WORKDIR ${APP_HOME}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      curl \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only dependency manifests first for caching
COPY pyproject.toml uv.lock* ./

# Install dependencies only (no project yet)
RUN uv sync --no-dev --no-install-project

# Copy the rest of the code
COPY . .

# Install the project package
RUN uv sync --no-dev

# Generate the demo dataset so the image runs out of the box.
# Remove this once you mount or bake in your own data.
RUN uv run python scripts/generate_sample_data.py

ENV PYTHONPATH="${APP_HOME}:${PYTHONPATH}"

CMD ["uv", "run", "pipeline"]
