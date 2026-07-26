#!/bin/bash
set -euo pipefail

if [[ -z "${PYTHON_VERSION:-}" ]]; then
  echo "PYTHON_VERSION must be set."
  echo "Usage: PYTHON_VERSION=<python_version> ./backend/setup.sh"
  exit 1
fi

# Always operate on the backend project, regardless of where this is invoked
# from. The virtual environment and lockfile belong to backend/; the .env and
# data/ directories it seeds belong to the repository root above it.
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${BACKEND_DIR}")"
cd "${BACKEND_DIR}"

VENV_DIR=".venv"

echo "Creating virtual environment (backend/${VENV_DIR}) using Python ${PYTHON_VERSION}"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

echo "Installing project dependencies with uv"
uv sync --all-extras

echo "Installing pre-commit hooks"
(cd "${PROJECT_ROOT}" && "${BACKEND_DIR}/${VENV_DIR}/bin/pre-commit" install --allow-missing-config)

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Creating .env from .env.example"
  cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
fi

echo "Generating the demo dataset (Breast Cancer Wisconsin)"
uv run python scripts/generate_sample_data.py

echo "Done."
echo "Activate the environment with:"
echo "  source backend/${VENV_DIR}/bin/activate"
echo "Run the pipeline with:"
echo "  cd backend && uv run pipeline"
