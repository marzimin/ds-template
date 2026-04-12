#!/bin/bash
set -euo pipefail

if [[ -z "${PYTHON_VERSION:-}" ]]; then
  echo "PYTHON_VERSION must be set."
  echo "Usage: PYTHON_VERSION=<python_version> ./setup.sh"
  exit 1
fi

VENV_DIR=".venv"

echo "Creating virtual environment (${VENV_DIR}) using Python ${PYTHON_VERSION}"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

echo "Installing project dependencies with uv"
uv sync --all-extras

echo "Installing pre-commit hooks"
uv run pre-commit install --allow-missing-config

echo "Done."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Run the project with:"
echo "  uv run pipeline"
