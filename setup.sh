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

echo "Installing Poetry (>=1.8) into the venv"
uv pip install --python "${VENV_DIR}/bin/python" "poetry>=1.8"

echo "Configuring Poetry to use the venv Python"
"${VENV_DIR}/bin/poetry" env use "${VENV_DIR}/bin/python"

echo "Installing project dependencies with Poetry"
"${VENV_DIR}/bin/poetry" install

echo "Installing pre-commit hooks"
"${VENV_DIR}/bin/poetry" run pre-commit install --allow-missing-config

echo "Done."
echo "Activate the environment with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Run the project with:"
echo "  poetry run pipeline"
