# DS Template

## Installation

After creating a new repository using **“Use this template”**, follow these steps:

---

## 0. Repository naming convention

All data science repositories should use the `data-` prefix.

- Format: `data-project-name`
- Example: `data-oa-classifier`

This keeps naming consistent and makes data team repos easily identifiable within the organisation.

---

## 1. Repository settings (do this first)

### Set the repository to Private

- Go to **Settings → General → Danger Zone**
- Ensure the repository visibility is set to **Private**

### Protect the `main` branch

- Go to **Settings → Branches → Add branch protection rule**
- Target branch: `main`
- Enable:
  - Require a pull request before merging
  - Require approvals
  - Required number of approvals: **1** (minimum)

This ensures:
- No direct pushes to `main`
- All changes go through a Pull Request
- At least one approval is required before merging

---

## 2. Rename the project

- Recursively replace:
  - `ds-template` → `project-name`
  - `ds_template` → `project_name`
- Rename the `ds_template/` directory to `project_name/`

---

## 3. Environment variables

- Copy the example file:
  ```bash
  cp .env.example .env
  ```
- Update values in `.env` for your local setup (paths, MLflow URI, repo name (= project name), etc.)

---

## Prerequisites

### Install Terraform

This project uses [Terraform](https://www.terraform.io/) for AWS infrastructure automation. You must install Terraform (>= 1.5.0) before running any deployment commands.

On macOS (recommended):

```bash
brew install terraform
```

---

### Install `uv`

This project uses **uv** to create and manage the Python virtual environment.

On macOS (recommended):

```bash
brew install uv
```

---

## Development environment setup (MacOS / Linux)

This project uses **Python 3.12**, **uv** for virtual environments, and **Poetry** for dependency management.

Run:

```bash
PYTHON_VERSION=3.12 ./setup.sh
```

This will:
- Create a `.venv` virtual environment using `uv`
- Install Poetry into the environment
- Install project dependencies
- Install pre-commit hooks
- Install the project as an editable package

Activate the environment manually later with:

```bash
source .venv/bin/activate
```

---

## Package management

Dependencies are managed with **Poetry**.

To add a new dependency:

```bash
poetry add <package-name>
```

For development-only dependencies:

```bash
poetry add --group dev <package-name>
```

Do **not** use `pip install` directly.

---

## Running the code locally

### MLflow

Before running training, start an MLflow server:

```bash
mlflow server
```

By default, this will be available at:

```
http://127.0.0.1:5000
```

Make sure `LOCAL_MLFLOW_TRACKING_URI` is set (either in `.env` or your shell).

---

### Running pipelines

The project exposes a CLI entrypoint.

Examples:

```bash
poetry run pipeline
```

Optional flags (depending on your implementation):
- `--prepare-data`
- `--train-model`
- `--run-name <name>`

MLflow will track metrics, models, and plots for each run.

---

## Pre-commit checks

This repository uses **pre-commit** with **Ruff** for formatting and linting.

### Install hooks
```bash
pre-commit install
```

### Run all checks manually
```bash
pre-commit run --all-files
```

---

## Testing

Tests live in the `tests/` directory.

Run all tests with:

```bash
poetry run pytest
```

Please add tests alongside each pipeline or major logic change.

---

## Schema checks

- File paths and types are defined in `cfg/files.yaml`
- Schema definitions live in `src/utils/schemas.py`

When data is read or written via the utility functions:
- Column presence is validated
- Basic sanity checks are applied (e.g. value ranges)

This ensures data consistency across pipeline steps.
