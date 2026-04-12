# DS Template

A general-purpose template for data science and machine learning workflows.

## Installation

After creating a new repository using **"Use this template"**, follow these steps:

---

## 1. Rename the project

- Recursively replace:
  - `ds-template` → `your-project-name`
  - `ds_template` → `your_project_name`

---

## 2. Environment variables

- Copy the example file:

  ```bash
  cp .env.example .env
  ```

- Update values in `.env` for your local setup (MLflow URI, etc.)

---

## Prerequisites

### Install `uv`

This project uses [uv](https://docs.astral.sh/uv/) for Python version management, virtual environments, and dependency management.

On macOS (recommended):

```bash
brew install uv
```

---

## Development environment setup

This project uses **Python 3.12** and **uv** for all dependency management.

Run:

```bash
PYTHON_VERSION=3.12 ./setup.sh
```

This will:

- Create a `.venv` virtual environment using `uv`
- Install all project dependencies (including dev extras)
- Install pre-commit hooks

Activate the environment manually later with:

```bash
source .venv/bin/activate
```

---

## Package management

Dependencies are managed with **uv**.

To add a new dependency:

```bash
uv add <package-name>
```

For development-only dependencies:

```bash
uv add --optional dev <package-name>
```

---

## Running the code locally

### MLflow

Before running training, start an MLflow server:

```bash
mlflow server
```

By default, this will be available at:

```text
http://127.0.0.1:5000
```

Make sure `MLFLOW_TRACKING_URI` is set (either in `.env` or your shell).

---

### Running pipelines

The project exposes a CLI entrypoint:

```bash
uv run pipeline
```

Optional flags:

- `--prepare-data` — run only the data preparation step
- `--train-model` — run only the training step
- `--run-name <name>` — custom MLflow run name

Without flags, both steps run sequentially.

MLflow will track metrics, models, and plots for each run.

---

## Project structure

```text
src/
├── main.py                 # CLI entry point
├── pipelines/
│   ├── pipeline.py         # Abstract Pipeline base class
│   ├── prepare_data.py     # Data loading and transformations
│   └── train_model.py      # Model training with MLflow tracking
└── utils/
    ├── schemas.py           # Pandera data validation schemas
    └── utils.py             # CSV I/O and config helpers
cfg/
└── config.yaml             # Dataset, model and training configuration
data/
└── input_data.csv           # Sample breast cancer dataset
tests/                       # Pytest test suite
```

---

## Pre-commit checks

This repository uses **pre-commit** with **Ruff**, **MyPy**, **Bandit**, and **pydocstyle**.

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
uv run pytest
```

---

## Schema checks

- Schema definitions live in `src/utils/schemas.py`

When data is read or written via the utility functions:

- Column presence is validated
- Basic sanity checks are applied (e.g. value ranges)

This ensures data consistency across pipeline steps.
