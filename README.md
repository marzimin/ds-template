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
- Generate the demo dataset into `data/raw/input_data.csv`

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
- `--eda` — run only the exploratory data analysis step
- `--train-model` — run only the training step
- `--run-name <name>` — custom MLflow run name

Without flags, all three steps run sequentially (prepare → EDA → train).

MLflow will track metrics, models, and plots for each run.

---

### Sample data

The template ships with the **Breast Cancer Wisconsin** dataset as a runnable
demo. `setup.sh` generates it automatically; you can regenerate it at any time
with:

```bash
uv run python scripts/generate_sample_data.py
```

To use your own data, drop a CSV into `data/raw/` named to match
`data.input_file` in `cfg/config.yaml`, then update the schemas in
`src/utils/schemas.py` to describe your columns.

---

## Project structure

```text
src/
├── main.py                 # CLI entry point
├── pipelines/
│   ├── pipeline.py         # Abstract Pipeline base class
│   ├── prepare_data.py     # Data loading and transformations
│   ├── eda.py              # Exploratory plots logged to MLflow
│   └── train_model.py      # Model training with MLflow tracking
└── utils/
    ├── schemas.py          # Pandera data validation schemas
    └── utils.py            # CSV I/O and config helpers
cfg/
└── config.yaml             # Dataset, model and training configuration
scripts/
└── generate_sample_data.py # Writes the demo dataset to data/raw/
data/
├── raw/                    # Raw input CSV (demo: Breast Cancer Wisconsin)
└── processed/              # Prepared and trained outputs
tests/                      # Pytest test suite
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

Schemas validate in **non-strict** mode: only the columns you declare are
checked, and any extra columns pass through. The template declares just a couple
of representative columns from the demo dataset — extend or replace them with
your own when you swap in your data.
