# DS Template

A general-purpose template for data science and machine learning workflows.

The repository is split into a **backend** (Python: data pipelines, ML, and the
API that serves them) and, in due course, a **frontend** (TypeScript/React) that
displays model outputs. The directories you actually configure — `cfg/`,
`data/`, and `.env` — sit at the repository root, above both.

```text
cfg/config.yaml   ← configure your dataset, model, and training here
data/             ← drop your data here
outputs/          ← generated plots and reports
.env              ← shared environment for backend, Docker, and compose
backend/          ← Python code (pipelines, ML, API)
```

New to web application structure? [`docs/architecture.md`](docs/architecture.md)
explains how the terminal workflow and the browser interface relate, and
introduces the backend/frontend concepts in data science terms.

## Installation

After creating a new repository using **"Use this template"**, follow these steps:

---

## 1. Rename the project

- Recursively replace:
  - `ds-template` → `your-project-name`
  - `ds_template` → `your_project_name`
- The starter package is currently imported as `src` (that is, `backend/src`).
  If you want a project-specific package name, rename that package and update
  imports plus the `pipeline` entry point in `backend/pyproject.toml`.

---

## 2. Environment variables

- Copy the example file **at the repository root**:

  ```bash
  cp .env.example .env
  ```

  (`./backend/setup.sh` does this for you if `.env` does not already exist.)

- Update values in `.env` for your local setup (MLflow URI, etc.)
- The default `MLFLOW_TRACKING_URI` expects a running MLflow server at
  `http://127.0.0.1:5000`.
- Paths in `.env` and `cfg/config.yaml` resolve relative to the repository
  root. Set `DS_PROJECT_ROOT` to override that root explicitly; the container
  image does exactly this.

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

Run, from the repository root:

```bash
PYTHON_VERSION=3.12 ./backend/setup.sh
```

This will:

- Create a `backend/.venv` virtual environment using `uv`
- Install all project dependencies (including dev extras)
- Install pre-commit hooks
- Create `.env` from `.env.example` if it does not already exist
- Generate the demo dataset into `data/raw/input_data.csv`

Activate the environment manually later with:

```bash
source backend/.venv/bin/activate
```

### Make shortcuts

A `Makefile` at the repository root wraps the common tasks so you do not have to
change directory:

```bash
make setup        # everything above
make test         # backend test suite
make lint         # all pre-commit hooks
make mlflow       # start a local tracking server
make pipeline     # prepare -> EDA -> train
make api          # start the FastAPI server (docs at /docs)
make hooks        # reinstall the git pre-commit hook
make help         # list every target
```

### Editor setup

`.vscode/settings.json` is tracked and points VS Code at
`backend/.venv/bin/python`. Editors cannot guess this from the repository root,
and without it every dependency is reported as missing. If you use a different
editor, point its interpreter at that path.

If a commit ever fails with `pre-commit not found`, the installed git hook is
referencing a virtual environment that no longer exists. Run `make hooks`.

---

## Package management

Dependencies are managed with **uv**, from the `backend/` directory (that is
where `pyproject.toml` and `uv.lock` live).

To add a new dependency:

```bash
cd backend && uv add <package-name>
```

For development-only dependencies:

```bash
cd backend && uv add --optional dev <package-name>
```

---

## Running the code locally

All `uv` commands below are run from the `backend/` directory.

### MLflow

Before running any pipeline through the CLI, start an MLflow server:

```bash
cd backend && uv run mlflow server --host 127.0.0.1 --port 5000
```

> On macOS, port 5000 is often taken by the AirPlay Receiver. If the health
> check fails with a 403, either disable AirPlay Receiver in System Settings or
> pick another port and set `MLFLOW_TRACKING_URI` to match.

By default, this will be available at:

```text
http://127.0.0.1:5000
```

Make sure `MLFLOW_TRACKING_URI` is set (either in `.env` or your shell).
If the server is not reachable, `uv run pipeline` fails before running pipeline
steps and prints the configured tracking URI plus the command above.

---

### Running pipelines

The project exposes a CLI entrypoint:

```bash
cd backend && uv run pipeline
```

Optional flags:

- `--prepare-data` — run only the data preparation step
- `--eda` — run only the exploratory data analysis step
- `--train-model` — run only the training step
- `--run-name <name>` — custom MLflow run name

Use at most one step flag at a time. Run without step flags to execute all
steps sequentially.

Without flags, all three steps run sequentially (prepare → EDA → train).

MLflow will track metrics, models, and plots for each run.

---

### Serving the model over HTTP

The pipeline is one way in; the API is the other. It reads the latest registered
model and exposes it over HTTP so a browser — or any other client — can use it.

```bash
make mlflow     # terminal 1
make api        # terminal 2
```

Then open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for
interactive documentation with a form to call every endpoint. This is the
quickest way to exercise the backend, and it works before any frontend exists.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Is the service up, and is a model loaded? |
| `GET /api/predict/schema` | The features a prediction needs, from the model signature |
| `POST /api/predict` | Predict for one record |
| `POST /api/predict/reload` | Pick up a newly trained version without restarting |
| `GET /api/runs` | List runs with their metrics |
| `GET /api/runs/{id}` | One run with parameters and tags |
| `GET /api/runs/{id}/artifacts` | List a run's artifacts |
| `GET /api/runs/{id}/artifacts/file` | Download one artifact, such as an EDA plot |

**Before a model has been trained**, the API still starts. `/api/health` reports
`model_available: false` and prediction endpoints return `503` explaining how to
fix it. Train with `make pipeline` and the next request picks the model up — no
restart needed.

**After retraining while the API is running**, call `POST /api/predict/reload`.
A server holds one model in memory for the life of the process, so without this
it would keep serving the previous version.

No feature name is hardcoded anywhere in the API. `GET /api/predict/schema`
reports what the current model expects, including representative values from the
logged input example, which is what lets a frontend build its form at runtime
and keep working when you swap datasets.

---

### Choosing a model

Models are declared in `cfg/config.yaml`, not in code. `model_registry` maps a
short name to a fully qualified class path, which is imported dynamically at
training time:

```yaml
model_registry:
  xgboost: "xgboost.XGBClassifier"
  random_forest: "sklearn.ensemble.RandomForestClassifier"
  lightgbm: "lightgbm.LGBMClassifier"   # add your own

model_name: "xgboost"                   # pick one from the registry
model_params:
  n_estimators: 50
  max_depth: 10
```

To use a different estimator, add a line to `model_registry` and point
`model_name` at it — no Python changes needed. Any class implementing the
scikit-learn `fit`/`predict` API works.

The correct MLflow flavor is selected automatically from the root module of the
import path (`xgboost.*` → `mlflow.xgboost`, `lightgbm.*` → `mlflow.lightgbm`,
and so on, defaulting to `mlflow.sklearn`). This matters because
`mlflow.sklearn` cannot serialise non-sklearn estimators such as XGBoost.

Every model is logged with a **signature** and registered in the MLflow Model
Registry, so it can be loaded without knowing which flavor produced it:

```python
import mlflow
model = mlflow.pyfunc.load_model("models:/<project-name>/latest")
model.metadata.get_input_schema()   # the features a prediction needs
```

The experiment and registered model names default to `[project].name` in
`backend/pyproject.toml`, so renaming the project renames them too. Override
either via the `tracking` block in `cfg/config.yaml`.

Note that the default pipeline assumes **binary classification** throughout its
metrics, plots, and schemas. Swapping estimators is a config change; moving to
regression or multiclass means editing `TrainModelPipeline` and `schemas.py`.

---

### Sample data

The template ships with the **Breast Cancer Wisconsin** dataset as a runnable
demo. `backend/setup.sh` generates it automatically; you can regenerate it at
any time with:

```bash
cd backend && uv run python scripts/generate_sample_data.py
```

To use your own data, drop a CSV into `data/raw/` (at the repository root) named
to match `data.input_file` in `cfg/config.yaml`, then update the schemas in
`backend/src/schemas.py` to describe your columns.

The starter training pipeline intentionally assumes:

- A binary classification target with values listed in `target_values`
- A target column named by `target_column`; names are normalised on read, so
  `target`, `Target`, and `TARGET` all become `TARGET`
- Numeric feature columns only
- No missing feature or target values by training time

If your project has categoricals, datetimes, nulls, multiclass labels, or a
regression target, update `PrepareDataPipeline`, `TrainModelPipeline`, and the
schemas before training. The template raises explicit errors for these cases so
you do not have to interpret lower-level scikit-learn tracebacks.

---

### Docker

The image needs both `backend/` and the root `cfg/`, so it is **built from the
repository root** with an explicit Dockerfile path:

```bash
docker build -f backend/Dockerfile -t <image-name> .
```

Point the container at a reachable MLflow server when running it:

```bash
docker run --rm \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  <image-name>
```

Inside the image, the layout mirrors the repository: code at `/opt/app/backend`,
configuration and data at `/opt/app`, with `DS_PROJECT_ROOT=/opt/app`.

---

## Project structure

```text
cfg/
└── config.yaml             # Dataset, model and training configuration
data/
├── raw/                    # Raw input CSV (demo: Breast Cancer Wisconsin)
└── processed/              # Prepared and trained outputs
outputs/                    # EDA plots, training plots, reports
.env                        # Shared by backend, Docker, and compose
backend/
├── pyproject.toml          # Python dependencies and tooling config
├── setup.sh                # Development environment bootstrap
├── Dockerfile              # Built from the repository root
├── scripts/
│   └── generate_sample_data.py  # Writes the demo dataset to data/raw/
├── notebooks/              # Exploratory notebooks
├── tests/                  # Pytest test suite
└── src/
    ├── main.py             # CLI entry point
    ├── config.py           # Paths, .env, and cfg/config.yaml access
    ├── schemas.py          # Pandera data validation schemas
    ├── ml/
    │   ├── pipeline.py     # Abstract Pipeline base class
    │   ├── prepare_data.py # Data loading and transformations
    │   ├── eda.py          # Exploratory plots logged to MLflow
    │   ├── train_model.py  # Model training with MLflow tracking
    │   ├── inference.py    # Loading a registered model and predicting
    │   ├── io.py           # CSV read/write with schema validation
    │   ├── plots.py        # Matplotlib/seaborn plotting helpers
    │   └── tracking.py     # MLflow setup and flavor-aware model logging
    └── api/
        ├── app.py          # FastAPI application factory
        ├── deps.py         # Shared dependencies (model, config, MLflow client)
        ├── models.py       # Request/response contracts (Pydantic)
        └── routers/        # health, predict, runs
```

`config.py` and `schemas.py` sit at the top of the package because they are the
two modules you are most likely to edit, and because they are deliberately light
— neither imports matplotlib, seaborn, or mlflow. That keeps them cheap for
consumers (such as an API layer) that need configuration and data contracts but
not the modelling and plotting stack.

`cfg/`, `data/`, and `outputs/` deliberately sit at the repository root rather
than inside `backend/`: they are the part of the template you configure and the
part the frontend will eventually read, so they stay above the code that
consumes them. Paths inside them resolve against the repository root, which is
inferred from the backend package location or set via `DS_PROJECT_ROOT`.

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

Tests live in the `backend/tests/` directory.

Run all tests with:

```bash
cd backend && uv run pytest
```

---

## Schema checks

- Schema definitions live in `backend/src/schemas.py`

When data is read or written via the utility functions:

- Column presence is validated
- Basic sanity checks are applied (e.g. value ranges)

This ensures data consistency across pipeline steps.

Schemas validate in **non-strict** mode: only the columns you declare are
checked, and any extra columns pass through. The template declares just a couple
of representative columns from the demo dataset — extend or replace them with
your own when you swap in your data.
