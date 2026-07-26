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
frontend/         ← TypeScript/React app that displays model outputs
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
- Generate the demo dataset into `data/raw/breast_cancer.csv`

Activate the environment manually later with:

```bash
source backend/.venv/bin/activate
```

### Make shortcuts

A `Makefile` at the repository root wraps the common tasks so you do not have to
change directory:

```bash
make setup        # install both halves, hooks, and the demo dataset
make test         # every test suite (backend + frontend)
make lint         # lint everything
make pipeline     # prepare -> EDA -> train
make mlflow       # start a local tracking server        (terminal 1)
make api          # start the FastAPI server             (terminal 2)
make web          # start the frontend dev server        (terminal 3)
make types        # regenerate frontend types from the API
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

## The web application

The full app is three processes, one per terminal:

```bash
make mlflow     # terminal 1 — tracking server and model registry
make api        # terminal 2 — loads the model, answers requests
make web        # terminal 3 — serves the page
```

Then open **[http://localhost:5173](http://localhost:5173)**.

> Use `localhost`, not `127.0.0.1`. Vite binds the hostname `localhost`, which
> often resolves to IPv6 `::1`, so `127.0.0.1:5173` can refuse the connection.

Three screens:

| Screen | What it does |
| --- | --- |
| **Predict** | A form generated from the model's signature, pre-filled with a real row from the logged input example. Submit for a prediction and class probabilities. |
| **Runs** | Every training run with its metrics. Columns are derived from what was logged. |
| **Run detail** | Metrics, parameters, and a gallery of that run's EDA and evaluation plots. |

The header shows whether the API is reachable and which model version is loaded,
with a **reload** button that picks up a newly trained version without
restarting anything.

### Keeping the frontend dataset-agnostic

New to frontend work? [`docs/architecture.md`](docs/architecture.md) explains
React, components, and state in data science terms. The four rules that keep
this template reusable:

1. **No feature name appears in frontend code.** The form is built from
   `GET /api/predict/schema` at runtime. Retrain on different data and the form
   changes by itself.
2. **Table columns come from the data.** The runs dashboard shows the union of
   metric keys it actually receives, so a regression pipeline logging
   `test_rmse` displays with no code change.
3. **Types are generated, never hand-written.** `make types` regenerates
   `frontend/src/api/schema.d.ts` from the API's OpenAPI document. Rename a
   Pydantic field and the frontend fails to compile at exactly the line that
   needs updating.
4. **"No model yet" is a normal state, not an error.** A fresh clone shows
   guidance and the command to run, not a red failure.

Both generated files (`openapi.json` and `schema.d.ts`) are committed so a fresh
clone builds without a running backend. Re-run `make types` after changing any
request or response shape.

### Restyling

`frontend/src/styles.css` is plain CSS with custom properties at the top. Change
the variables in `:root` to restyle the whole app; light and dark are both
handled. There is no CSS framework to learn.

---

### Choosing a model

Models are declared in `cfg/config.yaml`, not in code. `model_registry` maps a
short name to a fully qualified class path, which is imported dynamically at
training time:

```yaml
model_registry:
  # Classifiers
  xgboost: "xgboost.XGBClassifier"
  random_forest: "sklearn.ensemble.RandomForestClassifier"
  lightgbm: "lightgbm.LGBMClassifier"        # add your own
  # Regressors
  rf_regressor: "sklearn.ensemble.RandomForestRegressor"
  linear_regression: "sklearn.linear_model.LinearRegression"

model_name: "xgboost"                        # pick one from the registry
model_params:
  xgboost:                                   # nested per model, see below
    n_estimators: 50
    max_depth: 10
```

Use a **classifier** for class labels and a **regressor** for continuous
targets. Pairing them the wrong way round is caught before training with an
error naming both.

To use a different estimator, add a line to `model_registry` and point
`model_name` at it — no Python changes needed. Any class implementing the
scikit-learn `fit`/`predict` API works, classifier or regressor.

**`model_params` accepts two shapes.** Nested by model name keeps settings for
several models side by side, so switching is a one-word change to `model_name`:

```yaml
model_params:
  xgboost:
    n_estimators: 50
    max_depth: 10
  rf_regressor:
    n_estimators: 200
```

A flat mapping also works and is passed to whichever model is selected — simpler,
but then every model must accept the same arguments. Keys have to be real
constructor arguments for that estimator: `LinearRegression` has no
`n_estimators`.

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

---

### What kind of problem is this? Classification or regression

The template handles **binary classification, multiclass classification, and
regression**. You do not normally have to say which — it reads the target column
and works it out, then logs the decision on every run:

```text
Task inferred as regression from the target (1067 distinct values, dtype float64).
Set `task:` in cfg/config.yaml to override.
```

That one decision drives everything downstream:

| | Metrics | Evaluation plots |
| --- | --- | --- |
| **Binary classification** | accuracy, precision, recall, f1 | confusion matrix, classification report, ROC curve, precision-recall curve |
| **Multiclass classification** | accuracy, macro precision / recall / f1 | confusion matrix, classification report |
| **Regression** | RMSE, MAE, R² | predicted-vs-actual, residuals |

Two details worth knowing:

**Multiclass gets no ROC or precision-recall curve.** Those curves are defined
for two classes. Drawing one for three would quietly give you a one-vs-rest
curve against an arbitrary class — a plot that looks perfectly reasonable and
tells you something you did not ask. The template omits it rather than mislead.

**Macro averaging for multiclass** weights every class equally, so a large class
cannot hide poor performance on a small one.

#### When inference guesses wrong

The rules: a boolean, text, or categorical target is always classes. A numeric
target is measurements if it has fractional values, or more than 20 distinct
whole numbers; otherwise it is classes.

That gets the common cases right and will occasionally be wrong — integer counts
you genuinely want to regress onto look exactly like class labels. Set `task:`
in `cfg/config.yaml` when it does:

```yaml
task: regression       # or: classification, or: auto (the default)
target_values: null    # a continuous target has no fixed value set
```

Setting `task` explicitly never hurts. If you already know what you are
modelling, saying so is clearer than relying on a heuristic.

#### What changes downstream when the task changes

You set `task` (or let it infer) in one place. Here is what each pipeline step
does differently as a result — useful when reading the code or extending it:

| Step | Changes with the task? | What it does |
| --- | --- | --- |
| `prepare_data.py` | **No** | Loads, transforms, and writes the CSV. Its schema check validates the target's values only when `target_values` is set, which is what lets the same code path serve a continuous target. Your own transforms go here. |
| `eda.py` | **Slightly** | Feature plots are identical. The target gets a class-balance bar chart for classification and a histogram for regression — counting occurrences of a continuous target would draw one bar per row. |
| `train_model.py` | **Yes** | Picks the metrics and evaluation plots, rejects an estimator whose family does not match the target, and ignores `stratify` for regression, where there are no classes to balance. |

Two guards exist because their absence was confusing:

- **Mismatched estimator.** A regressor on class labels trains without
  complaint and only fails later, deep inside scikit-learn's metrics, with a
  message that never names the real mistake. The template now stops at model
  construction and tells you which model, which task, and both ways to fix it.
- **`stratify` on regression.** Almost every value in a continuous target is
  unique, so scikit-learn would refuse with "the least populated class has only
  1 member". The setting is ignored, with a log line saying so.

#### Adding a metric

`compute_metrics` in `backend/src/ml/task.py` returns a plain dictionary, and
everything downstream follows it. Add an entry and it is logged to MLflow *and*
appears as a column in the dashboard — no other file changes.

---

### Sample data

Three datasets ship with the template, one per supported task type, so you can
try each without finding your own data first. `backend/setup.sh` writes them all
into `data/raw/`; regenerate at any time with `make sample-data`.

| File in `data/raw/` | Task | Shape | Config to select it |
| --- | --- | --- | --- |
| `breast_cancer.csv` | binary classification | 569 × 31 | *the shipped default* |
| `iris.csv` | multiclass classification | 150 × 5 | `target_values: [0, 1, 2]`, `model_name: "random_forest"` |
| `california_housing.csv` | regression | 20,640 × 9 | `target_values: null`, `model_name: "rf_regressor"` |

**Only the file named by `data.input_file` is read.** The other two sit in
`data/raw/` doing nothing until you point the config at one. Set `input_file`,
`target_values`, and `model_name` from the table — `task` can stay `auto`, which
infers correctly for all three. The same three settings are repeated as a
copy-paste block in `cfg/config.yaml`.

Every dataset names its target column `target`, so the shipped
`target_column: "TARGET"` covers all three.

> `california_housing.csv` is downloaded by scikit-learn rather than bundled
> with it, and cached afterwards. On a machine without network access it is
> skipped with a warning and the other two still generate.

### Using your own data

Two steps, and neither requires editing Python:

1. Drop a CSV into `data/raw/` (at the repository root).
2. Point `data.input_file` and `target_column` at it in `cfg/config.yaml`, and
   set `target_values` to your class labels — or `null` for regression.

Then `make pipeline`. Column names are normalised on read, so `mean radius`,
`Mean Radius`, and `MEAN_RADIUS` all become `MEAN_RADIUS`; `target_column` is
matched after that.

`backend/src/schemas.py` validates only the target by default, so it does not
need touching. Add checks there for the columns whose silent corruption would
ruin a model — an example is in the module docstring.

The pipeline still assumes two things it will not do for you:

- **Numeric features only.** Encode or drop categoricals in
  `PrepareDataPipeline`.
- **No missing values by training time.** Impute or drop them in the same place.

Both raise a clear error naming the offending columns, rather than letting
scikit-learn fail with something harder to read.

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
    ├── cli.py              # Entry point for the terminal workflow
    ├── config.py           # Paths, .env, and cfg/config.yaml access
    ├── schemas.py          # Pandera data validation schemas
    ├── ml/
    │   ├── pipeline.py     # Abstract Pipeline base class
    │   ├── prepare_data.py # Data loading and transformations
    │   ├── eda.py          # Exploratory plots logged to MLflow
    │   ├── train_model.py  # Model training with MLflow tracking
    │   ├── task.py         # Classification vs regression, and its metrics
    │   ├── inference.py    # Loading a registered model and predicting
    │   ├── io.py           # CSV read/write with schema validation
    │   ├── plots.py        # Matplotlib/seaborn plotting helpers
    │   └── tracking.py     # MLflow setup and flavor-aware model logging
    └── api/
        ├── app.py          # FastAPI application factory
        ├── deps.py         # Shared dependencies (model, config, MLflow client)
        ├── contracts.py    # Request/response payload shapes (Pydantic)
        └── routers/        # health, predict, runs
frontend/
├── package.json            # Node dependencies and scripts
├── vite.config.ts          # Dev server, /api proxy, test config
└── src/
    ├── main.tsx            # Startup: mounts React into index.html
    ├── App.tsx             # Routing table
    ├── styles.css          # Plain CSS; restyle via the variables at the top
    ├── api/
    │   ├── openapi.json    # Generated from the backend — do not hand-edit
    │   ├── schema.d.ts     # Generated TypeScript types — do not hand-edit
    │   ├── client.ts       # fetch wrapper and typed errors
    │   └── hooks.ts        # One data-fetching hook per endpoint
    ├── components/         # FeatureForm, ArtifactGallery, Layout, States
    └── pages/              # Predict, Runs, RunDetail
```

Three words are easy to confuse in a repository that has both ML and web code,
so each has exactly one meaning here:

| Term | Means | Lives in |
| --- | --- | --- |
| **model** | A trained estimator | MLflow; loaded by `ml/inference.py` |
| **schema** | A Pandera DataFrame contract | `src/schemas.py` |
| **contract** | An HTTP request/response shape | `src/api/contracts.py` |

The layering rule is that **`api/` translates and `ml/` decides** — nothing in
`ml/` imports from `api/`. Prediction rules and MLflow access live in `ml/`; the
API only maps their results and exceptions onto HTTP status codes. That is what
lets the pipelines be tested without a web server and the API without MLflow.

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

Backend tests live in `backend/tests/`, frontend tests alongside the code they
cover in `frontend/src/`.

```bash
make test              # both suites
make test-backend      # pytest
make test-frontend     # vitest
```

---

## Schema checks

Schemas live in `backend/src/schemas.py` and are applied automatically whenever
the pipeline reads or writes a CSV, so a corrupted intermediate file is caught
at the boundary rather than three steps later.

They validate in **non-strict** mode: only the columns you declare are checked,
and anything else passes through untouched.

**Out of the box only the target column is checked**, which is what lets any
dataset run without editing this file. Naming specific feature columns here
would make the template fail on the first run with anyone else's data.

Add checks for the columns whose silent corruption would ruin a model — a
negative age, a probability above 1, a category that should never appear:

```python
FEATURE_COLUMNS = {
    "MEAN_RADIUS": Column(float, checks=Check.ge(0)),
    "AGE": Column(int, checks=Check.in_range(0, 120)),
}
```

The target's permitted values come from `target_values` in `cfg/config.yaml`.
Leave it `null` for regression: a continuous target has no fixed value set, and
checking one would reject every valid row.
