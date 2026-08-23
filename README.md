# DS Template

A starting point for data science projects that need to become more than a
notebook: a configurable ML pipeline, experiment tracking, and a web interface
so people who do not use a terminal can see the results.

It handles **binary classification, multiclass classification, and regression**,
and works out which one you have from your data.

```text
cfg/config.yaml   ← configure your dataset, model, and training here
data/             ← put your data here
outputs/          ← plots and reports land here
backend/          ← Python: pipelines, ML, and the API
frontend/         ← TypeScript/React: the web interface
```

You only edit the first two to use it with your own data.

---

## Quick start (DS/ML Development)

**Prerequisites:** [uv](https://docs.astral.sh/uv/) (`brew install uv`) and
Node.js 22+.

Or just Docker (Desktop) if you're running as an end user — see [Running it in containers](#running-it-in-containers).

```bash
make setup
```

That creates the Python environment, installs both halves, sets up git hooks,
and writes three demo datasets into `data/raw/`.

Then, in **two terminals**:

```bash
make mlflow      # terminal 1 — the experiment tracking server
make pipeline    # terminal 2 — prepare → explore → train (retrain)
```

> On macOS, port 5000 is often taken by the AirPlay Receiver. If MLflow appears
> to start but the pipeline reports it unreachable, disable AirPlay Receiver in
> System Settings, or use another port and set `MLFLOW_TRACKING_URI` to match.

---

## What you get

A full run of the shipped demo produces **67 artifacts**, all of them logged to
MLflow *and* written to disk:

| Location | Contents |
| --- | --- |
| `outputs/eda/` | 63 exploratory plots — one distribution and one feature-vs-target plot per column, plus a correlation heatmap, target balance, and missing values |
| `outputs/plots/` | Evaluation plots for the task: confusion matrix, ROC and precision-recall curves for binary classification; predicted-vs-actual and residuals for regression |
| `outputs/reports/` | `classification_report.txt` (classification only) |
| `data/processed/` | `*_prepared.csv` after your transformations, `*_trained.csv` with a `PREDICTION` column |
| MLflow | 8 metrics, 4 parameters, all 67 artifacts, and the trained model registered as a new version |

In the terminal you will see the task it inferred and the scores:

```text
Task inferred as binary_classification from the target (2 distinct values, dtype int64).
MODEL DRIFT: test accuracy = 0.9561
MODEL DRIFT: test precision = 0.9583
MODEL DRIFT: test recall = 0.9718
MODEL DRIFT: test f1_score = 0.9650
```

Open **<http://127.0.0.1:5000>** to browse runs in MLflow.

---

## Running the web application (ML + Frontend / Backend Development)

Three terminals, one per process:

```bash
make mlflow      # terminal 1 — tracking server        (port 5000)
make api         # terminal 2 — serves the model       (port 8000)
make web         # terminal 3 — serves the page        (port 5173)
```

Then open **<http://localhost:5173>**.

> Use `localhost`, not `127.0.0.1`. The dev server binds the hostname
> `localhost`, which often resolves to IPv6, so `127.0.0.1:5173` can refuse the
> connection.

Three screens:

| Screen | What it shows |
| --- | --- |
| **Predict** | A form built from your model's own feature list, pre-filled with a real row. Submit it for a prediction, with class probabilities when the model has them. |
| **Runs** | Every training run and its metrics, newest first. |
| **Run detail** | One run's metrics and parameters, plus a gallery of its plots. |

The header shows whether the API is reachable and which model version is loaded.

**Nothing trained yet?** The app still runs. It tells you to run `make pipeline`
rather than showing an error, and picks the model up on the next request — no
restart needed.

`make api` alone also gives you **<http://127.0.0.1:8000/redoc>**: an interactive
page listing every endpoint with a form to try it. Useful for checking the
backend without the frontend.

---

## Running it in containers (End User Demo)

Everything at once, with nothing installed but Docker:

```bash
make demo        # start the services AND train a model
```

That is the one to run first. It leaves you with a working app at
**<http://localhost:5173>**, `:8000/redoc` and `:5000`.

The pieces separately, once you know your way around:

```bash
make up               # services only — no model until you train one
make docker-pipeline  # train, as a batch job rather than a service
make logs             # follow all three
make down             # stop (runs, models and artifacts are preserved)
```

`cfg/`, `data/` and `outputs/` are bind-mounted, so edits take effect without a
rebuild and the plots and CSVs land in your working tree. MLflow's runs and
artifacts live in a named volume — `make down` keeps them, `make reset` deletes
them and starts fresh.

> **Docker and local runs keep separate MLflow stores.** Containers use the
> `mlflow_data` volume; `make mlflow` uses `backend/mlflow.db` beside the code.
> A model trained in one is invisible to the other, so train in whichever mode
> you are running. Pick one for a given session rather than mixing them.
>
> Port 5000 is often taken by the AirPlay Receiver on macOS. Set `MLFLOW_PORT`
> in `.env`, or `make demo MLFLOW_PORT=5001`.

The image also runs standalone. It defaults to the pipeline; override the
command to serve instead:

```bash
make build                                            # rebuild it
docker run --rm -p 8000:8000 ds-template-backend \
    uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

CI publishes the same image to `ghcr.io/marzimin/ds-template` on every push to
`main`, tagged with the version it bumped.

---

## Using your own data

Two steps to point it at your file:

1. Put a CSV in `data/raw/`.
2. In `cfg/config.yaml`, set `data.input_file`, `target_column`, and either
   `target_values` (your class labels) or `null` for regression. Pick a
   `model_name` from `model_registry` — a **classifier** for labels, a
   **regressor** for continuous targets.

Then `make pipeline`. Column names are normalised on read, so `mean radius`,
`Mean Radius`, and `MEAN_RADIUS` all become `MEAN_RADIUS`.

**If your data is already numeric and complete, that is the whole job.** If it
is not, expect to write some Python. The pipeline requires **numeric features
with no missing values** by training time and stops with an error naming the
offending columns, because encoding and imputation change what a model learns
and the template will not guess for you. That work goes in
`PrepareDataPipeline` (`backend/src/ml/prepare_data.py`), which ships as four
empty steps for you to fill in — real-world data usually needs at least one of
them.

See [`docs/ml.md`](docs/ml.md) for task types, model configuration, and schemas.

---

## Commands

```bash
make setup      # install everything and generate the demo datasets
make pipeline   # prepare → EDA → train
make mlflow     # tracking server        (terminal 1)
make api        # API server             (terminal 2)
make web        # web interface          (terminal 3)
make demo       # …or all three in containers, with a model trained
make down       # stop them
make test       # both test suites
make lint       # format and lint everything
make check      # lint and test — everything CI runs
make types      # regenerate frontend types after changing the API
make bundle     # production frontend bundle
make help       # every target
```

Target names match [de-template](https://github.com/marzimin/de-template) where
the two overlap, so moving between them does not mean relearning the verbs.

---

## Documentation

| Document | Read it when |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | You want to understand how the pieces fit together. Written for a data scientist who has not built a web application before. |
| [`docs/ml.md`](docs/ml.md) | You are changing the data, the model, or the metrics. |
| [`docs/backend.md`](docs/backend.md) | You are changing the API. |
| [`docs/frontend.md`](docs/frontend.md) | You are changing the web interface. |
| [`docs/handoff.md`](docs/handoff.md) | Your data arrives from a warehouse run by [de-template](https://github.com/marzimin/de-template), rather than from a CSV you put there yourself. |

---

## When something looks wrong

| Symptom | Likely cause | Try |
| --- | --- | --- |
| Pipeline says MLflow is unreachable | Tracking server not running, or port 5000 taken | `make mlflow`; on macOS check AirPlay Receiver |
| Web page panels are all empty | API not running | `make api`, then open `/docs` |
| Predictions return 503 | No model trained yet | `make pipeline` |
| Browser cannot reach the dev server | IPv6 vs IPv4 | Use `localhost:5173` |
| Frontend fails to compile after an API change | Types are stale | `make types` |
| Commit fails with `pre-commit not found` | Hook points at a deleted virtualenv | `make hooks` |
| Metrics look wrong for your problem | Task inferred incorrectly | Check the "Task inferred as…" log line; set `task:` in `cfg/config.yaml` |
| "Target column not found", naming a column you can see in the CSV | Column names are normalised to upper case on read | Write `target_column` in `UPPER_SNAKE`; see [`docs/handoff.md`](docs/handoff.md) |
| `make up` fails to bind port 5000 | AirPlay Receiver | `make up MLFLOW_PORT=5001`, or set it in `.env` |

---

## Making it your own

- **Rename the project.** Replace `ds-template` throughout. The MLflow
  experiment and registered model names follow `[project].name` in
  `backend/pyproject.toml` automatically.
- **Environment.** `cp .env.example .env` (`make setup` does this) and set
  `MLFLOW_TRACKING_URI` if your tracking server is not local.
- **Restyle the web interface.** Edit the CSS variables at the top of
  `frontend/src/styles.css`. No framework, no component changes.
- **Quality gates.** `make lint` runs Ruff, MyPy, Bandit, and pydocstyle on the
  backend, and TypeScript, ESLint, and Prettier on the frontend. The same
  checks run in CI.
