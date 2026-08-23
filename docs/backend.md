# The backend

Python: the pipelines, the ML code, and the API that serves them.

For the mental model of why an API exists at all, read
[`architecture.md`](architecture.md) first. For data and models, see
[`ml.md`](ml.md).

---

## Layout

```text
backend/src/
├── cli.py          Entry point for the terminal (uv run pipeline)
├── config.py       Paths, .env, and cfg/config.yaml. Dependency-light.
├── schemas.py      Pandera contracts for DataFrames
├── ml/
│   ├── pipeline.py       The abstract base class the steps implement
│   ├── prepare_data.py   Raw CSV in, transformed CSV out
│   ├── eda.py            Exploratory plots
│   ├── train_model.py    Fit, evaluate, log
│   ├── task.py           Classification vs regression, and its metrics
│   ├── inference.py      Load a registered model and predict
│   ├── io.py             CSV read/write with schema validation
│   ├── plots.py          All matplotlib and seaborn lives here
│   └── tracking.py       MLflow setup and flavour-aware model logging
└── api/
    ├── app.py            Application factory, CORS, startup
    ├── deps.py           Shared dependencies (model, config, MLflow client)
    ├── contracts.py      Request/response shapes (Pydantic)
    └── routers/          health, predict, runs
```

### The one rule

> **`api/` translates, `ml/` decides — and never the reverse.**

Nothing in `ml/` imports from `api/`. Prediction rules, validation, and MLflow
access live in `ml/`; the API only maps their results and exceptions onto HTTP
status codes.

That property is what lets the pipelines be tested without a web server, the API
be tested without MLflow, and the CLI keep working regardless of the API.

### Three words that would otherwise collide

| Term | Means | Lives in |
| --- | --- | --- |
| **model** | A trained estimator | MLflow, loaded by `ml/inference.py` |
| **schema** | A Pandera DataFrame contract | `src/schemas.py` |
| **contract** | An HTTP request/response shape | `src/api/contracts.py` |

The API's payload module is called `contracts.py` rather than the conventional
`models.py` precisely because "model" already means something here.

---

## Running it

```bash
make api      # http://127.0.0.1:8000
```

Then open **<http://127.0.0.1:8000/redoc>** — FastAPI generates an interactive
page from your function signatures, listing every endpoint with a form to call
it. It is the fastest way to exercise the backend, and it works before any
frontend exists.

---

## Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Is the service up, and is a model loaded? |
| `GET /api/predict/schema` | The features a prediction needs, from the model signature |
| `POST /api/predict` | Predict for one record |
| `POST /api/predict/reload` | Pick up a newly trained version without restarting |
| `GET /api/runs` | Runs with their metrics |
| `GET /api/runs/{id}` | One run, with parameters and tags |
| `GET /api/runs/{id}/artifacts` | List a run's artifacts |
| `GET /api/runs/{id}/artifacts/file` | Download one artifact |

### Status codes

| Code | Meaning | When |
| --- | --- | --- |
| `200` | OK | Normal response |
| `422` | Your input was invalid | Missing feature, or text where a number belongs |
| `404` | Not found | Unknown run id or artifact |
| `503` | Service unavailable | **No model trained yet** |
| `502` | Upstream failed | MLflow unreachable |
| `500` | We have a bug | Genuine server fault |

---

## Two behaviours worth knowing

### No model yet is a normal state, not a failure

On a fresh clone nobody has trained anything, so there is no registered model.
The API **starts anyway**: `/api/health` reports `model_available: false` and
prediction endpoints return `503` naming the fix.

A failed load is not cached, so training makes the *next* request succeed with
no restart. The frontend shows this as guidance rather than an error, because
nothing has failed.

### A running server holds one model

Loading a model takes a second or two — fine at startup, unacceptable per
request — so it is cached for the life of the process. That means **retraining
while the API runs would otherwise be invisible**. `POST /api/predict/reload`
drops the cache and picks up the newest version; the web interface exposes it as
a button in the header.

---

## Adding an endpoint

1. Declare the request and response shapes in `api/contracts.py`.
2. Add the handler to a router in `api/routers/`, using dependencies from
   `deps.py` rather than fetching things yourself.
3. Put any real logic in `ml/`, not the router. The router should translate,
   not decide.
4. Run `make types` so the frontend's TypeScript matches.

Declaring shapes as Pydantic classes means FastAPI validates input, documents it
at `/docs`, and publishes it in the OpenAPI schema — all from the same
declaration. You never write validation code by hand.

---

## Testing

```bash
make test-backend
```

Tests split the same way the code does:

- **`tests/api/`** runs against FastAPI's `TestClient` with dependencies
  overridden, so it needs **no MLflow server**.
- **`tests/test_*.py`** covers the pipelines and ML logic with no web server.

`tests/test_tracking.py` is the exception and deliberately so: it exercises a
real log-and-reload round trip against a temporary SQLite-backed MLflow, because
that contract is what the API depends on.

### Things pinned by tests because they were bugs once

- Artifact paths are validated against MLflow's own listing before download —
  the path is attacker-controlled, so it is never joined onto a local path
  directly.
- MLflow's schema-enforcement failures become concise `422`s rather than `500`s
  that echo the entire signature back to the caller.
- A regressor paired with a classification target is rejected at model
  construction, not deep inside the metrics.

---

## Configuration and paths

`config.py` resolves everything against the **repository root** — the directory
holding `cfg/`, `data/`, and `outputs/`. It is inferred from the package
location, or set explicitly with `DS_PROJECT_ROOT` (which the container image
does).

`.env` at the repository root is loaded once, explicitly, rather than depending
on the working directory. `MLFLOW_TRACKING_URI` and `CORS_ALLOW_ORIGINS` are the
two settings you are most likely to change.

---

## Quality gates

`make lint-backend` runs Ruff (lint and format), [ty](https://docs.astral.sh/ty/)
for type checking, Bandit, and pydocstyle via pre-commit. The same checks run in
CI, and the git hook runs them before each commit. See
[dev-practices.md](dev-practices.md) for the reasoning behind this toolchain.

If a commit ever fails with `pre-commit not found`, the installed hook points at
a virtualenv that no longer exists — run `make hooks`.
