# Convenience entry points so common tasks work from the repository root.
# The backend is a uv project rooted at backend/, the frontend an npm project
# rooted at frontend/; these targets forward to whichever is relevant.
#
# Target names are kept in step with de-template, the upstream data engineering
# project, so that moving between the two does not mean relearning the verbs.
# `build` in particular means "rebuild the container image" in both; the
# frontend bundle is `bundle`.

# Reads one key out of .env, or nothing if the file or the key is absent.
# Commented-out lines do not match: the key has to start the line. A trailing
# ` # comment` is dropped, and it has to be whitespace-preceded to count as one,
# which is the same rule docker compose applies to its own .env parsing — the
# value ends up exported, so anything left on it would override that parse.
env_value = $(strip $(shell [ -f .env ] && sed -n 's/^[[:space:]]*$(1)[[:space:]]*=[[:space:]]*//p' .env | tail -1 | sed 's/[[:space:]][[:space:]]*\#.*$$//' | tr -d "\"'"))

# Precedence for the three port knobs: command line, then the environment, then
# .env, then the default here. .env has to be read explicitly — make does not
# read it, and docker compose gives an exported environment variable precedence
# over the .env file it reads itself. So without the $(call env_value,...) the
# `export` below would pin 5000 and silently beat whatever .env says, making the
# knob that .env.example and docker-compose.yml both advertise do nothing.
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= $(or $(call env_value,MLFLOW_PORT),5000)
API_HOST ?= 127.0.0.1
API_PORT ?= $(or $(call env_value,API_PORT),8000)
WEB_PORT ?= $(or $(call env_value,WEB_PORT),5173)
BACKEND := backend
FRONTEND := frontend
COMPOSE := docker compose

# Exported so that `make up MLFLOW_PORT=5001` reaches docker compose, which
# reads these as ${MLFLOW_PORT} in docker-compose.yml. Without `export` only the
# `MLFLOW_PORT=5001 make up` form would work, and the difference is invisible.
export MLFLOW_PORT API_PORT WEB_PORT

.DEFAULT_GOAL := help
.PHONY: help setup setup-backend setup-frontend hooks test test-backend test-frontend \
        lint lint-backend lint-frontend format typecheck check pipeline mlflow api web \
        types bundle demo up down logs build reset docker-pipeline sample-data clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

## ── Setup ────────────────────────────────────────────────────────────────────

setup: setup-backend setup-frontend ## Set up both halves of the project

setup-backend: ## Create the venv, install dependencies and hooks, generate demo data
	./$(BACKEND)/setup.sh

setup-frontend: ## Install frontend dependencies
	cd $(FRONTEND) && npm install

hooks: ## (Re)install the git pre-commit hook against backend/.venv
	$(BACKEND)/.venv/bin/pre-commit install --allow-missing-config

## ── Running things ───────────────────────────────────────────────────────────
# Either run the three processes yourself, one per terminal:
#   make mlflow   (5000)  stores runs and serves the model registry
#   make api      (8000)  loads the model, answers requests
#   make web      (5173)  serves the page you open in a browser
# or let docker compose run all three at once with `make up`.

mlflow: ## Start a local MLflow tracking server
	cd $(BACKEND) && uv run mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

api: ## Start the FastAPI server (interactive docs at /docs)
	@echo "API docs: http://$(API_HOST):$(API_PORT)/docs"
	cd $(BACKEND) && uv run uvicorn src.api.app:app --reload \
		--host $(API_HOST) --port $(API_PORT)

web: ## Start the frontend dev server
	@echo "App: http://localhost:$(WEB_PORT)"
	cd $(FRONTEND) && npm run dev -- --port $(WEB_PORT)

pipeline: ## Run the full ML pipeline (prepare -> EDA -> train)
	cd $(BACKEND) && uv run pipeline

sample-data: ## Regenerate the demo datasets into data/raw/
	cd $(BACKEND) && uv run python scripts/generate_sample_data.py

## ── Docker ───────────────────────────────────────────────────────────────────

demo: ## Start everything AND train a model, so the app has something to show
	$(COMPOSE) up -d
	@echo ""
	@echo "Training the demo model — a minute or so the first time."
	$(COMPOSE) run --rm pipeline
	@echo ""
	@echo "  App        →  http://localhost:$(WEB_PORT)"
	@echo "  API docs   →  http://localhost:$(API_PORT)/docs"
	@echo "  MLflow     →  http://localhost:$(MLFLOW_PORT)"
	@echo ""
	@echo "Stop it with 'make down'."

up: ## Start MLflow, the API and the web interface (no model until you train one)
	@echo "App        →  http://localhost:$(WEB_PORT)"
	@echo "API docs   →  http://localhost:$(API_PORT)/docs"
	@echo "MLflow     →  http://localhost:$(MLFLOW_PORT)"
	@echo ""
	@echo "Nothing is trained yet — run 'make docker-pipeline', or use 'make demo'"
	@echo "next time to do both in one step."
	$(COMPOSE) up -d

down: ## Stop all services (runs, models and artifacts are preserved)
	$(COMPOSE) down

logs: ## Follow the logs of all services
	$(COMPOSE) logs -f

build: ## Rebuild the container image
	$(COMPOSE) build

reset: ## Stop everything and DELETE the MLflow volume, then start fresh
	$(COMPOSE) down -v
	$(COMPOSE) up -d --build

docker-pipeline: ## Run the pipeline in a container against the compose MLflow
	$(COMPOSE) run --rm pipeline

## ── Contract between the halves ──────────────────────────────────────────────

types: ## Regenerate frontend types from the API's OpenAPI schema
	cd $(BACKEND) && uv run python -c \
		"import json; from src.api.app import app; print(json.dumps(app.openapi(), indent=2))" \
		> ../$(FRONTEND)/src/api/openapi.json
	cd $(FRONTEND) && npx openapi-typescript src/api/openapi.json -o src/api/schema.d.ts
	@echo "Types regenerated. Commit both files so a fresh clone builds without a running API."

## ── Checks ───────────────────────────────────────────────────────────────────

check: lint test ## Everything CI runs

test: test-backend test-frontend ## Run every test suite

test-backend: ## Run the backend test suite
	cd $(BACKEND) && uv run pytest

test-frontend: ## Run the frontend test suite
	cd $(FRONTEND) && npm test

lint: lint-backend lint-frontend ## Lint everything

lint-backend: ## Run all pre-commit hooks across the repository
	cd $(BACKEND) && uv run pre-commit run --all-files

lint-frontend: ## Type-check, lint, and format-check the frontend
	cd $(FRONTEND) && npm run typecheck && npm run lint

format: ## Format both halves in place
	cd $(BACKEND) && uv run ruff format .
	cd $(FRONTEND) && npm run format

typecheck: ## Type-check both halves
	# tests/ is included so the `ignore_errors` override in pyproject.toml
	# applies; mypy warns the section is unused if they are left out.
	cd $(BACKEND) && uv run mypy src scripts tests
	cd $(FRONTEND) && npm run typecheck

bundle: ## Produce the production frontend bundle
	cd $(FRONTEND) && npm run build

clean: ## Remove caches and build artefacts
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
	rm -rf $(BACKEND)/htmlcov $(BACKEND)/.coverage
	find $(BACKEND) -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf $(FRONTEND)/dist $(FRONTEND)/.vite
