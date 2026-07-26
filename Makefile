# Convenience entry points so common tasks work from the repository root.
# The backend is a uv project rooted at backend/, the frontend an npm project
# rooted at frontend/; these targets forward to whichever is relevant.

PYTHON_VERSION ?= 3.12
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
API_HOST ?= 127.0.0.1
API_PORT ?= 8000
WEB_PORT ?= 5173
BACKEND := backend
FRONTEND := frontend

.DEFAULT_GOAL := help
.PHONY: help setup setup-backend setup-frontend hooks test test-backend test-frontend \
        lint lint-backend lint-frontend pipeline mlflow api web types build \
        sample-data docker-build clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## Set up both halves of the project

setup-backend: ## Create the venv, install dependencies and hooks, generate demo data
	PYTHON_VERSION=$(PYTHON_VERSION) ./$(BACKEND)/setup.sh

setup-frontend: ## Install frontend dependencies
	cd $(FRONTEND) && npm install

hooks: ## (Re)install the git pre-commit hook against backend/.venv
	$(BACKEND)/.venv/bin/pre-commit install --allow-missing-config

# --- Running things -------------------------------------------------------
# The full application needs three processes, one per terminal:
#   make mlflow   (5000)  stores runs and serves the model registry
#   make api      (8000)  loads the model, answers requests
#   make web      (5173)  serves the page you open in a browser

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

sample-data: ## Regenerate the demo dataset into data/raw/
	cd $(BACKEND) && uv run python scripts/generate_sample_data.py

# --- Contract between the halves ------------------------------------------

types: ## Regenerate frontend types from the API's OpenAPI schema
	cd $(BACKEND) && uv run python -c \
		"import json; from src.api.app import app; print(json.dumps(app.openapi(), indent=2))" \
		> ../$(FRONTEND)/src/api/openapi.json
	cd $(FRONTEND) && npx openapi-typescript src/api/openapi.json -o src/api/schema.d.ts
	@echo "Types regenerated. Commit both files so a fresh clone builds without a running API."

# --- Checks ---------------------------------------------------------------

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

build: ## Produce the production frontend bundle
	cd $(FRONTEND) && npm run build

docker-build: ## Build the backend image (context is the repository root)
	docker build -f $(BACKEND)/Dockerfile -t ds-template-backend .

clean: ## Remove caches and build artefacts
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
	find $(BACKEND) -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf $(FRONTEND)/dist $(FRONTEND)/.vite
