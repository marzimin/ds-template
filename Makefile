# Convenience entry points so common tasks work from the repository root.
# The backend is a uv project rooted at backend/; these targets forward to it.

PYTHON_VERSION ?= 3.12
MLFLOW_HOST ?= 127.0.0.1
MLFLOW_PORT ?= 5000
BACKEND := backend

.DEFAULT_GOAL := help
.PHONY: help setup test lint pipeline mlflow sample-data docker-build clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install dependencies and hooks, generate demo data
	PYTHON_VERSION=$(PYTHON_VERSION) ./$(BACKEND)/setup.sh

test: ## Run the backend test suite
	cd $(BACKEND) && uv run pytest

lint: ## Run all pre-commit hooks across the repository
	cd $(BACKEND) && uv run pre-commit run --all-files

pipeline: ## Run the full ML pipeline (prepare -> EDA -> train)
	cd $(BACKEND) && uv run pipeline

mlflow: ## Start a local MLflow tracking server
	cd $(BACKEND) && uv run mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT)

sample-data: ## Regenerate the demo dataset into data/raw/
	cd $(BACKEND) && uv run python scripts/generate_sample_data.py

docker-build: ## Build the backend image (context is the repository root)
	docker build -f $(BACKEND)/Dockerfile -t ds-template-backend .

clean: ## Remove caches and build artefacts
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
	find $(BACKEND) -name '__pycache__' -type d -prune -exec rm -rf {} +
