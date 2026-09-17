# ==========================================================================
# Makefile — Telco Churn MLOps pipeline
#
# Typical flow:
#   make install      # deps
#   make pipeline     # data -> train -> evaluate   (everything logged to MLflow)
#   make mlflow-ui    # browse experiments / registry at http://localhost:5000
#   make serve        # FastAPI on http://localhost:8000/docs
#
# Run `make help` to list every command.
# ==========================================================================

-include .env
export

PY      ?= python
CONFIG  ?= configs/config.yaml
MODEL   ?=                      # e.g. MODEL=random_forest to override config
PORT    ?= 8000
MLFLOW_PORT ?= 5000
IMAGE   ?= telco-churn-api:latest
export MLFLOW_DISABLE_AGENT_HINT=1

.DEFAULT_GOAL := help
.PHONY: help install data train evaluate predict pipeline notebook report lab test lint format \
        mlflow-ui serve docker-build docker-run docker-up docker-down clean

help: ## Show this help
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

# --- Setup -----------------------------------------------------------------

install: ## Install dependencies + the project in editable mode
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

# --- Pipeline stages ---------------------------------------------------------

data: ## Download the raw CSV and build the clean dataset
	$(PY) src/data.py --config $(CONFIG)

train: ## GridSearchCV + MLflow tracking + Model Registry (MODEL=logreg|random_forest)
	$(PY) src/train.py --config $(CONFIG) $(if $(MODEL),--model-type $(MODEL),)

evaluate: ## Score the @champion model, log plots & report to MLflow
	$(PY) src/evaluate.py --config $(CONFIG)

predict: ## Batch inference. Usage: make predict INPUT=file.csv [OUTPUT=out.csv]
	@test -n "$(INPUT)" || (echo "Usage: make predict INPUT=file.csv [OUTPUT=out.csv]" && exit 1)
	$(PY) src/predict.py --config $(CONFIG) --input $(INPUT) $(if $(OUTPUT),--output $(OUTPUT),)

pipeline: data train evaluate ## Run the whole pipeline end to end

# --- Analysis & reporting ------------------------------------------------------

notebook: ## Execute the EDA notebook in place (figures -> reports/figures/)
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/eda.ipynb --ExecutePreprocessor.timeout=300

report: ## Build docs/rapport.html + docs/rapport.pdf from MLflow runs and figures
	$(PY) scripts/build_report.py

lab: ## Open Jupyter Lab on the notebooks
	$(PY) -m jupyter lab notebooks/

# --- Quality -----------------------------------------------------------------

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Static checks (ruff)
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format: ## Auto-format + fix lint issues
	$(PY) -m ruff check --fix .
	$(PY) -m ruff format .

# --- Serving -----------------------------------------------------------------

mlflow-ui: ## Open the MLflow UI on the local SQLite backend
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

serve: ## Run the FastAPI prediction service (loads models:/TelcoChurnClassifier@champion)
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

# --- Docker ------------------------------------------------------------------

docker-build: ## Build the API image (embeds artifacts/model.joblib produced by `make train`)
	docker build -t $(IMAGE) .

docker-run: docker-build ## Run the API container on port $(PORT)
	docker run --rm -p $(PORT):8000 $(IMAGE)

docker-up: ## Start MLflow server + API with docker compose
	docker compose up --build

docker-down: ## Stop the compose stack
	docker compose down

# --- Housekeeping -------------------------------------------------------------

clean: ## Remove caches, reports and generated artifacts (keeps data & mlflow.db)
	$(PY) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['reports','artifacts','.pytest_cache','.ruff_cache','build','src.egg-info','telco_churn_mlops.egg-info']]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
