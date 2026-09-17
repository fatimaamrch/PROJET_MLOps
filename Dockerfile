# Serving image for the Telco Churn API.
# The model is baked in from artifacts/ (produced by `make train`) so the
# container is self-contained: no MLflow server needed at runtime.
#
#   make train && make docker-build && make docker-run
#   curl http://localhost:8000/health

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_SOURCE=local \
    MODEL_PATH=artifacts/model.joblib

WORKDIR /app

# Dependencies first (cached layer)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Source + config + trained model
COPY src ./src
COPY app ./app
COPY configs ./configs
COPY artifacts ./artifacts
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
