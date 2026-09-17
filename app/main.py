"""FastAPI microservice serving the churn model.

Model resolution order (first that works):
  1. MLflow Model Registry  ->  models:/<MODEL_NAME>@<MODEL_ALIAS>
  2. Local export           ->  <MODEL_PATH> (artifacts/model.joblib), used in Docker / offline

Endpoints:
  GET  /health         liveness + which model is loaded
  GET  /model-info     metadata of the loaded model (version, metrics...)
  POST /predict        score one customer
  POST /predict/batch  score a list of customers
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import coerce_feature_types, load_config, resolve, setup_mlflow

CONFIG = load_config(os.getenv("CONFIG_PATH"))
FEATURES = CONFIG["features"]["numeric"] + CONFIG["features"]["categorical"]
THRESHOLD = float(os.getenv("THRESHOLD", CONFIG["evaluation"]["threshold"]))

STATE: dict = {"model": None, "source": None, "info": {}}


# ---------- request / response schemas ----------


class Customer(BaseModel):
    """One Telco customer. Field names match the dataset columns exactly."""

    tenure: float = Field(..., ge=0, description="months with the company")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float | None = Field(None, ge=0, description="may be null for new customers")
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal["0", "1"] = "0"
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tenure": 2,
                    "MonthlyCharges": 85.7,
                    "TotalCharges": 171.4,
                    "gender": "Female",
                    "SeniorCitizen": "0",
                    "Partner": "No",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                }
            ]
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    churn: bool
    threshold: float


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    model_source: str


# ---------- model loading ----------


def _load_from_registry():
    import mlflow

    setup_mlflow(CONFIG)
    name = os.getenv("MODEL_NAME", CONFIG["mlflow"]["registered_model_name"])
    alias = os.getenv("MODEL_ALIAS", CONFIG["mlflow"]["alias"])
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    info = {"registered_model": name, "alias": alias, "version": mv.version, "run_id": mv.run_id, **mv.tags}
    return model, info


def _load_from_local():
    path = resolve(os.getenv("MODEL_PATH", CONFIG["serving"]["model_export_path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    info_file = path.parent / "model_info.json"
    info = json.loads(info_file.read_text()) if info_file.exists() else {}
    return joblib.load(path), {"path": str(path), **info}


def load_model() -> None:
    source = os.getenv("MODEL_SOURCE", "auto")  # auto | registry | local
    errors = []
    if source in ("auto", "registry"):
        try:
            STATE["model"], STATE["info"] = _load_from_registry()
            STATE["source"] = "mlflow-registry"
            print(f"[api] model loaded from registry: {STATE['info']}")
            return
        except Exception as exc:  # noqa: BLE001 - fall back to local export
            errors.append(f"registry: {exc}")
            print(f"[api] registry unavailable ({exc}); trying local export")
    if source in ("auto", "local"):
        try:
            STATE["model"], STATE["info"] = _load_from_local()
            STATE["source"] = "local-export"
            print(f"[api] model loaded from local export {STATE['info'].get('path')}")
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"local: {exc}")
    print("[api] WARNING no model loaded: " + " | ".join(errors))


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    description="Serves the TelcoChurnClassifier trained with the MLOps pipeline (MLflow registry).",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- endpoints ----------


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": STATE["model"] is not None, "model_source": STATE["source"]}


@app.get("/model-info")
def model_info():
    _require_model()
    return {"model_source": STATE["source"], "threshold": THRESHOLD, **STATE["info"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    return _score([customer])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    return BatchResponse(predictions=_score(req.customers), model_source=STATE["source"])


def _require_model():
    if STATE["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")


def _score(customers: list[Customer]) -> list[Prediction]:
    _require_model()
    df = pd.DataFrame([c.model_dump() for c in customers])
    X = coerce_feature_types(df, CONFIG)[FEATURES]
    proba = STATE["model"].predict_proba(X)[:, 1]
    return [
        Prediction(churn_probability=float(p), churn=bool(p >= THRESHOLD), threshold=THRESHOLD) for p in proba
    ]
