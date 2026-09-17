"""API tests. The model is loaded from the local joblib export produced by a
training run on the synthetic dataset (no dependency on the real registry)."""

import pytest
from fastapi.testclient import TestClient

EXAMPLE = {
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


@pytest.fixture(scope="module")
def client(trained_run, session_env):
    _, cfg = trained_run
    session_env.setenv("MODEL_SOURCE", "local")
    session_env.setenv("MODEL_PATH", cfg["serving"]["model_export_path"])
    from app.main import app

    with TestClient(app) as c:  # lifespan -> model loaded once for the module
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_source"] == "local-export"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_type"] == "logreg"


def test_predict_single(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)


def test_predict_accepts_null_total_charges(client):
    r = client.post("/predict", json={**EXAMPLE, "TotalCharges": None})
    assert r.status_code == 200


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"customers": [EXAMPLE, {**EXAMPLE, "tenure": 60}]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_rejects_invalid_payload(client):
    r = client.post("/predict", json={**EXAMPLE, "Contract": "Forever"})
    assert r.status_code == 422
    r = client.post("/predict", json={**EXAMPLE, "tenure": -1})
    assert r.status_code == 422
