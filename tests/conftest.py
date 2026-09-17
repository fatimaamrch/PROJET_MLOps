"""Shared fixtures: a small synthetic Telco-like dataset and a config pointing
to a temporary directory, so tests never touch real data or the real MLflow DB."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from src.utils import load_config

N_ROWS = 300


@pytest.fixture(scope="session")
def base_config() -> dict:
    return load_config()


@pytest.fixture(scope="session")
def synthetic_df(base_config) -> pd.DataFrame:
    """Random but schema-valid dataset with a learnable signal (tenure vs churn)."""
    rng = np.random.default_rng(0)
    cats = {
        "gender": ["Male", "Female"],
        "SeniorCitizen": ["0", "1"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }
    df = pd.DataFrame({c: rng.choice(v, N_ROWS) for c, v in cats.items()})
    df["tenure"] = rng.integers(0, 72, N_ROWS).astype(float)
    df["MonthlyCharges"] = rng.uniform(20, 120, N_ROWS).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2)
    df.loc[:3, "TotalCharges"] = np.nan  # like the real data (new customers)
    churn_p = 1 / (1 + np.exp((df["tenure"] - 20) / 10))
    df["Churn"] = np.where(rng.uniform(size=N_ROWS) < churn_p, "Yes", "No")
    return df


@pytest.fixture(scope="session")
def session_env():
    """Make sure a developer's .env / shell does not redirect tests to the real DB."""
    mp = pytest.MonkeyPatch()
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MODEL_NAME", "MODEL_ALIAS", "MODEL_PATH"):
        mp.delenv(var, raising=False)
    mp.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def tmp_config(tmp_path_factory, base_config, synthetic_df, session_env) -> str:
    """Config copy whose data / MLflow / artifact paths all live under a temp dir.
    Session-scoped so the (slow) training happens once for the whole test run."""
    tmp_path = tmp_path_factory.mktemp("mlops")
    cfg = copy.deepcopy(base_config)
    processed = tmp_path / "processed.csv"
    synthetic_df.to_csv(processed, index=False)
    cfg["data"]["processed_path"] = str(processed)
    cfg["data"]["raw_path"] = str(tmp_path / "raw.csv")
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    cfg["mlflow"]["experiment_name"] = "test-exp"
    cfg["mlflow"]["registered_model_name"] = "TestChurnClassifier"
    cfg["serving"]["model_export_path"] = str(tmp_path / "artifacts" / "model.joblib")
    cfg["evaluation"]["reports_dir"] = str(tmp_path / "reports")
    # Tiny grid so the test stays fast.
    cfg["model"]["param_grid"]["logreg"] = {"C": [1.0], "solver": ["lbfgs"]}
    cfg["cv"]["n_splits"] = 3
    cfg["cv"]["n_jobs"] = 1
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


@pytest.fixture(scope="session")
def trained_run(tmp_config) -> tuple[str, dict]:
    """Run the real training stage once against the temp MLflow backend."""
    from src.train import train

    run_id = train(tmp_config)
    return run_id, load_config(tmp_config)
