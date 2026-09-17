"""End-to-end test of the training + evaluation stages against a temporary
MLflow SQLite backend (slow-ish: a couple of seconds)."""

from pathlib import Path

import mlflow
import pandas as pd

from src.evaluate import evaluate
from src.utils import coerce_feature_types


def test_train_logs_everything_to_mlflow_and_registers_model(trained_run):
    run_id, cfg = trained_run
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)

    # params & metrics
    assert "best_model__C" in run.data.params
    assert "best_cv_roc_auc" in run.data.metrics
    assert 0.0 <= run.data.metrics["test_roc_auc"] <= 1.0
    assert run.data.tags["model_type"] == "logreg"

    # artifacts: config + autolog outputs
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"config", "cv_results.csv"} <= artifacts

    # MLflow 3: the model is a LoggedModel entity linked to the run
    logged = mlflow.search_logged_models(experiment_ids=[run.info.experiment_id], output_format="list")
    assert any(m.source_run_id == run_id for m in logged)

    # registry: version 1 exists and carries the alias
    name = cfg["mlflow"]["registered_model_name"]
    mv = client.get_model_version_by_alias(name, cfg["mlflow"]["alias"])
    assert mv.run_id == run_id
    assert mv.tags["model_type"] == "logreg"

    # local export for serving
    export = Path(cfg["serving"]["model_export_path"])
    assert export.exists()
    assert (export.parent / "model_info.json").exists()


def test_registered_model_predicts(trained_run):
    _, cfg = trained_run
    model = mlflow.sklearn.load_model(
        f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['alias']}"
    )
    df = pd.read_csv(cfg["data"]["processed_path"])
    X = coerce_feature_types(df, cfg)[cfg["features"]["numeric"] + cfg["features"]["categorical"]]
    assert model.predict_proba(X.head(3)).shape == (3, 2)


def test_evaluate_logs_artifacts_on_training_run(trained_run, tmp_config):
    run_id, cfg = trained_run
    metrics = evaluate(tmp_config)
    assert 0.0 <= metrics["eval_roc_auc"] <= 1.0

    client = mlflow.MlflowClient()
    eval_artifacts = {Path(a.path).name for a in client.list_artifacts(run_id, "evaluation")}
    assert {"roc_curve.png", "pr_curve.png", "confusion_matrix.png", "test_predictions.csv"} <= eval_artifacts
    assert "eval_roc_auc" in client.get_run(run_id).data.metrics
