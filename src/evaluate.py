"""Evaluation stage: score the registered model on the hold-out set and log
ROC / PR / confusion-matrix plots + classification report + predictions CSV
as artifacts *on the same MLflow run that produced the model*.

Usage:
    python src/evaluate.py --config configs/config.yaml [--alias champion]
"""

from __future__ import annotations

import argparse
import json

import mlflow
from sklearn.metrics import classification_report, roc_auc_score

from src.utils import (
    load_config,
    load_processed,
    resolve,
    save_evaluation_plots,
    setup_mlflow,
    split_features_target,
    train_test_split_cfg,
)


def load_registered_model(name: str, alias: str):
    """Return (pyfunc-free sklearn pipeline, model version object)."""
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, alias)
    model = mlflow.sklearn.load_model(f"models:/{name}@{alias}")
    return model, mv


def evaluate(config_path: str | None = None, alias: str | None = None) -> dict:
    cfg = load_config(config_path)
    setup_mlflow(cfg)
    name = cfg["mlflow"]["registered_model_name"]
    alias = alias or cfg["mlflow"]["alias"]
    threshold = cfg["evaluation"]["threshold"]

    model, mv = load_registered_model(name, alias)
    print(f"[evaluate] loaded {name} v{mv.version} (@{alias}) from run {mv.run_id}")

    # Same deterministic split as training -> the test set was never seen by the model.
    df = load_processed(cfg)
    X, y = split_features_target(df, cfg)
    _, X_test, _, y_test = train_test_split_cfg(X, y, cfg)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    metrics = {
        "eval_roc_auc": roc_auc_score(y_test, proba),
        "eval_accuracy": report["accuracy"],
        "eval_precision": report["1"]["precision"],
        "eval_recall": report["1"]["recall"],
        "eval_f1": report["1"]["f1-score"],
        "eval_threshold": threshold,
    }

    reports_dir = resolve(cfg["evaluation"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = save_evaluation_plots(y_test, pred, proba, reports_dir)
    (reports_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds_path = reports_dir / "test_predictions.csv"
    X_test.assign(y_true=y_test.values, y_proba=proba, y_pred=pred).to_csv(preds_path, index=False)

    # Attach everything to the training run for full traceability.
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("evaluated_version", mv.version)
        for p in plot_paths:
            mlflow.log_artifact(str(p), artifact_path="evaluation")
        mlflow.log_artifact(str(reports_dir / "classification_report.json"), artifact_path="evaluation")
        mlflow.log_artifact(str(reports_dir / "metrics.json"), artifact_path="evaluation")
        mlflow.log_artifact(str(preds_path), artifact_path="evaluation")

    print("[evaluate] " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"[evaluate] reports written to {reports_dir} and logged to MLflow run {mv.run_id}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--alias", default=None, help="registry alias to evaluate (default: config)")
    args = parser.parse_args()
    evaluate(args.config, args.alias)
