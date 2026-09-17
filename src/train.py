"""Training stage: hyper-parameter search + full MLflow tracking + Model Registry.

What gets stored in MLflow for every run:
  - params      : every pipeline/estimator param + the CV grid (autolog)
  - metrics     : CV score per candidate (child runs), best CV score, hold-out metrics
  - artifacts   : config.yaml, the fitted model (with signature + input example),
                  the dataset reference (hash + schema)
  - registry    : a new version of the registered model, given the configured alias

Usage:
    python src/train.py --config configs/config.yaml [--model-type random_forest]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline, to_search_grid
from src.utils import (
    DEFAULT_CONFIG,
    load_config,
    load_processed,
    resolve,
    setup_mlflow,
    split_features_target,
    train_test_split_cfg,
)


def model_requirements() -> list[str]:
    """Exact versions of the libraries the serialized pipeline depends on."""
    import cloudpickle
    import numpy
    import pandas
    import sklearn

    return [
        f"scikit-learn=={sklearn.__version__}",
        f"pandas=={pandas.__version__}",
        f"numpy=={numpy.__version__}",
        f"cloudpickle=={cloudpickle.__version__}",
    ]


def holdout_metrics(model, X, y, prefix: str = "test") -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return {
        f"{prefix}_roc_auc": roc_auc_score(y, proba),
        f"{prefix}_accuracy": accuracy_score(y, pred),
        f"{prefix}_precision": precision_score(y, pred, zero_division=0),
        f"{prefix}_recall": recall_score(y, pred),
        f"{prefix}_f1": f1_score(y, pred),
    }


def train(config_path: str | None = None, model_type: str | None = None) -> str:
    cfg = load_config(config_path)
    model_type = model_type or cfg["model"]["type"]
    experiment = setup_mlflow(cfg)

    df = load_processed(cfg)
    X, y = split_features_target(df, cfg)
    X_train, X_test, y_train, y_test = train_test_split_cfg(X, y, cfg)

    pipe = build_pipeline(
        cfg["features"]["numeric"],
        cfg["features"]["categorical"],
        model_type,
        random_state=cfg["data"]["random_state"],
    )
    grid = to_search_grid(cfg["model"]["param_grid"][model_type])
    cv = StratifiedKFold(
        n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["data"]["random_state"]
    )
    search = GridSearchCV(
        pipe,
        grid,
        cv=cv,
        scoring=cfg["cv"]["scoring"],
        n_jobs=cfg["cv"]["n_jobs"],
        refit=True,
        return_train_score=True,
    )

    # Autolog captures params, CV results (incl. child runs per candidate) and
    # cv_results_.csv. We log the final model ourselves to control signature/registry.
    mlflow.sklearn.autolog(log_models=False, log_datasets=False, max_tuning_runs=None)

    run_name = f"{model_type}-gridsearch"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "stage": "train",
                "model_type": model_type,
                "dataset": Path(cfg["data"]["processed_path"]).name,
            }
        )
        # Dataset lineage: MLflow computes a digest + schema so every run
        # records exactly which data it saw.
        train_ds = mlflow.data.from_pandas(
            X_train.assign(**{cfg["data"]["target"]: y_train}),
            source=str(resolve(cfg["data"]["processed_path"])),
            targets=cfg["data"]["target"],
            name="telco-train",
        )
        mlflow.log_input(train_ds, context="training")
        mlflow.log_artifact(str(Path(config_path) if config_path else DEFAULT_CONFIG), "config")

        print(f"[train] GridSearchCV on {len(X_train)} rows, model={model_type}, grid={grid}")
        search.fit(X_train, y_train)
        best = search.best_estimator_

        mlflow.log_metric("best_cv_" + cfg["cv"]["scoring"], search.best_score_)
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})

        metrics = holdout_metrics(best, X_test, y_test)
        mlflow.log_metrics(metrics)
        print(f"[train] best params: {search.best_params_}")
        print(f"[train] best CV {cfg['cv']['scoring']}: {search.best_score_:.4f}")
        print("[train] hold-out: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        # Log + register the best pipeline (preprocessing included).
        signature = infer_signature(X_train, best.predict(X_train))
        registered_name = cfg["mlflow"]["registered_model_name"]
        info = mlflow.sklearn.log_model(
            best,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=registered_name,
            serialization_format="cloudpickle",
            # Pin the exact runtime deps instead of letting MLflow infer them in a
            # subprocess (slow, and inference may drift between machines).
            pip_requirements=model_requirements(),
        )

        # Point the alias to this fresh version so predict.py / the API pick it up.
        client = mlflow.MlflowClient()
        version = info.registered_model_version
        alias = cfg["mlflow"]["alias"]
        client.set_registered_model_alias(registered_name, alias, version)
        client.set_model_version_tag(
            registered_name, version, "test_roc_auc", f"{metrics['test_roc_auc']:.4f}"
        )
        client.set_model_version_tag(registered_name, version, "model_type", model_type)

        # Local export for the API / Docker image (no registry needed at serve time).
        export_path = resolve(cfg["serving"]["model_export_path"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, export_path)
        (export_path.parent / "model_info.json").write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "experiment": experiment,
                    "registered_model": registered_name,
                    "version": version,
                    "alias": alias,
                    "model_type": model_type,
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        print(f"[train] run_id={run.info.run_id}  registered {registered_name} v{version} (@{alias})")
        print(f"[train] model exported to {export_path}")
        return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--model-type", default=os.getenv("MODEL_TYPE"), help="override model.type")
    args = parser.parse_args()
    train(args.config, args.model_type)
