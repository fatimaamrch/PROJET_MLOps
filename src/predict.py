"""Batch inference: score a CSV with the registered model (or the local export).

Usage:
    python src/predict.py --input data/processed/telco_churn_clean.csv --output reports/predictions.csv
    python src/predict.py --input some.csv --source local   # use artifacts/model.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import pandas as pd

from src.utils import coerce_feature_types, load_config, resolve, setup_mlflow


def load_model(cfg: dict, source: str = "registry"):
    """Load from the MLflow registry alias, or from the joblib export (offline / Docker)."""
    if source == "registry":
        setup_mlflow(cfg)
        uri = f"models:/{cfg['mlflow']['registered_model_name']}@{cfg['mlflow']['alias']}"
        print(f"[predict] loading {uri}")
        return mlflow.sklearn.load_model(uri)
    path = resolve(cfg["serving"]["model_export_path"])
    print(f"[predict] loading {path}")
    return joblib.load(path)


def predict_df(model, df: pd.DataFrame, cfg: dict, threshold: float | None = None) -> pd.DataFrame:
    threshold = cfg["evaluation"]["threshold"] if threshold is None else threshold
    features = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X = coerce_feature_types(df, cfg)[features]
    proba = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["churn_proba"] = proba
    out["churn_pred"] = (proba >= threshold).astype(int)
    return out


def main(input_path: str, output_path: str | None, config_path: str | None, source: str) -> Path:
    cfg = load_config(config_path)
    model = load_model(cfg, source)
    df = pd.read_csv(input_path)
    scored = predict_df(model, df, cfg)
    out = Path(output_path) if output_path else Path(input_path).with_name("predictions.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    print(f"[predict] {len(scored)} rows scored -> {out}  (positive rate={scored['churn_pred'].mean():.3f})")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with the feature columns")
    parser.add_argument("--output", default=None, help="where to write the scored CSV")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--source", choices=["registry", "local"], default="registry")
    args = parser.parse_args()
    main(args.input, args.output, args.config, args.source)
