"""Small shared helpers: config loading, MLflow setup, split and plots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import mlflow
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Read the YAML config. Relative paths in it are resolved from the project root."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike) -> Path:
    """Make a config path absolute relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Configure tracking URI + experiment. Env vars win over config (12-factor style)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", cfg["mlflow"]["experiment_name"])
    # Keep the SQLite file next to the project, whatever the cwd is.
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        db_file = tracking_uri.removeprefix("sqlite:///")
        tracking_uri = f"sqlite:///{resolve(db_file).as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return experiment


def load_processed(cfg: dict[str, Any]) -> pd.DataFrame:
    """Read the processed CSV and enforce dtypes (CSV round-trips lose them,
    e.g. SeniorCitizen "0"/"1" would come back as int)."""
    path = resolve(cfg["data"]["processed_path"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `make data` (python src/data.py) first.")
    df = pd.read_csv(path)
    return coerce_feature_types(df, cfg)


def coerce_feature_types(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Categorical columns -> str, numeric columns -> float. Shared by training and serving."""
    df = df.copy()
    for col in cfg["features"]["categorical"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in cfg["features"]["numeric"]:
        if col in df.columns:
            # float64 (not int) so the MLflow signature tolerates NaN at inference time
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def split_features_target(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with y encoded as 0/1 from the configured positive label."""
    target = cfg["data"]["target"]
    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    X = df[feature_cols].copy()
    y = (df[target] == cfg["data"]["positive_label"]).astype(int)
    return X, y


def train_test_split_cfg(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]):
    """Stratified hold-out split driven by the config."""
    return train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )


def save_evaluation_plots(y_true, y_pred, y_proba, out_dir: Path) -> list[Path]:
    """Write ROC, PR and confusion-matrix PNGs and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC curve")
    paths.append(_save(fig, out_dir / "roc_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Precision-Recall curve")
    paths.append(_save(fig, out_dir / "pr_curve.png"))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    paths.append(_save(fig, out_dir / "confusion_matrix.png"))

    return paths


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
