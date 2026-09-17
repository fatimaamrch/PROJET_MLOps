"""Data stage: download the raw CSV and produce a clean, typed dataset.

Usage:
    python src/data.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import pandas as pd

from src.utils import load_config, resolve


def download(url: str, dest: Path) -> Path:
    """Fetch the CSV once; skip if it is already on disk (keeps the demo offline-friendly)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[data] raw file already present: {dest}")
        return dest
    print(f"[data] downloading {url}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - trusted public URL from config
    return dest


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Telco-specific cleaning, kept deliberately small and explicit.

    - TotalCharges is stored as text with blanks for brand-new customers -> numeric + NaN
      (the pipeline's imputer handles the NaN).
    - SeniorCitizen is 0/1 int but semantically categorical -> string.
    - Drop the customer id (leak-free, non predictive).
    - Drop exact duplicates.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    id_col = cfg["data"].get("id_column")
    if id_col and id_col in df.columns:
        df = df.drop(columns=[id_col])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def validate(df: pd.DataFrame, cfg: dict) -> None:
    """Fail fast if the dataset does not match the config (schema check)."""
    expected = set(cfg["features"]["numeric"] + cfg["features"]["categorical"] + [cfg["data"]["target"]])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    labels = set(df[cfg["data"]["target"]].unique())
    if cfg["data"]["positive_label"] not in labels:
        raise ValueError(
            f"positive_label {cfg['data']['positive_label']!r} not found in target values {labels}"
        )


def main(config_path: str | None = None) -> Path:
    cfg = load_config(config_path)
    raw = download(cfg["data"]["url"], resolve(cfg["data"]["raw_path"]))
    df = pd.read_csv(raw)
    df = clean(df, cfg)
    validate(df, cfg)
    out = resolve(cfg["data"]["processed_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    target = cfg["data"]["target"]
    print(f"[data] processed dataset written to {out}  shape={df.shape}")
    print(f"[data] target distribution:\n{df[target].value_counts(normalize=True).round(3).to_string()}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args()
    main(args.config)
