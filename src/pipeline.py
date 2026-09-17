"""Build the scikit-learn Pipeline: preprocessing (ColumnTransformer) + estimator.

Keeping preprocessing inside the Pipeline guarantees that the exact same
transformations are applied at training, evaluation and serving time.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUPPORTED_MODELS = ("logreg", "random_forest")


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
    )


def build_estimator(model_type: str, random_state: int = 42):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unsupported model_type {model_type!r}. Choose from {SUPPORTED_MODELS}")


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    random_state: int = 42,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(numeric, categorical)),
            ("model", build_estimator(model_type, random_state)),
        ]
    )


def to_search_grid(param_grid: dict) -> dict:
    """Prefix params with the pipeline step name so GridSearchCV can address them."""
    return {f"model__{k}": v for k, v in param_grid.items()}
