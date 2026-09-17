import pandas as pd
import pytest

from src.data import clean, validate
from src.utils import coerce_feature_types, split_features_target


def test_clean_converts_total_charges_and_drops_id(base_config):
    raw = pd.DataFrame(
        {
            "customerID": ["a", "b", "c"],
            "TotalCharges": ["10.5", " ", "20"],
            "SeniorCitizen": [0, 1, 1],
            "Churn": ["Yes", "No", "No"],
        }
    )
    out = clean(raw, base_config)
    assert "customerID" not in out.columns
    assert out["TotalCharges"].dtype.kind == "f"
    assert out["TotalCharges"].isna().sum() == 1
    assert out["SeniorCitizen"].dtype == object
    assert len(out) == 3


def test_clean_drops_exact_duplicates(base_config):
    raw = pd.DataFrame({"TotalCharges": ["20", "20"], "SeniorCitizen": [1, 1], "Churn": ["No", "No"]})
    assert len(clean(raw, base_config)) == 1


def test_validate_accepts_schema_valid_dataset(synthetic_df, base_config):
    validate(synthetic_df, base_config)


def test_validate_rejects_missing_columns(synthetic_df, base_config):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(synthetic_df.drop(columns=["tenure"]), base_config)


def test_validate_rejects_unknown_positive_label(synthetic_df, base_config):
    df = synthetic_df.copy()
    df["Churn"] = "maybe"
    with pytest.raises(ValueError, match="positive_label"):
        validate(df, base_config)


def test_split_features_target_encodes_binary_target(synthetic_df, base_config):
    X, y = split_features_target(synthetic_df, base_config)
    assert set(y.unique()) <= {0, 1}
    assert "Churn" not in X.columns
    assert list(X.columns) == base_config["features"]["numeric"] + base_config["features"]["categorical"]


def test_coerce_feature_types(base_config):
    df = pd.DataFrame({"tenure": ["3"], "SeniorCitizen": [1], "TotalCharges": ["x"]})
    out = coerce_feature_types(df, base_config)
    assert out["tenure"].dtype == "float64"
    assert out["SeniorCitizen"].iloc[0] == "1"
    assert pd.isna(out["TotalCharges"].iloc[0])
