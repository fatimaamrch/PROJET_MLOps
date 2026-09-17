import numpy as np
import pytest

from src.pipeline import SUPPORTED_MODELS, build_pipeline, to_search_grid
from src.utils import split_features_target


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert list(pipe.named_steps) == ["pre", "model"]


@pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
def test_pipeline_fits_and_predicts_probabilities(model_type, synthetic_df, base_config):
    X, y = split_features_target(synthetic_df, base_config)
    pipe = build_pipeline(
        base_config["features"]["numeric"], base_config["features"]["categorical"], model_type
    )
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(pipe.predict(X)) <= {0, 1}


def test_pipeline_handles_missing_and_unknown_categories(synthetic_df, base_config):
    X, y = split_features_target(synthetic_df, base_config)
    pipe = build_pipeline(
        base_config["features"]["numeric"], base_config["features"]["categorical"], "logreg"
    ).fit(X, y)
    X_new = X.head(2).copy()
    X_new.loc[X_new.index[0], "TotalCharges"] = np.nan  # imputer
    X_new.loc[X_new.index[1], "Contract"] = "Unseen"  # handle_unknown="ignore"
    assert pipe.predict_proba(X_new).shape == (2, 2)


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(["a"], ["b"], "svm")


def test_to_search_grid_prefixes_step_name():
    assert to_search_grid({"C": [1, 2]}) == {"model__C": [1, 2]}
