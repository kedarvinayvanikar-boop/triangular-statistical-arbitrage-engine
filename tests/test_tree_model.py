import numpy as np
import pandas as pd
import pytest

from src.tree_model import (
    DecisionTreeConfig,
    compare_tree_to_logistic,
    dataset_large_enough,
    entropy_impurity,
    feature_split_summary,
    fit_decision_tree,
    gini_impurity,
    information_gain,
    leaf_summary,
    make_tree_prediction_frame,
    predict_labels_tree,
    predict_proba_tree,
    prepare_tree_frame,
    train_event_decision_tree,
    weighted_impurity,
)


def tree_frame(n=160):
    rng = np.random.default_rng(11)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(0.0, 1.0, n)
    x3 = rng.normal(0.0, 1.0, n)
    nonlinear = ((x1 > 0.3) & (x2 < 0.6)) | ((x1 < -0.8) & (x3 > 0.0))
    label = nonlinear.astype(int)
    return pd.DataFrame(
        {
            "event_id": [f"evt_{idx:03d}" for idx in range(n)],
            "triplet_id": "NVDA_SMH_QQQ",
            "method": "kalman_random_walk",
            "event_date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "residual_z_score": x1,
            "residual_volatility": x2,
            "beta_stability": x3,
            "label": label,
        }
    )


def test_impurity_measures_are_zero_for_pure_nodes():
    pure = np.ones(10, dtype=int)

    assert gini_impurity(pure) == pytest.approx(0.0)
    assert entropy_impurity(pure) == pytest.approx(0.0)


def test_information_gain_is_positive_for_useful_split():
    y_parent = np.array([0, 0, 0, 1, 1, 1])
    y_left = np.array([0, 0, 0])
    y_right = np.array([1, 1, 1])

    assert weighted_impurity(y_left, y_right, criterion="gini") == pytest.approx(0.0)
    assert information_gain(y_parent, y_left, y_right, criterion="gini") > 0.0


def test_decision_tree_fits_nonlinear_rule_and_predicts_probabilities():
    frame = tree_frame()
    cfg = DecisionTreeConfig(max_depth=4, min_samples_split=20, min_samples_leaf=5)
    model = fit_decision_tree(frame[["residual_z_score", "residual_volatility", "beta_stability"]], frame["label"], config=cfg)

    probabilities = predict_proba_tree(model, frame[["residual_z_score", "residual_volatility", "beta_stability"]])
    labels = predict_labels_tree(model, frame[["residual_z_score", "residual_volatility", "beta_stability"]])

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert (labels == frame["label"].to_numpy()).mean() > 0.75


def test_feature_split_and_leaf_summaries_have_expected_schema():
    frame = tree_frame()
    model = fit_decision_tree(frame[["residual_z_score", "residual_volatility"]], frame["label"], config=DecisionTreeConfig(max_depth=3, min_samples_split=20, min_samples_leaf=5))
    splits = feature_split_summary(model)
    leaves = leaf_summary(model)

    assert {"feature", "threshold", "information_gain", "n_samples"}.issubset(splits.columns)
    assert {"node_id", "predicted_probability", "n_samples"}.issubset(leaves.columns)
    assert not leaves.empty


def test_tree_prediction_frame_preserves_event_metadata():
    frame = tree_frame()
    features = ["residual_z_score", "residual_volatility", "beta_stability"]
    model = fit_decision_tree(frame[features], frame["label"], config=DecisionTreeConfig(max_depth=3, min_samples_split=20, min_samples_leaf=5))
    predictions = make_tree_prediction_frame(frame, model, features, split_name="validation")

    assert {"event_id", "split", "model_type", "predicted_reversion_probability", "predicted_label", "label"}.issubset(predictions.columns)
    assert predictions["model_type"].eq("decision_tree").all()


def test_prepare_tree_frame_imputes_missing_numeric_features():
    frame = tree_frame(40)
    frame.loc[0, "residual_z_score"] = np.nan
    prepared, features = prepare_tree_frame(frame, feature_columns=["residual_z_score", "residual_volatility"])

    assert features == ["residual_z_score", "residual_volatility"]
    assert prepared["residual_z_score"].isna().sum() == 0


def test_dataset_size_gate_rejects_small_event_samples():
    small = tree_frame(30)

    assert not dataset_large_enough(small, min_events=100, min_positive=10, min_negative=10)
    with pytest.raises(ValueError):
        train_event_decision_tree(small, min_events_required=100)


def test_training_pipeline_returns_expected_outputs():
    frame = tree_frame(180)
    outputs = train_event_decision_tree(
        frame,
        feature_columns=["residual_z_score", "residual_volatility", "beta_stability"],
        config=DecisionTreeConfig(max_depth=4, min_samples_split=20, min_samples_leaf=5),
        min_events_required=100,
    )

    assert set(outputs) == {
        "model",
        "feature_columns",
        "decision_tree_predictions",
        "decision_tree_validation_metrics",
        "feature_split_summary",
        "leaf_summary",
    }
    assert not outputs["decision_tree_predictions"].empty
    assert not outputs["decision_tree_validation_metrics"].empty


def test_tree_can_be_compared_to_logistic_prediction_frame():
    frame = tree_frame(120)
    features = ["residual_z_score", "residual_volatility", "beta_stability"]
    model = fit_decision_tree(frame[features], frame["label"], config=DecisionTreeConfig(max_depth=3, min_samples_split=20, min_samples_leaf=5))
    tree_predictions = make_tree_prediction_frame(frame, model, features, split_name="validation")
    logistic_predictions = tree_predictions.copy()
    logistic_predictions["predicted_reversion_probability"] = 0.5

    comparison = compare_tree_to_logistic(tree_predictions, logistic_predictions, threshold=0.5)

    assert set(comparison["model_type"]) == {"decision_tree", "logistic_regression"}
    assert {"accuracy", "precision", "recall", "brier_score"}.issubset(comparison.columns)


def test_invalid_tree_config_is_rejected():
    frame = tree_frame(20)
    with pytest.raises(ValueError):
        fit_decision_tree(frame[["residual_z_score"]], frame["label"], config=DecisionTreeConfig(max_depth=0))
    with pytest.raises(ValueError):
        fit_decision_tree(frame[["residual_z_score"]], frame["label"], config=DecisionTreeConfig(criterion="bad"))
