import numpy as np
import pandas as pd
import pytest

from src.logistic_model import (
    LogisticConfig,
    binary_cross_entropy,
    calibration_table,
    fit_logistic_regression,
    loss_history_frame,
    make_prediction_frame,
    model_coefficients_frame,
    predict_labels,
    predict_proba,
    prepare_model_frame,
    sigmoid,
    time_ordered_split,
    train_event_logistic_model,
    validation_metrics,
    walk_forward_splits,
)


def separable_frame(n=80):
    rng = np.random.default_rng(7)
    x1 = np.linspace(-2.5, 2.5, n)
    x2 = rng.normal(0.0, 0.4, n)
    score = 1.4 * x1 - 0.5 * x2
    label = (score > 0.0).astype(int)
    return pd.DataFrame(
        {
            "event_id": [f"evt_{i:03d}" for i in range(n)],
            "triplet_id": "A_B_C",
            "method": "kalman_random_walk",
            "event_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "x1": x1,
            "x2": x2,
            "label": label,
        }
    )


def test_sigmoid_is_numerically_stable():
    values = np.array([-1000.0, 0.0, 1000.0])
    out = sigmoid(values)

    assert np.all(np.isfinite(out))
    assert out[0] < 1e-10
    assert out[1] == pytest.approx(0.5)
    assert out[2] > 1.0 - 1e-10


def test_binary_cross_entropy_penalizes_bad_probabilities():
    y = np.array([1, 0, 1, 0])
    good = np.array([0.9, 0.1, 0.8, 0.2])
    bad = 1.0 - good

    assert binary_cross_entropy(y, good) < binary_cross_entropy(y, bad)


def test_logistic_fit_reduces_loss_and_predicts_probabilities():
    frame = separable_frame()
    cfg = LogisticConfig(learning_rate=0.15, max_iter=1_000, l2_penalty=0.01)

    model = fit_logistic_regression(frame[["x1", "x2"]], frame["label"], config=cfg)
    probabilities = predict_proba(model, frame[["x1", "x2"]])
    labels = predict_labels(model, frame[["x1", "x2"]])

    assert model.losses[-1] < model.losses[0]
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert (labels == frame["label"].to_numpy()).mean() > 0.9


def test_l2_regularization_reduces_coefficient_norm():
    frame = separable_frame()
    weak = fit_logistic_regression(frame[["x1", "x2"]], frame["label"], config=LogisticConfig(learning_rate=0.15, max_iter=600, l2_penalty=0.0))
    strong = fit_logistic_regression(frame[["x1", "x2"]], frame["label"], config=LogisticConfig(learning_rate=0.15, max_iter=600, l2_penalty=20.0))

    assert np.linalg.norm(strong.coefficients) < np.linalg.norm(weak.coefficients)


def test_time_ordered_split_does_not_shuffle_events():
    frame = separable_frame(30).sample(frac=1.0, random_state=3).reset_index(drop=True)
    split = time_ordered_split(frame, train_size=0.5, validation_size=0.25)

    assert split.train["event_date"].max() < split.validation["event_date"].min()
    assert split.validation["event_date"].max() < split.test["event_date"].min()


def test_walk_forward_splits_expand_training_set():
    frame = separable_frame(24)
    splits = walk_forward_splits(frame, initial_train_size=10, validation_size=4, step_size=5)

    assert len(splits) == 3
    assert splits[1][0].shape[0] > splits[0][0].shape[0]
    assert splits[0][0]["event_date"].max() < splits[0][1]["event_date"].min()


def test_prediction_and_metric_outputs_have_expected_schema():
    frame = separable_frame(60)
    split = time_ordered_split(frame)
    model = fit_logistic_regression(split.train[["x1", "x2"]], split.train["label"], config=LogisticConfig(max_iter=300))

    predictions = make_prediction_frame(split.validation, model, ["x1", "x2"], split_name="validation")
    metrics = validation_metrics(predictions["label"], predictions["predicted_reversion_probability"])
    coefficients = model_coefficients_frame(model)
    losses = loss_history_frame(model)
    calibration = calibration_table(predictions["label"], predictions["predicted_reversion_probability"], n_bins=3)

    assert {"event_id", "predicted_reversion_probability", "predicted_label", "label"}.issubset(predictions.columns)
    assert {"accuracy", "log_loss", "brier_score", "roc_auc"}.issubset(metrics.columns)
    assert coefficients.iloc[0]["feature"] == "intercept"
    assert {"iteration", "loss"}.issubset(losses.columns)
    assert {"probability_bucket", "observed_success_rate"}.issubset(calibration.columns)


def test_training_pipeline_returns_model_outputs():
    frame = separable_frame(70)
    outputs = train_event_logistic_model(frame, config=LogisticConfig(learning_rate=0.1, max_iter=500, l2_penalty=0.1))

    assert set(outputs) == {
        "model",
        "feature_columns",
        "split",
        "model_coefficients",
        "training_loss",
        "predicted_reversion_probabilities",
        "validation_metrics",
    }
    assert not outputs["model_coefficients"].empty
    assert not outputs["predicted_reversion_probabilities"].empty


def test_prepare_model_frame_imputes_numeric_features_and_rejects_missing_label():
    frame = separable_frame(20)
    frame.loc[0, "x1"] = np.nan

    prepared, cols = prepare_model_frame(frame, feature_columns=["x1", "x2"])

    assert cols == ["x1", "x2"]
    assert prepared["x1"].isna().sum() == 0
    with pytest.raises(KeyError):
        prepare_model_frame(frame.drop(columns=["label"]))


def test_invalid_config_is_rejected():
    frame = separable_frame(10)
    with pytest.raises(ValueError):
        fit_logistic_regression(frame[["x1"]], frame["label"], config=LogisticConfig(learning_rate=-0.1))
    with pytest.raises(ValueError):
        validation_metrics(frame["label"], np.full(frame.shape[0], 0.5), threshold=1.2)
