import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    brier_score,
    calibration_curve_frame,
    classification_summary,
    confusion_matrix_frame,
    evaluate_predictions,
    probability_bucket_summary,
    roc_auc_score,
    roc_curve_frame,
)
from src.plotting import (
    plot_confusion_matrix,
    plot_precision_by_probability_bucket,
    plot_probability_calibration_curve,
)


def sample_predictions():
    return pd.DataFrame(
        {
            "event_id": [f"evt_{i:03d}" for i in range(10)],
            "split": ["validation"] * 5 + ["test"] * 5,
            "label": [0, 0, 1, 1, 1, 0, 1, 0, 1, 1],
            "predicted_reversion_probability": [0.05, 0.20, 0.35, 0.65, 0.90, 0.15, 0.45, 0.55, 0.80, 0.95],
        }
    )


def test_confusion_matrix_counts_are_correct():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.7, 0.8, 0.2])

    matrix = confusion_matrix_frame(y, p, threshold=0.5)

    assert matrix["count"].sum() == 4
    assert int(matrix.query("actual_label == 1 and predicted_label == 1")["count"].iloc[0]) == 1
    assert int(matrix.query("actual_label == 0 and predicted_label == 1")["count"].iloc[0]) == 1


def test_classification_summary_contains_core_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.8, 0.9])

    summary = classification_summary(y, p, threshold=0.5, split="validation")

    assert summary.loc[0, "split"] == "validation"
    assert summary.loc[0, "accuracy"] == pytest.approx(1.0)
    assert summary.loc[0, "precision"] == pytest.approx(1.0)
    assert summary.loc[0, "recall"] == pytest.approx(1.0)
    assert summary.loc[0, "brier_score"] == pytest.approx(brier_score(y, p))
    assert 0.0 <= summary.loc[0, "roc_auc"] <= 1.0


def test_roc_auc_orders_good_model_above_bad_model():
    y = np.array([0, 0, 1, 1])
    good = np.array([0.1, 0.2, 0.8, 0.9])
    bad = 1.0 - good

    assert roc_auc_score(y, good) == pytest.approx(1.0)
    assert roc_auc_score(y, bad) == pytest.approx(0.0)
    assert {"false_positive_rate", "true_positive_rate"}.issubset(roc_curve_frame(y, good).columns)


def test_probability_bucket_and_calibration_outputs_have_expected_schema():
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])

    buckets = probability_bucket_summary(y, p, n_bins=3)
    calibration = calibration_curve_frame(y, p, n_bins=3)

    assert {"probability_bucket", "precision", "mean_predicted_probability"}.issubset(buckets.columns)
    assert {"calibration_error", "absolute_calibration_error"}.issubset(calibration.columns)
    assert buckets["n_events"].sum() == len(y)


def test_evaluate_predictions_groups_by_split():
    predictions = sample_predictions()

    outputs = evaluate_predictions(predictions, threshold=0.5, n_bins=4)

    assert set(outputs) == {
        "model_evaluation_summary",
        "confusion_matrix",
        "probability_bucket_summary",
        "calibration_curve",
        "roc_curve",
    }
    assert set(outputs["model_evaluation_summary"]["split"]) == {"validation", "test"}
    assert set(outputs["confusion_matrix"]["split"]) == {"validation", "test"}


def test_plotting_functions_create_files(tmp_path):
    predictions = sample_predictions()
    outputs = evaluate_predictions(predictions, threshold=0.5, n_bins=4)

    calibration_path = plot_probability_calibration_curve(outputs["calibration_curve"], tmp_path / "calibration.png", split="validation")
    bucket_path = plot_precision_by_probability_bucket(outputs["probability_bucket_summary"], tmp_path / "bucket.png", split="validation")
    matrix_path = plot_confusion_matrix(outputs["confusion_matrix"], tmp_path / "matrix.png", split="validation")

    assert calibration_path.exists()
    assert bucket_path.exists()
    assert matrix_path.exists()


def test_invalid_metric_inputs_raise_clear_errors():
    with pytest.raises(ValueError):
        classification_summary([0, 1], [0.2, 0.8], threshold=1.0)
    with pytest.raises(ValueError):
        probability_bucket_summary([0, 1], [0.2, 0.8], n_bins=1)
    assert np.isnan(roc_auc_score([1, 1], [0.2, 0.8]))
    with pytest.raises(KeyError):
        evaluate_predictions(pd.DataFrame({"label": [0, 1]}))
