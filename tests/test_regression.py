import numpy as np
import pandas as pd
import pytest

from src.regression import (
    estimate_dynamic_hedges_for_triplets,
    fit_ols,
    ols_predict,
    rolling_ols,
    static_out_of_sample_residuals,
)
from src.residuals import compare_residual_stability


def test_fit_ols_recovers_coefficients():
    x = np.array(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [3.0, 1.5],
            [4.0, 2.0],
            [5.0, 2.5],
            [6.0, 3.0],
        ]
    )
    y = -0.3 + 0.8 * x[:, 0] + 1.2 * x[:, 1]

    result = fit_ols(x, y)

    assert result.intercept == pytest.approx(-0.3)
    assert result.coefficients[0] == pytest.approx(0.8)
    assert result.coefficients[1] == pytest.approx(1.2)


def test_ols_predict_checks_dimensions():
    with pytest.raises(ValueError):
        ols_predict(np.ones((3, 3)), 0.0, [1.0, 2.0])


def test_rolling_ols_does_not_include_prediction_row_in_training_window():
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    b = np.linspace(1.0, 10.0, 10)
    c = np.linspace(2.0, 11.0, 10) ** 1.01
    a = 0.4 + 1.1 * b - 0.2 * c
    frame = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)

    out = rolling_ols(frame, "A", ["B", "C"], window=6)
    manual = fit_ols(frame.iloc[:6][["B", "C"]], frame.iloc[:6]["A"])
    expected = ols_predict(frame.iloc[[6]][["B", "C"]], manual.intercept, manual.coefficients)[0]

    assert out.iloc[0]["date"] == idx[6]
    assert out.iloc[0]["train_end"] == idx[5]
    assert out.iloc[0]["fitted_log_price"] == pytest.approx(expected)


def test_static_oos_residuals_align_with_rolling_dates():
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = pd.DataFrame(
        {
            "A": np.linspace(10.0, 12.0, 12),
            "B": np.linspace(7.0, 8.0, 12),
            "C": np.linspace(3.0, 4.2, 12),
        },
        index=idx,
    )

    static = static_out_of_sample_residuals(frame, "A", ["B", "C"], window=5)
    rolling = rolling_ols(frame, "A", ["B", "C"], window=5)

    assert list(static["date"]) == list(rolling["date"])
    assert static["method"].unique().tolist() == ["static_ols_initial_window"]


def test_estimate_dynamic_hedges_returns_expected_tables():
    idx = pd.date_range("2024-01-01", periods=90, freq="D")
    b = np.cumsum(np.full(90, 0.01)) + 5.0
    c = b * 0.9 + 0.02 * np.sin(np.arange(90)) + 1.0
    a = 0.2 + 0.7 * b + 0.3 * c + 0.01 * np.cos(np.arange(90))
    frame = pd.DataFrame({"A": a, "B": b, "C": c}, index=idx)
    triplets = [{"triplet_id": "A_B_C", "target": "A", "hedge_1": "B", "hedge_2": "C"}]

    tables = estimate_dynamic_hedges_for_triplets(frame, triplets, window=30, ridge_alpha=1.0)

    assert set(tables) == {"rolling_coefficients", "ridge_coefficients", "dynamic_residuals"}
    assert not tables["rolling_coefficients"].empty
    assert not tables["ridge_coefficients"].empty
    assert set(tables["dynamic_residuals"]["method"]) == {
        "static_ols_initial_window",
        "rolling_ols",
        "rolling_ridge",
    }


def test_compare_residual_stability_has_ratios():
    residuals = pd.DataFrame(
        {
            "triplet_id": ["T"] * 8,
            "method": ["static_ols_initial_window"] * 4 + ["rolling_ols"] * 4,
            "residual": [1.0, -1.0, 0.5, -0.5, 0.2, -0.1, 0.1, -0.2],
        }
    )
    out = compare_residual_stability(residuals)

    assert "std_ratio_vs_static" in out.columns
    assert set(out["method"]) == {"static_ols_initial_window", "rolling_ols"}
