import numpy as np
import pandas as pd
import pytest

from src.kalman import (
    KalmanConfig,
    compare_kalman_residuals,
    estimate_kalman_for_triplets,
    kalman_filter_dynamic_regression,
    kalman_predict,
    kalman_update,
)
from src.regression import fit_ols, ols_predict, rolling_ols


def _sample_log_price_frame(periods=80):
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    h1 = 4.0 + np.cumsum(0.01 + 0.002 * np.sin(np.arange(periods) / 5.0))
    h2 = 3.0 + np.cumsum(0.008 + 0.002 * np.cos(np.arange(periods) / 7.0))
    beta_1 = 0.6 + 0.12 * np.sin(np.arange(periods) / 20.0)
    beta_2 = 0.25 + 0.08 * np.cos(np.arange(periods) / 18.0)
    target = 0.15 + beta_1 * h1 + beta_2 * h2 + 0.002 * np.sin(np.arange(periods))
    return pd.DataFrame({"A": target, "B": h1, "C": h2}, index=idx)


def test_kalman_predict_adds_process_noise_to_covariance():
    state = np.array([0.1, 0.5, 0.2])
    covariance = np.eye(3)

    predicted_state, predicted_covariance = kalman_predict(state, covariance, process_noise=0.25)

    np.testing.assert_allclose(predicted_state, state)
    np.testing.assert_allclose(predicted_covariance, np.eye(3) * 1.25)


def test_kalman_update_returns_finite_state_and_residual():
    state = np.array([0.1, 0.5, 0.2])
    covariance = np.eye(3)
    observation_vector = np.array([1.0, 2.0, 3.0])

    out = kalman_update(state, covariance, 2.0, observation_vector, measurement_noise=0.1)

    assert out.filtered_state.shape == (3,)
    assert out.filtered_covariance.shape == (3, 3)
    assert np.isfinite(out.innovation)
    assert out.innovation_variance > 0
    assert out.kalman_gain.shape == (3,)


def test_kalman_filter_returns_expected_rows_and_columns():
    frame = _sample_log_price_frame(90)
    config = KalmanConfig(process_noise=1e-5, measurement_noise=1e-4, initial_window=20)

    out = kalman_filter_dynamic_regression(frame, "A", ["B", "C"], config=config, triplet_id="A_B_C")

    assert out.shape[0] == 70
    assert out.iloc[0]["date"] == frame.index[20]
    assert out.iloc[0]["triplet_id"] == "A_B_C"
    assert {"alpha", "beta_1", "beta_2", "residual", "residual_variance"}.issubset(out.columns)
    assert out["method"].unique().tolist() == ["kalman_random_walk"]


def test_kalman_first_residual_is_one_step_ahead_prediction_error():
    frame = _sample_log_price_frame(70)
    config = KalmanConfig(process_noise=0.0, measurement_noise=1e-4, initial_window=15)

    out = kalman_filter_dynamic_regression(frame, "A", ["B", "C"], config=config)
    initial_fit = fit_ols(frame.iloc[:15][["B", "C"]], frame.iloc[:15]["A"])
    expected_fitted = ols_predict(frame.iloc[[15]][["B", "C"]], initial_fit.intercept, initial_fit.coefficients)[0]

    assert out.iloc[0]["date"] == frame.index[15]
    assert out.iloc[0]["fitted_log_price"] == pytest.approx(expected_fitted)
    assert out.iloc[0]["residual"] == pytest.approx(frame.iloc[15]["A"] - expected_fitted)


def test_kalman_filter_rejects_invalid_noise_and_state():
    frame = _sample_log_price_frame(20)

    with pytest.raises(ValueError):
        kalman_filter_dynamic_regression(
            frame,
            "A",
            ["B", "C"],
            config=KalmanConfig(process_noise=1e-5, measurement_noise=0.0),
        )

    with pytest.raises(ValueError):
        kalman_filter_dynamic_regression(frame, "A", ["B", "C"], initial_state=[0.0, 1.0])


def test_estimate_kalman_for_triplets_returns_state_and_residual_tables():
    frame = _sample_log_price_frame(80)
    triplets = [{"triplet_id": "A_B_C", "target": "A", "hedge_1": "B", "hedge_2": "C"}]
    config = KalmanConfig(process_noise=1e-5, measurement_noise=1e-4, initial_window=20)

    tables = estimate_kalman_for_triplets(frame, triplets, config=config)

    assert set(tables) == {"kalman_states", "kalman_residuals"}
    assert not tables["kalman_states"].empty
    assert not tables["kalman_residuals"].empty
    assert "actual_log_price" not in tables["kalman_states"].columns
    assert "state_cov_trace" not in tables["kalman_residuals"].columns


def test_compare_kalman_residuals_includes_rolling_and_kalman_methods():
    frame = _sample_log_price_frame(75)
    rolling = rolling_ols(frame, "A", ["B", "C"], window=20, triplet_id="A_B_C")
    kalman = kalman_filter_dynamic_regression(
        frame,
        "A",
        ["B", "C"],
        config=KalmanConfig(process_noise=1e-5, measurement_noise=1e-4, initial_window=20),
        triplet_id="A_B_C",
    )

    summary = compare_kalman_residuals(kalman, rolling)

    assert set(summary["method"]) == {"rolling_ols", "kalman_random_walk"}
    assert "residual_std" in summary.columns
