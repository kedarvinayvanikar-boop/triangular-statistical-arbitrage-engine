import numpy as np
import pandas as pd
import pytest

from src.features import (
    FeatureConfig,
    build_event_feature_matrix,
    collinear_feature_pairs,
    create_feature_outputs,
    feature_correlation_matrix,
    feature_missingness_report,
    feature_summary_statistics,
    fit_feature_scaler,
    transform_feature_matrix,
)


def sample_inputs():
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    residual = np.linspace(-0.2, 0.3, len(dates)) + np.sin(np.arange(len(dates)) / 3.0) * 0.05
    residuals = pd.DataFrame(
        {
            "date": dates,
            "triplet_id": "A_B_C",
            "method": "kalman_random_walk",
            "target_symbol": "A",
            "hedge_symbol_1": "B",
            "hedge_symbol_2": "C",
            "actual_log_price": 5.0 + residual,
            "fitted_log_price": 5.0,
            "residual": residual,
            "z_score": np.linspace(-1.5, 2.5, len(dates)),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": ["evt1", "evt2"],
            "triplet_id": ["A_B_C", "A_B_C"],
            "method": ["kalman_random_walk", "kalman_random_walk"],
            "event_date": [dates[20], dates[30]],
            "target_symbol": ["A", "A"],
            "hedge_symbol_1": ["B", "B"],
            "hedge_symbol_2": ["C", "C"],
            "side": ["short_spread", "short_spread"],
            "entry_z_score": [2.1, 2.3],
            "entry_abs_z": [2.1, 2.3],
        }
    )
    returns = pd.DataFrame(
        {
            "date": dates,
            "A": np.sin(np.arange(len(dates))) / 100.0,
            "B": np.sin(np.arange(len(dates)) + 0.2) / 120.0,
            "C": np.cos(np.arange(len(dates))) / 130.0,
            "QQQ": np.sin(np.arange(len(dates)) + 0.4) / 150.0,
        }
    )
    log_prices = pd.DataFrame(
        {
            "date": dates,
            "A": np.log(100 + np.arange(len(dates)) + np.sin(np.arange(len(dates)))),
            "B": np.log(90 + np.arange(len(dates)) * 0.5),
            "C": np.log(80 + np.arange(len(dates)) * 0.4),
            "QQQ": np.log(110 + np.arange(len(dates)) * 0.3),
        }
    )
    volumes = pd.DataFrame(
        {
            "date": dates,
            "A": 1_000_000 + np.arange(len(dates)) * 10_000,
            "B": 800_000 + np.arange(len(dates)) * 8_000,
            "C": 700_000 + np.arange(len(dates)) * 7_000,
        }
    )
    coefficients = pd.DataFrame(
        {
            "date": dates,
            "triplet_id": "A_B_C",
            "method": "kalman_random_walk",
            "beta_1": 0.5 + np.sin(np.arange(len(dates)) / 4.0) * 0.05,
            "beta_2": 0.3 + np.cos(np.arange(len(dates)) / 5.0) * 0.04,
        }
    )
    labels = pd.DataFrame(
        {
            "event_id": ["evt1", "evt2"],
            "label": [1, 0],
            "outcome": ["success", "failure"],
            "exit_reason": ["reversion", "max_holding_period"],
            "holding_period": [3, 10],
        }
    )
    return events, residuals, returns, log_prices, volumes, coefficients, labels


def test_build_event_feature_matrix_contains_core_features():
    events, residuals, returns, log_prices, volumes, coefficients, labels = sample_inputs()
    cfg = FeatureConfig(residual_window=8, volatility_window=8, stability_window=8, correlation_window=8, moving_average_window=8, min_periods=4)

    features = build_event_feature_matrix(
        events,
        residuals,
        config=cfg,
        labels=labels,
        returns=returns,
        log_prices=log_prices,
        volumes=volumes,
        coefficients=coefficients,
    )

    assert features.shape[0] == 2
    assert {"residual_change", "residual_volatility", "half_life_estimate", "beta_stability"}.issubset(features.columns)
    assert {"target_return_volatility", "market_return", "recent_drawdown", "volume_shock"}.issubset(features.columns)
    assert features["label"].tolist() == [1, 0]


def test_features_are_matched_on_event_date_without_future_residuals():
    events, residuals, *_ = sample_inputs()
    event_date = pd.Timestamp(events.loc[0, "event_date"])
    expected = residuals.loc[residuals["date"].eq(event_date), "residual"].iloc[0]
    future_value = residuals.loc[residuals["date"].gt(event_date), "residual"].iloc[0]

    features = build_event_feature_matrix(events.iloc[[0]], residuals, config=FeatureConfig(min_periods=3))

    assert features.loc[0, "residual"] == pytest.approx(expected)
    assert features.loc[0, "residual"] != pytest.approx(future_value)


def test_create_feature_outputs_returns_matrix_reports_and_correlation():
    events, residuals, returns, log_prices, volumes, coefficients, labels = sample_inputs()

    outputs = create_feature_outputs(
        events,
        residuals,
        labels=labels,
        returns=returns,
        log_prices=log_prices,
        volumes=volumes,
        coefficients=coefficients,
        config=FeatureConfig(min_periods=3),
    )

    assert set(outputs) == {"feature_matrix", "feature_summary_statistics", "feature_missingness_report", "feature_correlation_matrix"}
    assert not outputs["feature_matrix"].empty
    assert not outputs["feature_summary_statistics"].empty
    assert not outputs["feature_missingness_report"].empty


def test_summary_and_missingness_reports_have_expected_schema():
    events, residuals, *_ = sample_inputs()
    features = build_event_feature_matrix(events, residuals, config=FeatureConfig(min_periods=3))

    summary = feature_summary_statistics(features)
    missingness = feature_missingness_report(features)
    corr = feature_correlation_matrix(features, min_non_null=1)

    assert {"feature", "count", "mean", "std"}.issubset(summary.columns)
    assert {"column", "missing_count", "missing_rate"}.issubset(missingness.columns)
    assert isinstance(corr, pd.DataFrame)


def test_feature_scaler_standardizes_selected_columns():
    events, residuals, returns, log_prices, *_ = sample_inputs()
    features = build_event_feature_matrix(
        events,
        residuals,
        config=FeatureConfig(min_periods=3),
        returns=returns,
        log_prices=log_prices,
    )
    cols = ["entry_z_score", "residual", "target_return_volatility"]

    params = fit_feature_scaler(features, columns=cols)
    scaled = transform_feature_matrix(features, params)

    assert set(params.columns) == set(cols)
    assert abs(float(scaled["entry_z_score"].mean())) < 1e-12


def test_collinear_feature_pairs_flags_high_correlation_only():
    matrix = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.01, 2.02, 2.99, 4.01, 5.02],   # near-duplicate of a
        "c": [5.0, 1.0, 4.0, 2.0, 3.0],         # unrelated
    })
    corr = feature_correlation_matrix(matrix)
    flagged = collinear_feature_pairs(corr, threshold=0.85)

    assert len(flagged) == 1
    assert {flagged.loc[0, "feature_a"], flagged.loc[0, "feature_b"]} == {"a", "b"}
    assert flagged.loc[0, "abs_correlation"] > 0.85

    assert collinear_feature_pairs(pd.DataFrame()).empty


def test_missing_inputs_raise_clear_errors():
    events, residuals, *_ = sample_inputs()
    with pytest.raises(KeyError):
        build_event_feature_matrix(events.drop(columns=["event_id"]), residuals)
    with pytest.raises(ValueError):
        build_event_feature_matrix(events, pd.DataFrame())
