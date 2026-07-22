import numpy as np
import pandas as pd
import pytest

from src.hmm import (
    HMMConfig,
    apply_regime_trade_filter,
    fit_gaussian_hmm,
    fit_hmm_by_triplet,
    forward_algorithm,
    gaussian_emission_matrix,
    infer_regime_labels,
    make_regime_probability_table,
    posterior_state_probabilities,
    summarize_strategy_performance_by_regime,
    viterbi_decode,
)


def test_gaussian_emission_matrix_matches_normal_pdf_and_respects_variance_floor():
    values = np.array([-1.0, 0.0, 1.0])
    means = np.array([0.0, 0.0])
    variances = np.array([1.0, 0.0])  # second state exercises the variance floor
    emissions = gaussian_emission_matrix(values, means, variances, variance_floor=1e-6)
    assert emissions.shape == (3, 2)
    expected_state0 = np.exp(-0.5 * values ** 2) / np.sqrt(2 * np.pi)
    assert np.allclose(emissions[:, 0], expected_state0, atol=1e-8)
    assert np.all(emissions > 0.0)  # floored, never exactly zero -- forward algorithm divides by row sums
    with pytest.raises(ValueError):
        gaussian_emission_matrix(values, means, np.array([1.0, 1.0, 1.0]))


def test_forward_algorithm_returns_scaled_probabilities():
    emissions = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7]])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    alpha, scales, log_likelihood = forward_algorithm(emissions, np.array([0.5, 0.5]), transition)
    assert alpha.shape == emissions.shape
    assert np.allclose(alpha.sum(axis=1), 1.0)
    assert np.all(scales > 0.0)
    assert np.isfinite(log_likelihood)


def test_gaussian_hmm_fit_outputs_valid_parameters():
    rng = np.random.default_rng(42)
    values = np.r_[rng.normal(-1.5, 0.2, 40), rng.normal(0.0, 0.15, 40), rng.normal(1.5, 0.3, 40)]
    result = fit_gaussian_hmm(values, HMMConfig(n_states=3, max_iter=20, random_state=1))
    assert result.transition_matrix.shape == (3, 3)
    assert np.allclose(result.transition_matrix.sum(axis=1), 1.0)
    assert np.all(result.variances > 0.0)
    assert len(result.log_likelihoods) >= 1
    assert set(result.state_labels) == {"mean_reverting", "trending", "volatile_breakdown"}


def test_posterior_probabilities_and_viterbi_shapes():
    values = np.sin(np.linspace(0.0, 8.0, 60))
    result = fit_gaussian_hmm(values, HMMConfig(n_states=3, max_iter=10))
    probabilities, log_likelihood = posterior_state_probabilities(values, result)
    path = viterbi_decode(values, result)
    assert probabilities.shape == (60, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert path.shape == (60,)
    assert np.isfinite(log_likelihood)


def test_regime_probability_table_contains_named_regime_probabilities():
    dates = pd.date_range("2024-01-01", periods=75, freq="B")
    values = np.r_[np.zeros(25), np.linspace(0.2, 1.5, 25), np.random.default_rng(4).normal(0, 1.2, 25)]
    residuals = pd.DataFrame({"date": dates, "triplet_id": "A_B_C", "residual_z_score": values})
    result = fit_gaussian_hmm(residuals["residual_z_score"], HMMConfig(n_states=3, max_iter=15))
    table = make_regime_probability_table(residuals, result, triplet_id="A_B_C")
    assert {"mean_reverting_probability", "trending_probability", "volatile_breakdown_probability"}.issubset(table.columns)
    assert table.shape[0] == residuals.shape[0]
    assert table["most_likely_regime"].notna().all()


def test_fit_hmm_by_triplet_returns_tables():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    frame = pd.concat(
        [
            pd.DataFrame({"date": dates, "triplet_id": "T1", "residual_z_score": np.sin(np.linspace(0, 4, 40))}),
            pd.DataFrame({"date": dates, "triplet_id": "T2", "residual_z_score": np.cos(np.linspace(0, 4, 40))}),
        ],
        ignore_index=True,
    )
    output = fit_hmm_by_triplet(frame, config=HMMConfig(n_states=3, max_iter=8))
    assert set(output.keys()) == {"models", "regime_probabilities", "regime_parameters"}
    assert output["regime_probabilities"].shape[0] == 80
    assert output["regime_parameters"].shape[0] == 6


def test_apply_regime_trade_filter_flags_allowed_events():
    events = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "triplet_id": ["T", "T"],
            "event_date": ["2024-01-02", "2024-01-03"],
        }
    )
    regimes = pd.DataFrame(
        {
            "triplet_id": ["T", "T"],
            "date": ["2024-01-02", "2024-01-03"],
            "mean_reverting_probability": [0.7, 0.4],
            "most_likely_regime": ["mean_reverting", "volatile_breakdown"],
            "viterbi_regime": ["mean_reverting", "volatile_breakdown"],
        }
    )
    filtered = apply_regime_trade_filter(events, regimes, threshold=0.6)
    assert filtered["allowed_by_regime_filter"].tolist() == [True, False]


def test_summarize_strategy_performance_by_regime():
    trades = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "triplet_id": ["T", "T"],
            "event_date": ["2024-01-02", "2024-01-03"],
            "strategy": ["baseline_rule_based", "baseline_rule_based"],
            "net_pnl": [1.0, -0.5],
            "label": [1, 0],
            "turnover": [2.0, 2.0],
        }
    )
    regimes = pd.DataFrame(
        {
            "triplet_id": ["T", "T"],
            "date": ["2024-01-02", "2024-01-03"],
            "mean_reverting_probability": [0.8, 0.2],
            "most_likely_regime": ["mean_reverting", "trending"],
            "viterbi_regime": ["mean_reverting", "trending"],
        }
    )
    summary = summarize_strategy_performance_by_regime(trades, regimes, threshold=0.6)
    assert summary["trade_count"].sum() == 2
    assert set(summary["regime_bucket"]) == {
        "mean_reverting_probability_above_threshold",
        "mean_reverting_probability_below_threshold",
    }


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        fit_gaussian_hmm([1.0, 2.0, 3.0], HMMConfig(n_states=1))


def test_infer_regime_labels_for_non_three_state_model():
    labels = infer_regime_labels(np.array([0.0, 1.0]), np.array([0.2, 0.3]))
    assert labels == ["state_0", "state_1"]
