import numpy as np
import pandas as pd
import pytest

from src.cointegration import (
    augmented_dickey_fuller,
    benjamini_hochberg,
    cointegration_report,
)


def test_adf_does_not_flag_pure_random_walk_as_stationary():
    rng = np.random.default_rng(1)
    walk = np.cumsum(rng.normal(0, 1, 1000))
    result = augmented_dickey_fuller(walk, triplet_id="rw")
    assert not result.is_stationary_5pct
    assert result.p_value > 0.05


def test_adf_flags_strongly_mean_reverting_series_as_stationary():
    rng = np.random.default_rng(2)
    n = 1000
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.5 * ar1[t - 1] + rng.normal(0, 1)
    result = augmented_dickey_fuller(ar1, triplet_id="ar1")
    assert result.is_stationary_5pct
    assert result.p_value < 0.05
    assert result.statistic < result.critical_value_1pct


def test_adf_raises_on_too_few_observations():
    with pytest.raises(ValueError):
        augmented_dickey_fuller(np.arange(5.0))


def test_adf_handles_nans_by_dropping_them():
    rng = np.random.default_rng(3)
    n = 500
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.5 * ar1[t - 1] + rng.normal(0, 1)
    ar1_with_gaps = ar1.copy()
    ar1_with_gaps[10:15] = np.nan
    result = augmented_dickey_fuller(ar1_with_gaps, triplet_id="ar1_gaps")
    assert result.n_obs > 0
    assert np.isfinite(result.statistic)


def test_p_value_is_monotonic_in_the_test_statistic():
    # more negative tau (stronger evidence against a unit root) must never
    # produce a larger p-value than a less negative tau
    from src.cointegration import _approximate_p_value
    taus = [-6.0, -3.43, -2.86, -2.57, -1.0, 0.0, 2.0]
    p_values = [_approximate_p_value(t) for t in taus]
    assert p_values == sorted(p_values)
    assert all(0.0 <= p <= 1.0 for p in p_values)


def test_benjamini_hochberg_matches_hand_worked_example():
    # classic worked example: only the single smallest p-value survives
    # correction at alpha=0.05 with n=20 tests
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212,
                  0.216, 0.222, 0.251, 0.269, 0.275, 0.34, 0.341, 0.384, 0.569, 0.594, 0.696])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert reject.sum() == 1
    assert reject[0]  # the smallest p-value


def test_benjamini_hochberg_rejects_more_with_all_small_p_values():
    p = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert reject.all()


def test_benjamini_hochberg_rejects_none_with_all_large_p_values():
    p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    reject = benjamini_hochberg(p, alpha=0.05)
    assert not reject.any()


def test_benjamini_hochberg_handles_empty_input():
    assert benjamini_hochberg(np.array([])).size == 0


def test_cointegration_report_flags_a_known_mixture_correctly():
    rng = np.random.default_rng(4)
    n = 800
    rows = []
    # two genuinely stationary triplets
    for tid in ["STATIONARY_A", "STATIONARY_B"]:
        ar1 = np.zeros(n)
        for t in range(1, n):
            ar1[t] = 0.4 * ar1[t - 1] + rng.normal(0, 1)
        for v in ar1:
            rows.append({"triplet_id": tid, "method": "test", "residual": v})
    # two genuine random walks
    for tid in ["RANDOM_A", "RANDOM_B"]:
        walk = np.cumsum(rng.normal(0, 1, n))
        for v in walk:
            rows.append({"triplet_id": tid, "method": "test", "residual": v})

    residuals = pd.DataFrame(rows)
    report = cointegration_report(residuals)

    assert set(report["triplet_id"]) == {"STATIONARY_A", "STATIONARY_B", "RANDOM_A", "RANDOM_B"}
    stationary_flagged = set(report.loc[report["fdr_reject"], "triplet_id"])
    assert stationary_flagged == {"STATIONARY_A", "STATIONARY_B"}


def test_cointegration_report_raises_on_missing_columns():
    with pytest.raises(KeyError):
        cointegration_report(pd.DataFrame({"triplet_id": ["A"]}))
