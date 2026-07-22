import numpy as np
import pandas as pd
import pytest

from src.fractional_diff import (
    find_minimum_stationary_d,
    fractional_diff,
    fractional_diff_weights,
)


def test_d_zero_weights_are_trivial():
    assert np.allclose(fractional_diff_weights(0.0), [1.0])


def test_d_one_weights_are_exact_first_difference():
    assert np.allclose(fractional_diff_weights(1.0), [1.0, -1.0])


def test_d_one_matches_np_diff_exactly():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 500))
    diffed = fractional_diff(series, 1.0).dropna().to_numpy()
    manual = np.diff(series)
    assert np.allclose(diffed, manual[len(manual) - len(diffed):])


def test_negative_d_rejected():
    with pytest.raises(ValueError):
        fractional_diff_weights(-0.1)


def test_leading_values_are_nan_not_partially_computed():
    series = pd.Series(np.arange(10.0))
    diffed = fractional_diff(series, 1.0)
    assert diffed.iloc[0:1].isna().all()  # width=2 for d=1 -> first observation can't be computed
    assert diffed.iloc[1:].notna().all()


def test_preserves_pandas_index():
    idx = pd.date_range("2024-01-01", periods=20)
    series = pd.Series(np.arange(20.0), index=idx)
    diffed = fractional_diff(series, 1.0)
    assert diffed.index.equals(idx)


def test_memory_preservation_increases_as_d_decreases():
    # this is the entire point of the technique: smaller d should retain
    # more correlation with the original level than d=1 (full differencing)
    rng = np.random.default_rng(7)
    n = 4000
    walk = np.cumsum(rng.normal(0, 1, n)) + 100

    def corr_with_level(d):
        diffed = fractional_diff(walk, d).dropna()
        if len(diffed) < 20:
            return None
        level = walk[len(walk) - len(diffed):]
        return abs(np.corrcoef(diffed.to_numpy(), level)[0, 1])

    corr_d1 = corr_with_level(1.0)
    corr_d05 = corr_with_level(0.5)
    assert corr_d1 is not None and corr_d05 is not None
    assert corr_d05 > corr_d1  # less differencing keeps more memory


def test_find_minimum_stationary_d_on_random_walk_needs_partial_differencing():
    rng = np.random.default_rng(8)
    walk = np.cumsum(rng.normal(0, 1, 3000)) + 100
    result = find_minimum_stationary_d(walk, d_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert result["d"] is not None
    assert result["d"] > 0.0  # raw levels (d=0) must not pass -- it's a random walk
    assert result["adf_p_value"] <= 0.05
    assert 0.0 <= result["correlation_with_level"] <= 1.0


def test_find_minimum_stationary_d_on_already_stationary_series_returns_zero():
    rng = np.random.default_rng(9)
    n = 1000
    ar1 = np.zeros(n)
    for t in range(1, n):
        ar1[t] = 0.3 * ar1[t - 1] + rng.normal(0, 1)
    result = find_minimum_stationary_d(ar1, d_grid=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    assert result["d"] == 0.0  # already stationary -- no differencing needed
    assert result["correlation_with_level"] == pytest.approx(1.0)


def test_find_minimum_stationary_d_returns_none_when_nothing_in_grid_passes():
    # an empty/degenerate grid should not crash, just report no result
    result = find_minimum_stationary_d(np.arange(30.0), d_grid=())
    assert result["d"] is None
