"""
Fixed-width fractional differentiation (FFD), implemented from scratch.

Standard practice for stationarizing a price or residual series is
integer differencing (d=1: simple returns/differences). That achieves
stationarity but destroys essentially all memory of the series' level --
a first difference has no information about where the price actually was,
only how much it moved. Marcos Lopez de Prado's fractional differentiation
(Advances in Financial Machine Learning, 2018) generalizes differencing to
non-integer d, so a series can be differenced only as much as needed to
pass a stationarity test, preserving more predictive memory than a blunt
d=1 difference would.

This module answers a concrete question this project didn't previously
ask: is d=1 (or the ad hoc log-price differencing used elsewhere) actually
the *minimum* differencing needed for stationarity, or is it throwing away
usable signal? `find_minimum_stationary_d` searches for the smallest d
that passes the ADF test from src/cointegration.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cointegration import augmented_dickey_fuller


def fractional_diff_weights(d: float, threshold: float = 1e-5, max_size: int = 10_000) -> np.ndarray:
    """Weights w_0, w_1, ... for the binomial-series expansion of (1-L)^d,
    truncated once a weight's magnitude drops below `threshold`. For
    d=0 this is exactly [1.0] (no differencing); for d=1 it collapses to
    exactly [1.0, -1.0] (exact first differencing, not an approximation --
    see tests/test_fractional_diff.py).

    Derivation in plain terms: the binomial series for (1-L)^d expands to
    an infinite sum of terms, and each successive weight can be computed
    from the previous one via w_k = -w_{k-1} * (d-k+1)/k rather than
    recomputing a binomial coefficient from scratch every time. For
    d close to 0, that ratio stays close to 1 for a long time, so the
    weights decay very slowly (long memory, many terms needed to reach
    the threshold); for d=1, the ratio becomes exactly 0 after the second
    term, which is exactly why d=1 collapses to only two nonzero weights.
    """
    if d < 0:
        raise ValueError("d must be non-negative")
    weights = [1.0]  # w_0 is always 1 by definition of the binomial series
    k = 1
    while k < max_size:
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            # weight has decayed to negligible -- truncating here is what
            # makes this "fixed-width" rather than requiring the entire
            # history of the series for every single output point
            break
        weights.append(w_k)
        k += 1
    return np.array(weights)


def fractional_diff(series: pd.Series | np.ndarray, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fixed-width fractionally differenced series. The first
    `len(weights) - 1` observations are NaN (not enough history yet to
    apply the full weight window) rather than partially computed with a
    truncated weight set, which would silently bias the earliest values.
    """
    is_series = isinstance(series, pd.Series)
    index = series.index if is_series else None
    values = np.asarray(series, dtype=float)

    weights = fractional_diff_weights(d, threshold)
    width = len(weights)
    # weights[0] applies to the most recent observation in each window,
    # so the weight array is reversed once up front to line up with a
    # window slice ordered oldest-to-newest (see the dot product below)
    reversed_weights = weights[::-1]

    n = len(values)
    result = np.full(n, np.nan)
    if width <= n:
        # slide a window of exactly `width` observations across the
        # series; each output point is a weighted sum of that window,
        # equivalent to convolving the series with the weight kernel
        for t in range(width - 1, n):
            window = values[t - width + 1: t + 1]
            if np.isnan(window).any():
                # a gap anywhere in this window makes the weighted sum
                # undefined -- skip rather than silently drop the NaN and
                # compute a biased partial sum
                continue
            result[t] = float(np.dot(reversed_weights, window))

    return pd.Series(result, index=index) if is_series else pd.Series(result)


def find_minimum_stationary_d(
    series: pd.Series | np.ndarray,
    d_grid: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    significance: float = 0.05,
    threshold: float = 1e-5,
) -> dict:
    """Searches `d_grid` in increasing order and returns the smallest d
    whose fractionally-differenced series passes the ADF test, along with
    the correlation between the differenced series and the original
    levels -- the usual way de Prado's approach is presented, since that
    correlation is the practical measure of how much memory survived.
    Returns d=None if nothing in the grid passes, meaning even d=1 (full
    differencing) wasn't tried or didn't help -- extend d_grid past 1.0 to
    check further.
    """
    values = pd.Series(series).astype(float).reset_index(drop=True)
    # search smallest-to-largest so the first d that passes is genuinely
    # the *minimum* differencing needed, not just any d that happens to work
    for d in sorted(d_grid):
        diffed = fractional_diff(values, d, threshold=threshold)
        valid = diffed.dropna()
        if len(valid) < 20:
            # too few usable observations left after the weight window's
            # leading NaNs to run a meaningful ADF test at this d
            continue
        try:
            result = augmented_dickey_fuller(valid.to_numpy())
        except ValueError:
            continue
        if result.p_value <= significance:
            # this is the practical "how much memory survived" number:
            # correlate the differenced series against the original
            # levels over whatever dates both have valid data for
            aligned = pd.concat([values, diffed], axis=1).dropna()
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])) if len(aligned) > 1 else np.nan
            return {
                "d": d, "adf_statistic": result.statistic, "adf_p_value": result.p_value,
                "n_obs": result.n_obs, "correlation_with_level": corr,
            }
    return {"d": None, "adf_statistic": np.nan, "adf_p_value": np.nan, "n_obs": 0, "correlation_with_level": np.nan}
