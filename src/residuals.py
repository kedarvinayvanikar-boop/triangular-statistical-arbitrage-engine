from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def calculate_residuals(actual: object, fitted: object) -> np.ndarray:
    # Just actual minus fitted -- the "gap" between what really happened
    # and what the hedge-ratio model predicted. Kept as its own function
    # (rather than inlined everywhere) so every hedge-ratio method
    # computes this identically.
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    fitted_array = np.asarray(fitted, dtype=float).reshape(-1)
    if actual_array.shape[0] != fitted_array.shape[0]:
        raise ValueError("actual and fitted arrays must have the same length")
    if not np.isfinite(actual_array).all() or not np.isfinite(fitted_array).all():
        raise ValueError("actual and fitted arrays must be finite")
    return actual_array - fitted_array


def residual_autocorrelation(residuals: object, lag: int = 1) -> float:
    """Correlation between the residual series and itself, shifted by
    `lag` days. High positive autocorrelation at lag 1 means today's
    residual is a good predictor of tomorrow's -- i.e. the gap tends to
    persist rather than bounce around randomly, a sign the residual isn't
    behaving like noise. Near zero suggests the gap moves unpredictably
    day to day.
    """
    values = pd.Series(np.asarray(residuals, dtype=float).reshape(-1)).dropna()
    if lag <= 0:
        raise ValueError("lag must be positive")
    if values.shape[0] <= lag:
        return np.nan
    left = values.iloc[:-lag].to_numpy()
    right = values.iloc[lag:].to_numpy()
    if np.std(left) == 0 or np.std(right) == 0:
        # a constant series has undefined correlation (division by zero
        # in the correlation formula) -- return NaN rather than crash
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def residual_summary(
    residual_table: pd.DataFrame,
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> pd.DataFrame:
    """Descriptive statistics (mean, spread, percentiles, lag-1
    autocorrelation) for every (triplet, hedge-ratio method) combination
    in one table -- the first diagnostic pass to sanity-check a residual
    series before trusting it: is it centered near zero, how wide is its
    typical range, does it look autocorrelated.
    """
    required = set(group_cols).union({"residual"})
    missing = [col for col in required if col not in residual_table.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    rows = []
    for keys, group in residual_table.dropna(subset=["residual"]).groupby(list(group_cols), sort=True):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        residuals = group["residual"].astype(float)
        row = {col: value for col, value in zip(group_cols, key_tuple)}
        row.update(
            {
                "n_obs": int(residuals.shape[0]),
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std(ddof=1)) if residuals.shape[0] > 1 else 0.0,
                "residual_abs_mean": float(residuals.abs().mean()),
                "residual_min": float(residuals.min()),
                "residual_q05": float(residuals.quantile(0.05)),
                "residual_median": float(residuals.median()),
                "residual_q95": float(residuals.quantile(0.95)),
                "residual_max": float(residuals.max()),
                "autocorr_1": residual_autocorrelation(residuals, lag=1),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_residual_stability(residual_table: pd.DataFrame) -> pd.DataFrame:
    """Expresses every method's residual spread as a ratio relative to
    the static (fixed, initial-window) fit's spread, per triplet. A ratio
    below 1.0 means that method (rolling OLS, ridge, Kalman) produced a
    tighter, more stable residual than just using one fixed hedge ratio
    for the whole period -- evidence the adaptive method is actually
    tracking a real drift in the relationship, not just adding noise.
    """
    summary = residual_summary(residual_table, group_cols=("triplet_id", "method"))
    if summary.empty:
        return summary

    base = summary[summary["method"] == "static_ols_initial_window"]
    base = base.loc[:, ["triplet_id", "residual_std", "residual_abs_mean"]].rename(
        columns={
            "residual_std": "static_residual_std",
            "residual_abs_mean": "static_abs_mean",
        }
    )
    comparison = summary.merge(base, on="triplet_id", how="left")
    comparison["std_ratio_vs_static"] = comparison["residual_std"] / comparison["static_residual_std"]
    comparison["abs_mean_ratio_vs_static"] = comparison["residual_abs_mean"] / comparison["static_abs_mean"]
    return comparison.sort_values(["triplet_id", "method"]).reset_index(drop=True)


def zscore_residuals(residuals: object, window: int) -> pd.Series:
    """Converts a raw residual (a dollar/log-price gap, whose "normal"
    size differs from one triplet to another) into a z-score: how many
    standard deviations away from its own recent average the gap
    currently is. This is what makes "wide gap" comparable across
    triplets with very different price scales and volatilities, and it's
    the number entry/exit thresholds are defined against everywhere else
    in this project (e.g. "enter when |z| > 2").

    The rolling mean/std are computed with `.shift(1)` applied before
    `.rolling(...)` -- meaning today's z-score is calculated using only
    the window of days *strictly before* today, never including today's
    own value. Without that shift, today's residual would be part of the
    baseline it's being compared against, which quietly leaks today's
    answer into today's own signal.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    series = pd.Series(np.asarray(residuals, dtype=float).reshape(-1))
    rolling_mean = series.shift(1).rolling(window, min_periods=window).mean()
    rolling_std = series.shift(1).rolling(window, min_periods=window).std(ddof=1)
    # a rolling window with zero variance (residual was perfectly flat)
    # would divide by zero -- treated as undefined (NaN) rather than
    # producing an infinite z-score
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)
