"""
Augmented Dickey-Fuller (ADF) stationarity test, implemented from scratch,
and a Benjamini-Hochberg false-discovery-rate correction for testing many
triplets at once.

This exists to close a real gap: every other module in this pipeline
assumes a triplet's residual is mean-reverting because the triplet was
picked to look economically related. Nothing actually tested that
assumption before treating a triplet as tradeable. A triplet whose
residual is not stationary is a triplet where "mean reversion" is not a
real property of the data -- trading it on that assumption is trading
noise, exactly the risk the null-hypothesis check (scripts/
null_hypothesis_check.py) surfaced from a different angle.

Precision note: the reported p-value is a linear interpolation against the
standard asymptotic Dickey-Fuller critical values (constant, no trend
specification) at the 1%, 5%, and 10% levels -- not the full MacKinnon
(1994) response-surface regression that packages like statsmodels use.
That interpolation is a reasonable approximation for ranking and FDR
correction, but is intentionally not presented as exact. Reimplementing
MacKinnon's response surface from memory risked silently shipping wrong
p-values, which is worse than a clearly-labeled approximation; if exact
finite-sample p-values matter, use statsmodels.tsa.stattools.adfuller and
treat this module's output as a fast, dependency-free first pass.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Asymptotic critical values for the ADF test, constant-no-trend
# specification (Dickey & Fuller 1979 / as commonly tabulated in Davidson &
# MacKinnon, "Econometric Theory and Methods"). Well-established, widely
# cited constants -- not derived, just standard reference values.
_DF_CRITICAL_VALUES = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


@dataclass(frozen=True)
class ADFResult:
    triplet_id: str
    statistic: float
    p_value: float
    lag_used: int
    n_obs: int
    critical_value_1pct: float = _DF_CRITICAL_VALUES[0.01]
    critical_value_5pct: float = _DF_CRITICAL_VALUES[0.05]
    critical_value_10pct: float = _DF_CRITICAL_VALUES[0.10]

    @property
    def is_stationary_5pct(self) -> bool:
        return self.statistic < self.critical_value_5pct


def _approximate_p_value(tau: float) -> float:
    """Linear interpolation of tau against the three known critical points.
    See module docstring for why this is an approximation, not the exact
    MacKinnon response-surface p-value.
    """
    points = sorted(_DF_CRITICAL_VALUES.items(), key=lambda kv: kv[1])  # (alpha, tau) sorted by tau ascending
    taus = [t for _, t in points]
    alphas = [a for a, _ in points]

    if tau <= taus[0]:
        # more extreme than the 1% critical value -- extrapolate the same
        # slope as the steepest known segment, floored at a small p-value
        slope = (alphas[1] - alphas[0]) / (taus[1] - taus[0])
        return float(max(0.001, alphas[0] + slope * (tau - taus[0])))
    if tau >= taus[-1]:
        # less extreme than the 10% critical value -- the tail flattens out
        # well before p=1; cap rather than let the same slope run away
        slope = (alphas[-1] - alphas[-2]) / (taus[-1] - taus[-2])
        extrapolated = alphas[-1] + slope * (tau - taus[-1])
        return float(min(0.995, max(alphas[-1], extrapolated)))
    return float(np.interp(tau, taus, alphas))


def _select_lag_by_aic(delta_y: np.ndarray, y_lag1: np.ndarray, max_lag: int) -> int:
    """Chooses the number of augmenting lagged-difference terms by AIC,
    rather than a fixed lag -- residual autocorrelation left over from an
    under-specified lag structure biases the test statistic."""
    best_lag, best_aic = 0, np.inf
    n_total = len(delta_y)
    for lag in range(max_lag + 1):
        start = max_lag  # align all candidate lag models on the same sample
        y_dep = delta_y[start:]
        design_cols = [np.ones(n_total - start), y_lag1[start:]]
        for i in range(1, lag + 1):
            design_cols.append(delta_y[start - i: n_total - i])
        design = np.column_stack(design_cols)
        try:
            beta, residuals, *_ = np.linalg.lstsq(design, y_dep, rcond=None)
        except np.linalg.LinAlgError:
            continue
        fitted = design @ beta
        rss = float(np.sum((y_dep - fitted) ** 2))
        n = len(y_dep)
        k = design.shape[1]
        if rss <= 0 or n <= k:
            continue
        aic = n * np.log(rss / n) + 2 * k
        if aic < best_aic:
            best_aic, best_lag = aic, lag
    return best_lag


def augmented_dickey_fuller(
    series: np.ndarray | pd.Series | list[float],
    max_lag: int | None = None,
    triplet_id: str = "",
) -> ADFResult:
    """ADF test with a constant term, no trend -- appropriate for a
    residual series that's expected to be mean-reverting around zero
    rather than around a deterministic trend line.
    """
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 20:
        raise ValueError(f"need at least 20 observations for a meaningful ADF test, got {n}")

    y = values[1:]
    y_lag1 = values[:-1]
    delta_y = y - y_lag1

    if max_lag is None:
        # Schwert's (1989) rule of thumb, a standard default lag ceiling
        max_lag = int(np.floor(12 * (n / 100.0) ** 0.25))
    max_lag = max(0, min(max_lag, len(delta_y) - 3))

    lag = _select_lag_by_aic(delta_y, y_lag1, max_lag) if max_lag > 0 else 0

    start = max_lag
    y_dep = delta_y[start:]
    design_cols = {"const": np.ones(len(y_dep)), "y_lag1": y_lag1[start:]}
    for i in range(1, lag + 1):
        design_cols[f"delta_lag_{i}"] = delta_y[start - i: len(delta_y) - i]
    design = np.column_stack(list(design_cols.values()))

    beta, _, _, _ = np.linalg.lstsq(design, y_dep, rcond=None)
    fitted = design @ beta
    resid = y_dep - fitted
    n_obs, k = design.shape
    dof = n_obs - k
    if dof <= 0:
        raise ValueError("not enough observations relative to the number of lags to estimate the ADF regression")
    sigma2 = float(np.sum(resid ** 2)) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se_gamma = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    gamma_hat = float(beta[1])
    tau = gamma_hat / se_gamma if se_gamma > 0 else np.nan

    return ADFResult(
        triplet_id=triplet_id,
        statistic=tau,
        p_value=_approximate_p_value(tau) if np.isfinite(tau) else 1.0,
        lag_used=lag,
        n_obs=n_obs,
    )


def cointegration_report(
    residuals: pd.DataFrame,
    value_column: str = "residual",
    group_columns: tuple[str, ...] = ("triplet_id", "method"),
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Runs the ADF test per (triplet, method) group and applies a
    Benjamini-Hochberg FDR correction across all of them jointly.

    Testing 82 triplets (or 82 x 3 hedge methods) individually at a raw 5%
    significance level means an expected ~4-12 of them look "stationary"
    by chance alone even if none actually are. The FDR correction adjusts
    the rejection threshold for the number of simultaneous tests, so
    `fdr_reject` is the column that should gate which triplets are treated
    as tradeable -- not the raw uncorrected p-value or the boolean
    `is_stationary_5pct` alone.
    """
    required = {value_column, *group_columns}
    missing = required.difference(residuals.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    rows = []
    for key, group in residuals.groupby(list(group_columns), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        label = "_".join(str(k) for k in key)
        try:
            result = augmented_dickey_fuller(group[value_column], triplet_id=label)
        except ValueError:
            continue
        row = {col: k for col, k in zip(group_columns, key)}
        row.update({
            "statistic": result.statistic,
            "p_value": result.p_value,
            "lag_used": result.lag_used,
            "n_obs": result.n_obs,
            "is_stationary_5pct": result.is_stationary_5pct,
        })
        rows.append(row)

    report = pd.DataFrame(rows)
    if report.empty:
        report["fdr_reject"] = pd.Series(dtype=bool)
        return report

    report["fdr_reject"] = benjamini_hochberg(report["p_value"].to_numpy(), alpha=fdr_alpha)
    return report.sort_values("p_value").reset_index(drop=True)


def benjamini_hochberg(p_values: np.ndarray | list[float], alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg (1995) false-discovery-rate correction.

    Returns a boolean array (same order as input) marking which hypotheses
    are rejected -- i.e. which p-values are small enough to survive
    correction for the number of simultaneous tests. Rejecting the null
    here means "stationary" in the ADF context: this is the array that
    should gate triplet inclusion, not the raw p-value.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool)
    # largest rank k such that p_(k) <= (k/n)*alpha; reject all ranks <= k
    max_rank = np.max(np.where(below)[0])
    reject_sorted = np.zeros(n, dtype=bool)
    reject_sorted[: max_rank + 1] = True
    reject = np.zeros(n, dtype=bool)
    reject[order] = reject_sorted
    return reject
