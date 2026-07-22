from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeResult:
    intercept: float
    coefficients: np.ndarray
    alpha: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    feature_names: Optional[Tuple[str, ...]] = None

    @property
    def params(self) -> np.ndarray:
        return np.concatenate(([self.intercept], self.coefficients))


def _as_2d_float_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X must be one- or two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("X contains NaN or infinite values")
    return array


def _as_1d_float_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("y contains NaN or infinite values")
    return array


def _design_matrix(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    if not fit_intercept:
        return X
    return np.column_stack((np.ones(X.shape[0]), X))


def ridge_fit(
    X: object,
    y: object,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    penalize_intercept: bool = False,
    feature_names: Optional[Sequence[str]] = None,
) -> RidgeResult:
    """Ridge regression: ordinary least squares with an added penalty
    that discourages large coefficients.

    Why this exists alongside plain OLS: the two hedge legs in a triplet
    are often similar to each other by construction (e.g. QQQ and XLK
    both track tech stocks). When two predictors are highly correlated,
    OLS can respond by assigning one an inflated positive coefficient and
    the other an inflated negative one that happen to cancel out on the
    training data -- a fit that looks fine in-sample but is unstable and
    won't generalize. Ridge counteracts this by adding `alpha * beta^2`
    to what's being minimized, which shrinks coefficients toward zero and
    is penalized more the larger they get, discouraging that kind of
    unstable cancellation.

    The closed-form solution is beta = (X'X + alpha*I)^-1 X'y -- the
    normal-equations solution for OLS, with `alpha*I` added to the matrix
    being inverted. That addition is also what keeps the matrix invertible
    even when the hedge legs are so correlated that X'X alone would be
    (numerically) singular.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    x_array = _as_2d_float_array(X)
    y_array = _as_1d_float_array(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if x_array.shape[0] == 0:
        raise ValueError("at least one observation is required")

    x_design = _design_matrix(x_array, fit_intercept=fit_intercept)
    # The penalty matrix is alpha on the diagonal, zero elsewhere -- this
    # is what "alpha * beta^2 per coefficient" looks like in matrix form
    # once folded into the normal equations.
    penalty = np.eye(x_design.shape[1]) * float(alpha)
    if fit_intercept and not penalize_intercept:
        # The intercept represents a price-level offset, not a
        # relationship strength -- shrinking it toward zero the way the
        # slope coefficients are shrunk wouldn't make conceptual sense,
        # so its penalty row/column is zeroed out by default.
        penalty[0, 0] = 0.0

    lhs = x_design.T @ x_design + penalty
    rhs = x_design.T @ y_array

    try:
        # np.linalg.solve is the direct route (Ax=b); it's only used when
        # the matrix is well-conditioned enough to invert cleanly.
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        # Falls back to the least-squares solver (SVD-based, tolerant of
        # a singular or near-singular matrix) if direct solving fails --
        # this should be rare given the ridge penalty already improves
        # conditioning, but a hedge pair could still be pathological.
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    fitted = x_design @ beta
    residuals = y_array - fitted
    ss_total = float(np.sum((y_array - y_array.mean()) ** 2))
    ss_resid = float(np.sum(residuals ** 2))
    r_squared = 0.0 if ss_total == 0 else 1.0 - ss_resid / ss_total

    if fit_intercept:
        intercept = float(beta[0])
        coefficients = beta[1:].astype(float)
    else:
        intercept = 0.0
        coefficients = beta.astype(float)

    names = tuple(feature_names) if feature_names is not None else None
    if names is not None and len(names) != coefficients.shape[0]:
        raise ValueError("feature_names must match the number of columns in X")

    return RidgeResult(
        intercept=intercept,
        coefficients=coefficients,
        alpha=float(alpha),
        fitted_values=fitted.astype(float),
        residuals=residuals.astype(float),
        r_squared=float(r_squared),
        feature_names=names,
    )


def ridge_predict(X: object, intercept: float, coefficients: Iterable[float]) -> np.ndarray:
    x_array = _as_2d_float_array(X)
    coef = np.asarray(list(coefficients), dtype=float).reshape(-1)
    if x_array.shape[1] != coef.shape[0]:
        raise ValueError("coefficient count must match the number of X columns")
    return float(intercept) + x_array @ coef


def rolling_ridge(
    log_prices: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    window: int,
    alpha: float = 1.0,
    min_obs: Optional[int] = None,
    triplet_id: Optional[str] = None,
) -> pd.DataFrame:
    """Same rolling-window, refit-every-day structure as
    `regression.rolling_ols`, but using `ridge_fit` instead of plain OLS
    at each step -- the regularized hedge-ratio series used to compare
    against the unregularized rolling OLS series and check whether the
    ridge penalty actually produces smoother, more stable coefficients
    for a given triplet.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if len(hedge_cols) != 2:
        raise ValueError("triangular regression requires exactly two hedge columns")

    required = [target_col, *hedge_cols]
    missing = [col for col in required if col not in log_prices.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    clean = log_prices[required].dropna().copy()
    if clean.shape[0] <= window:
        return pd.DataFrame(columns=_rolling_output_columns())

    min_required = window if min_obs is None else int(min_obs)
    if min_required <= len(hedge_cols):
        raise ValueError("min_obs must exceed the number of hedge columns")
    if min_required > window:
        raise ValueError("min_obs cannot exceed window")

    rows = []
    # Same walk-forward pattern as rolling_ols: fit only on the trailing
    # `window` days, score only against the single next day, then slide
    # the window forward one day and repeat.
    for i in range(window, clean.shape[0]):
        train = clean.iloc[i - window : i]
        if train.dropna().shape[0] < min_required:
            continue

        result = ridge_fit(
            train.loc[:, hedge_cols].to_numpy(),
            train.loc[:, target_col].to_numpy(),
            alpha=alpha,
            feature_names=hedge_cols,
        )
        prediction_row = clean.iloc[[i]]
        fitted = ridge_predict(
            prediction_row.loc[:, hedge_cols].to_numpy(),
            result.intercept,
            result.coefficients,
        )[0]
        actual = float(prediction_row[target_col].iloc[0])
        rows.append(
            {
                "date": clean.index[i],
                "triplet_id": triplet_id or f"{target_col}_{hedge_cols[0]}_{hedge_cols[1]}",
                "target_symbol": target_col,
                "hedge_symbol_1": hedge_cols[0],
                "hedge_symbol_2": hedge_cols[1],
                "alpha": result.intercept,
                "beta_1": float(result.coefficients[0]),
                "beta_2": float(result.coefficients[1]),
                "actual_log_price": actual,
                "fitted_log_price": float(fitted),
                "residual": actual - float(fitted),
                "train_start": train.index[0],
                "train_end": train.index[-1],
                "window": int(window),
                "ridge_alpha": float(alpha),
                "method": "rolling_ridge",
            }
        )

    return pd.DataFrame(rows, columns=_rolling_output_columns())


def _rolling_output_columns() -> list[str]:
    # Matches src/regression.py's schema exactly (same column names, same
    # order) so rolling OLS, rolling ridge, and static results can be
    # concatenated into one table downstream without reconciling schemas.
    return [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "actual_log_price",
        "fitted_log_price",
        "residual",
        "train_start",
        "train_end",
        "window",
        "ridge_alpha",
        "method",
    ]
