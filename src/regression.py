from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .ridge import rolling_ridge


@dataclass(frozen=True)
class OLSResult:
    """Container for a single ordinary-least-squares fit.

    `intercept` and `coefficients` together define the fitted line/plane:
    y = intercept + coefficients[0]*x0 + coefficients[1]*x1 + ...
    `fitted_values` and `residuals` are computed on whatever data the fit
    was trained on (in-sample), not held-out predictions.
    """
    intercept: float
    coefficients: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    r_squared: float
    feature_names: Optional[tuple[str, ...]] = None

    @property
    def params(self) -> np.ndarray:
        # intercept and slope coefficients combined into one vector, in the
        # same order the design matrix expects them (intercept column first)
        return np.concatenate(([self.intercept], self.coefficients))


def _as_2d_float_array(values: object) -> np.ndarray:
    # Accepts a plain 1D array (single predictor) or a 2D array/DataFrame
    # (multiple predictors) and normalizes both into a 2D float array, so
    # the rest of this module only has to handle one shape.
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("X must be one- or two-dimensional")
    if not np.isfinite(array).all():
        # A silent NaN here would propagate through the whole regression
        # and produce a fit that looks valid but isn't -- better to fail
        # loudly at the point the bad data entered.
        raise ValueError("X contains NaN or infinite values")
    return array


def _as_1d_float_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(array).all():
        raise ValueError("y contains NaN or infinite values")
    return array


def _design_matrix(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    # OLS solves y = X @ beta. To also fit an intercept (the "alpha" in
    # the triangular regression), a column of 1s is prepended to X so the
    # corresponding beta coefficient becomes the intercept term.
    if not fit_intercept:
        return X
    return np.column_stack((np.ones(X.shape[0]), X))


def fit_ols(
    X: object,
    y: object,
    fit_intercept: bool = True,
    feature_names: Optional[Sequence[str]] = None,
) -> OLSResult:
    """Ordinary least squares via the normal, closed-form solution
    (no gradient descent needed here -- unlike logistic regression, OLS
    has an exact solution).

    Uses `np.linalg.lstsq` rather than the textbook (X'X)^-1 X'y formula:
    lstsq is solved via SVD internally, which stays numerically stable
    even when hedge legs are highly correlated (near-singular X'X) --
    exactly the situation ridge regression exists to handle more
    deliberately, but plain OLS still needs to not blow up in the
    meantime.
    """
    x_array = _as_2d_float_array(X)
    y_array = _as_1d_float_array(y)
    if x_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if x_array.shape[0] == 0:
        raise ValueError("at least one observation is required")

    x_design = _design_matrix(x_array, fit_intercept=fit_intercept)
    # lstsq minimizes ||y - X@beta||^2 -- the "least squares" the method
    # is named for -- and returns the beta vector that does it.
    beta = np.linalg.lstsq(x_design, y_array, rcond=None)[0]
    fitted = x_design @ beta
    residuals = y_array - fitted

    # R^2 = 1 - (unexplained variance / total variance). 1.0 means the
    # fit passes through every point exactly; 0.0 means it does no better
    # than just predicting the mean of y every time.
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
        raise ValueError("feature_names must match the number of X columns")

    return OLSResult(
        intercept=intercept,
        coefficients=coefficients,
        fitted_values=fitted.astype(float),
        residuals=residuals.astype(float),
        r_squared=float(r_squared),
        feature_names=names,
    )


def ols_predict(X: object, intercept: float, coefficients: Sequence[float]) -> np.ndarray:
    # Applies an already-fitted line/plane to new X values -- this is how
    # a rolling window's fit gets scored against the one day just outside
    # that window (see rolling_ols below), rather than re-deriving fitted
    # values from fit_ols's own in-sample output.
    x_array = _as_2d_float_array(X)
    coef = np.asarray(coefficients, dtype=float).reshape(-1)
    if x_array.shape[1] != coef.shape[0]:
        raise ValueError("coefficient count must match the number of X columns")
    return float(intercept) + x_array @ coef


def fit_static_triplet(
    log_prices: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    triplet_id: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fits log(target) = alpha + beta_1*log(hedge_1) + beta_2*log(hedge_2)
    once, using the entire available history. This is the "static" hedge
    ratio -- a single fixed relationship for the whole sample, as opposed
    to `rolling_ols`/`rolling_ridge`/the Kalman filter, which re-estimate
    it over time. Useful as a baseline to compare the adaptive methods
    against, but a static ratio can't respond if the true relationship
    between the three assets drifts.
    """
    if len(hedge_cols) != 2:
        # This is a triangular relationship by construction (one target,
        # two hedge legs) -- the whole project's math assumes exactly two
        # hedge columns, not a general N-asset regression.
        raise ValueError("triangular regression requires exactly two hedge columns")

    required = [target_col, *hedge_cols]
    missing = [col for col in required if col not in log_prices.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    # Rows where any of the three series has a gap (missing price) are
    # dropped entirely rather than filled -- an interpolated price would
    # be a fabricated data point feeding into the fit.
    clean = log_prices[required].dropna().copy()
    result = fit_ols(
        clean.loc[:, hedge_cols].to_numpy(),
        clean.loc[:, target_col].to_numpy(),
        feature_names=hedge_cols,
    )
    identifier = triplet_id or f"{target_col}_{hedge_cols[0]}_{hedge_cols[1]}"
    coefficient_table = pd.DataFrame(
        [
            {
                "triplet_id": identifier,
                "target_symbol": target_col,
                "hedge_symbol_1": hedge_cols[0],
                "hedge_symbol_2": hedge_cols[1],
                "alpha": result.intercept,
                "beta_1": float(result.coefficients[0]),
                "beta_2": float(result.coefficients[1]),
                "r_squared": result.r_squared,
                "n_obs": int(clean.shape[0]),
                "method": "static_ols",
            }
        ]
    )
    residual_table = pd.DataFrame(
        {
            "date": clean.index,
            "triplet_id": identifier,
            "target_symbol": target_col,
            "hedge_symbol_1": hedge_cols[0],
            "hedge_symbol_2": hedge_cols[1],
            "actual_log_price": clean.loc[:, target_col].to_numpy(dtype=float),
            "fitted_log_price": result.fitted_values,
            # residual = actual - fitted: the "gap" the whole strategy
            # trades. Positive means the target is trading rich relative
            # to what the hedge basket predicts; negative means cheap.
            "residual": result.residuals,
            "method": "static_ols_full_sample",
        }
    )
    return coefficient_table, residual_table


def static_out_of_sample_residuals(
    log_prices: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    window: int,
    triplet_id: Optional[str] = None,
) -> pd.DataFrame:
    """Fits the hedge ratio ONCE on the first `window` observations, then
    applies that same fixed fit to every subsequent day without
    refitting. This sits between `fit_static_triplet` (one fit, scored
    on its own training data) and `rolling_ols` (refits every day): it's
    still a single fixed relationship, but the residuals it reports are
    genuinely out-of-sample -- computed on data the fit never saw.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if len(hedge_cols) != 2:
        raise ValueError("triangular regression requires exactly two hedge columns")

    required = [target_col, *hedge_cols]
    missing = [col for col in required if col not in log_prices.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    clean = log_prices[required].dropna().copy()
    if clean.shape[0] <= window:
        # Not enough history to both train on `window` days and still
        # have at least one out-of-sample day left to score.
        return pd.DataFrame(columns=_rolling_output_columns())

    # Train once on the first `window` rows...
    train = clean.iloc[:window]
    result = fit_ols(
        train.loc[:, hedge_cols].to_numpy(),
        train.loc[:, target_col].to_numpy(),
        feature_names=hedge_cols,
    )
    rows = []
    identifier = triplet_id or f"{target_col}_{hedge_cols[0]}_{hedge_cols[1]}"
    # ...then walk forward one day at a time, scoring that same fixed fit
    # against each day in turn. None of these days were used to estimate
    # alpha/beta_1/beta_2, so the residual here reflects genuine
    # prediction error, not curve-fitting to the same data.
    for i in range(window, clean.shape[0]):
        prediction_row = clean.iloc[[i]]
        fitted = ols_predict(
            prediction_row.loc[:, hedge_cols].to_numpy(),
            result.intercept,
            result.coefficients,
        )[0]
        actual = float(prediction_row[target_col].iloc[0])
        rows.append(
            {
                "date": clean.index[i],
                "triplet_id": identifier,
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
                "ridge_alpha": np.nan,
                "method": "static_ols_initial_window",
            }
        )
    return pd.DataFrame(rows, columns=_rolling_output_columns())


def rolling_ols(
    log_prices: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    window: int,
    min_obs: Optional[int] = None,
    triplet_id: Optional[str] = None,
) -> pd.DataFrame:
    """Re-fits the hedge ratio from scratch every day, using only the
    trailing `window` days of history each time, then scores that day's
    fresh fit against the very next observation.

    This is the adaptive alternative to `fit_static_triplet`: if the true
    relationship between the target and its hedge legs drifts over time
    (a company's business mix changes, sector composition shifts, etc.),
    a rolling fit tracks that drift instead of staying anchored to
    whatever was true when the static fit was estimated. The tradeoff is
    noise -- a short window makes the coefficients jumpy; a long window
    makes them slow to react.
    """
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if len(hedge_cols) != 2:
        raise ValueError("triangular regression requires exactly two hedge columns")

    required = [target_col, *hedge_cols]
    missing = [col for col in required if col not in log_prices.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    clean = log_prices[required].dropna().copy()
    if clean.shape[0] <= window:
        return pd.DataFrame(columns=_rolling_output_columns())

    # min_obs guards against fitting a 2-predictor regression on a window
    # that's mostly gaps after dropna -- e.g. a window of 60 calendar days
    # that only has 5 valid trading days in it because of missing data.
    min_required = window if min_obs is None else int(min_obs)
    if min_required <= len(hedge_cols):
        raise ValueError("min_obs must exceed the number of hedge columns")
    if min_required > window:
        raise ValueError("min_obs cannot exceed window")

    rows = []
    identifier = triplet_id or f"{target_col}_{hedge_cols[0]}_{hedge_cols[1]}"
    # Slide a `window`-day training block forward one day at a time. At
    # each step, everything up to (but not including) day i is used to
    # fit, and the fit is scored only against day i -- so day i's own
    # price never leaks into its own fitted value.
    for i in range(window, clean.shape[0]):
        train = clean.iloc[i - window : i]
        if train.dropna().shape[0] < min_required:
            continue

        result = fit_ols(
            train.loc[:, hedge_cols].to_numpy(),
            train.loc[:, target_col].to_numpy(),
            feature_names=hedge_cols,
        )
        prediction_row = clean.iloc[[i]]
        fitted = ols_predict(
            prediction_row.loc[:, hedge_cols].to_numpy(),
            result.intercept,
            result.coefficients,
        )[0]
        actual = float(prediction_row[target_col].iloc[0])
        rows.append(
            {
                "date": clean.index[i],
                "triplet_id": identifier,
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
                "ridge_alpha": np.nan,
                "method": "rolling_ols",
            }
        )

    return pd.DataFrame(rows, columns=_rolling_output_columns())


def estimate_dynamic_hedges_for_triplets(
    log_prices: pd.DataFrame,
    triplets: Sequence[dict],
    window: int,
    ridge_alpha: float,
) -> dict[str, pd.DataFrame]:
    """Runs rolling OLS, rolling ridge, and the static-out-of-sample
    baseline for every triplet in `triplets`, and stacks the results into
    three combined tables. This is the batch entry point the pipeline
    scripts call instead of looping over triplets by hand -- one triplet
    with a data problem (see the KeyError below) currently stops the
    whole batch, which is why callers upstream of this function (e.g.
    scripts/run_universe_pipeline.py) pre-filter to triplets with all
    required symbols present before calling it.
    """
    rolling_frames = []
    ridge_frames = []
    residual_frames = []

    for triplet in triplets:
        target = triplet["target"]
        # accepts either hedge_1/hedge_2 or the older anchor_1/anchor_2
        # naming, so callers built against either convention both work
        hedge_cols = [triplet.get("hedge_1", triplet.get("anchor_1")), triplet.get("hedge_2", triplet.get("anchor_2"))]
        if hedge_cols[0] is None or hedge_cols[1] is None:
            raise KeyError("triplet dictionaries must include hedge_1/hedge_2 or anchor_1/anchor_2")
        triplet_id = triplet.get("triplet_id", f"{target}_{hedge_cols[0]}_{hedge_cols[1]}")

        static_frame = static_out_of_sample_residuals(
            log_prices=log_prices,
            target_col=target,
            hedge_cols=hedge_cols,
            window=window,
            triplet_id=triplet_id,
        )
        rolling_frame = rolling_ols(
            log_prices=log_prices,
            target_col=target,
            hedge_cols=hedge_cols,
            window=window,
            triplet_id=triplet_id,
        )
        ridge_frame = rolling_ridge(
            log_prices=log_prices,
            target_col=target,
            hedge_cols=hedge_cols,
            window=window,
            alpha=ridge_alpha,
            triplet_id=triplet_id,
        )

        rolling_frames.append(rolling_frame)
        ridge_frames.append(ridge_frame)
        # dynamic_residuals combines all three methods' residuals into one
        # table -- downstream code (cointegration testing, labeling) can
        # then treat "method" as just another grouping column rather than
        # needing three separate code paths.
        residual_frames.extend([static_frame, rolling_frame, ridge_frame])

    return {
        "rolling_coefficients": _concat_or_empty(rolling_frames, _rolling_output_columns()),
        "ridge_coefficients": _concat_or_empty(ridge_frames, _rolling_output_columns()),
        "dynamic_residuals": _concat_or_empty(residual_frames, _rolling_output_columns()),
    }


def _concat_or_empty(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    # pd.concat on an all-empty list of frames raises; this returns a
    # correctly-shaped empty frame instead so callers don't need a
    # separate empty-input special case.
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=columns)
    return pd.concat(valid, ignore_index=True)


def _rolling_output_columns() -> list[str]:
    # Single source of truth for the schema every hedge-ratio table in
    # this module shares (static, rolling OLS, rolling ridge) -- keeping
    # them column-identical is what lets downstream code concatenate and
    # group across methods without special-casing any one of them.
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
