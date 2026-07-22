from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .regression import fit_ols
from .residuals import residual_summary


@dataclass(frozen=True)
class KalmanConfig:
    # process_noise controls how much the hedge ratios are allowed to
    # drift day to day -- larger values let alpha/beta_1/beta_2 change
    # faster, at the cost of a noisier estimate. Can be one number applied
    # to all three state components, or three separate numbers.
    process_noise: float | Sequence[float] = 1e-5
    # measurement_noise represents how much of the day-to-day price
    # movement is treated as pure observation noise rather than a real
    # shift in the relationship -- higher values make the filter trust
    # each individual observation less and change its estimate more
    # slowly.
    measurement_noise: float = 1e-4
    initial_covariance: float = 1.0
    # if set, the filter's starting alpha/beta_1/beta_2 come from an
    # ordinary OLS fit on the first `initial_window` days, rather than
    # starting from a guess and letting the filter converge on its own
    initial_window: Optional[int] = None


@dataclass(frozen=True)
class KalmanStepResult:
    predicted_state: np.ndarray
    filtered_state: np.ndarray
    predicted_covariance: np.ndarray
    filtered_covariance: np.ndarray
    innovation: float
    innovation_variance: float
    kalman_gain: np.ndarray
    fitted_value: float


def kalman_predict(
    state: object,
    covariance: object,
    process_noise: float | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The Kalman filter's "predict" half-step.

    The hidden state here is [alpha, beta_1, beta_2] -- the triangular
    hedge ratio -- modeled as a random walk: tomorrow's true value is
    expected to equal today's, plus some uncertainty. So the predicted
    state is just a copy of the current state (a random walk's best
    forecast of tomorrow is today), while the covariance (how uncertain
    we are about that state) grows by `process_noise` to reflect that a
    day has passed and the true hedge ratio could have drifted.
    """
    state_array = _as_state_vector(state)
    covariance_array = _as_covariance_matrix(covariance)
    q_matrix = _process_noise_matrix(process_noise)
    return state_array.copy(), covariance_array + q_matrix


def kalman_update(
    predicted_state: object,
    predicted_covariance: object,
    observation: float,
    observation_vector: object,
    measurement_noise: float,
) -> KalmanStepResult:
    """The Kalman filter's "update" half-step: incorporates one new price
    observation into the predicted state from `kalman_predict`.

    Step by step:
      1. `fitted` is what the predicted state says today's target log
         price *should* be, given today's hedge leg prices (observation_vector
         is [1, hedge_1_price, hedge_2_price], so this is just alpha +
         beta_1*hedge_1 + beta_2*hedge_2 using the predicted coefficients).
      2. `innovation` is the surprise: actual minus predicted. This IS the
         Kalman-filtered residual used everywhere downstream as the
         trading signal -- the gap between what happened and what the
         current hedge ratio estimate expected to happen.
      3. `innovation_variance` measures how much uncertainty was in that
         prediction, combining the state's own uncertainty with the
         measurement noise.
      4. `gain` (the Kalman gain) decides how much to move the state
         estimate in response to the innovation: high when the state was
         very uncertain relative to measurement noise (trust today's
         observation more), low when the state was already confident
         (trust the observation less, treat it as noise).
      5. The state and its covariance are both updated using that gain --
         the coefficients shift toward explaining today's price better,
         and the uncertainty shrinks now that one more day of evidence has
         been folded in.
    """
    if measurement_noise <= 0:
        raise ValueError("measurement_noise must be positive")

    state_pred = _as_state_vector(predicted_state)
    cov_pred = _as_covariance_matrix(predicted_covariance)
    h = np.asarray(observation_vector, dtype=float).reshape(1, -1)
    if h.shape != (1, 3):
        raise ValueError("observation_vector must contain intercept, hedge_1, and hedge_2 terms")
    if not np.isfinite(h).all():
        raise ValueError("observation_vector contains NaN or infinite values")

    fitted = float((h @ state_pred)[0])
    innovation = float(observation) - fitted
    innovation_variance = float((h @ cov_pred @ h.T)[0, 0] + measurement_noise)
    if innovation_variance <= 0 or not np.isfinite(innovation_variance):
        raise ValueError("innovation variance must be positive and finite")

    # Kalman gain: how much of the innovation to actually apply to the
    # state estimate, scaled by how the state's own uncertainty compares
    # to the total (state + measurement) uncertainty.
    gain = (cov_pred @ h.T / innovation_variance).reshape(-1)
    state_filtered = state_pred + gain * innovation

    # Joseph-form covariance update: numerically more stable than the
    # textbook-simplest form (cov_filtered = (I - gain@h) @ cov_pred),
    # because it stays symmetric and positive-semidefinite even when
    # floating-point rounding would otherwise slowly corrupt it over many
    # iterations -- and this filter runs over years of daily data, so
    # small per-step errors would otherwise compound.
    identity = np.eye(3)
    adjustment = identity - np.outer(gain, h.reshape(-1))
    cov_filtered = adjustment @ cov_pred @ adjustment.T + np.outer(gain, gain) * measurement_noise
    # Forces exact symmetry (cov_filtered should be symmetric
    # mathematically, but floating-point arithmetic can leave tiny
    # asymmetries that compound over thousands of steps).
    cov_filtered = 0.5 * (cov_filtered + cov_filtered.T)

    return KalmanStepResult(
        predicted_state=state_pred,
        filtered_state=state_filtered.astype(float),
        predicted_covariance=cov_pred.astype(float),
        filtered_covariance=cov_filtered.astype(float),
        innovation=float(innovation),
        innovation_variance=float(innovation_variance),
        kalman_gain=gain.astype(float),
        fitted_value=float(fitted),
    )


def kalman_filter_dynamic_regression(
    log_prices: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    config: Optional[KalmanConfig] = None,
    triplet_id: Optional[str] = None,
    initial_state: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Runs the predict/update cycle across an entire price history for
    one triplet, day by day, producing a continuously-updated hedge ratio
    instead of the step-function behavior a rolling window produces
    (rolling OLS's coefficients jump abruptly every time the window
    slides past an old, influential data point; the Kalman filter's
    coefficients move smoothly, one day's evidence at a time).
    """
    if len(hedge_cols) != 2:
        raise ValueError("triangular regression requires exactly two hedge columns")

    required = [target_col, *hedge_cols]
    missing = [col for col in required if col not in log_prices.columns]
    if missing:
        raise KeyError(f"missing columns: {missing}")

    cfg = config or KalmanConfig()
    if cfg.measurement_noise <= 0:
        raise ValueError("measurement_noise must be positive")
    if cfg.initial_covariance <= 0:
        raise ValueError("initial_covariance must be positive")

    clean = log_prices.loc[:, required].dropna().copy()
    if clean.empty:
        return pd.DataFrame(columns=_kalman_output_columns())

    identifier = triplet_id or f"{target_col}_{hedge_cols[0]}_{hedge_cols[1]}"
    state, start_index = _initial_state_and_start(
        clean=clean,
        target_col=target_col,
        hedge_cols=hedge_cols,
        initial_state=initial_state,
        initial_window=cfg.initial_window,
    )
    # Starting uncertainty: a diagonal matrix means the filter starts out
    # assuming alpha, beta_1, and beta_2 are uncertain but unrelated to
    # each other -- any correlation between them gets learned from the
    # data as the filter runs, not assumed up front.
    covariance = np.eye(3) * float(cfg.initial_covariance)

    rows = []
    for i in range(start_index, clean.shape[0]):
        # predict today's state from yesterday's filtered state, then
        # immediately correct that prediction using today's actual prices
        state_pred, cov_pred = kalman_predict(state, covariance, cfg.process_noise)
        row = clean.iloc[i]
        observation_vector = np.array([1.0, float(row[hedge_cols[0]]), float(row[hedge_cols[1]])])
        step = kalman_update(
            predicted_state=state_pred,
            predicted_covariance=cov_pred,
            observation=float(row[target_col]),
            observation_vector=observation_vector,
            measurement_noise=cfg.measurement_noise,
        )

        rows.append(
            {
                "date": clean.index[i],
                "triplet_id": identifier,
                "target_symbol": target_col,
                "hedge_symbol_1": hedge_cols[0],
                "hedge_symbol_2": hedge_cols[1],
                "alpha": float(step.filtered_state[0]),
                "beta_1": float(step.filtered_state[1]),
                "beta_2": float(step.filtered_state[2]),
                "predicted_alpha": float(step.predicted_state[0]),
                "predicted_beta_1": float(step.predicted_state[1]),
                "predicted_beta_2": float(step.predicted_state[2]),
                "actual_log_price": float(row[target_col]),
                "fitted_log_price": float(step.fitted_value),
                "residual": float(step.innovation),
                "residual_variance": float(step.innovation_variance),
                "kalman_gain_alpha": float(step.kalman_gain[0]),
                "kalman_gain_beta_1": float(step.kalman_gain[1]),
                "kalman_gain_beta_2": float(step.kalman_gain[2]),
                "state_cov_trace": float(np.trace(step.filtered_covariance)),
                "process_noise": _noise_value_for_storage(cfg.process_noise),
                "measurement_noise": float(cfg.measurement_noise),
                "initial_window": np.nan if cfg.initial_window is None else int(cfg.initial_window),
                "method": "kalman_random_walk",
            }
        )
        # this step's filtered output becomes next step's starting point
        state = step.filtered_state
        covariance = step.filtered_covariance

    return pd.DataFrame(rows, columns=_kalman_output_columns())


def estimate_kalman_for_triplets(
    log_prices: pd.DataFrame,
    triplets: Sequence[dict],
    config: Optional[KalmanConfig] = None,
) -> dict[str, pd.DataFrame]:
    # Batch entry point: runs the filter independently for every triplet
    # (no information is shared between triplets -- each one's hedge
    # ratio evolves on its own) and splits the combined output into a
    # "states" table (the coefficient path itself) and a "residuals"
    # table (the innovations used for trading signals), since downstream
    # consumers usually only need one or the other.
    frames = []
    for triplet in triplets:
        target = triplet["target"]
        hedge_cols = [triplet.get("hedge_1", triplet.get("anchor_1")), triplet.get("hedge_2", triplet.get("anchor_2"))]
        if hedge_cols[0] is None or hedge_cols[1] is None:
            raise KeyError("triplet dictionaries must include hedge_1/hedge_2 or anchor_1/anchor_2")
        triplet_id = triplet.get("triplet_id", f"{target}_{hedge_cols[0]}_{hedge_cols[1]}")
        frames.append(
            kalman_filter_dynamic_regression(
                log_prices=log_prices,
                target_col=target,
                hedge_cols=hedge_cols,
                config=config,
                triplet_id=triplet_id,
            )
        )

    kalman = _concat_or_empty(frames, _kalman_output_columns())
    return {
        "kalman_states": kalman.loc[:, _kalman_state_columns()].copy() if not kalman.empty else pd.DataFrame(columns=_kalman_state_columns()),
        "kalman_residuals": kalman.loc[:, _kalman_residual_columns()].copy() if not kalman.empty else pd.DataFrame(columns=_kalman_residual_columns()),
    }


def compare_kalman_residuals(
    kalman_residuals: pd.DataFrame,
    rolling_residuals: pd.DataFrame,
) -> pd.DataFrame:
    # Puts the Kalman filter's residual behavior side by side with rolling
    # OLS/ridge's, summarized the same way (residual_summary), so the
    # three hedge-ratio methods can be compared on equal footing rather
    # than needing method-specific analysis code.
    frames = []
    if rolling_residuals is not None and not rolling_residuals.empty:
        rolling = rolling_residuals.copy()
        if "method" in rolling.columns:
            rolling = rolling[rolling["method"].isin(["rolling_ols", "rolling_ridge"])]
        frames.append(rolling)
    if kalman_residuals is not None and not kalman_residuals.empty:
        frames.append(kalman_residuals.copy())
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    return residual_summary(combined, group_cols=("triplet_id", "method"))


def _initial_state_and_start(
    clean: pd.DataFrame,
    target_col: str,
    hedge_cols: Sequence[str],
    initial_state: Optional[Sequence[float]],
    initial_window: Optional[int],
) -> tuple[np.ndarray, int]:
    # Three ways to start the filter, in priority order:
    #   1. an explicit starting state was given -- use it as-is
    #   2. an initial_window was given -- run a one-off OLS fit on that
    #      many days first, so the filter doesn't start from a blind
    #      guess and waste its first several updates just converging
    #   3. neither -- start with beta_1=beta_2=0 (no relationship assumed
    #      yet) and let the filter learn everything from scratch
    if initial_state is not None:
        return _as_state_vector(initial_state), 0

    if initial_window is not None:
        if initial_window <= len(hedge_cols):
            raise ValueError("initial_window must exceed the number of hedge columns")
        if clean.shape[0] <= initial_window:
            return np.array([float(clean[target_col].iloc[0]), 0.0, 0.0]), 0
        train = clean.iloc[:initial_window]
        result = fit_ols(train.loc[:, hedge_cols].to_numpy(), train.loc[:, target_col].to_numpy())
        return result.params.astype(float), int(initial_window)

    return np.array([float(clean[target_col].iloc[0]), 0.0, 0.0]), 0


def _as_state_vector(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != 3:
        raise ValueError("state must contain alpha, beta_1, and beta_2")
    if not np.isfinite(array).all():
        raise ValueError("state contains NaN or infinite values")
    return array.astype(float)


def _as_covariance_matrix(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3, 3):
        raise ValueError("covariance must be a 3 by 3 matrix")
    if not np.isfinite(array).all():
        raise ValueError("covariance contains NaN or infinite values")
    return array.astype(float)


def _process_noise_matrix(noise: float | Sequence[float] | np.ndarray) -> np.ndarray:
    # Normalizes the three accepted forms of process_noise (single number
    # applied uniformly, a length-3 vector for per-parameter noise, or an
    # already-built 3x3 covariance matrix) into one consistent matrix
    # shape used by kalman_predict.
    array = np.asarray(noise, dtype=float)
    if array.ndim == 0:
        value = float(array)
        if value < 0:
            raise ValueError("process_noise must be non-negative")
        return np.eye(3) * value
    if array.shape == (3,):
        if (array < 0).any():
            raise ValueError("process_noise values must be non-negative")
        return np.diag(array.astype(float))
    if array.shape == (3, 3):
        if not np.isfinite(array).all():
            raise ValueError("process_noise contains NaN or infinite values")
        return array.astype(float)
    raise ValueError("process_noise must be a scalar, length-3 vector, or 3 by 3 matrix")


def _noise_value_for_storage(noise: float | Sequence[float] | np.ndarray) -> float:
    # Collapses whatever process_noise shape was configured down to one
    # representative number, purely for logging in the output table --
    # doesn't affect the filter's actual math.
    array = np.asarray(noise, dtype=float)
    if array.ndim == 0:
        return float(array)
    return float(np.mean(np.diag(_process_noise_matrix(noise))))


def _concat_or_empty(frames: Sequence[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame(columns=columns)
    return pd.concat(valid, ignore_index=True)


def _kalman_state_columns() -> list[str]:
    return [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "predicted_alpha",
        "predicted_beta_1",
        "predicted_beta_2",
        "kalman_gain_alpha",
        "kalman_gain_beta_1",
        "kalman_gain_beta_2",
        "state_cov_trace",
        "process_noise",
        "measurement_noise",
        "initial_window",
        "method",
    ]


def _kalman_residual_columns() -> list[str]:
    return [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "method",
        "actual_log_price",
        "fitted_log_price",
        "residual",
        "residual_variance",
        "process_noise",
        "measurement_noise",
        "initial_window",
    ]


def _kalman_output_columns() -> list[str]:
    return [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "predicted_alpha",
        "predicted_beta_1",
        "predicted_beta_2",
        "actual_log_price",
        "fitted_log_price",
        "residual",
        "residual_variance",
        "kalman_gain_alpha",
        "kalman_gain_beta_1",
        "kalman_gain_beta_2",
        "state_cov_trace",
        "process_noise",
        "measurement_noise",
        "initial_window",
        "method",
    ]
