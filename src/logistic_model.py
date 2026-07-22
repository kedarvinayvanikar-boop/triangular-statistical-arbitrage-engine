from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LogisticConfig:
    learning_rate: float = 0.05   # how big a step gradient descent takes each iteration
    max_iter: int = 2_000         # hard cap on training iterations, in case convergence never triggers
    l2_penalty: float = 0.0       # ridge-style penalty discouraging large weights; 0 disables it
    fit_intercept: bool = True
    tolerance: float = 1e-8       # stop early once the loss stops improving by more than this
    standardize: bool = True      # rescale features to mean 0 / std 1 before training
    threshold: float = 0.5        # default probability cutoff for converting a probability to a 0/1 prediction


@dataclass(frozen=True)
class LogisticModelResult:
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    intercept: float
    losses: tuple[float, ...]      # loss at every training iteration, kept for inspecting convergence
    feature_means: np.ndarray      # needed to standardize new data the same way at prediction time
    feature_scales: np.ndarray
    config: LogisticConfig


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Squashes any real number into the (0, 1) range -- this is what
    turns a raw weighted sum of features into something that can be read
    as a probability. sigmoid(0) = 0.5, large positive inputs approach 1,
    large negative inputs approach 0.

    Computed in two branches (for x >= 0 and x < 0) rather than the single
    textbook formula 1/(1+e^-x): for a large negative x, e^-x explodes to
    a huge number and can overflow; the x < 0 branch instead computes
    e^x/(1+e^x), which is mathematically identical but never evaluates
    exp() on a large positive number. Same function, numerically safer.
    """
    x = np.asarray(values, dtype=float)
    out = np.empty_like(x, dtype=float)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binary_cross_entropy(y_true: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    """The loss function logistic regression minimizes -- also called
    log loss. For each example it's -log(p) if the true label is 1, or
    -log(1-p) if the true label is 0: a confident WRONG prediction (e.g.
    predicting p=0.99 for something that was actually a 0) is penalized
    far more heavily than a mildly wrong one, which is what pushes the
    model toward well-calibrated probabilities rather than just "mostly
    right" ones.

    Probabilities are clipped away from exactly 0 or 1 (`eps`) because
    log(0) is undefined -- a model that ever became perfectly confident
    would otherwise crash the loss calculation instead of just being
    heavily penalized if wrong.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), eps, 1.0 - eps)
    if y.shape[0] == 0:
        raise ValueError("at least one observation is required")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_logistic_regression(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray | pd.Series,
    config: Optional[LogisticConfig] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> LogisticModelResult:
    """Trains a logistic regression classifier from scratch using batch
    gradient descent -- no scikit-learn involved.

    The training loop, in plain terms, repeated up to `max_iter` times:
      1. Make a prediction with the current weights (sigmoid of a
         weighted sum of features).
      2. Measure how wrong those predictions were (binary_cross_entropy).
      3. Compute the gradient: the direction and size of the "nudge" to
         each weight that would have reduced the loss.
      4. Move every weight a small step (`learning_rate`) in that
         direction.
      5. Stop early if the loss barely changed since last iteration
         (`tolerance`) -- there's nothing left to gain from more steps.

    The gradient formula used here, `X' @ (probabilities - target) / n`,
    is the textbook logistic-regression gradient -- it drops out cleanly
    from the calculus of binary cross-entropy combined with the sigmoid,
    which is part of why logistic regression + log loss is such a common
    pairing (the gradient has no leftover sigmoid-derivative term to carry
    around, unlike squared-error loss would).
    """
    cfg = config or LogisticConfig()
    _validate_config(cfg)
    matrix, names = _coerce_matrix(X, feature_names=feature_names)
    target = _coerce_target(y)
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")

    # standardizing puts every feature on a comparable scale (mean 0,
    # std 1) before training -- without this, a feature measured in the
    # thousands (e.g. volume) would dominate the gradient purely because
    # of its scale, not because it's actually more predictive
    scaled, means, scales = _standardize_matrix(matrix, standardize=cfg.standardize)
    X_design = _add_intercept(scaled) if cfg.fit_intercept else scaled
    weights = np.zeros(X_design.shape[1], dtype=float)  # start every weight at zero -- no prior assumption about direction
    losses: list[float] = []
    previous_loss: Optional[float] = None

    # the intercept isn't penalized by L2 regularization -- same reasoning
    # as ridge regression not penalizing its intercept (see src/ridge.py):
    # it's a baseline offset, not a "relationship strength" that should be
    # shrunk toward zero
    penalty_mask = np.ones_like(weights)
    if cfg.fit_intercept:
        penalty_mask[0] = 0.0

    n_obs = float(target.shape[0])
    for _ in range(cfg.max_iter):
        linear = X_design @ weights
        probabilities = sigmoid(linear)
        base_loss = binary_cross_entropy(target, probabilities)
        # L2 penalty term: 0.5 * alpha * sum(weight^2), divided by n_obs
        # to keep it on the same scale as the mean loss above regardless
        # of how many training examples there are
        penalty_loss = 0.5 * cfg.l2_penalty * float(np.sum((weights * penalty_mask) ** 2)) / n_obs
        loss = base_loss + penalty_loss
        losses.append(loss)

        # the gradient of binary cross-entropy w.r.t. the weights,
        # evaluated at the current predictions -- "probabilities - target"
        # is literally the prediction error for every example, and this
        # matrix multiply is how that error gets distributed back across
        # every weight in proportion to that feature's value
        gradient = (X_design.T @ (probabilities - target)) / n_obs
        gradient += (cfg.l2_penalty / n_obs) * weights * penalty_mask
        weights = weights - cfg.learning_rate * gradient  # the actual "descent" step

        if previous_loss is not None and abs(previous_loss - loss) < cfg.tolerance:
            # loss has plateaued -- further iterations would just be
            # spending compute for no real improvement
            break
        previous_loss = loss

    if cfg.fit_intercept:
        intercept = float(weights[0])
        coefficients = weights[1:].copy()
    else:
        intercept = 0.0
        coefficients = weights.copy()

    return LogisticModelResult(
        feature_names=names,
        coefficients=coefficients,
        intercept=intercept,
        losses=tuple(float(x) for x in losses),
        feature_means=means,
        feature_scales=scales,
        config=cfg,
    )


def predict_proba(model: LogisticModelResult, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    # New data must be standardized using the SAME mean/scale the model
    # was trained with (model.feature_means / feature_scales), not
    # recomputed from the new data -- otherwise the same raw feature
    # value would map to a different standardized value than it did
    # during training, silently changing what the model has learned.
    matrix, _ = _coerce_matrix(X, feature_names=model.feature_names)
    if matrix.shape[1] != len(model.feature_names):
        raise ValueError("X has a different number of columns than the fitted model")
    scaled = (matrix - model.feature_means) / model.feature_scales
    linear = scaled @ model.coefficients + model.intercept
    return sigmoid(linear)


def predict_labels(model: LogisticModelResult, X: pd.DataFrame | np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
    # Converts a continuous probability into a hard 0/1 call -- but note
    # this is a different, more permissive threshold than the trade
    # filter's own probability_threshold used elsewhere in the pipeline;
    # this one is purely for computing classification metrics like
    # precision/recall/accuracy.
    cutoff = model.config.threshold if threshold is None else float(threshold)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (predict_proba(model, X) >= cutoff).astype(int)


def model_coefficients_frame(model: LogisticModelResult) -> pd.DataFrame:
    # A readable table of what the model actually learned -- which
    # features push the prediction up vs. down, and by how much. This is
    # the practical payoff of using a simple linear model instead of
    # something more opaque: you can look at this table and sanity-check
    # whether the learned relationships make economic sense.
    rows = [{"feature": "intercept", "coefficient": model.intercept}]
    rows.extend(
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in zip(model.feature_names, model.coefficients)
    )
    return pd.DataFrame(rows)


def loss_history_frame(model: LogisticModelResult) -> pd.DataFrame:
    # Plotting this (loss vs. iteration) is how you visually confirm
    # gradient descent actually converged rather than diverging or
    # oscillating -- a healthy run looks like a smooth decreasing curve
    # that flattens out.
    return pd.DataFrame(
        {
            "iteration": np.arange(1, len(model.losses) + 1, dtype=int),
            "loss": np.asarray(model.losses, dtype=float),
        }
    )


def make_prediction_frame(
    frame: pd.DataFrame,
    model: LogisticModelResult,
    feature_columns: Sequence[str],
    split_name: str,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    # Wraps raw probabilities with the identifying columns (event_id,
    # triplet_id, etc.) needed to join predictions back to the trade log
    # later -- a bare array of probabilities on its own can't be matched
    # back to which specific event each one belongs to.
    required = ["event_id", "triplet_id", "method", "event_date"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"missing prediction metadata columns: {missing}")
    probabilities = predict_proba(model, frame.loc[:, list(feature_columns)])
    cutoff = model.config.threshold if threshold is None else float(threshold)
    predictions = (probabilities >= cutoff).astype(int)
    output = frame.loc[:, required].copy()
    output["split"] = split_name
    output["predicted_reversion_probability"] = probabilities
    output["classification_threshold"] = cutoff
    output["predicted_label"] = predictions
    if "label" in frame.columns:
        output["label"] = frame["label"].astype(int).to_numpy()
    return output


def validation_metrics(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Standard binary classification metrics, computed without
    scikit-learn so the evaluation stays consistent with the from-scratch
    modeling approach used everywhere else in this project.

    - true/false positive/negative: of the events predicted "will
      revert" (positive), how many actually did (true positive) vs.
      didn't (false positive)? And of the events predicted "won't
      revert," how many actually didn't (true negative) vs. did (false
      negative)?
    - precision = true positives / everything predicted positive. "When
      the model says yes, how often is it right?"
    - recall = true positives / everything that was actually positive.
      "Of all the real successes, how many did the model catch?"
    - f1 = the harmonic mean of precision and recall -- a single number
      that penalizes an imbalance between the two rather than letting one
      metric look great while the other is poor.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    y = _coerce_target(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp > 0 else np.nan
    recall = tp / (tp + fn) if tp + fn > 0 else np.nan
    f1 = 2.0 * precision * recall / (precision + recall) if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0 else np.nan
    metrics = {
        "n_obs": int(y.shape[0]),
        "positive_rate": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(np.mean(pred == y)),
        "precision": float(precision) if np.isfinite(precision) else np.nan,
        "recall": float(recall) if np.isfinite(recall) else np.nan,
        "f1": float(f1) if np.isfinite(f1) else np.nan,
        "log_loss": binary_cross_entropy(y, p),
        "brier_score": float(np.mean((p - y) ** 2)),
        "roc_auc": _roc_auc(y, p),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    return pd.DataFrame([metrics])


def calibration_table(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    """Groups predictions into probability buckets (e.g. 0.6-0.7,
    0.7-0.8, ...) and checks whether the model's confidence matches
    reality: among events the model gave a ~70% probability, did roughly
    70% of them actually succeed? A well-calibrated model's buckets track
    closely along the diagonal (predicted ~= observed); a model that's
    "right in aggregate but wrong locally" -- confident and wrong more
    than it should be, or under-confident -- shows up here as a
    systematic gap between the two columns.
    """
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y = _coerce_target(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = pd.cut(p, bins=bins, include_lowest=True, right=True)
    frame = pd.DataFrame({"probability": p, "label": y, "probability_bucket": bucket})
    grouped = frame.groupby("probability_bucket", observed=False)
    out = grouped.agg(
        n_events=("label", "size"),
        mean_predicted_probability=("probability", "mean"),
        observed_success_rate=("label", "mean"),
    ).reset_index()
    out["probability_bucket"] = out["probability_bucket"].astype(str)
    return out


def time_ordered_split(
    frame: pd.DataFrame,
    date_col: str = "event_date",
    train_size: float = 0.6,
    validation_size: float = 0.2,
) -> TimeSplit:
    """Splits events into train/validation/test chunks ordered strictly
    by date -- train on the earliest 60%, validate on the next 20%, test
    on the most recent 20%. This is the opposite of the usual "shuffle
    randomly then split" approach taught for most ML problems, and
    deliberately so: shuffling time-series data before splitting would
    let the model train on events that happened *after* some of its
    validation events chronologically, which is a form of lookahead bias
    -- in the real world you can never train on the future to predict the
    past.
    """
    if date_col not in frame.columns:
        raise KeyError(f"missing date column: {date_col}")
    if not 0.0 < train_size < 1.0 or not 0.0 < validation_size < 1.0:
        raise ValueError("split sizes must be between 0 and 1")
    if train_size + validation_size >= 1.0:
        raise ValueError("train_size + validation_size must be less than 1")
    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col, "event_id" if "event_id" in ordered.columns else date_col]).reset_index(drop=True)
    n = ordered.shape[0]
    if n < 3:
        raise ValueError("at least three observations are required for train/validation/test split")
    train_end = max(1, int(np.floor(n * train_size)))
    validation_end = max(train_end + 1, int(np.floor(n * (train_size + validation_size))))
    validation_end = min(validation_end, n - 1)
    return TimeSplit(
        train=ordered.iloc[:train_end].reset_index(drop=True),
        validation=ordered.iloc[train_end:validation_end].reset_index(drop=True),
        test=ordered.iloc[validation_end:].reset_index(drop=True),
    )


def walk_forward_splits(
    frame: pd.DataFrame,
    date_col: str = "event_date",
    initial_train_size: int = 20,
    validation_size: int = 5,
    step_size: int = 5,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generates a sequence of (train, validation) pairs that slide
    forward through time -- train on the first `initial_train_size`
    events, validate on the next `validation_size`; then grow the
    training set by `step_size` events and validate on the next chunk
    after that; repeat until the data runs out. This simulates
    periodically retraining a model as new data becomes available (which
    is closer to how a model would actually be operated) rather than
    evaluating one single fixed train/test split.
    """
    if initial_train_size < 1 or validation_size < 1 or step_size < 1:
        raise ValueError("walk-forward sizes must be positive")
    if date_col not in frame.columns:
        raise KeyError(f"missing date column: {date_col}")
    ordered = frame.copy()
    ordered[date_col] = pd.to_datetime(ordered[date_col])
    ordered = ordered.sort_values([date_col, "event_id" if "event_id" in ordered.columns else date_col]).reset_index(drop=True)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start = initial_train_size
    while start + validation_size <= ordered.shape[0]:
        train = ordered.iloc[:start].reset_index(drop=True)
        validation = ordered.iloc[start : start + validation_size].reset_index(drop=True)
        splits.append((train, validation))
        start += step_size
    return splits


def select_feature_columns(
    frame: pd.DataFrame,
    label_col: str = "label",
    exclude: Optional[Sequence[str]] = None,
) -> list[str]:
    # Auto-selects every numeric column except known identifier/metadata
    # columns and the label itself -- so adding a new engineered feature
    # in src/features.py doesn't require also updating a hardcoded list
    # here.
    default_exclude = {
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "side",
        "outcome",
        "exit_reason",
        "holding_period",
        label_col,
    }
    if exclude is not None:
        default_exclude.update(exclude)
    numeric = []
    for col in frame.columns:
        if col in default_exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            numeric.append(col)
    if not numeric:
        raise ValueError("no numeric feature columns were found")
    return numeric


def prepare_model_frame(
    feature_matrix: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    label_col: str = "label",
) -> tuple[pd.DataFrame, list[str]]:
    if label_col not in feature_matrix.columns:
        raise KeyError(f"missing label column: {label_col}")
    frame = feature_matrix.copy()
    frame = frame.loc[frame[label_col].notna()].copy()
    if frame.empty:
        raise ValueError("at least one labeled event is required")
    cols = list(feature_columns) if feature_columns is not None else select_feature_columns(frame, label_col=label_col)
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    for col in cols:
        # missing feature values are filled with that column's own
        # median rather than dropped -- dropping would throw away the
        # whole event (including its label) just because one feature was
        # unavailable, which is wasteful when most of that event's other
        # features are still informative
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
        median = frame[col].median()
        fill_value = 0.0 if pd.isna(median) else float(median)
        frame[col] = frame[col].fillna(fill_value)
    frame[label_col] = frame[label_col].astype(int)
    return frame, cols


def train_event_logistic_model(
    feature_matrix: pd.DataFrame,
    config: Optional[LogisticConfig] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> dict[str, object]:
    # End-to-end entry point: prepare data, split by time, train on the
    # earliest chunk only, then score on validation and test (both of
    # which the model never saw during training) -- this is the function
    # scripts/run_universe_pipeline.py actually calls.
    model_frame, cols = prepare_model_frame(feature_matrix, feature_columns=feature_columns)
    split = time_ordered_split(model_frame)
    model = fit_logistic_regression(split.train.loc[:, cols], split.train["label"], config=config, feature_names=cols)
    validation_prob = predict_proba(model, split.validation.loc[:, cols])
    test_prob = predict_proba(model, split.test.loc[:, cols])
    validation_pred = make_prediction_frame(split.validation, model, cols, split_name="validation")
    test_pred = make_prediction_frame(split.test, model, cols, split_name="test")
    metrics = pd.concat(
        [
            validation_metrics(split.validation["label"], validation_prob).assign(split="validation"),
            validation_metrics(split.test["label"], test_prob).assign(split="test"),
        ],
        ignore_index=True,
        sort=False,
    )
    predictions = pd.concat([validation_pred, test_pred], ignore_index=True, sort=False)
    return {
        "model": model,
        "feature_columns": cols,
        "split": split,
        "model_coefficients": model_coefficients_frame(model),
        "training_loss": loss_history_frame(model),
        "predicted_reversion_probabilities": predictions,
        "validation_metrics": metrics,
    }


def _standardize_matrix(matrix: np.ndarray, standardize: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not standardize:
        means = np.zeros(matrix.shape[1], dtype=float)
        scales = np.ones(matrix.shape[1], dtype=float)
        return matrix.astype(float), means, scales
    means = np.nanmean(matrix, axis=0)
    scales = np.nanstd(matrix, axis=0)
    # a feature that's constant across all training examples has zero
    # variance -- dividing by that zero would produce inf/NaN, so its
    # scale is treated as 1 instead (the feature just stays at its
    # original, already-constant value after "standardizing")
    scales = np.where(scales == 0.0, 1.0, scales)
    scaled = (matrix - means) / scales
    return scaled, means, scales


def _coerce_matrix(X: pd.DataFrame | np.ndarray, feature_names: Optional[Sequence[str]] = None) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        if feature_names is not None:
            missing = [col for col in feature_names if col not in X.columns]
            if missing:
                raise KeyError(f"missing feature columns: {missing}")
            frame = X.loc[:, list(feature_names)]
            names = tuple(feature_names)
        else:
            frame = X
            names = tuple(str(col) for col in X.columns)
        matrix = frame.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if feature_names is None:
            names = tuple(f"x_{i}" for i in range(matrix.shape[1]))
        else:
            names = tuple(feature_names)
    if matrix.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length must match the number of columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X contains non-finite values")
    return matrix, names


def _coerce_target(y: Sequence[int] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(y, dtype=float).reshape(-1)
    if arr.shape[0] == 0:
        raise ValueError("target cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("target contains non-finite values")
    unique = set(np.unique(arr))
    if not unique.issubset({0.0, 1.0}):
        # this is a binary classifier -- a label of 2, -1, or a
        # continuous value would silently produce nonsensical gradients
        # rather than an obvious error, so it's rejected up front instead
        raise ValueError("target must contain only 0 and 1")
    return arr.astype(float)


def _add_intercept(matrix: np.ndarray) -> np.ndarray:
    # same trick as OLS/ridge: prepend a column of 1s so the intercept
    # can be estimated as just another weight, rather than needing
    # special-cased handling throughout the gradient descent loop
    return np.column_stack([np.ones(matrix.shape[0], dtype=float), matrix])


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve, computed via the rank-sum (Mann-Whitney
    U) shortcut rather than actually building the curve point by point.
    Intuition: AUC equals the probability that a randomly chosen positive
    example gets a higher predicted score than a randomly chosen negative
    example. 0.5 means the model is no better than random guessing; 1.0
    means it perfectly ranks every positive above every negative.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        # AUC is undefined without at least one example of each class --
        # there's nothing to rank against
        return np.nan
    ranks = pd.Series(s).rank(method="average").to_numpy(dtype=float)
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _validate_config(cfg: LogisticConfig) -> None:
    if cfg.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if cfg.max_iter < 1:
        raise ValueError("max_iter must be positive")
    if cfg.l2_penalty < 0.0:
        raise ValueError("l2_penalty cannot be negative")
    if cfg.tolerance < 0.0:
        raise ValueError("tolerance cannot be negative")
    if not 0.0 < cfg.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
