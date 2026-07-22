from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float = 0.5
    n_probability_buckets: int = 5


def confusion_matrix_frame(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    # The four-cell breakdown every other classification metric here is
    # built from: for each (actual, predicted) combination, how many
    # events fell into it. E.g. actual=1, predicted=1 is a true positive;
    # actual=0, predicted=1 is a false positive (the model cried wolf).
    y, p = _coerce_binary_inputs(y_true, probabilities)
    cutoff = _validate_threshold(threshold)
    pred = (p >= cutoff).astype(int)
    rows = []
    for actual in (0, 1):
        for predicted in (0, 1):
            rows.append(
                {
                    "actual_label": actual,
                    "predicted_label": predicted,
                    "count": int(np.sum((y == actual) & (pred == predicted))),
                    "threshold": cutoff,
                }
            )
    return pd.DataFrame(rows)


def classification_summary(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    threshold: float = 0.5,
    split: str | None = None,
) -> pd.DataFrame:
    """One-row summary of every standard binary classification metric.
    Precision/recall/f1 are explained in src/logistic_model.py's
    validation_metrics docstring; the two added here:
      - specificity = true negatives / all actual negatives -- the
        mirror image of recall, asking "of the events that really
        didn't revert, how many did the model correctly call?"
      - negative predictive value = true negatives / everything
        predicted negative -- the mirror image of precision, "when the
        model says no, how often is it right?"
    """
    y, p = _coerce_binary_inputs(y_true, probabilities)
    cutoff = _validate_threshold(threshold)
    pred = (p >= cutoff).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    negative_predictive_value = _safe_divide(tn, tn + fn)

    row = {
        "split": split if split is not None else "all",
        "n_obs": int(y.shape[0]),
        "positive_rate": float(np.mean(y)),
        "threshold": cutoff,
        "accuracy": float(np.mean(pred == y)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "f1": f1,
        "log_loss": log_loss(y, p),
        "brier_score": brier_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    return pd.DataFrame([row])


def brier_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    # Mean squared error between predicted probability and the actual
    # 0/1 outcome -- a simpler, more intuitive calibration measure than
    # log loss (no log involved, so it doesn't blow up near 0 or 1), at
    # the cost of being somewhat less sensitive to badly miscalibrated
    # extreme predictions.
    y, p = _coerce_binary_inputs(y_true, probabilities)
    return float(np.mean((p - y) ** 2))


def log_loss(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    eps: float = 1e-12,
) -> float:
    # Same binary cross-entropy formula the logistic model is trained to
    # minimize (see src/logistic_model.py), reimplemented here as a
    # standalone evaluation metric so it can be computed on ANY
    # probability predictions -- including the decision tree's, which
    # isn't trained via gradient descent on this loss at all.
    y, p = _coerce_binary_inputs(y_true, probabilities)
    clipped = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def roc_curve_frame(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    """The ROC curve itself: true positive rate vs. false positive rate
    at every possible classification threshold. Sweeping the threshold
    from "classify everything as positive" (top-left of the parameter
    range) down to "classify nothing as positive" traces out the curve;
    a model that's better at ranking positives above negatives produces a
    curve that bulges further toward the top-left corner (high true
    positive rate achievable while keeping false positive rate low).
    Every unique predicted probability is tried as a threshold (plus +-inf
    as the two endpoints), rather than a fixed grid, so the curve is exact
    rather than an approximation.
    """
    y, p = _coerce_binary_inputs(y_true, probabilities)
    thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]
    rows = []
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        rows.append(
            {
                "threshold": float(threshold) if np.isfinite(threshold) else threshold,
                "true_positive_rate": _safe_divide(tp, positives),
                "false_positive_rate": _safe_divide(fp, negatives),
                "true_positive": tp,
                "false_positive": fp,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["false_positive_rate", "true_positive_rate"]).reset_index(drop=True)


def roc_auc_score(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    # Area under the ROC curve via direct numerical integration
    # (trapezoidal rule) of the curve computed above -- a more literal,
    # if slightly slower, route to the same number the rank-sum shortcut
    # in src/logistic_model.py:_roc_auc computes; kept as a separate
    # implementation here since this module builds the full curve anyway
    # for roc_curve_frame.
    curve = roc_curve_frame(y_true, probabilities)
    if curve["false_positive_rate"].isna().any() or curve["true_positive_rate"].isna().any():
        return np.nan
    return float(np.trapezoid(curve["true_positive_rate"], curve["false_positive_rate"]))


def probability_bucket_summary(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    # Groups predictions into probability ranges and reports the actual
    # outcome rate within each -- the raw data behind a calibration
    # curve/plot (see calibration_curve_frame below), and useful on its
    # own for spotting which probability range has too few observations
    # to trust.
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _coerce_binary_inputs(y_true, probabilities)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket = pd.cut(p, bins=bins, include_lowest=True, right=True)
    frame = pd.DataFrame({"label": y, "probability": p, "probability_bucket": bucket})
    grouped = frame.groupby("probability_bucket", observed=False)
    output = grouped.agg(
        n_events=("label", "size"),
        mean_predicted_probability=("probability", "mean"),
        realized_success_rate=("label", "mean"),
        success_count=("label", "sum"),
    ).reset_index()
    output["failure_count"] = output["n_events"] - output["success_count"]
    output["precision"] = output["realized_success_rate"]
    output["probability_bucket"] = output["probability_bucket"].astype(str)
    output["bucket_lower"] = bins[:-1]
    output["bucket_upper"] = bins[1:]
    return output.loc[
        :,
        [
            "probability_bucket",
            "bucket_lower",
            "bucket_upper",
            "n_events",
            "success_count",
            "failure_count",
            "mean_predicted_probability",
            "realized_success_rate",
            "precision",
        ],
    ]


def calibration_curve_frame(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 5,
) -> pd.DataFrame:
    # calibration_error = observed minus predicted, per bucket: positive
    # means the model is under-confident in that range (things happened
    # more often than it predicted), negative means over-confident.
    buckets = probability_bucket_summary(y_true, probabilities, n_bins=n_bins)
    calibration = buckets.loc[
        buckets["n_events"].gt(0),
        ["probability_bucket", "n_events", "mean_predicted_probability", "realized_success_rate"],
    ].copy()
    calibration["calibration_error"] = calibration["realized_success_rate"] - calibration["mean_predicted_probability"]
    calibration["absolute_calibration_error"] = calibration["calibration_error"].abs()
    return calibration.reset_index(drop=True)


def evaluate_predictions(
    predictions: pd.DataFrame,
    probability_col: str = "predicted_reversion_probability",
    label_col: str = "label",
    split_col: str = "split",
    threshold: float = 0.5,
    n_bins: int = 5,
) -> dict[str, pd.DataFrame]:
    # Bundles every evaluation table above into one call, computed
    # separately per split (train/validation/test) when a split column is
    # present -- so validation and test performance never get silently
    # blended into one misleading combined number.
    missing = [col for col in [probability_col, label_col] if col not in predictions.columns]
    if missing:
        raise KeyError(f"missing prediction columns: {missing}")
    frame = predictions.loc[predictions[label_col].notna()].copy()
    if frame.empty:
        raise ValueError("at least one labeled prediction is required")

    if split_col in frame.columns:
        summaries = []
        matrices = []
        buckets = []
        curves = []
        roc_curves = []
        for split, group in frame.groupby(split_col, sort=False):
            y = group[label_col]
            p = group[probability_col]
            summaries.append(classification_summary(y, p, threshold=threshold, split=str(split)))
            matrices.append(confusion_matrix_frame(y, p, threshold=threshold).assign(split=str(split)))
            buckets.append(probability_bucket_summary(y, p, n_bins=n_bins).assign(split=str(split)))
            curves.append(calibration_curve_frame(y, p, n_bins=n_bins).assign(split=str(split)))
            roc_curves.append(roc_curve_frame(y, p).assign(split=str(split)))
        summary = pd.concat(summaries, ignore_index=True, sort=False)
        matrix = pd.concat(matrices, ignore_index=True, sort=False)
        bucket_frame = pd.concat(buckets, ignore_index=True, sort=False)
        calibration = pd.concat(curves, ignore_index=True, sort=False)
        roc_curve = pd.concat(roc_curves, ignore_index=True, sort=False)
    else:
        y = frame[label_col]
        p = frame[probability_col]
        summary = classification_summary(y, p, threshold=threshold)
        matrix = confusion_matrix_frame(y, p, threshold=threshold)
        bucket_frame = probability_bucket_summary(y, p, n_bins=n_bins)
        calibration = calibration_curve_frame(y, p, n_bins=n_bins)
        roc_curve = roc_curve_frame(y, p)

    return {
        "model_evaluation_summary": summary,
        "confusion_matrix": matrix,
        "probability_bucket_summary": bucket_frame,
        "calibration_curve": calibration,
        "roc_curve": roc_curve,
    }


def _coerce_binary_inputs(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    p = np.asarray(probabilities, dtype=float).reshape(-1)
    if y.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if y.shape[0] != p.shape[0]:
        raise ValueError("y_true and probabilities must have the same length")
    if not np.all(np.isin(y, [0.0, 1.0])):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probabilities must be finite values between 0 and 1")
    return y.astype(int), p.astype(float)


def _validate_threshold(threshold: float) -> float:
    cutoff = float(threshold)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return cutoff


def _safe_divide(numerator: float, denominator: float) -> float:
    # A metric like precision (tp / (tp+fp)) is undefined, not zero, when
    # its denominator is zero (e.g. the model never predicted positive at
    # all) -- returning NaN rather than 0.0 or crashing keeps that
    # distinction visible downstream instead of silently implying "0%
    # precision," which would be a different, wrong claim.
    return float(numerator / denominator) if denominator else np.nan


def cost_drag_summary(
    summary: pd.DataFrame,
    strategy_col: str = "strategy",
    gross_col: str = "gross_pnl",
    net_col: str = "net_pnl",
) -> pd.DataFrame:
    # How much of the gross PnL was eaten by costs, in both absolute and
    # relative terms -- the first thing to check when a strategy looks
    # good gross but the net numbers disappoint.
    required = {strategy_col, gross_col, net_col}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"missing summary columns: {sorted(missing)}")
    frame = summary.copy()
    frame["cost_drag"] = frame[gross_col].astype(float) - frame[net_col].astype(float)
    denominator = frame[gross_col].abs().replace(0.0, np.nan)
    frame["cost_drag_ratio"] = frame["cost_drag"] / denominator
    columns = [strategy_col]
    for optional in ["cost_scenario", "total_cost_per_unit", "commission_per_trade", "bid_ask_spread_proxy", "slippage"]:
        if optional in frame.columns:
            columns.append(optional)
    columns += [gross_col, net_col, "cost_drag", "cost_drag_ratio"]
    return frame.loc[:, columns].copy()


def threshold_sensitivity_pivot(
    sensitivity: pd.DataFrame,
    metric: str = "net_pnl",
) -> pd.DataFrame:
    # Reshapes the long-format parameter sweep (one row per strategy x
    # entry x exit x stop x holding combination) into a wide grid --
    # entry thresholds as rows, exit thresholds as columns -- which is
    # exactly the shape the dashboard's heatmap panel renders directly.
    # Note: see CHANGELOG.md for the finding that exit_threshold has no
    # actual effect on net_pnl/sharpe in the current pipeline, which this
    # pivot itself doesn't know or check -- it just reshapes whatever
    # values are there.
    required = {"strategy", "entry_threshold", "exit_threshold", metric}
    missing = required.difference(sensitivity.columns)
    if missing:
        raise KeyError(f"missing sensitivity columns: {sorted(missing)}")
    return (
        sensitivity.pivot_table(
            index=["strategy", "entry_threshold"],
            columns="exit_threshold",
            values=metric,
            aggfunc="mean",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def robustness_summary_from_sensitivity(
    threshold_sensitivity: pd.DataFrame,
    cost_sensitivity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Collapses the entire parameter grid down to one row per strategy:
    # not just the average performance across all the parameter
    # combinations tried, but also the WORST one (worst_net_pnl,
    # worst_drawdown) -- a strategy whose average looks good but whose
    # worst-case parameter combination is a disaster is a strategy that's
    # fragile to the exact parameters chosen, not genuinely robust.
    required = {"strategy", "net_pnl", "sharpe", "max_drawdown", "trade_count", "turnover"}
    missing = required.difference(threshold_sensitivity.columns)
    if missing:
        raise KeyError(f"missing threshold sensitivity columns: {sorted(missing)}")
    rows = []
    for strategy, group in threshold_sensitivity.groupby("strategy", sort=False):
        pnl = group["net_pnl"].astype(float)
        sharpe = group["sharpe"].astype(float)
        dd = group["max_drawdown"].astype(float)
        turnover = group["turnover"].astype(float)
        trade_count = group["trade_count"].astype(float)
        rows.append(
            {
                "strategy": strategy,
                "threshold_scenarios": int(group.shape[0]),
                "average_net_pnl": float(pnl.mean()),
                "median_net_pnl": float(pnl.median()),
                "worst_net_pnl": float(pnl.min()),
                "best_net_pnl": float(pnl.max()),
                "profitable_scenario_rate": float((pnl > 0.0).mean()),
                "average_sharpe": float(sharpe.mean()),
                "worst_drawdown": float(dd.min()),
                "average_turnover": float(turnover.mean()),
                "minimum_trade_count": float(trade_count.min()),
                "maximum_trade_count": float(trade_count.max()),
            }
        )
    output = pd.DataFrame(rows)
    if cost_sensitivity is not None and not cost_sensitivity.empty:
        cost_required = {"strategy", "cost_scenario", "net_pnl", "gross_pnl"}
        cost_missing = cost_required.difference(cost_sensitivity.columns)
        if cost_missing:
            raise KeyError(f"missing cost sensitivity columns: {sorted(cost_missing)}")
        cost = cost_sensitivity.copy()
        cost["cost_drag"] = cost["gross_pnl"].astype(float) - cost["net_pnl"].astype(float)
        cost_summary = (
            cost.groupby("strategy", as_index=False)
            .agg(
                cost_scenarios=("cost_scenario", "nunique"),
                worst_cost_adjusted_net_pnl=("net_pnl", "min"),
                average_cost_drag=("cost_drag", "mean"),
                maximum_cost_drag=("cost_drag", "max"),
            )
        )
        output = output.merge(cost_summary, on="strategy", how="left")
    return output


def cost_adjusted_performance_summary(cost_sensitivity: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "cost_scenario", "gross_pnl", "net_pnl", "trade_count", "turnover", "sharpe", "max_drawdown"}
    missing = required.difference(cost_sensitivity.columns)
    if missing:
        raise KeyError(f"missing cost sensitivity columns: {sorted(missing)}")
    frame = cost_sensitivity.copy()
    frame["cost_drag"] = frame["gross_pnl"].astype(float) - frame["net_pnl"].astype(float)
    frame["net_to_gross_ratio"] = frame["net_pnl"] / frame["gross_pnl"].replace(0.0, np.nan)
    columns = [
        "cost_scenario",
        "strategy",
        "trade_count",
        "gross_pnl",
        "net_pnl",
        "cost_drag",
        "net_to_gross_ratio",
        "sharpe",
        "max_drawdown",
        "turnover",
    ]
    optional = ["total_cost_per_unit", "commission_per_trade", "bid_ask_spread_proxy", "slippage"]
    return frame.loc[:, columns + [col for col in optional if col in frame.columns]].copy()
