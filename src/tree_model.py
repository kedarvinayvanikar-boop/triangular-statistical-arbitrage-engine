from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.metrics import classification_summary


@dataclass(frozen=True)
class DecisionTreeConfig:
    max_depth: int = 3            # kept shallow deliberately -- a few hundred events can't support a deep tree without overfitting
    min_samples_split: int = 20   # a node needs at least this many events before considering splitting it further
    min_samples_leaf: int = 10    # neither child of a split may end up with fewer events than this
    min_impurity_decrease: float = 1e-6  # a split must improve purity by at least this much to be worth making
    criterion: str = "gini"       # "gini" or "entropy" -- see gini_impurity/entropy_impurity below
    threshold: float = 0.5
    max_thresholds: int = 64      # caps how many candidate split points are tried per feature, for speed on continuous features


@dataclass(frozen=True)
class TreeNode:
    node_id: int
    depth: int
    n_samples: int
    positive_count: int
    probability: float          # fraction of this node's training events that were label=1
    impurity: float
    feature_index: Optional[int] = None
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    information_gain: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


@dataclass(frozen=True)
class DecisionTreeModelResult:
    feature_names: tuple[str, ...]
    root: TreeNode
    config: DecisionTreeConfig
    n_features: int
    n_training_events: int
    positive_rate: float


def gini_impurity(y: Sequence[int] | np.ndarray | pd.Series) -> float:
    """Gini impurity: how "mixed" a group of labels is. 0 means every
    label in the group is identical (perfectly pure -- either all wins
    or all losses); 0.5 is the most mixed a binary group can be (a coin
    flip's worth of each). Formula: 1 - p^2 - (1-p)^2, where p is the
    fraction of positives -- this is the probability that two randomly
    picked labels from the group would disagree, which is the intuitive
    meaning behind "impurity."
    """
    target = _coerce_target(y)
    p = float(np.mean(target))
    return 1.0 - p**2 - (1.0 - p) ** 2


def entropy_impurity(y: Sequence[int] | np.ndarray | pd.Series, eps: float = 1e-12) -> float:
    """Shannon entropy, the other common impurity measure: also 0 for a
    pure group and maximal (1.0, in bits) for a 50/50 mix, but penalizes
    a mixed group somewhat more aggressively than Gini does -- the choice
    between the two is usually a minor detail in practice, offered here
    as `criterion="entropy"` for comparison against the Gini default.
    """
    target = _coerce_target(y)
    p = float(np.mean(target))
    if p <= 0.0 or p >= 1.0:
        return 0.0
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def weighted_impurity(
    y_left: Sequence[int] | np.ndarray | pd.Series,
    y_right: Sequence[int] | np.ndarray | pd.Series,
    criterion: str = "gini",
) -> float:
    # A candidate split produces two child groups -- this combines their
    # individual impurities into one number, weighted by how many
    # observations landed in each, so a split that creates one huge pure
    # group and one tiny mixed group is scored fairly against a split
    # that creates two medium, moderately pure groups.
    left = _coerce_target(y_left)
    right = _coerce_target(y_right)
    total = left.shape[0] + right.shape[0]
    if total == 0:
        raise ValueError("at least one observation is required")
    impurity_fn = _impurity_function(criterion)
    return float((left.shape[0] / total) * impurity_fn(left) + (right.shape[0] / total) * impurity_fn(right))


def information_gain(
    y_parent: Sequence[int] | np.ndarray | pd.Series,
    y_left: Sequence[int] | np.ndarray | pd.Series,
    y_right: Sequence[int] | np.ndarray | pd.Series,
    criterion: str = "gini",
) -> float:
    # How much purer the two children are, combined, than the single
    # parent group was -- this is exactly what _best_split searches to
    # maximize at every node: the split that reduces impurity the most.
    parent = _coerce_target(y_parent)
    left = _coerce_target(y_left)
    right = _coerce_target(y_right)
    if left.shape[0] + right.shape[0] != parent.shape[0]:
        raise ValueError("child samples must partition the parent samples")
    impurity_fn = _impurity_function(criterion)
    return float(impurity_fn(parent) - weighted_impurity(left, right, criterion=criterion))


def fit_decision_tree(
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray | pd.Series,
    config: Optional[DecisionTreeConfig] = None,
    feature_names: Optional[Sequence[str]] = None,
) -> DecisionTreeModelResult:
    """Grows a binary decision tree from scratch: starting at the root
    (all training events), repeatedly finds the single best
    "feature <= threshold" question to split on, and recurses into each
    resulting half, stopping a branch once it hits max_depth, has too few
    samples left to keep splitting, is already pure, or the best
    available split wouldn't improve things enough to be worth it.
    """
    cfg = config or DecisionTreeConfig()
    _validate_config(cfg)
    matrix, names = _coerce_matrix(X, feature_names=feature_names)
    target = _coerce_target(y)
    if matrix.shape[0] != target.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    node_counter = [0]  # mutable list used as a simple shared counter across the recursive calls below
    root = _build_node(matrix, target, names, cfg, depth=0, node_counter=node_counter)
    return DecisionTreeModelResult(
        feature_names=names,
        root=root,
        config=cfg,
        n_features=matrix.shape[1],
        n_training_events=matrix.shape[0],
        positive_rate=float(np.mean(target)),
    )


def predict_proba_tree(model: DecisionTreeModelResult, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    # Each row is pushed down the tree independently (see _predict_row),
    # arriving at exactly one leaf, whose training-data win rate becomes
    # that row's predicted probability.
    matrix, _ = _coerce_matrix(X, feature_names=model.feature_names)
    if matrix.shape[1] != model.n_features:
        raise ValueError("X has a different number of columns than the fitted tree")
    probabilities = np.empty(matrix.shape[0], dtype=float)
    for row_idx in range(matrix.shape[0]):
        probabilities[row_idx] = _predict_row(model.root, matrix[row_idx])
    return probabilities


def predict_labels_tree(
    model: DecisionTreeModelResult,
    X: pd.DataFrame | np.ndarray,
    threshold: Optional[float] = None,
) -> np.ndarray:
    cutoff = model.config.threshold if threshold is None else float(threshold)
    if not 0.0 < cutoff < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (predict_proba_tree(model, X) >= cutoff).astype(int)


def make_tree_prediction_frame(
    frame: pd.DataFrame,
    model: DecisionTreeModelResult,
    feature_columns: Sequence[str],
    split_name: str,
    threshold: Optional[float] = None,
) -> pd.DataFrame:
    required = ["event_id", "triplet_id", "method", "event_date"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise KeyError(f"missing prediction metadata columns: {missing}")
    cutoff = model.config.threshold if threshold is None else float(threshold)
    probabilities = predict_proba_tree(model, frame.loc[:, list(feature_columns)])
    output = frame.loc[:, required].copy()
    output["split"] = split_name
    output["model_type"] = "decision_tree"
    output["predicted_reversion_probability"] = probabilities
    output["classification_threshold"] = cutoff
    output["predicted_label"] = (probabilities >= cutoff).astype(int)
    if "label" in frame.columns:
        output["label"] = frame["label"].astype(int).to_numpy()
    return output


def feature_split_summary(model: DecisionTreeModelResult) -> pd.DataFrame:
    # A readable table of every internal (non-leaf) node's split -- which
    # feature it used, at what threshold, and how much purity that split
    # bought. This is the tree's equivalent of the logistic model's
    # coefficient table: the human-inspectable summary of what the model
    # actually learned.
    rows: list[dict[str, object]] = []
    _collect_splits(model.root, rows)
    columns = [
        "node_id",
        "depth",
        "feature",
        "threshold",
        "information_gain",
        "n_samples",
        "positive_rate",
        "left_node_id",
        "right_node_id",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame.loc[:, columns].sort_values(["depth", "node_id"]).reset_index(drop=True)


def leaf_summary(model: DecisionTreeModelResult) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    _collect_leaves(model.root, rows)
    if not rows:
        return pd.DataFrame(columns=["node_id", "depth", "n_samples", "positive_count", "predicted_probability", "impurity"])
    return pd.DataFrame(rows).sort_values(["depth", "node_id"]).reset_index(drop=True)


def compare_tree_to_logistic(
    tree_predictions: pd.DataFrame,
    logistic_predictions: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    # Puts the tree's and the logistic model's validation metrics side by
    # side -- checking whether the extra nonlinearity a tree can capture
    # (interactions between features, not just a weighted sum of them)
    # actually helps on this particular problem, or whether the simpler
    # linear model does just as well.
    required = {"event_id", "label", "predicted_reversion_probability"}
    missing_tree = required.difference(tree_predictions.columns)
    missing_logistic = required.difference(logistic_predictions.columns)
    if missing_tree:
        raise KeyError(f"missing decision tree prediction columns: {sorted(missing_tree)}")
    if missing_logistic:
        raise KeyError(f"missing logistic prediction columns: {sorted(missing_logistic)}")

    rows = []
    tree_metrics = classification_summary(
        tree_predictions["label"],
        tree_predictions["predicted_reversion_probability"],
        threshold=threshold,
        split="decision_tree",
    )
    logistic_metrics = classification_summary(
        logistic_predictions["label"],
        logistic_predictions["predicted_reversion_probability"],
        threshold=threshold,
        split="logistic_regression",
    )
    for model_name, metrics in (("decision_tree", tree_metrics), ("logistic_regression", logistic_metrics)):
        row = metrics.iloc[0].to_dict()
        row["model_type"] = model_name
        rows.append(row)
    return pd.DataFrame(rows).loc[
        :,
        [
            "model_type",
            "n_obs",
            "positive_rate",
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
            "roc_auc",
            "true_positive",
            "false_positive",
            "true_negative",
            "false_negative",
        ],
    ]


def dataset_large_enough(
    frame: pd.DataFrame,
    label_col: str = "label",
    min_events: int = 100,
    min_positive: int = 20,
    min_negative: int = 20,
) -> bool:
    # A tree with even a modest max_depth can fit noise if there aren't
    # enough examples of BOTH classes to constrain each split -- this
    # gate exists to keep the tree from being trained (and treated as
    # meaningful) on too little data to support it.
    if label_col not in frame.columns:
        raise KeyError(f"missing label column: {label_col}")
    labels = _coerce_target(frame[label_col])
    return bool(
        labels.shape[0] >= min_events
        and int(np.sum(labels == 1)) >= min_positive
        and int(np.sum(labels == 0)) >= min_negative
    )


def train_event_decision_tree(
    feature_matrix: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    config: Optional[DecisionTreeConfig] = None,
    train_fraction: float = 0.7,
    min_events_required: int = 100,
) -> dict[str, pd.DataFrame | DecisionTreeModelResult | list[str]]:
    # End-to-end entry point, mirroring train_event_logistic_model's
    # structure: gate on dataset size, split by time (not randomly --
    # same lookahead-avoidance reasoning as the logistic model's
    # time_ordered_split), train, then score on the held-out chunk.
    if "label" not in feature_matrix.columns:
        raise KeyError("feature_matrix must contain label")
    if not dataset_large_enough(feature_matrix, min_events=min_events_required, min_positive=5, min_negative=5):
        raise ValueError("event dataset is too small for the optional decision tree phase")
    prepared, cols = prepare_tree_frame(feature_matrix, feature_columns=feature_columns)
    ordered = prepared.sort_values("event_date").reset_index(drop=True) if "event_date" in prepared.columns else prepared.reset_index(drop=True)
    split_idx = max(1, min(ordered.shape[0] - 1, int(np.floor(ordered.shape[0] * train_fraction))))
    train = ordered.iloc[:split_idx].copy()
    validation = ordered.iloc[split_idx:].copy()
    model = fit_decision_tree(train.loc[:, cols], train["label"], config=config)
    predictions = make_tree_prediction_frame(validation, model, cols, split_name="validation")
    metrics = classification_summary(predictions["label"], predictions["predicted_reversion_probability"], threshold=model.config.threshold, split="validation")
    return {
        "model": model,
        "feature_columns": cols,
        "decision_tree_predictions": predictions,
        "decision_tree_validation_metrics": metrics,
        "feature_split_summary": feature_split_summary(model),
        "leaf_summary": leaf_summary(model),
    }


def prepare_tree_frame(
    frame: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    label_col: str = "label",
) -> tuple[pd.DataFrame, list[str]]:
    if label_col not in frame.columns:
        raise KeyError(f"missing label column: {label_col}")
    metadata = {"event_id", "triplet_id", "method", "event_date", label_col}
    if feature_columns is None:
        cols = [
            col
            for col in frame.columns
            if col not in metadata and pd.api.types.is_numeric_dtype(frame[col])
        ]
    else:
        cols = list(feature_columns)
    if not cols:
        raise ValueError("at least one numeric feature column is required")
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    prepared = frame.copy()
    for col in cols:
        # median-fill missing values, same reasoning as
        # logistic_model.prepare_model_frame: keeps the event (and its
        # label) in the training set rather than dropping it entirely
        # over one missing feature
        values = pd.to_numeric(prepared[col], errors="coerce")
        median = values.median()
        if not np.isfinite(median):
            median = 0.0
        prepared[col] = values.fillna(median)
    prepared[label_col] = prepared[label_col].astype(int)
    return prepared, cols


def _build_node(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    config: DecisionTreeConfig,
    depth: int,
    node_counter: list[int],
) -> TreeNode:
    # Recursive tree construction. At every call: compute this node's own
    # stats first (so even a node that ends up a leaf has a valid
    # probability/impurity recorded), then check the stopping conditions,
    # and only search for a split if none of them apply.
    node_id = node_counter[0]
    node_counter[0] += 1
    impurity = _impurity_function(config.criterion)(y)
    positive_count = int(np.sum(y == 1))
    probability = float(np.mean(y))

    if (
        depth >= config.max_depth
        or X.shape[0] < config.min_samples_split
        or impurity == 0.0
    ):
        # stopping conditions: too deep, too few samples to justify
        # another split, or already perfectly pure -- nothing left to gain
        return TreeNode(node_id, depth, X.shape[0], positive_count, probability, impurity)

    split = _best_split(X, y, feature_names, config)
    if split is None or split["information_gain"] < config.min_impurity_decrease:
        # no candidate split cleared min_samples_leaf on both sides, or
        # the best one available wasn't worth the added complexity
        return TreeNode(node_id, depth, X.shape[0], positive_count, probability, impurity)

    mask = X[:, split["feature_index"]] <= split["threshold"]
    left = _build_node(X[mask], y[mask], feature_names, config, depth + 1, node_counter)
    right = _build_node(X[~mask], y[~mask], feature_names, config, depth + 1, node_counter)
    return TreeNode(
        node_id=node_id,
        depth=depth,
        n_samples=X.shape[0],
        positive_count=positive_count,
        probability=probability,
        impurity=impurity,
        feature_index=int(split["feature_index"]),
        feature_name=str(split["feature_name"]),
        threshold=float(split["threshold"]),
        information_gain=float(split["information_gain"]),
        left=left,
        right=right,
    )


def _best_split(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    config: DecisionTreeConfig,
) -> Optional[dict[str, object]]:
    # Exhaustive search: try every candidate threshold on every feature,
    # and keep whichever single (feature, threshold) pair produces the
    # highest information gain. This is what "greedy" tree-building
    # means -- the best split at THIS node, not necessarily the split
    # that leads to the best tree overall a few levels down (which would
    # require an intractable search over every possible tree shape).
    parent_impurity = _impurity_function(config.criterion)(y)
    best: Optional[dict[str, object]] = None
    for feature_idx in range(X.shape[1]):
        thresholds = _candidate_thresholds(X[:, feature_idx], max_thresholds=config.max_thresholds)
        for threshold in thresholds:
            mask = X[:, feature_idx] <= threshold
            left_count = int(np.sum(mask))
            right_count = int(mask.shape[0] - left_count)
            if left_count < config.min_samples_leaf or right_count < config.min_samples_leaf:
                continue
            child_impurity = weighted_impurity(y[mask], y[~mask], criterion=config.criterion)
            gain = float(parent_impurity - child_impurity)
            if best is None or gain > float(best["information_gain"]):
                best = {
                    "feature_index": feature_idx,
                    "feature_name": feature_names[feature_idx],
                    "threshold": float(threshold),
                    "information_gain": gain,
                }
    return best


def _candidate_thresholds(values: np.ndarray, max_thresholds: int) -> np.ndarray:
    # Candidate split points are midpoints between consecutive distinct
    # sorted values (the only thresholds that could possibly separate two
    # different values from each other). If a feature has more unique
    # values than `max_thresholds` allows testing individually, the
    # candidates are subsampled at evenly spaced quantiles instead of
    # trying every single one -- a speed/thoroughness tradeoff for
    # continuous features with many distinct values.
    finite = np.sort(np.unique(values[np.isfinite(values)]))
    if finite.shape[0] <= 1:
        return np.array([], dtype=float)
    mids = (finite[:-1] + finite[1:]) / 2.0
    if mids.shape[0] <= max_thresholds:
        return mids
    quantiles = np.linspace(0.0, 1.0, max_thresholds + 2)[1:-1]
    return np.unique(np.quantile(mids, quantiles))


def _predict_row(node: TreeNode, row: np.ndarray) -> float:
    # Walks down the tree by repeatedly asking "is this row's value for
    # the split feature <= the threshold?" until landing on a leaf, then
    # returns that leaf's training-data win rate as the prediction.
    current = node
    while not current.is_leaf:
        if current.feature_index is None or current.threshold is None:
            break
        current = current.left if row[current.feature_index] <= current.threshold else current.right
        if current is None:
            break
    return float(current.probability if current is not None else node.probability)


def _collect_splits(node: TreeNode, rows: list[dict[str, object]]) -> None:
    if node.is_leaf:
        return
    rows.append(
        {
            "node_id": node.node_id,
            "depth": node.depth,
            "feature": node.feature_name,
            "threshold": node.threshold,
            "information_gain": node.information_gain,
            "n_samples": node.n_samples,
            "positive_rate": node.probability,
            "left_node_id": node.left.node_id if node.left else None,
            "right_node_id": node.right.node_id if node.right else None,
        }
    )
    if node.left is not None:
        _collect_splits(node.left, rows)
    if node.right is not None:
        _collect_splits(node.right, rows)


def _collect_leaves(node: TreeNode, rows: list[dict[str, object]]) -> None:
    if node.is_leaf:
        rows.append(
            {
                "node_id": node.node_id,
                "depth": node.depth,
                "n_samples": node.n_samples,
                "positive_count": node.positive_count,
                "predicted_probability": node.probability,
                "impurity": node.impurity,
            }
        )
        return
    if node.left is not None:
        _collect_leaves(node.left, rows)
    if node.right is not None:
        _collect_leaves(node.right, rows)


def _coerce_matrix(
    X: pd.DataFrame | np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if isinstance(X, pd.DataFrame):
        matrix = X.to_numpy(dtype=float)
        names = tuple(str(col) for col in X.columns)
        if feature_names is not None:
            names = tuple(str(col) for col in feature_names)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        names = tuple(feature_names) if feature_names is not None else tuple(f"x_{idx}" for idx in range(matrix.shape[1]))
    if matrix.ndim != 2:
        raise ValueError("X must be two-dimensional")
    if matrix.shape[1] != len(names):
        raise ValueError("number of feature names must match number of columns")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("X must contain finite numeric values")
    return matrix, tuple(str(name) for name in names)


def _coerce_target(y: Sequence[int] | np.ndarray | pd.Series) -> np.ndarray:
    target = np.asarray(y, dtype=float).reshape(-1)
    if target.shape[0] == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isin(target, [0.0, 1.0])):
        raise ValueError("y must contain only 0 and 1")
    return target.astype(int)


def _impurity_function(criterion: str):
    if criterion == "gini":
        return gini_impurity
    if criterion == "entropy":
        return entropy_impurity
    raise ValueError("criterion must be either 'gini' or 'entropy'")


def _validate_config(config: DecisionTreeConfig) -> None:
    if config.max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if config.min_samples_split < 2:
        raise ValueError("min_samples_split must be at least 2")
    if config.min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")
    if config.min_samples_split < 2 * config.min_samples_leaf:
        raise ValueError("min_samples_split must be at least twice min_samples_leaf")
    if config.min_impurity_decrease < 0.0:
        raise ValueError("min_impurity_decrease must be non-negative")
    if config.max_thresholds < 1:
        raise ValueError("max_thresholds must be at least 1")
    if not 0.0 < config.threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    _impurity_function(config.criterion)
