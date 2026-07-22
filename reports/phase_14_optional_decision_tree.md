# Phase 14: Optional Decision Tree from Scratch

This optional phase adds a simple decision tree classifier for event outcome modeling. The model is intentionally constrained because the event dataset can be small and tree models can overfit residual-event samples.

The target remains the Phase 9 label:

- `1`: residual reverted before stop-loss inside the maximum holding period
- `0`: stop-loss hit first or no reversion inside the maximum holding period

The decision tree uses Phase 10 event-time features only. It does not use exit dates, future residual paths, realized post-entry outcomes, or label-derived columns as features.

## Files

- `src/tree_model.py`
- `tests/test_tree_model.py`
- `notebooks/optional_decision_tree.ipynb`

## Outputs

- `data/processed/decision_tree_predictions.csv`
- `data/processed/decision_tree_vs_logistic_comparison.csv`
- `data/processed/decision_tree_feature_split_summary.csv`
- `data/processed/decision_tree_leaf_summary.csv`

The included outputs are placeholder artifacts generated from synthetic event data because the available packaged event feature matrix is too small for a reliable optional tree comparison. The notebook will use real project features when enough labeled events are available.

## Research caveats

Decision trees can capture nonlinear relationships and threshold effects, but they are not automatically superior to logistic regression. For small event datasets, shallow trees with minimum leaf-size constraints are more defensible than deep trees. If validation performance is unstable across time splits, the logistic regression baseline should remain the primary ML model.
