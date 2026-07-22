# Phase 10: Feature Engineering for ML

This phase converts candidate spread events into an event-level feature matrix for supervised learning.

The target remains the Phase 9 event label:

\[
P(\text{residual mean-reverts before stop-loss}\mid\text{current setup features}).
\]

The features describe the residual state, relationship stability, volatility regime, market movement, recent drawdown, moving-average distance, and optional volume pressure at the candidate event date.

## Design constraints

Features must be known at signal time. Forward-looking information is allowed only in the label, not in the feature vector.

The feature matrix keeps event metadata, feature values, and the label in one table for modeling, while the SQL schema still stores event construction, labels, and features as separate reproducible objects.

## Core outputs

- `data/processed/event_feature_matrix.csv`
- `data/processed/feature_summary_statistics.csv`
- `data/processed/feature_missingness_report.csv`
- `data/processed/feature_correlation_matrix.csv`
- `figures/feature_correlation_heatmap.png`

The included sample outputs are placeholders produced from synthetic data. They are present only to show expected file format. Local project runs should overwrite them using real residual, event, price, return, and volume data.

## Limitations

Feature engineering can improve the representation of the trading setup, but it does not prove predictive power. Later phases still need walk-forward validation, probability calibration, and ML-filtered backtesting after transaction costs.
