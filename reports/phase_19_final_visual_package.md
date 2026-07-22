# Phase 19: Final visual package

This phase adds a polished final visualization layer for the triangular statistical arbitrage research system.

The final figure set is designed to tell the research story from data coverage through residual construction, diagnostics, event labels, ML model behavior, strategy comparison, regime diagnostics, and transaction-cost robustness.

The plotting code is data-driven. Placeholder figures are only included so the repo has expected outputs when the market-data database is unavailable. Local runs should regenerate the figures from real pipeline outputs before interpretation.

## Figure set

- price_coverage_summary.png
- triplet_price_relationship.png
- hedge_ratio_stability.png
- residual_zscore_example.png
- residual_distribution.png
- residual_autocorrelation.png
- half_life_by_triplet.png
- baseline_equity_curve.png
- baseline_drawdown.png
- event_label_distribution.png
- feature_correlation_heatmap.png
- logistic_loss_curve.png
- probability_calibration_curve.png
- precision_by_probability_bucket.png
- ml_filtered_vs_baseline_equity.png
- performance_by_triplet.png
- performance_by_regime.png
- transaction_cost_sensitivity.png

## Research rule

The visual package should not cherry-pick only favorable outputs. Equity curves should be paired with drawdowns, model predictions should be paired with calibration, and strategy results should be paired with cost sensitivity.
