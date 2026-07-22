# Phase 9: Event Labeling

This phase converts dynamic residual histories into supervised learning examples. The labeling target is not a next-day price forecast. The target is whether an entry signal mean-reverts before a stop-loss barrier within a fixed holding period.

The event definition is tied to the trading problem:

\[
P(\text{residual reverts before stop-loss} \mid \text{entry-time setup features})
\]

Candidate events are created when the residual z-score crosses the entry threshold. Short-spread events occur when the z-score crosses above the positive threshold. Long-spread events occur when the z-score crosses below the negative threshold.

The label is forward-looking and must be used only as a training target. It should not be included as an entry-time feature.

## Labeling rule

For a short-spread event, success occurs if the z-score falls to the reversion threshold before it reaches the stop-loss threshold.

For a long-spread event, success occurs if the z-score rises to the negative reversion threshold before it widens to the negative stop-loss threshold.

If neither condition occurs before the maximum holding period, the event is labeled as a failure.

## Main outputs

- `data/processed/candidate_events_table.csv`
- `data/processed/event_labels_table.csv`
- `data/processed/event_success_rate_by_triplet.csv`
- `data/processed/event_success_rate_by_z_bucket.csv`
- `figures/event_label_distribution.png`

The included outputs are placeholders generated from synthetic data when real residual histories are not available. The notebook overwrites these files when run with real project outputs from the earlier phases.

## Limitations

A high success rate in this table does not imply profitability. It does not include position sizing, transaction costs, liquidity constraints, borrow constraints, slippage, or portfolio interaction. The labels define a supervised learning target, not a complete trading rule.
