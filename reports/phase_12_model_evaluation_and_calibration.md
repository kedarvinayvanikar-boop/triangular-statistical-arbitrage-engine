# Phase 12: Model Evaluation and Calibration

This phase evaluates the logistic regression trade-event model from Phase 11. The model output is a predicted probability that a candidate residual event will revert before hitting the stop-loss barrier within the maximum holding period.

The evaluation layer focuses on probability quality, not only classification accuracy. This is important because the later ML-filtered backtest can use probability thresholds to decide which events to trade.

## Evaluation metrics

The primary metrics are:

- accuracy
- precision
- recall
- confusion matrix counts
- ROC-AUC
- Brier score
- probability calibration by bucket
- realized success rate by predicted-probability bucket

Accuracy is included, but it is not treated as the main objective. If failed events dominate the dataset, a model can have high accuracy by mostly predicting failure. For a trading filter, false positives are costly because they allow weak candidate trades into the backtest. Precision is therefore more relevant than raw accuracy.

## Calibration

Calibration compares predicted probabilities with observed event success rates. A calibrated model should have realized success rates close to the predicted probabilities inside each probability bucket.

This matters because the next phase may use rules such as:

```text
trade only if predicted_reversion_probability >= 0.60
```

or use probability as an input to position sizing. Poor calibration can make these rules misleading even when ranking quality is acceptable.

## Artifacts

Phase 12 writes:

```text
data/processed/model_evaluation_summary.csv
data/processed/confusion_matrix.csv
data/processed/probability_calibration_curve.csv
data/processed/precision_by_probability_bucket.csv
data/processed/roc_curve.csv
figures/probability_calibration_curve.png
figures/precision_by_probability_bucket.png
figures/confusion_matrix.png
```

The placeholder artifacts included in the package are based on the available sample predictions from Phase 11. They should be regenerated locally with real labeled event predictions before interpretation.

## Limitations

The evaluation only measures historical labeled events. It does not prove live profitability. Small sample sizes can make precision and calibration estimates unstable, especially inside high-probability buckets with few events.
