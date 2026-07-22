# Logistic Regression from Scratch

This adds a supervised classification model for event-level trade filtering. The model estimates the probability that a candidate residual event reverts before its stop-loss within the maximum holding period.

The target is the event label defined during event labeling. The inputs are the event features built during feature engineering. The model does not predict next-day stock returns.

## Model

The fitted probability is

\[
P(y_i = 1 \mid x_i) = \sigma(w^\top x_i + b)
\]

where \(y_i = 1\) is a successful reversion event and \(\sigma\) is the sigmoid function.

The implementation minimizes binary cross-entropy with optional L2 regularization. Optimization is done with batch gradient descent.

## Validation

Events are split by time order. The data is not shuffled because event outcomes are part of a time-series process. Random shuffling would mix market regimes and create overly optimistic validation results.

## Outputs

- `data/processed/logistic_model_coefficients.csv`
- `data/processed/logistic_training_loss.csv`
- `data/processed/predicted_reversion_probabilities.csv`
- `data/processed/logistic_validation_metrics.csv`
- `data/processed/logistic_probability_calibration.csv`
- `figures/logistic_loss_curve.png`

The included outputs are placeholder artifacts generated from the available sample feature matrix. They should be regenerated on real event features before interpreting model quality.
