# Rolling OLS and Ridge Regression

## Intuition

A static hedge ratio assumes that the relationship between the target asset and its hedge basket is stable through the whole sample. That is convenient, but equity and ETF relationships can shift when sector exposure, market volatility, funding conditions, or single-name news changes.

Rolling regression lets the hedge ratio adapt by estimating the relationship from a trailing window. The coefficient used on date `t` is estimated only from dates before `t`, so the fitted value and residual are out-of-sample for that date.

Ridge regression adds a penalty against large coefficients. This matters when the two hedge assets are correlated. For example, QQQ and XLK can move together because both contain large technology exposure. Ordinary least squares can then split exposure between the two anchors in unstable ways. Ridge reduces that instability by accepting a small amount of bias in exchange for lower variance.

## Finance concept

The triangular relationship is still:

```text
log(P_A,t) = alpha + beta_1 log(P_B,t) + beta_2 log(P_C,t) + epsilon_t
```

The residual measures how far the target is from the estimated hedge basket. This section changes how the coefficients are estimated, not the definition of the residual.

## Math

Rolling OLS solves this problem on a trailing window ending before date `t`:

```text
min_beta ||y_train - X_train beta||^2
```

Ridge solves:

```text
min_beta ||y_train - X_train beta||^2 + lambda ||beta||^2
```

With an intercept excluded from the penalty, the closed-form ridge estimate is:

```text
beta_ridge = (X'X + lambda D)^(-1) X'y
```

where `D` penalizes slope coefficients but not the intercept.

## Build task

- Estimate rolling OLS coefficients for each triplet.
- Estimate rolling ridge coefficients for each triplet.
- Generate out-of-sample fitted values and residuals.
- Compare static initial-window OLS, rolling OLS, and rolling ridge residuals.
- Store coefficients and residuals in SQLite.
- Save a hedge-ratio stability plot.

## Outputs

- `rolling_coefficients` SQL table
- `ridge_coefficients` SQL table
- `dynamic_residuals` SQL table
- `residual_method_summary` SQL table
- `figures/hedge_ratio_stability.png`
- `data/processed/static_vs_dynamic_residual_summary.csv`
- `data/processed/backtest_comparison_table.csv`

## Self-test checklist

- The prediction row is not included in the rolling training window.
- Static comparison residuals are evaluated on the same dates as the rolling residuals.
- Ridge with `alpha = 0` matches OLS when the system is well-conditioned.
- Ridge coefficients shrink when the hedge assets are highly collinear.
- All SQL tables have triplet, date, method, and coefficient or residual identifiers.
- Placeholder outputs are replaced by real project data before interpreting results.

## Limitation

A lower residual standard deviation does not prove the strategy is better. Dynamic coefficients can overfit recent noise, especially with short windows. Any improvement must later be checked through transaction-cost-aware backtests and walk-forward validation.
