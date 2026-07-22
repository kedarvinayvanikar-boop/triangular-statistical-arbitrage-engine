# Phase 8: Kalman Filter for Dynamic Hedge Ratios

## Research purpose

Phase 8 extends the dynamic hedge-ratio framework from rolling regressions to a state-space model. The Kalman filter treats the triangular regression coefficients as latent variables that evolve through time. This is useful when the relationship between a target asset and two hedge assets changes gradually rather than remaining fixed across the full sample.

The triangular relationship remains

```text
log(P_A,t) = alpha_t + beta_1,t log(P_B,t) + beta_2,t log(P_C,t) + residual_t
```

but the coefficients are now dynamic states.

## State equation

The state vector is

```text
theta_t = [alpha_t, beta_1,t, beta_2,t]'
```

The default transition is a random walk:

```text
theta_t = theta_{t-1} + w_t
```

where `w_t` is process noise. Larger process noise allows the hedge ratios to adapt faster, while smaller process noise forces smoother state paths.

## Observation equation

The observation equation is

```text
y_t = H_t theta_t + v_t
```

where

```text
y_t = log(P_A,t)
H_t = [1, log(P_B,t), log(P_C,t)]
```

and `v_t` is measurement noise. Larger measurement noise makes the filter trust each new observed price less.

## Residual definition

The residual used in this phase is the one-step-ahead innovation:

```text
residual_t = y_t - H_t theta_{t|t-1}
```

This is calculated before the current observation updates the state. That convention keeps the residual out-of-sample with respect to the current date and makes it more comparable to rolling OLS residuals.

## Comparison to rolling OLS

Rolling OLS re-estimates the regression over a trailing window. The Kalman filter recursively updates the previous state using the newest prediction error. Rolling OLS can adjust sharply when the training window changes. The Kalman filter tends to produce smoother paths when process noise is small, but it may react too slowly if the relationship changes abruptly.

Neither method is assumed to be superior. The comparison should be based on coefficient stability, residual diagnostics, sensitivity to parameter assumptions, and later backtest performance after transaction costs.

## Added implementation

```text
src/kalman.py
```

Core functions:

```text
kalman_predict
kalman_update
kalman_filter_dynamic_regression
estimate_kalman_for_triplets
compare_kalman_residuals
```

## Outputs

```text
data/processed/kalman_hedge_ratio_table.csv
data/processed/kalman_residual_table.csv
data/processed/kalman_vs_rolling_residual_summary.csv
figures/kalman_hedge_ratio_comparison.png
figures/kalman_residual_comparison.png
```

The included outputs are placeholders generated from synthetic data. They are not empirical market findings.

## Limitations

The random-walk state equation is a modeling assumption. Process noise and measurement noise can materially change the estimated hedge ratios. The filter can over-smooth during breaks or overreact when noise settings are too large. The model also assumes linear observation structure and Gaussian noise, which may not hold during stress regimes or single-name event shocks.
