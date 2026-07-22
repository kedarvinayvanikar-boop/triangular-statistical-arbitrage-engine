# Optional C Rolling Regression Kernel

## Research purpose

Rolling regression can be computationally repetitive because the same small matrix calculation is performed for every date and every triplet. For each rolling window, the regression rebuilds the normal-equation matrices, solves for alpha and hedge ratios, and computes the one-step residual.

This isolates that repeated calculation in an optional C kernel while keeping Python as the reference implementation.

## What the C kernel computes

For each out-of-sample date, the kernel computes a rolling triangular regression using the previous window:

```text
y_t = alpha + beta_1 x_{1,t} + beta_2 x_{2,t} + residual_t
```

The C function forms the small matrix products:

```text
XTX = X'X
XTy = X'y
```

Then it solves the 3-by-3 system for:

```text
alpha, beta_1, beta_2
```

A second residual kernel computes fitted values and residuals from precomputed dynamic coefficients.

## Why C is optional

The project is primarily a reproducible quant research system, not a low-latency trading engine. SQL design, validation, feature correctness, model evaluation, and look-ahead protection matter more than raw speed.

C is appropriate only for isolated, repeated, numerical kernels where:

```text
inputs are simple arrays
outputs are easy to validate
Python remains the reference path
the performance gain is measurable
```

It is not appropriate for high-level research logic where clarity and auditability are more valuable.

## Validation result

The included validation table compares Python and C outputs for:

```text
alpha
beta_1
beta_2
fitted_log_price
residual
```

The sample output was generated on synthetic data. It validates numerical agreement and expected file format, not market-data performance.

## Fallback behavior

If the C shared library is missing or compilation fails, `src/c_bindings.py` returns the Python implementation. This keeps the project portable across machines.

## Interpretation

This should be treated as an engineering extension. The correct standard is not whether C is faster on one sample. The correct standard is whether C matches Python first, remains optional, and does not make the research pipeline harder to audit.
