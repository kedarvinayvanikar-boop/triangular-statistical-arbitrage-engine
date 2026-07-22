# Transaction Costs, Slippage, and Robustness

## Objective

This tests whether residual-based event strategies remain robust after execution frictions and parameter variation. The goal is not to claim tradability. The goal is to make the backtest less dependent on optimistic assumptions.

## Cost model

Total round-trip cost is modeled as:

```text
total_cost_per_unit = commission_per_trade + bid_ask_spread_proxy + slippage
```

The placeholder implementation applies that cost to residual PnL proxy units. When execution-level prices, hedge weights, and share quantities are available, the same interface should be replaced with dollar or basis-point cost calculations.

## Robustness grid

The threshold sensitivity grid varies:

```text
entry threshold
exit threshold
stop-loss level
maximum holding period
```

This checks whether results depend on one fragile parameter combination.

## Main outputs

```text
transaction_cost_sensitivity.csv
threshold_sensitivity_table.csv
cost_adjusted_performance_summary.csv
robustness_summary.csv
transaction_cost_sensitivity.png
threshold_sensitivity.png
```

## Interpretation

A strategy that looks strong before costs can fail after costs if turnover is high or average edge per trade is small. The ML-filtered strategy is useful only if it removes enough poor trades to offset fewer opportunities and any remaining cost drag.

The strongest result would be stable performance across a range of cost and threshold assumptions. A weak result would be performance that disappears under small cost increases or only works for one specific entry/exit combination.

## Limitations

The included sample outputs are synthetic placeholders. They should be regenerated with real event labels and logistic regression predictions before drawing any research conclusions.
