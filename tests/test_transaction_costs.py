import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    RobustnessGrid,
    TransactionCostAssumption,
    apply_transaction_cost_assumption,
    build_strategy_trades,
    prepare_ml_backtest_events,
    relabel_events_for_threshold_scenario,
    run_cost_and_threshold_robustness,
    run_threshold_sensitivity,
    run_transaction_cost_sensitivity,
)
from src.metrics import (
    cost_adjusted_performance_summary,
    cost_drag_summary,
    robustness_summary_from_sensitivity,
    threshold_sensitivity_pivot,
)


def _labels():
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "triplet_id": ["A_B_C"] * 6,
            "method": ["kalman"] * 6,
            "event_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]),
            "exit_date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]),
            "side": ["short_spread", "long_spread", "short_spread", "long_spread", "short_spread", "long_spread"],
            "label": [1, 1, 0, 0, 1, 0],
            "outcome": ["success", "success", "failure", "failure", "success", "failure"],
            "exit_reason": ["reversion", "reversion", "stop_loss", "max_holding", "reversion", "stop_loss"],
            "holding_period": [2, 2, 2, 6, 2, 2],
            "entry_z_score": [2.2, -2.1, 2.8, -2.3, 1.7, -2.9],
            "exit_z_score": [0.4, -0.3, 3.2, -2.0, 0.2, -3.4],
        }
    )


def _predictions():
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4", "e5", "e6"],
            "split": ["test"] * 6,
            "predicted_reversion_probability": [0.82, 0.71, 0.41, 0.35, 0.64, 0.25],
        }
    )


def test_apply_transaction_cost_assumption_reprices_trade_log():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    trades = build_strategy_trades(events, strategy="baseline_rule_based", transaction_cost=0.0)
    costed = apply_transaction_cost_assumption(
        trades,
        TransactionCostAssumption("base", commission_per_trade=0.01, bid_ask_spread_proxy=0.02, slippage=0.01),
    )

    assert costed["cost_scenario"].eq("base").all()
    assert np.allclose(costed["transaction_cost"], 0.04 * costed["position_size"].abs())
    assert np.allclose(costed["net_pnl"], costed["gross_pnl"] - costed["transaction_cost"])


def test_borrow_cost_scales_with_holding_period_and_is_backward_compatible():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    trades = build_strategy_trades(events, strategy="baseline_rule_based", transaction_cost=0.0)
    assert "holding_period" in trades.columns  # sanity check the fixture actually carries it

    borrow_cost = TransactionCostAssumption("with_borrow", commission_per_trade=0.0, borrow_cost_bps_per_day=5.0)
    costed = apply_transaction_cost_assumption(trades, borrow_cost)

    expected_borrow = trades["holding_period"].astype(float).apply(borrow_cost.borrow_cost_for_holding_period)
    expected_borrow = expected_borrow * trades["position_size"].abs().to_numpy()
    assert np.allclose(costed["borrow_cost"], expected_borrow)
    assert np.allclose(costed["transaction_cost"], costed["borrow_cost"])  # no other cost components set
    # a trade held longer must cost more in borrow than one held for less time, all else equal
    longest = trades["holding_period"].idxmax()
    shortest = trades["holding_period"].idxmin()
    if trades.loc[longest, "holding_period"] != trades.loc[shortest, "holding_period"]:
        assert costed.loc[longest, "borrow_cost"] >= costed.loc[shortest, "borrow_cost"]

    # zero borrow rate (the default) must not add any cost -- existing scenarios must be unaffected
    no_borrow = TransactionCostAssumption("no_borrow", commission_per_trade=0.01)
    costed_no_borrow = apply_transaction_cost_assumption(trades, no_borrow)
    assert (costed_no_borrow["borrow_cost"] == 0.0).all()

    # missing holding_period column must not crash -- borrow cost simply doesn't apply
    trades_no_holding = trades.drop(columns=["holding_period"])
    costed_missing_col = apply_transaction_cost_assumption(trades_no_holding, borrow_cost)
    assert (costed_missing_col["borrow_cost"] == 0.0).all()


def test_transaction_cost_sensitivity_worsens_net_pnl_when_costs_increase():
    outputs = run_transaction_cost_sensitivity(
        _labels(),
        _predictions(),
        cost_assumptions=[
            {"scenario": "zero", "commission_per_trade": 0.0, "bid_ask_spread_proxy": 0.0, "slippage": 0.0},
            {"scenario": "high", "commission_per_trade": 0.05, "bid_ask_spread_proxy": 0.05, "slippage": 0.05},
        ],
        probability_threshold=0.6,
    )
    table = outputs["transaction_cost_sensitivity"]
    zero = table.query("cost_scenario == 'zero' and strategy == 'baseline_rule_based'")["net_pnl"].iloc[0]
    high = table.query("cost_scenario == 'high' and strategy == 'baseline_rule_based'")["net_pnl"].iloc[0]

    assert high < zero
    assert {"cost_scenario", "strategy", "cost_drag"}.issubset(table.columns)


def test_relabel_events_for_threshold_scenario_respects_thresholds():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    relabeled = relabel_events_for_threshold_scenario(events, entry_threshold=2.0, exit_threshold=0.5, stop_loss_level=3.0, max_holding_period=4)

    assert relabeled["entry_z_score"].abs().ge(2.0).all()
    assert set(relabeled["label"]).issubset({0, 1})
    assert "scenario_exit_reason" in relabeled.columns
    assert relabeled.loc[relabeled["event_id"].eq("e3"), "label"].iloc[0] == 0


def test_threshold_sensitivity_and_robustness_outputs_have_schema():
    grid = RobustnessGrid(entry_thresholds=(1.5, 2.0), exit_thresholds=(0.5,), stop_loss_levels=(3.0,), max_holding_periods=(4,))
    sensitivity = run_threshold_sensitivity(_labels(), _predictions(), grid=grid, probability_threshold=0.6, transaction_cost=0.01)
    cost = run_transaction_cost_sensitivity(_labels(), _predictions(), probability_threshold=0.6)["transaction_cost_sensitivity"]
    robust = robustness_summary_from_sensitivity(sensitivity, cost)

    assert {"entry_threshold", "exit_threshold", "stop_loss_level", "max_holding_period"}.issubset(sensitivity.columns)
    assert set(sensitivity["strategy"]).issuperset({"baseline_rule_based", "ml_filtered"})
    assert {"strategy", "profitable_scenario_rate", "worst_cost_adjusted_net_pnl"}.issubset(robust.columns)


def test_metric_helpers_return_cost_and_threshold_summaries():
    outputs = run_transaction_cost_sensitivity(_labels(), _predictions(), probability_threshold=0.6)
    cost_table = outputs["transaction_cost_sensitivity"]
    cost_summary = cost_adjusted_performance_summary(cost_table)
    drag = cost_drag_summary(cost_table)

    grid = RobustnessGrid(entry_thresholds=(1.5, 2.0), exit_thresholds=(0.5, 0.75), stop_loss_levels=(3.0,), max_holding_periods=(4,))
    threshold = run_threshold_sensitivity(_labels(), _predictions(), grid=grid, probability_threshold=0.6, transaction_cost=0.01)
    pivot = threshold_sensitivity_pivot(threshold, metric="net_pnl")

    assert {"cost_drag", "net_to_gross_ratio"}.issubset(cost_summary.columns)
    assert drag["cost_drag"].notna().all()
    assert not pivot.empty


def test_run_cost_and_threshold_robustness_returns_all_tables():
    grid = {"entry_thresholds": [1.5], "exit_thresholds": [0.5], "stop_loss_levels": [3.0], "max_holding_periods": [4]}
    outputs = run_cost_and_threshold_robustness(_labels(), _predictions(), grid=grid, probability_threshold=0.6)

    assert set(outputs) == {
        "transaction_cost_trade_log",
        "transaction_cost_sensitivity",
        "threshold_sensitivity_table",
        "robustness_summary",
    }


def test_invalid_cost_and_threshold_inputs_raise():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    trades = build_strategy_trades(events, strategy="baseline_rule_based", transaction_cost=0.0)
    with pytest.raises(ValueError):
        apply_transaction_cost_assumption(trades, {"scenario": "bad", "slippage": -0.1})
    with pytest.raises(ValueError):
        relabel_events_for_threshold_scenario(events, entry_threshold=2.0, exit_threshold=0.5, stop_loss_level=1.0, max_holding_period=5)
