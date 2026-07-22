"""
Recomputes every processed table whose `sharpe` column was produced before
the fix to `strategy_performance_summary` (see src/portfolio.py --
Sharpe was previously annualized off trade-only days, which overstates
performance for low-frequency strategies by treating idle days as if they
never happened).

This script does NOT introduce new synthetic data or change any random
seed. It reproduces the exact same inputs the original notebooks used
(same event universes, same placeholder-generation recipe where one was
used) and re-runs them through the corrected backtest/portfolio code, so
the only thing that changes in the output is the Sharpe methodology --
everything else (trade counts, win rates, net PnL, which trades were
taken) is unchanged.

Run from the repository root: python scripts/fix_sharpe_regeneration.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.backtest import RobustnessGrid, run_cost_and_threshold_robustness, run_ml_backtest_comparison
from src.metrics import cost_adjusted_performance_summary

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def regenerate_ml_backtest() -> None:
    """Mirrors notebooks/12_ml_filtered_backtest.ipynb cell 3-5 exactly,
    including its placeholder-prediction fallback (seed=13), so the trade
    universe is identical to what's currently shipped -- only Sharpe changes.
    """
    labels = pd.read_csv(DATA_DIR / "event_labels_table.csv")
    predictions = pd.read_csv(DATA_DIR / "predicted_reversion_probabilities.csv")

    overlap = set(labels["event_id"]).intersection(set(predictions["event_id"]))
    if not overlap:
        rng = np.random.default_rng(13)
        placeholder = labels.loc[:, ["event_id", "triplet_id", "method", "event_date", "label"]].copy()
        base = 0.38 + 0.32 * placeholder["label"].astype(float).to_numpy() + rng.normal(0.0, 0.16, size=len(placeholder))
        placeholder["predicted_reversion_probability"] = np.clip(base, 0.05, 0.95)
        placeholder["split"] = "placeholder"
        placeholder["classification_threshold"] = 0.60
        placeholder["predicted_label"] = (placeholder["predicted_reversion_probability"] > 0.60).astype(int)
        predictions = placeholder

    result = run_ml_backtest_comparison(
        labels=labels, predictions=predictions, probability_threshold=0.60, transaction_cost=0.05,
    )
    result["strategy_trade_log"].to_csv(DATA_DIR / "ml_backtest_trade_log.csv", index=False)
    result["strategy_equity_curve"].to_csv(DATA_DIR / "ml_backtest_equity_curve.csv", index=False)
    result["strategy_summary"].to_csv(DATA_DIR / "ml_backtest_strategy_summary.csv", index=False)
    result["avoided_bad_trades"].to_csv(DATA_DIR / "avoided_bad_trades_analysis.csv", index=False)

    summary = result["strategy_summary"]
    trade_count = summary[["strategy", "trade_count"]].copy()
    base_n = trade_count.loc[trade_count["strategy"] == "baseline_rule_based", "trade_count"].iloc[0]
    trade_count["baseline_trade_count"] = base_n
    trade_count["trades_avoided_vs_baseline"] = base_n - trade_count["trade_count"]
    trade_count["trade_count_ratio_to_baseline"] = trade_count["trade_count"] / base_n
    trade_count.to_csv(DATA_DIR / "baseline_vs_ml_trade_count.csv", index=False)

    pnl = summary[["strategy", "gross_pnl", "net_pnl", "sharpe", "max_drawdown", "turnover", "win_rate"]].copy()
    pnl.to_csv(DATA_DIR / "baseline_vs_ml_pnl.csv", index=False)

    print("regenerated: ml_backtest_trade_log, ml_backtest_equity_curve, ml_backtest_strategy_summary,")
    print("             baseline_vs_ml_trade_count, baseline_vs_ml_pnl, avoided_bad_trades_analysis")


def regenerate_cost_and_threshold_sensitivity() -> None:
    """Mirrors notebooks/transaction_cost_sensitivity.ipynb exactly, using
    the same synthetic placeholder event universe the original run actually
    consumed (event_labels_table.csv / predicted_reversion_probabilities.csv
    did not overlap by event_id at generation time, so the notebook's
    existence-based fallback to the synthetic event/prediction files is what produced the
    currently-shipped 90-trade baseline grid; reproduced here identically).
    """
    label_path = DATA_DIR / "18_synthetic_event_labels.csv"
    prediction_path = DATA_DIR / "18_synthetic_predictions.csv"
    labels = pd.read_csv(label_path, parse_dates=["event_date", "exit_date"])
    predictions = pd.read_csv(prediction_path)

    cost_assumptions = [
        {"scenario": "zero_cost", "commission_per_trade": 0.000, "bid_ask_spread_proxy": 0.000, "slippage": 0.000},
        {"scenario": "low_cost", "commission_per_trade": 0.005, "bid_ask_spread_proxy": 0.010, "slippage": 0.005},
        {"scenario": "base_cost", "commission_per_trade": 0.010, "bid_ask_spread_proxy": 0.025, "slippage": 0.015},
        {"scenario": "stress_cost", "commission_per_trade": 0.020, "bid_ask_spread_proxy": 0.060, "slippage": 0.040},
    ]
    grid = RobustnessGrid(
        entry_thresholds=(1.5, 2.0, 2.5),
        exit_thresholds=(0.25, 0.5, 0.75),
        stop_loss_levels=(2.75, 3.0, 3.5),
        max_holding_periods=(10, 15, 20),
    )

    outputs = run_cost_and_threshold_robustness(
        labels, predictions, cost_assumptions=cost_assumptions, grid=grid, probability_threshold=0.60,
    )
    transaction_cost_sensitivity = outputs["transaction_cost_sensitivity"]
    threshold_sensitivity_table = outputs["threshold_sensitivity_table"]
    robustness_summary = outputs["robustness_summary"]
    cost_adjusted_summary = cost_adjusted_performance_summary(transaction_cost_sensitivity)

    transaction_cost_sensitivity.to_csv(DATA_DIR / "transaction_cost_sensitivity.csv", index=False)
    threshold_sensitivity_table.to_csv(DATA_DIR / "threshold_sensitivity_table.csv", index=False)
    robustness_summary.to_csv(DATA_DIR / "robustness_summary.csv", index=False)
    cost_adjusted_summary.to_csv(DATA_DIR / "cost_adjusted_performance_summary.csv", index=False)

    print("regenerated: transaction_cost_sensitivity, threshold_sensitivity_table,")
    print("             robustness_summary, cost_adjusted_performance_summary")


if __name__ == "__main__":
    regenerate_ml_backtest()
    regenerate_cost_and_threshold_sensitivity()
