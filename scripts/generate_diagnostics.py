"""
Generates three diagnostic tables that are not part of the original phase
pipeline but should be checked before trusting any headline metric here:

  1. feature_collinearity_flags.csv  -- ML trade-filter input features with
     |correlation| >= 0.85, which inflate logistic-regression coefficient
     variance without adding information.
  2. triplet_correlation_clusters.csv -- triplets that share a hedge-leg
     symbol (and are therefore not independent bets when PnL is aggregated
     across the book).
  3. bootstrap_confidence_intervals.csv -- win rate (Wilson score interval)
     and mean PnL (percentile bootstrap) with a 90% confidence interval,
     computed from the actual ML backtest trade log.

Run from the repository root, after scripts/fix_sharpe_regeneration.py (or
any other step that changes ml_backtest_trade_log.csv or
event_feature_matrix.csv) so these stay consistent with current data:

    python scripts/generate_diagnostics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features import collinear_feature_pairs, feature_correlation_matrix
from src.portfolio import bootstrap_trade_metric_ci, shared_leg_groups, wilson_score_interval

DATA_DIR = PROJECT_ROOT / "data" / "processed"

# The full set of triplets this project tracks hedge ratios for -- kept
# here rather than inferred from one table, since different phase outputs
# cover different subsets (see CHANGELOG.md on HMM coverage being partial).
TRIPLETS = [
    "AAPL_QQQ_XLK", "AMD_SMH_QQQ", "AMZN_QQQ_XLY", "BAC_XLF_KRE", "CVX_XLE_USO",
    "JPM_XLF_KRE", "MSFT_QQQ_XLK", "NVDA_SMH_QQQ", "TSLA_QQQ_XLY", "XOM_XLE_USO",
]


def generate_feature_collinearity() -> None:
    feature_matrix = pd.read_csv(DATA_DIR / "event_feature_matrix.csv")
    corr = feature_correlation_matrix(feature_matrix)
    flags = collinear_feature_pairs(corr, threshold=0.85)
    flags.to_csv(DATA_DIR / "feature_collinearity_flags.csv", index=False)
    print(f"feature_collinearity_flags.csv: {len(flags)} pairs flagged (threshold=0.85)")


def generate_triplet_clusters() -> None:
    clusters = shared_leg_groups(TRIPLETS)
    clusters = clusters.copy()
    clusters["triplets"] = clusters["triplets"].apply(lambda ids: ";".join(ids))
    clusters["shared_legs"] = clusters["shared_legs"].apply(lambda legs: ";".join(legs))
    clusters.to_csv(DATA_DIR / "triplet_correlation_clusters.csv", index=False)
    n_independent = int(clusters["independent_bet"].sum())
    n_clusters = len(clusters)
    print(f"triplet_correlation_clusters.csv: {len(TRIPLETS)} triplets -> {n_clusters} clusters ({n_independent} fully independent)")


def generate_confidence_intervals() -> None:
    trades = pd.read_csv(DATA_DIR / "ml_backtest_trade_log.csv")
    rows = []
    for strategy, group in trades.groupby("strategy"):
        pnl = group["net_pnl"].to_numpy()
        n = len(pnl)
        wins = int((pnl > 0).sum())

        wilson = wilson_score_interval(wins, n, confidence=0.90)
        rows.append({
            "strategy": strategy, "metric": "win_rate", "method": "wilson_score",
            "point_estimate": wilson["rate"], "ci_low": wilson["ci_low"], "ci_high": wilson["ci_high"],
            "confidence": 0.90, "n_trades": n,
        })

        boot = bootstrap_trade_metric_ci(pnl, metric="mean_pnl", n_resamples=2000, random_state=0)
        rows.append({
            "strategy": strategy, "metric": "mean_pnl", "method": "percentile_bootstrap",
            "point_estimate": boot["point_estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            "confidence": 0.90, "n_trades": n,
        })
    pd.DataFrame(rows).to_csv(DATA_DIR / "bootstrap_confidence_intervals.csv", index=False)
    print("bootstrap_confidence_intervals.csv: win_rate (Wilson) + mean_pnl (bootstrap) per strategy")


if __name__ == "__main__":
    generate_feature_collinearity()
    generate_triplet_clusters()
    generate_confidence_intervals()
