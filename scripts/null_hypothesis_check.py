"""
Null-hypothesis check for the event-labeling methodology (src/labeling.py).

Runs the exact entry/exit/stop-loss/holding-period rule used everywhere
else in this project against genuinely independent random walks -- no
real cointegrating relationship, no engineered mean reversion -- across
many seeds, and reports the win rate distribution. If the labeling method
has a structural bias, the mean win rate here will sit well away from 50%
consistently. If it doesn't, it will center on 50% with ordinary sampling
noise.

This exists because an earlier one-off test (a single seed, during
development of scripts/run_universe_pipeline.py) showed a 75.7% win rate
on random-walk data and was initially written up as evidence of a
structural flaw. Re-running it properly across 20 seeds showed a mean of
51.7% -- the original result was one noisy draw, not a bias. See
CHANGELOG.md for the full account. This script is the reusable version of
that check, so the question can be re-asked cheaply whenever the labeling
config, window length, or anything upstream of it changes.

Usage: python scripts/null_hypothesis_check.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.labeling import LabelingConfig, generate_event_labels
from src.regression import rolling_ols
from src.residuals import zscore_residuals


def run_single_trial(seed: int, n_days: int, window: int, config: LabelingConfig) -> dict:
    """One trial: three independent random walks, rolling OLS regression,
    rolling z-score, the project's real entry/exit/stop-loss rule."""
    rng = np.random.default_rng(seed)
    target = np.cumsum(rng.normal(0, 1, n_days)) + 100
    hedge1 = np.cumsum(rng.normal(0, 1, n_days)) + 100
    hedge2 = np.cumsum(rng.normal(0, 1, n_days)) + 100
    log_prices = pd.DataFrame(
        {"T": target, "H1": hedge1, "H2": hedge2},
        index=pd.bdate_range("2020-01-01", periods=n_days),
    )
    rolled = rolling_ols(log_prices, "T", ["H1", "H2"], window=window, triplet_id=f"null_seed_{seed}")
    if rolled.empty:
        return {"seed": seed, "n_events": 0, "win_rate": np.nan}
    rolled["method"] = "rolling_ols"
    rolled["z_score"] = zscore_residuals(rolled["residual"], window=window).to_numpy()

    result = generate_event_labels(rolled, config=config)
    labels = result["event_labels"]
    return {
        "seed": seed,
        "n_events": len(labels),
        "win_rate": float(labels["label"].mean()) if len(labels) else np.nan,
    }


def run_null_hypothesis_check(
    n_seeds: int = 30, n_days: int = 2000, window: int = 60,
    config: LabelingConfig | None = None, min_events: int = 5,
) -> pd.DataFrame:
    cfg = config or LabelingConfig()
    trials = [run_single_trial(seed, n_days, window, cfg) for seed in range(n_seeds)]
    return pd.DataFrame(trials)


def main() -> None:
    print("Running null-hypothesis check: independent random walks, no real")
    print("mean reversion, same labeling rule used throughout this project.\n")

    results = run_null_hypothesis_check()
    usable = results.loc[results["n_events"] >= 5]
    print(f"{len(usable)}/{len(results)} seeds produced >=5 events\n")

    if usable.empty:
        print("no seeds produced enough events to evaluate -- try more days or a looser entry threshold")
        return

    win_rates = usable["win_rate"]
    print(f"mean win rate:   {win_rates.mean():.3f}")
    print(f"median win rate: {win_rates.median():.3f}")
    print(f"std:             {win_rates.std():.3f}")
    print(f"range:           {win_rates.min():.3f} - {win_rates.max():.3f}")
    print(f"mean events/seed: {usable['n_events'].mean():.1f}")

    out_path = PROJECT_ROOT / "data" / "processed" / "null_hypothesis_check.csv"
    results.to_csv(out_path, index=False)
    print(f"\nfull results written to {out_path}")

    if abs(win_rates.mean() - 0.5) > 0.10:
        print("\nWARNING: mean win rate is more than 10 points away from 50% -- "
              "this warrants investigation before trusting real backtest results.")
    else:
        print("\nNo evidence of a systematic labeling bias: mean win rate is close to 50%.")


if __name__ == "__main__":
    main()
