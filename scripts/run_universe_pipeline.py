"""
Runs the full research pipeline -- hedge ratios, residuals, event labeling,
feature engineering, the logistic trade filter, HMM regime detection, and
the ML-filtered backtest -- across every triplet in TRIPLET_DEFINITIONS
(82, spanning 18 sector themes), not just the original 10.

This intentionally does NOT fall back to synthetic placeholder data the
way some of the earlier phase notebooks do when their preferred input is
missing. If data/processed/adjusted_prices_clean.csv doesn't exist, this
script stops with a clear error rather than silently generating fake
prices -- run scripts/ingest_prices.py first (on a machine with real
internet access).

Output goes to data/processed/full_universe_*.csv, deliberately separate
from the original ml_backtest_*.csv / threshold_sensitivity_table.csv
etc. produced by the notebooks against the original 10 triplets, so this
doesn't silently overwrite or get confused with that existing, already
cost/threshold-validated dataset.

Usage: python scripts/run_universe_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DEFAULT_RIDGE_ALPHA, DEFAULT_ROLLING_WINDOW, PROCESSED_DATA_DIR, TRIPLET_DEFINITIONS
from src.cointegration import cointegration_report
from src.features import build_event_feature_matrix, collinear_feature_pairs, feature_correlation_matrix
from src.hmm import fit_hmm_by_triplet
from src.labeling import LabelingConfig, generate_event_labels
from src.logistic_model import train_event_logistic_model
from src.regression import estimate_dynamic_hedges_for_triplets, fit_static_triplet
from src.backtest import run_ml_backtest_comparison

PRICES_PATH = PROCESSED_DATA_DIR / "adjusted_prices_clean.csv"
OUT_PREFIX = "full_universe_"


def _require_real_prices() -> pd.DataFrame:
    if not PRICES_PATH.exists():
        raise SystemExit(
            f"{PRICES_PATH} not found.\n\n"
            "This pipeline requires real ingested price data and will not run\n"
            "against synthetic placeholders. Run this first, on a machine with\n"
            "real internet access:\n\n"
            "    python scripts/ingest_prices.py\n"
        )
    return pd.read_csv(PRICES_PATH, parse_dates=["date"])


def _wide_log_prices(clean_prices: pd.DataFrame) -> pd.DataFrame:
    clean_prices = clean_prices.loc[clean_prices.get("quality_flag", "clean") == "clean"] if "quality_flag" in clean_prices.columns else clean_prices
    wide = clean_prices.pivot_table(index="date", columns="symbol", values="adj_close", aggfunc="last")
    return np.log(wide).sort_index()


def run_hedge_ratios(log_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    available_cols = set(log_prices.columns)
    fittable, skipped = [], []
    for triplet in TRIPLET_DEFINITIONS:
        required = {triplet["target"], triplet["hedge_1"], triplet["hedge_2"]}
        (fittable if required.issubset(available_cols) else skipped).append(triplet)
    if skipped:
        print(f"    skipping {len(skipped)} triplet(s) missing a required symbol in ingested prices: "
              f"{[t['triplet_id'] for t in skipped]}")

    dynamic = estimate_dynamic_hedges_for_triplets(
        log_prices=log_prices, triplets=fittable,
        window=DEFAULT_ROLLING_WINDOW, ridge_alpha=DEFAULT_RIDGE_ALPHA,
    ) if fittable else {"rolling_coefficients": pd.DataFrame(), "ridge_coefficients": pd.DataFrame(), "dynamic_residuals": pd.DataFrame()}

    static_coeffs, static_residuals = [], []
    for triplet in fittable:
        hedge_cols = [triplet["hedge_1"], triplet["hedge_2"]]
        try:
            coeffs, resid = fit_static_triplet(log_prices, triplet["target"], hedge_cols, triplet_id=triplet["triplet_id"])
        except (KeyError, ValueError):
            continue
        static_coeffs.append(coeffs)
        static_residuals.append(resid)

    dynamic["static_coefficients"] = pd.concat(static_coeffs, ignore_index=True) if static_coeffs else pd.DataFrame()
    dynamic["static_residuals"] = pd.concat(static_residuals, ignore_index=True) if static_residuals else pd.DataFrame()
    return dynamic


def run_labeling(residuals: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if residuals.empty:
        return {"scored_residuals": pd.DataFrame(), "candidate_events": pd.DataFrame(), "event_labels": pd.DataFrame()}
    return generate_event_labels(residuals, config=LabelingConfig())


def run_features(
    candidates: pd.DataFrame, scored_residuals: pd.DataFrame, labels: pd.DataFrame,
    clean_prices: pd.DataFrame,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    returns_long = clean_prices.rename(columns={"adj_close": "_ac"}).copy()
    returns_long = returns_long.sort_values(["symbol", "date"])
    returns_long["return"] = returns_long.groupby("symbol")["_ac"].pct_change()
    log_prices_long = clean_prices.assign(log_price=np.log(clean_prices["adj_close"]))
    volumes_long = clean_prices[["symbol", "date", "volume"]] if "volume" in clean_prices.columns else None

    return build_event_feature_matrix(
        candidate_events=candidates, residuals=scored_residuals, labels=labels,
        returns=returns_long[["symbol", "date", "return"]],
        log_prices=log_prices_long[["symbol", "date", "log_price"]],
        volumes=volumes_long,
    )


def main() -> None:
    clean_prices = _require_real_prices()
    log_prices = _wide_log_prices(clean_prices)
    print(f"loaded {len(clean_prices)} clean price rows, {log_prices.shape[1]} symbols, "
          f"{log_prices.shape[0]} trading days")

    print("\n[1/7] hedge ratios (static OLS, rolling OLS, rolling ridge) ...")
    hedges = run_hedge_ratios(log_prices)
    for key in ["rolling_coefficients", "ridge_coefficients", "static_coefficients", "dynamic_residuals", "static_residuals"]:
        hedges[key].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}{key}.csv", index=False)
    n_fitted = hedges["static_coefficients"]["triplet_id"].nunique() if not hedges["static_coefficients"].empty else 0
    print(f"    fitted {n_fitted}/{len(TRIPLET_DEFINITIONS)} triplets "
          f"(remainder missing required symbols in ingested price data)")

    print("\n[2/7] cointegration gate (ADF test + Benjamini-Hochberg FDR correction) ...")
    residuals_for_labeling = hedges["dynamic_residuals"]
    if not residuals_for_labeling.empty:
        coint = cointegration_report(residuals_for_labeling, value_column="residual")
        coint.to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}cointegration_report.csv", index=False)
        passing = set(coint.loc[coint["fdr_reject"], "triplet_id"])
        n_tested = coint["triplet_id"].nunique()
        print(f"    {len(passing)}/{n_tested} triplets pass the FDR-corrected stationarity test "
              f"-- only these proceed to labeling/backtest")
        residuals_for_labeling = residuals_for_labeling.loc[residuals_for_labeling["triplet_id"].isin(passing)]
    else:
        print("    skipped -- no residuals to test")

    print("\n[3/7] event labeling (z-score entry/exit/stop-loss/holding-period rules) ...")
    labeling = run_labeling(residuals_for_labeling)
    for key in ["scored_residuals", "candidate_events", "event_labels"]:
        labeling[key].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}{key}.csv", index=False)
    print(f"    {len(labeling['candidate_events'])} candidate events across "
          f"{labeling['candidate_events']['triplet_id'].nunique() if not labeling['candidate_events'].empty else 0} triplets")

    print("\n[4/7] feature engineering ...")
    feature_matrix = run_features(labeling["candidate_events"], labeling["scored_residuals"], labeling["event_labels"], clean_prices)
    feature_matrix.to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}event_feature_matrix.csv", index=False)
    if not feature_matrix.empty:
        corr = feature_correlation_matrix(feature_matrix)
        collinear_feature_pairs(corr, threshold=0.85).to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}feature_collinearity_flags.csv", index=False)
    print(f"    {len(feature_matrix)} labeled events with features")

    print("\n[5/7] logistic trade filter (from scratch, walk-forward split) ...")
    if len(feature_matrix) >= 30:  # too few events makes the walk-forward split meaningless
        result = train_event_logistic_model(feature_matrix)
        result["model_coefficients"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}logistic_model_coefficients.csv", index=False)
        result["predicted_reversion_probabilities"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}predicted_reversion_probabilities.csv", index=False)
        result["validation_metrics"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}logistic_validation_metrics.csv", index=False)
        predictions = result["predicted_reversion_probabilities"]
        print(f"    trained on {len(feature_matrix)} events")
    else:
        predictions = pd.DataFrame()
        print(f"    skipped -- only {len(feature_matrix)} labeled events, need >=30 for a meaningful walk-forward split")

    print("\n[6/7] HMM regime detection (all triplets with sufficient history, not a fixed subset) ...")
    if not labeling["scored_residuals"].empty:
        hmm_result = fit_hmm_by_triplet(labeling["scored_residuals"], value_column="z_score")
        hmm_result["regime_probabilities"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}hmm_regime_probabilities.csv", index=False)
        hmm_result["regime_parameters"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}hmm_regime_parameters.csv", index=False)
        n_hmm = len(hmm_result["models"])
        print(f"    fit HMM for {n_hmm} triplets (skipped any with too little residual history)")
    else:
        print("    skipped -- no scored residuals")

    print("\n[7/7] ML-filtered backtest ...")
    if not predictions.empty and not labeling["event_labels"].empty:
        backtest = run_ml_backtest_comparison(labeling["event_labels"], predictions, probability_threshold=0.60, transaction_cost=0.05)
        backtest["strategy_trade_log"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}backtest_trade_log.csv", index=False)
        backtest["strategy_equity_curve"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}backtest_equity_curve.csv", index=False)
        backtest["strategy_summary"].to_csv(PROCESSED_DATA_DIR / f"{OUT_PREFIX}backtest_strategy_summary.csv", index=False)
        print(backtest["strategy_summary"][["strategy", "trade_count", "net_pnl", "sharpe", "win_rate"]].to_string(index=False))
    else:
        print("    skipped -- no trained model predictions available")

    print(f"\ndone. Outputs written to {PROCESSED_DATA_DIR}/{OUT_PREFIX}*.csv")


if __name__ == "__main__":
    main()
