import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location("run_universe_pipeline", SCRIPTS_DIR / "run_universe_pipeline.py")
pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pipeline)


def _synthetic_clean_prices(symbols, n_days=250, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    rows = []
    for symbol in symbols:
        start = rng.uniform(20, 300)
        log_returns = rng.normal(0.0003, 0.015, n_days)
        price = start * np.exp(np.cumsum(log_returns))
        for d, p in zip(dates, price):
            rows.append({
                "symbol": symbol, "date": d.strftime("%Y-%m-%d"),
                "close": p, "adj_close": p,
                "volume": rng.integers(1_000_000, 20_000_000),
                "quality_flag": "clean", "source": "synthetic_test",
            })
    return pd.DataFrame(rows)


def _two_triplets():
    return [
        {"triplet_id": "AAA_HH1_HH2", "target": "AAA", "hedge_1": "HH1", "hedge_2": "HH2", "theme": "test"},
        {"triplet_id": "BBB_HH1_HH2", "target": "BBB", "hedge_1": "HH1", "hedge_2": "HH2", "theme": "test"},
    ]


def test_wide_log_prices_pivots_and_takes_log():
    prices = _synthetic_clean_prices(["AAA", "HH1"], n_days=10)
    wide = pipeline._wide_log_prices(prices)
    assert set(wide.columns) == {"AAA", "HH1"}
    assert wide.shape[0] == 10
    # log of a positive price series must itself be finite and monotonically
    # related to price -- not, e.g., accidentally left un-logged
    raw_wide = prices.pivot_table(index="date", columns="symbol", values="adj_close")
    assert np.allclose(wide["AAA"].to_numpy(), np.log(raw_wide["AAA"].to_numpy()))


def test_run_hedge_ratios_fits_every_triplet_with_available_symbols():
    triplets = _two_triplets()
    prices = _synthetic_clean_prices(["AAA", "BBB", "HH1", "HH2"], n_days=200)
    log_prices = pipeline._wide_log_prices(prices)

    original_triplets = pipeline.TRIPLET_DEFINITIONS
    pipeline.TRIPLET_DEFINITIONS = triplets
    try:
        hedges = pipeline.run_hedge_ratios(log_prices)
    finally:
        pipeline.TRIPLET_DEFINITIONS = original_triplets

    assert set(hedges["static_coefficients"]["triplet_id"]) == {"AAA_HH1_HH2", "BBB_HH1_HH2"}
    assert not hedges["dynamic_residuals"].empty
    assert not hedges["rolling_coefficients"].empty
    assert not hedges["ridge_coefficients"].empty


def test_run_hedge_ratios_skips_triplets_with_missing_symbols_instead_of_crashing():
    triplets = _two_triplets() + [{"triplet_id": "ZZZ_MISSING_HH2", "target": "ZZZ", "hedge_1": "MISSING", "hedge_2": "HH2", "theme": "test"}]
    prices = _synthetic_clean_prices(["AAA", "BBB", "HH1", "HH2"], n_days=200)  # no ZZZ or MISSING
    log_prices = pipeline._wide_log_prices(prices)

    original_triplets = pipeline.TRIPLET_DEFINITIONS
    pipeline.TRIPLET_DEFINITIONS = triplets
    try:
        hedges = pipeline.run_hedge_ratios(log_prices)
    finally:
        pipeline.TRIPLET_DEFINITIONS = original_triplets

    fitted = set(hedges["static_coefficients"]["triplet_id"])
    assert fitted == {"AAA_HH1_HH2", "BBB_HH1_HH2"}
    assert "ZZZ_MISSING_HH2" not in fitted


def test_full_pipeline_stages_run_end_to_end_on_synthetic_data():
    # this is the same shape of test as the manual /tmp smoke test used
    # during development, made fast (2 triplets, 250 days) and automated
    triplets = _two_triplets()
    prices = _synthetic_clean_prices(["AAA", "BBB", "HH1", "HH2"], n_days=250)
    log_prices = pipeline._wide_log_prices(prices)

    original_triplets = pipeline.TRIPLET_DEFINITIONS
    pipeline.TRIPLET_DEFINITIONS = triplets
    try:
        hedges = pipeline.run_hedge_ratios(log_prices)
        labeling = pipeline.run_labeling(hedges["dynamic_residuals"])
        features = pipeline.run_features(
            labeling["candidate_events"], labeling["scored_residuals"], labeling["event_labels"], prices,
        )
    finally:
        pipeline.TRIPLET_DEFINITIONS = original_triplets

    assert isinstance(labeling["candidate_events"], pd.DataFrame)
    if not labeling["candidate_events"].empty:
        assert set(labeling["candidate_events"]["triplet_id"]).issubset({"AAA_HH1_HH2", "BBB_HH1_HH2"})
    # features frame should carry a label column when labels are present
    if not features.empty:
        assert "label" in features.columns


def test_require_real_prices_fails_loudly_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "PRICES_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(SystemExit, match="ingest_prices.py"):
        pipeline._require_real_prices()
