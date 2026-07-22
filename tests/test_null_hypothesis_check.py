import numpy as np
import pytest

from scripts.null_hypothesis_check import run_null_hypothesis_check, run_single_trial
from src.labeling import LabelingConfig


def test_single_trial_returns_expected_shape():
    result = run_single_trial(seed=0, n_days=500, window=60, config=LabelingConfig())
    assert set(result.keys()) == {"seed", "n_events", "win_rate"}
    assert result["seed"] == 0
    assert result["n_events"] >= 0
    if result["n_events"] > 0:
        assert 0.0 <= result["win_rate"] <= 1.0


def test_null_hypothesis_check_is_not_systematically_biased():
    # small run for test speed -- the full 30-seed/2000-day version is what
    # scripts/null_hypothesis_check.py runs when invoked directly
    results = run_null_hypothesis_check(n_seeds=8, n_days=800, window=60)
    usable = results.loc[results["n_events"] >= 3]
    if len(usable) < 4:
        pytest.skip("not enough seeds produced sufficient events at this reduced test scale")
    # this is a statistical check, not an exact one -- allow real sampling
    # noise but the mean should not be dramatically off 50% the way a true
    # structural bias would produce
    assert abs(usable["win_rate"].mean() - 0.5) < 0.25


def test_run_single_trial_handles_too_few_days_gracefully():
    # window (60) larger than available days should not crash, just
    # produce zero events
    result = run_single_trial(seed=0, n_days=30, window=60, config=LabelingConfig())
    assert result["n_events"] == 0
    assert np.isnan(result["win_rate"])
