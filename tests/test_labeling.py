import numpy as np
import pandas as pd
import pytest

from src.labeling import (
    LabelingConfig,
    generate_candidate_events,
    generate_event_labels,
    label_candidate_events,
    success_rate_by_z_bucket,
    summarize_event_labels,
)


def residual_frame(z_values, triplet_id="T_A_B_C", method="kalman_random_walk"):
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(z_values), freq="D"),
            "triplet_id": triplet_id,
            "method": method,
            "residual": np.asarray(z_values, dtype=float) / 10.0,
            "z_score": np.asarray(z_values, dtype=float),
        }
    )


def test_candidate_events_require_threshold_crossing():
    frame = residual_frame([0.2, 1.7, 2.1, 2.4, 1.1, -2.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=5, min_periods=2)

    candidates = generate_candidate_events(frame, config=cfg)

    assert candidates.shape[0] == 2
    assert candidates["side"].tolist() == ["short_spread", "long_spread"]
    assert candidates["entry_z_score"].round(1).tolist() == [2.1, -2.2]


def test_short_spread_label_succeeds_when_reversion_hits_before_stop():
    frame = residual_frame([0.1, 1.8, 2.2, 1.4, 0.4, 1.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=4, min_periods=2)
    candidates = generate_candidate_events(frame, config=cfg)

    labels = label_candidate_events(frame, candidates, config=cfg)

    assert labels.shape[0] == 1
    assert labels.loc[0, "label"] == 1
    assert labels.loc[0, "exit_reason"] == "reversion"
    assert labels.loc[0, "holding_period"] == 2


def test_short_spread_label_fails_when_stop_hits_first():
    frame = residual_frame([0.1, 1.8, 2.2, 2.7, 3.1, 0.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=5, min_periods=2)
    candidates = generate_candidate_events(frame, config=cfg)

    labels = label_candidate_events(frame, candidates, config=cfg)

    assert labels.loc[0, "label"] == 0
    assert labels.loc[0, "exit_reason"] == "stop_loss"
    assert labels.loc[0, "holding_period"] == 2


def test_event_label_fails_when_no_reversion_before_max_holding_period():
    frame = residual_frame([0.1, 1.8, 2.2, 1.9, 1.4, 1.0, 0.3])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=3, min_periods=2)
    candidates = generate_candidate_events(frame, config=cfg)

    labels = label_candidate_events(frame, candidates, config=cfg)

    assert labels.loc[0, "label"] == 0
    assert labels.loc[0, "exit_reason"] == "max_holding_period"
    assert labels.loc[0, "holding_period"] == 3


def test_long_spread_reversion_uses_symmetric_barrier():
    frame = residual_frame([0.1, -1.5, -2.1, -1.2, -0.4, -3.3])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=4, min_periods=2)
    candidates = generate_candidate_events(frame, config=cfg)

    labels = label_candidate_events(frame, candidates, config=cfg)

    assert candidates.loc[0, "side"] == "long_spread"
    assert labels.loc[0, "label"] == 1
    assert labels.loc[0, "exit_reason"] == "reversion"


def test_generate_event_labels_pipeline_returns_expected_tables():
    frame = residual_frame([0.0, 1.0, 2.1, 1.0, 0.3, -2.2, -1.0, -0.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=3, min_periods=2)

    result = generate_event_labels(frame, config=cfg)

    assert set(result) == {"scored_residuals", "candidate_events", "event_labels"}
    assert result["candidate_events"].shape[0] == 2
    assert result["event_labels"].shape[0] == 2


def test_event_summary_and_z_bucket_summary():
    frame = residual_frame([0.0, 1.0, 2.1, 1.0, 0.3, -2.6, -1.0, -0.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0, max_holding_period=3, min_periods=2)
    labels = generate_event_labels(frame, config=cfg)["event_labels"]

    triplet_summary = summarize_event_labels(labels)
    bucket_summary = success_rate_by_z_bucket(labels, buckets=[2.0, 2.5, 3.0])

    assert triplet_summary.loc[0, "n_events"] == 2
    assert 0.0 <= triplet_summary.loc[0, "success_rate"] <= 1.0
    assert bucket_summary["n_events"].sum() == 2


def test_invalid_config_rejects_inconsistent_barriers():
    frame = residual_frame([0.0, 2.1, 0.2])
    cfg = LabelingConfig(entry_z=2.0, exit_z=2.0, stop_loss_z=3.0, max_holding_period=3, min_periods=2)

    with pytest.raises(ValueError):
        generate_candidate_events(frame, config=cfg)
