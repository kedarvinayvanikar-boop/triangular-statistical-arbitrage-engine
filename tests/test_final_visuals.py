
from pathlib import Path

import numpy as np
import pandas as pd

from src import plotting


def test_final_caption_table(tmp_path: Path):
    path = plotting.write_chart_caption_table(tmp_path / "captions.csv")
    frame = pd.read_csv(path)
    assert frame.shape[0] == 18
    assert {"figure", "caption"}.issubset(frame.columns)


def test_residual_autocorrelation_output(tmp_path: Path):
    values = np.sin(np.linspace(0, 4, 60))
    table = plotting.compute_autocorrelation(values, max_lag=5)
    assert table.shape[0] == 5
    assert set(table.columns) == {"lag", "autocorrelation"}
    path = plotting.plot_residual_autocorrelation(table, tmp_path / "acf.png")
    assert path.exists()


def test_required_visual_functions_create_files(tmp_path: Path):
    dates = pd.bdate_range("2024-01-01", periods=40)
    coverage = pd.DataFrame({"symbol": ["A", "B"], "n_observations": [40, 38], "coverage_ratio": [1.0, 0.95]})
    prices = pd.DataFrame({"date": dates, "target": np.linspace(100, 120, 40), "anchor_1": np.linspace(100, 115, 40), "anchor_2": np.linspace(100, 110, 40)})
    z = pd.DataFrame({"date": dates, "z_score": np.sin(np.linspace(0, 8, 40))})
    labels = pd.DataFrame({"label": [0, 1, 1, 0, 1]})
    loss = pd.DataFrame({"iteration": [1, 2, 3], "loss": [0.7, 0.6, 0.55]})

    paths = [
        plotting.plot_price_coverage_summary(coverage, tmp_path / "coverage.png"),
        plotting.plot_triplet_price_relationship(prices, tmp_path / "prices.png"),
        plotting.plot_residual_zscore_example(z, tmp_path / "z.png"),
        plotting.plot_event_label_distribution(labels, tmp_path / "labels.png"),
        plotting.plot_logistic_loss_curve(loss, tmp_path / "loss.png"),
    ]
    assert all(path.exists() for path in paths)
