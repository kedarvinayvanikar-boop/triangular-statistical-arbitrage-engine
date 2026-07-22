"""
Renders the HMM regime-classification surface in three dimensions: elapsed
time in the rolling window, the standardized residual (z-score) that drives
the regime model, and the model's posterior probability that the residual
is in a mean-reverting state.

A 2D time series of z-score and a separate 2D time series of regime
probability each tell part of the story; overlaying them in 3D makes the
actual decision surface the ML filter is trading against visible in one
view -- specifically, whether large |z| dislocations coincide with high or
low mean-reversion confidence, which is the question the trade filter has
to answer at every entry signal.

Run from the repository root: python scripts/plot_regime_probability_surface.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.plotting import PALETTE, set_research_plot_style

REGIME_COLORS = {
    "mean_reverting": PALETTE["green"],
    "trending": PALETTE["blue"],
    "volatile_breakdown": PALETTE["red"],
}

DATA_PATH = PROJECT_ROOT / "data/processed/hmm_regime_probability_table.csv"
OUT_PATH = PROJECT_ROOT / "figures/final/regime_probability_3d_surface.png"


def _load() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    return df.sort_values(["triplet_id", "date"])


def _plot_triplet(ax, g: pd.DataFrame, title: str) -> None:
    t = np.arange(len(g))
    z = g["feature_value"].to_numpy()
    p = g["mean_reverting_probability"].to_numpy()
    regimes = g["most_likely_regime"].to_numpy()

    # color segments by regime rather than by point, so the transition
    # boundaries the HMM actually detected are visible as color breaks
    points = np.array([t, z, p]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = [REGIME_COLORS.get(r, PALETTE["muted"]) for r in regimes[:-1]]
    lc = Line3DCollection(segments, colors=seg_colors, linewidths=1.1, alpha=0.85)
    ax.add_collection3d(lc)

    # scatter overlay marks the discrete daily observations
    for regime, color in REGIME_COLORS.items():
        mask = regimes == regime
        if mask.any():
            ax.scatter(
                t[mask], z[mask], p[mask],
                s=9, color=color, depthshade=False, linewidths=0,
            )

    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(np.nanmin(z) - 0.3, np.nanmax(z) + 0.3)
    ax.set_zlim(0, 1)
    ax.set_xlabel("trading day index", labelpad=8, fontsize=8)
    ax.set_ylabel("residual z-score", labelpad=8, fontsize=8)
    ax.set_zlabel("P(mean-reverting)", labelpad=6, fontsize=8)
    ax.set_title(title, fontsize=10.5, color=PALETTE["ink"], pad=2)
    ax.view_init(elev=22, azim=-60)
    ax.xaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 0))
    ax.zaxis.pane.set_facecolor((0.97, 0.98, 0.99, 1))
    ax.tick_params(labelsize=7)


def main() -> None:
    set_research_plot_style()
    df = _load()
    triplets = sorted(df["triplet_id"].unique())

    fig = plt.figure(figsize=(15, 5.4))
    fig.suptitle(
        "Regime Probability Surface: Residual Dislocation vs. HMM Mean-Reversion Confidence",
        fontsize=13, color=PALETTE["ink"], y=1.03,
    )
    for i, trip in enumerate(triplets):
        ax = fig.add_subplot(1, len(triplets), i + 1, projection="3d")
        _plot_triplet(ax, df[df["triplet_id"] == trip], trip.replace("_", " / "))

    handles = [
        plt.Line2D([0], [0], color=c, lw=2, label=r.replace("_", " "))
        for r, c in REGIME_COLORS.items()
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, -0.04), fontsize=9,
    )
    fig.text(
        0.5, -0.11,
        "Synthetic placeholder data -- illustrates the regime-detection "
        "mechanism, not a claim about real market behavior.",
        ha="center", fontsize=8, color=PALETTE["muted"],
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
