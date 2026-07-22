
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = {
    "ink": "#172033",
    "muted": "#64748B",
    "grid": "#D9E2EC",
    "blue": "#2563EB",
    "cyan": "#0891B2",
    "green": "#059669",
    "amber": "#D97706",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "slate": "#334155",
    "panel": "#F8FAFC",
}

SERIES_COLORS = [
    PALETTE["blue"],
    PALETTE["green"],
    PALETTE["amber"],
    PALETTE["purple"],
    PALETTE["cyan"],
    PALETTE["red"],
    PALETTE["slate"],
]


def set_research_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "xtick.color": PALETTE["slate"],
            "ytick.color": PALETTE["slate"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _filter_split(frame: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None or "split" not in frame.columns:
        return frame
    filtered = frame.loc[frame["split"].astype(str).eq(split)].copy()
    if filtered.empty:
        raise ValueError(f"no rows found for split={split}")
    return filtered


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise KeyError(f"missing {label} columns: {sorted(missing)}")


def _finish_figure(fig: plt.Figure, output_path: str | Path) -> Path:
    path = _prepare_output_path(output_path)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _annotate_note(ax: plt.Axes, text: str | None) -> None:
    if text:
        ax.text(
            0.0,
            -0.20,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color=PALETTE["muted"],
        )


def _despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_price_coverage_summary(
    coverage: pd.DataFrame,
    output_path: str | Path,
    title: str = "Price coverage by asset",
    note: str | None = None,
) -> Path:
    _require_columns(coverage, {"symbol", "n_observations", "coverage_ratio"}, "coverage")
    set_research_plot_style()
    frame = coverage.copy()
    frame["coverage_ratio"] = frame["coverage_ratio"].astype(float).clip(0, 1)
    frame = frame.sort_values("coverage_ratio", ascending=True)
    path = _prepare_output_path(output_path)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(frame))))
    y = np.arange(len(frame))
    bars = ax.barh(y, frame["coverage_ratio"], color=PALETTE["blue"], alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["symbol"].astype(str))
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Coverage ratio")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.7)
    for bar, obs in zip(bars, frame["n_observations"].astype(int)):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2, f"{obs:,} obs", va="center", fontsize=8.5, color=PALETTE["muted"])
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, path)


def plot_triplet_price_relationship(
    prices: pd.DataFrame,
    output_path: str | Path,
    title: str = "Indexed triplet price relationship",
    note: str | None = None,
) -> Path:
    _require_columns(prices, {"date", "target", "anchor_1", "anchor_2"}, "price relationship")
    set_research_plot_style()
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    columns = ["target", "anchor_1", "anchor_2"]
    indexed = frame[columns].astype(float).div(frame[columns].astype(float).iloc[0]).mul(100.0)

    fig, ax = plt.subplots(figsize=(11, 6))
    labels = ["Target", "Anchor 1", "Anchor 2"]
    for idx, col in enumerate(columns):
        ax.plot(frame["date"], indexed[col], linewidth=2.2 if col == "target" else 1.75, color=SERIES_COLORS[idx], label=labels[idx])
    ax.set_ylabel("Indexed price level, first observation = 100")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_hedge_ratio_stability(
    coefficients: pd.DataFrame,
    output_path: str | Path,
    title: str = "Dynamic hedge-ratio stability",
    note: str | None = None,
) -> Path:
    _require_columns(coefficients, {"date", "model", "beta_1", "beta_2"}, "hedge ratio")
    set_research_plot_style()
    frame = coefficients.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for model_idx, (model, group) in enumerate(frame.groupby("model", sort=False)):
        ordered = group.sort_values("date")
        color = SERIES_COLORS[model_idx % len(SERIES_COLORS)]
        axes[0].plot(ordered["date"], ordered["beta_1"], label=str(model), color=color, linewidth=1.9)
        axes[1].plot(ordered["date"], ordered["beta_2"], label=str(model), color=color, linewidth=1.9)
    axes[0].set_title(title, loc="left", pad=12, fontweight="bold")
    axes[0].set_ylabel("Beta 1")
    axes[1].set_ylabel("Beta 2")
    axes[1].set_xlabel("Date")
    for ax in axes:
        ax.axhline(0, color="#94A3B8", linewidth=1.0, alpha=0.8)
        ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
        _despine(ax)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    _date_axis(axes[1])
    _annotate_note(axes[1], note)
    return _finish_figure(fig, output_path)


def plot_residual_zscore_example(
    residuals: pd.DataFrame,
    output_path: str | Path,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5,
    stop_loss: float = 3.0,
    title: str = "Residual z-score with trading bands",
    note: str | None = None,
) -> Path:
    _require_columns(residuals, {"date", "z_score"}, "residual z-score")
    set_research_plot_style()
    frame = residuals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(frame["date"], frame["z_score"].astype(float), color=PALETTE["blue"], linewidth=1.7)
    ax.fill_between(frame["date"], -exit_threshold, exit_threshold, color=PALETTE["green"], alpha=0.10, label="Exit zone")
    for value, label, color, linestyle in [
        (entry_threshold, "Entry", PALETTE["amber"], "--"),
        (-entry_threshold, "Entry", PALETTE["amber"], "--"),
        (stop_loss, "Stop", PALETTE["red"], ":"),
        (-stop_loss, "Stop", PALETTE["red"], ":"),
        (0, "Mean", PALETTE["slate"], "-"),
    ]:
        ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Residual z-score")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_residual_distribution(
    residuals: pd.DataFrame,
    output_path: str | Path,
    value_col: str = "z_score",
    title: str = "Residual distribution",
    note: str | None = None,
) -> Path:
    _require_columns(residuals, {value_col}, "residual distribution")
    set_research_plot_style()
    values = residuals[value_col].astype(float).dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.hist(values, bins=36, color=PALETTE["blue"], alpha=0.85, density=True)
    mean = float(np.mean(values)) if values.size else 0.0
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    ax.axvline(mean, color=PALETTE["red"], linewidth=2.0, label=f"Mean {mean:.2f}")
    if std > 0:
        ax.axvline(mean + std, color=PALETTE["amber"], linestyle="--", linewidth=1.2, label="±1 std")
        ax.axvline(mean - std, color=PALETTE["amber"], linestyle="--", linewidth=1.2)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def compute_autocorrelation(values: pd.Series | np.ndarray, max_lag: int = 20) -> pd.DataFrame:
    series = pd.Series(values).astype(float).dropna()
    rows: list[dict[str, float]] = []
    for lag in range(1, max_lag + 1):
        rows.append({"lag": lag, "autocorrelation": float(series.autocorr(lag=lag))})
    return pd.DataFrame(rows)


def plot_residual_autocorrelation(
    autocorr: pd.DataFrame | pd.Series | np.ndarray,
    output_path: str | Path,
    title: str = "Residual autocorrelation profile",
    note: str | None = None,
) -> Path:
    if isinstance(autocorr, pd.DataFrame):
        _require_columns(autocorr, {"lag", "autocorrelation"}, "autocorrelation")
        frame = autocorr.copy()
    else:
        frame = compute_autocorrelation(autocorr, max_lag=20)
    set_research_plot_style()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(frame["lag"].astype(int), frame["autocorrelation"].astype(float), color=PALETTE["cyan"], alpha=0.88)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1.0)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_half_life_by_triplet(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
    title: str = "Residual half-life by triplet",
    note: str | None = None,
) -> Path:
    _require_columns(diagnostics, {"triplet_id", "half_life"}, "half-life")
    set_research_plot_style()
    frame = diagnostics.copy().sort_values("half_life", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.38 * len(frame))))
    ax.barh(np.arange(len(frame)), frame["half_life"].astype(float), color=PALETTE["purple"], alpha=0.86)
    ax.set_yticks(np.arange(len(frame)))
    ax.set_yticklabels(frame["triplet_id"].astype(str))
    ax.set_xlabel("Estimated half-life, trading days")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_baseline_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: str | Path,
    title: str = "Baseline equity curve",
    note: str | None = None,
) -> Path:
    _require_columns(equity_curve, {"date", "equity"}, "baseline equity")
    set_research_plot_style()
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(frame["date"], frame["equity"].astype(float), color=PALETTE["blue"], linewidth=2.2)
    ax.fill_between(frame["date"], frame["equity"].astype(float), frame["equity"].astype(float).min(), color=PALETTE["blue"], alpha=0.10)
    ax.set_ylabel("Cumulative net PnL units")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_baseline_drawdown(
    equity_curve: pd.DataFrame,
    output_path: str | Path,
    title: str = "Baseline drawdown",
    note: str | None = None,
) -> Path:
    _require_columns(equity_curve, {"date", "drawdown"}, "baseline drawdown")
    set_research_plot_style()
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.fill_between(frame["date"], frame["drawdown"].astype(float), 0, color=PALETTE["red"], alpha=0.30)
    ax.plot(frame["date"], frame["drawdown"].astype(float), color=PALETTE["red"], linewidth=1.8)
    ax.set_ylabel("Drawdown in PnL units")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_event_label_distribution(
    labels: pd.DataFrame,
    output_path: str | Path,
    title: str = "Event label distribution",
    note: str | None = None,
) -> Path:
    if "label" in labels.columns:
        counts = labels["label"].map({0: "Failure", 1: "Success"}).fillna(labels["label"].astype(str)).value_counts().rename_axis("outcome").reset_index(name="n_events")
    else:
        _require_columns(labels, {"outcome", "n_events"}, "event-label distribution")
        counts = labels.copy()
    set_research_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = [PALETTE["red"] if str(x).lower().startswith("fail") else PALETTE["green"] for x in counts["outcome"]]
    bars = ax.bar(counts["outcome"].astype(str), counts["n_events"].astype(int), color=colors, alpha=0.88)
    total = int(counts["n_events"].sum())
    for bar in bars:
        pct = bar.get_height() / total if total else 0
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(bar.get_height())}\n{pct:.1%}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Candidate events")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_feature_correlation_heatmap(
    matrix_or_features: pd.DataFrame,
    output_path: str | Path,
    title: str = "Feature correlation heatmap",
    note: str | None = None,
) -> Path:
    set_research_plot_style()
    frame = matrix_or_features.copy()
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("no numeric columns available for correlation heatmap")
    corr = numeric.corr() if not (numeric.shape[0] == numeric.shape[1] and np.allclose(numeric.to_numpy(), numeric.to_numpy().T, equal_nan=True)) else numeric
    labels = [str(x).replace("_", " ") for x in corr.columns]
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(labels)), max(7, 0.50 * len(labels))))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            if np.isfinite(value) and len(labels) <= 16:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(value) > 0.55 else PALETTE["ink"])
    fig.colorbar(image, ax=ax, fraction=0.036, pad=0.03, label="Correlation")
    _annotate_note(ax, note)
    return _finish_figure(fig, output_path)


def plot_logistic_loss_curve(
    loss: pd.DataFrame,
    output_path: str | Path,
    title: str = "Logistic regression training loss",
    note: str | None = None,
) -> Path:
    _require_columns(loss, {"iteration", "loss"}, "loss curve")
    set_research_plot_style()
    frame = loss.copy().sort_values("iteration")
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(frame["iteration"].astype(int), frame["loss"].astype(float), color=PALETTE["blue"], linewidth=2.1)
    if "validation_loss" in frame.columns:
        ax.plot(frame["iteration"].astype(int), frame["validation_loss"].astype(float), color=PALETTE["amber"], linewidth=1.8, label="Validation")
        ax.legend(frameon=False)
    ax.set_xlabel("Gradient descent iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_probability_calibration_curve(
    calibration: pd.DataFrame,
    output_path: str | Path,
    split: str | None = None,
    title: str = "Probability calibration curve",
    note: str | None = None,
) -> Path:
    required = {"mean_predicted_probability", "realized_success_rate"}
    _require_columns(calibration, required, "calibration")
    set_research_plot_style()
    frame = _filter_split(calibration, split).dropna(subset=list(required)).copy().sort_values("mean_predicted_probability")
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1.2, label="Perfect calibration")
    ax.plot(frame["mean_predicted_probability"], frame["realized_success_rate"], marker="o", color=PALETTE["blue"], linewidth=2.2, label="Observed")
    if "n_events" in frame.columns:
        sizes = frame["n_events"].astype(float).clip(lower=1)
        sizes = 50 + 220 * sizes / sizes.max()
        ax.scatter(frame["mean_predicted_probability"], frame["realized_success_rate"], s=sizes, color=PALETTE["blue"], alpha=0.25, edgecolor=PALETTE["blue"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted reversion probability")
    ax.set_ylabel("Realized success rate")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, loc="upper left")
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_precision_by_probability_bucket(
    buckets: pd.DataFrame,
    output_path: str | Path,
    split: str | None = None,
    title: str = "Precision by probability bucket",
    note: str | None = None,
) -> Path:
    required = {"probability_bucket", "precision", "n_events"}
    _require_columns(buckets, required, "bucket")
    set_research_plot_style()
    frame = _filter_split(buckets, split).copy()
    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.arange(frame.shape[0])
    bars = ax.bar(x, frame["precision"].astype(float), color=PALETTE["green"], alpha=0.86)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["probability_bucket"].astype(str), rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability bucket")
    ax.set_ylabel("Realized success rate")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    for bar, n in zip(bars, frame["n_events"].astype(int)):
        ax.text(bar.get_x() + bar.get_width() / 2, min(0.98, bar.get_height() + 0.035), f"n={n}", ha="center", va="bottom", fontsize=8.5, color=PALETTE["slate"])
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_confusion_matrix(matrix: pd.DataFrame, output_path: str | Path, split: str | None = None) -> Path:
    required = {"actual_label", "predicted_label", "count"}
    _require_columns(matrix, required, "confusion-matrix")
    set_research_plot_style()
    frame = _filter_split(matrix, split).copy()
    pivot = frame.pivot_table(index="actual_label", columns="predicted_label", values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted failure", "Predicted success"], rotation=20, ha="right")
    ax.set_yticklabels(["Actual failure", "Actual success"])
    ax.set_title("Confusion matrix", loc="left", pad=12, fontweight="bold")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(pivot.iloc[i, j]), ha="center", va="center", fontsize=12, fontweight="bold", color=PALETTE["ink"])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _finish_figure(fig, output_path)


def plot_strategy_equity_curve(equity_curve: pd.DataFrame, output_path: str | Path, title: str = "Baseline versus ML-filtered equity", note: str | None = None) -> Path:
    _require_columns(equity_curve, {"date", "strategy", "equity"}, "equity")
    set_research_plot_style()
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(11, 5.8))
    for idx, (strategy, group) in enumerate(frame.groupby("strategy", sort=False)):
        ordered = group.sort_values("date")
        ax.plot(ordered["date"], ordered["equity"].astype(float), label=str(strategy), color=SERIES_COLORS[idx % len(SERIES_COLORS)], linewidth=2.1)
    ax.set_ylabel("Cumulative net PnL units")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_strategy_drawdown(equity_curve: pd.DataFrame, output_path: str | Path, title: str = "Drawdown comparison", note: str | None = None) -> Path:
    _require_columns(equity_curve, {"date", "strategy", "drawdown"}, "drawdown")
    set_research_plot_style()
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for idx, (strategy, group) in enumerate(frame.groupby("strategy", sort=False)):
        ordered = group.sort_values("date")
        ax.plot(ordered["date"], ordered["drawdown"].astype(float), label=str(strategy), color=SERIES_COLORS[idx % len(SERIES_COLORS)], linewidth=1.9)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1.0)
    ax.set_ylabel("Drawdown in PnL units")
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    _date_axis(ax)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_strategy_bar_comparison(summary: pd.DataFrame, metric: str, output_path: str | Path, title: str | None = None) -> Path:
    _require_columns(summary, {"strategy", metric}, "summary")
    set_research_plot_style()
    frame = summary.copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(frame.shape[0])
    ax.bar(x, frame[metric].astype(float), color=SERIES_COLORS[: frame.shape[0]], alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["strategy"].astype(str), rotation=25, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title if title is not None else metric.replace("_", " ").title(), loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_performance_by_triplet(
    performance: pd.DataFrame,
    output_path: str | Path,
    metric: str = "net_pnl",
    title: str = "Performance by triplet",
    note: str | None = None,
) -> Path:
    _require_columns(performance, {"triplet_id", "strategy", metric}, "triplet performance")
    set_research_plot_style()
    frame = performance.copy()
    order = frame.groupby("triplet_id")[metric].mean().sort_values().index.tolist()
    strategies = list(frame["strategy"].drop_duplicates())
    x = np.arange(len(order))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(max(10, 0.65 * len(order)), 5.8))
    for i, strategy in enumerate(strategies):
        values = frame.loc[frame["strategy"].eq(strategy)].set_index("triplet_id").reindex(order)[metric].astype(float)
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, values, width=width, label=strategy, color=SERIES_COLORS[i % len(SERIES_COLORS)], alpha=0.88)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, ncol=min(3, len(strategies)), loc="upper left")
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_performance_by_regime(
    performance: pd.DataFrame,
    output_path: str | Path,
    metric: str = "mean_trade_pnl",
    title: str = "Strategy performance by inferred regime",
    note: str | None = None,
) -> Path:
    _require_columns(performance, {"regime", metric}, "regime performance")
    set_research_plot_style()
    frame = performance.copy()
    if "strategy" not in frame.columns:
        frame["strategy"] = "strategy"
    regimes = list(frame["regime"].drop_duplicates())
    strategies = list(frame["strategy"].drop_duplicates())
    x = np.arange(len(regimes))
    width = 0.8 / max(1, len(strategies))
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for i, strategy in enumerate(strategies):
        values = frame.loc[frame["strategy"].eq(strategy)].set_index("regime").reindex(regimes)[metric].astype(float)
        ax.bar(x + (i - (len(strategies) - 1) / 2) * width, values, width=width, color=SERIES_COLORS[i % len(SERIES_COLORS)], label=strategy, alpha=0.88)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r).replace("_", " ") for r in regimes], rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False)
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_transaction_cost_sensitivity(
    sensitivity: pd.DataFrame,
    output_path: str | Path,
    metric: str = "net_pnl",
    title: str = "Transaction cost sensitivity",
    note: str | None = None,
) -> Path:
    _require_columns(sensitivity, {"cost_scenario", "strategy", metric}, "sensitivity")
    set_research_plot_style()
    frame = sensitivity.copy()
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for idx, (strategy, group) in enumerate(frame.groupby("strategy", sort=False)):
        ordered = group.sort_values("total_cost_per_unit") if "total_cost_per_unit" in group.columns else group
        ax.plot(ordered["cost_scenario"].astype(str), ordered[metric].astype(float), marker="o", markersize=5, linewidth=2.1, label=strategy, color=SERIES_COLORS[idx % len(SERIES_COLORS)])
    ax.axhline(0, color=PALETTE["slate"], linewidth=1.0)
    ax.set_xlabel("Cost scenario")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title, loc="left", pad=12, fontweight="bold")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, alpha=0.65)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _annotate_note(ax, note)
    _despine(ax)
    return _finish_figure(fig, output_path)


def plot_threshold_sensitivity(sensitivity: pd.DataFrame, output_path: str | Path, strategy: str = "ml_filtered", metric: str = "net_pnl") -> Path:
    _require_columns(sensitivity, {"strategy", "entry_threshold", "exit_threshold", metric}, "sensitivity")
    set_research_plot_style()
    frame = sensitivity.loc[sensitivity["strategy"].astype(str).eq(strategy)].copy()
    if frame.empty:
        raise ValueError(f"no rows found for strategy={strategy}")
    pivot = frame.pivot_table(index="entry_threshold", columns="exit_threshold", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("Exit threshold")
    ax.set_ylabel("Entry threshold")
    ax.set_title(f"Threshold sensitivity: {strategy}", loc="left", pad=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label=metric.replace("_", " "))
    return _finish_figure(fig, output_path)


FINAL_FIGURE_CAPTIONS = {
    "price_coverage_summary.png": "Shows whether all assets have enough aligned observations to support cross-asset regression and backtesting.",
    "triplet_price_relationship.png": "Indexes the target and anchor assets to compare the broad co-movement that motivates triangular relative-value modeling.",
    "hedge_ratio_stability.png": "Compares dynamic beta paths to evaluate whether hedge ratios are stable or regime-sensitive.",
    "residual_zscore_example.png": "Shows how residual dislocations map into entry, exit, and stop-loss zones.",
    "residual_distribution.png": "Checks whether residuals are centered, heavy-tailed, or skewed relative to the strategy assumptions.",
    "residual_autocorrelation.png": "Measures persistence in residual behavior and whether mean reversion is plausible across lags.",
    "half_life_by_triplet.png": "Compares estimated residual mean-reversion speed across triplets.",
    "baseline_equity_curve.png": "Displays the cumulative result of the rule-based baseline before ML filtering.",
    "baseline_drawdown.png": "Shows the path-dependent risk of the baseline strategy through peak-to-trough losses.",
    "event_label_distribution.png": "Summarizes the balance between successful and failed candidate events.",
    "feature_correlation_heatmap.png": "Identifies redundant or highly related ML features before model training.",
    "logistic_loss_curve.png": "Checks whether gradient descent training is numerically stable and loss decreases smoothly.",
    "probability_calibration_curve.png": "Compares predicted reversion probabilities to realized success rates.",
    "precision_by_probability_bucket.png": "Tests whether higher model probabilities correspond to higher realized trade success.",
    "ml_filtered_vs_baseline_equity.png": "Compares cumulative strategy results after applying the ML probability filter.",
    "performance_by_triplet.png": "Shows whether performance is broad-based or concentrated in a few relationships.",
    "performance_by_regime.png": "Compares trade performance across inferred residual regimes.",
    "transaction_cost_sensitivity.png": "Shows how much performance decays as execution-cost assumptions become more conservative.",
}


def write_chart_caption_table(output_path: str | Path) -> Path:
    path = _prepare_output_path(output_path)
    rows = [{"figure": name, "caption": caption} for name, caption in FINAL_FIGURE_CAPTIONS.items()]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
