from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelingConfig:
    # entry_z: how many standard deviations from normal the residual must
    # be before it's even considered a candidate trade (a "dislocation").
    entry_z: float = 2.0
    # exit_z: how close back to normal the residual must return to count
    # as a successful reversion -- doesn't have to hit exactly zero.
    exit_z: float = 0.5
    # stop_loss_z: if the residual gets even MORE extreme than this
    # (rather than reverting), the event is cut and marked a failure --
    # the position was wrong about the direction of reversion.
    stop_loss_z: float = 3.0
    # max_holding_period: if neither the exit nor the stop-loss triggers
    # within this many trading days, the event is marked a failure by
    # timeout rather than left open indefinitely.
    max_holding_period: int = 10
    # z_window: how many trailing days the rolling mean/std (used to
    # compute the z-score itself) are calculated over. None means use an
    # expanding window (all history to date) instead of a fixed one.
    z_window: Optional[int] = 60
    min_periods: int = 20
    # if True, a residual that starts the series already beyond entry_z
    # counts as a candidate event on its first appearance; if False (the
    # default), an event requires seeing the residual actually *cross*
    # the threshold from inside it, so a triplet whose first data point
    # coincidentally happens to be extreme doesn't generate a spurious
    # signal with no real "entry" moment behind it.
    include_initial_crossing: bool = False


def add_event_z_scores(
    residuals: pd.DataFrame,
    config: Optional[LabelingConfig] = None,
    residual_col: str = "residual",
    z_col: str = "z_score",
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> pd.DataFrame:
    """Attaches a rolling z-score column to a residual table, computed
    independently within each (triplet, method) group so one triplet's
    z-score never mixes in another triplet's residual history.
    """
    cfg = config or LabelingConfig()
    _validate_config(cfg)
    frame = _prepare_residual_frame(residuals, residual_col=residual_col)
    groups = _available_group_cols(frame, group_cols)

    if z_col in frame.columns and frame[z_col].notna().any():
        # z-scores were already computed upstream (e.g. this residual
        # table came from a source that scores its own residuals) --
        # don't silently overwrite them with a possibly different window
        return frame

    if not groups:
        # single-series case: no triplet_id/method columns to group by,
        # treat the whole frame as one series
        frame[z_col] = _rolling_z_score(
            frame[residual_col],
            window=cfg.z_window,
            min_periods=cfg.min_periods,
        )
        return frame

    pieces = []
    for _, group in frame.groupby(groups, sort=False, dropna=False):
        group = group.sort_values("date").copy()
        group[z_col] = _rolling_z_score(
            group[residual_col],
            window=cfg.z_window,
            min_periods=cfg.min_periods,
        )
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True, sort=False)


def generate_candidate_events(
    residuals: pd.DataFrame,
    config: Optional[LabelingConfig] = None,
    residual_col: str = "residual",
    z_col: str = "z_score",
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> pd.DataFrame:
    """Scans a scored residual series for every moment the |z-score|
    crosses above `entry_z` from below -- each crossing becomes one
    candidate trade event. This does not yet say whether the trade would
    have succeeded; that's `label_candidate_events`'s job. Splitting
    "find candidate setups" from "grade what happened next" keeps the
    entry-detection logic independent of the outcome-scoring logic, so
    each can be tested and reasoned about on its own.
    """
    cfg = config or LabelingConfig()
    _validate_config(cfg)
    frame = add_event_z_scores(
        residuals=residuals,
        config=cfg,
        residual_col=residual_col,
        z_col=z_col,
        group_cols=group_cols,
    )
    groups = _available_group_cols(frame, group_cols)
    metadata_cols = [col for col in ["target_symbol", "hedge_symbol_1", "hedge_symbol_2"] if col in frame.columns]

    rows = []
    if groups:
        grouped = frame.groupby(groups, sort=False, dropna=False)
    else:
        grouped = [((), frame)]

    for key, group in grouped:
        group = group.sort_values("date").reset_index(drop=True)
        previous_abs_z = group[z_col].abs().shift(1)
        current_abs_z = group[z_col].abs()
        if cfg.include_initial_crossing:
            crossed = current_abs_z.ge(cfg.entry_z) & (previous_abs_z.lt(cfg.entry_z) | previous_abs_z.isna())
        else:
            # a genuine crossing: today is beyond the threshold AND
            # yesterday was not -- this is what prevents every day of an
            # already-extended dislocation from separately re-triggering
            # as its own new candidate event
            crossed = current_abs_z.ge(cfg.entry_z) & previous_abs_z.lt(cfg.entry_z)

        for local_index, row in group.loc[crossed.fillna(False)].iterrows():
            z_value = float(row[z_col])
            # positive z means the target is trading rich relative to the
            # hedge basket -- the trade is to short that richness
            # (short_spread); negative z means cheap, so long the spread
            side = "short_spread" if z_value > 0 else "long_spread"
            base = {
                "triplet_id": row.get("triplet_id", "single_series"),
                "method": row.get("method", "unspecified"),
                "event_date": row["date"],
                "side": side,
                "entry_z_score": z_value,
                "entry_abs_z": abs(z_value),
                "entry_residual": float(row[residual_col]),
                "entry_threshold": float(cfg.entry_z),
                "exit_z": float(cfg.exit_z),
                "stop_loss_z": float(cfg.stop_loss_z),
                "max_holding_period": int(cfg.max_holding_period),
                "z_window": np.nan if cfg.z_window is None else int(cfg.z_window),
                "event_row": int(local_index),
            }
            for col in metadata_cols:
                base[col] = row[col]
            base["event_id"] = _event_id(base)
            rows.append(base)

    return pd.DataFrame(rows, columns=_candidate_columns(metadata_cols))


def label_candidate_events(
    residuals: pd.DataFrame,
    candidate_events: pd.DataFrame,
    config: Optional[LabelingConfig] = None,
    residual_col: str = "residual",
    z_col: str = "z_score",
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> pd.DataFrame:
    """For every candidate event, walks forward day by day through the
    holding window and determines the outcome: did the residual revert
    (success), hit the stop-loss (failure), or run out the max holding
    period without either (failure by timeout). This label -- not next-day
    price direction -- is the target the ML classifier is trained on.
    """
    cfg = config or LabelingConfig()
    _validate_config(cfg)
    if candidate_events is None or candidate_events.empty:
        return pd.DataFrame(columns=_label_columns())

    frame = add_event_z_scores(
        residuals=residuals,
        config=cfg,
        residual_col=residual_col,
        z_col=z_col,
        group_cols=group_cols,
    )
    groups = _available_group_cols(frame, group_cols)

    # index each triplet/method's full residual history once up front,
    # rather than re-filtering the whole frame for every single candidate
    # event -- this is what keeps labeling an 82-triplet universe fast
    lookup: dict[tuple, pd.DataFrame] = {}
    if groups:
        for key, group in frame.groupby(groups, sort=False, dropna=False):
            lookup[_tuple_key(key)] = group.sort_values("date").reset_index(drop=True)
    else:
        lookup[()] = frame.sort_values("date").reset_index(drop=True)

    rows = []
    for _, event in candidate_events.iterrows():
        key = _candidate_key(event, groups)
        group = lookup.get(key)
        if group is None or group.empty:
            rows.append(_missing_path_label(event, "missing_residual_path"))
            continue

        dates = pd.to_datetime(group["date"])
        event_date = pd.Timestamp(event["event_date"])
        positions = np.flatnonzero(dates.eq(event_date).to_numpy())
        if len(positions) == 0:
            rows.append(_missing_path_label(event, "missing_event_date"))
            continue

        position = int(positions[0])
        # the forward-looking window starts the day AFTER the entry --
        # the entry day itself is never part of its own outcome window,
        # since that would mean grading a trade using information from
        # the moment it was entered
        horizon = group.iloc[position + 1 : position + 1 + int(event["max_holding_period"])].copy()
        if horizon.empty:
            rows.append(_missing_path_label(event, "no_forward_observations"))
            continue

        rows.append(_label_single_event(event, horizon, z_col=z_col, residual_col=residual_col))

    return pd.DataFrame(rows, columns=_label_columns())


def generate_event_labels(
    residuals: pd.DataFrame,
    config: Optional[LabelingConfig] = None,
    residual_col: str = "residual",
    z_col: str = "z_score",
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> dict[str, pd.DataFrame]:
    # convenience wrapper chaining all three steps (score -> detect
    # candidates -> label outcomes) and returning all three intermediate
    # tables, since downstream code (feature engineering, diagnostics)
    # often needs more than just the final labels
    cfg = config or LabelingConfig()
    scored = add_event_z_scores(
        residuals=residuals,
        config=cfg,
        residual_col=residual_col,
        z_col=z_col,
        group_cols=group_cols,
    )
    candidates = generate_candidate_events(
        residuals=scored,
        config=cfg,
        residual_col=residual_col,
        z_col=z_col,
        group_cols=group_cols,
    )
    labels = label_candidate_events(
        residuals=scored,
        candidate_events=candidates,
        config=cfg,
        residual_col=residual_col,
        z_col=z_col,
        group_cols=group_cols,
    )
    return {
        "scored_residuals": scored,
        "candidate_events": candidates,
        "event_labels": labels,
    }


def summarize_event_labels(
    labels: pd.DataFrame,
    group_cols: Sequence[str] = ("triplet_id", "method"),
) -> pd.DataFrame:
    # Success rate here is a labeling diagnostic (how often did the
    # defined reversion actually happen), not a claim about trading
    # profitability -- src/portfolio.py's win_rate is a separate concept
    # computed from realized dollar PnL, and the two are not guaranteed
    # to agree.
    if labels is None or labels.empty:
        return pd.DataFrame(columns=[*group_cols, "n_events", "success_count", "failure_count", "success_rate"])
    groups = _available_group_cols(labels, group_cols)
    if not groups:
        total = int(labels.shape[0])
        success = int(labels["label"].sum())
        return pd.DataFrame(
            [{"n_events": total, "success_count": success, "failure_count": total - success, "success_rate": success / total}]
        )
    summary = (
        labels.groupby(groups, dropna=False)
        .agg(n_events=("label", "size"), success_count=("label", "sum"))
        .reset_index()
    )
    summary["success_count"] = summary["success_count"].astype(int)
    summary["failure_count"] = summary["n_events"] - summary["success_count"]
    summary["success_rate"] = summary["success_count"] / summary["n_events"]
    return summary


def success_rate_by_z_bucket(
    labels: pd.DataFrame,
    buckets: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    # Buckets events by how extreme their entry z-score was (e.g. 2.0-2.5,
    # 2.5-3.0, ...) and reports success rate per bucket -- checks whether
    # "more extreme dislocation" actually correlates with "more likely to
    # revert," which is the implicit assumption behind trading at all.
    columns = ["z_bucket", "n_events", "success_count", "failure_count", "success_rate"]
    if labels is None or labels.empty:
        return pd.DataFrame(columns=columns)
    clean = labels.copy()
    if buckets is None:
        # auto-derive half-point bucket edges spanning the observed range
        # of entry z-scores, so this works without the caller needing to
        # know the data's range in advance
        lower = float(np.floor(clean["entry_abs_z"].min() * 2.0) / 2.0)
        upper = float(np.ceil(clean["entry_abs_z"].max() * 2.0) / 2.0)
        if lower == upper:
            upper = lower + 0.5
        buckets = np.arange(lower, upper + 0.5001, 0.5)
        if len(buckets) < 2:
            buckets = [lower, lower + 0.5]
    clean["z_bucket"] = pd.cut(clean["entry_abs_z"], bins=list(buckets), include_lowest=True, right=False)
    summary = (
        clean.dropna(subset=["z_bucket"])
        .groupby("z_bucket", observed=True)
        .agg(n_events=("label", "size"), success_count=("label", "sum"))
        .reset_index()
    )
    summary["z_bucket"] = summary["z_bucket"].astype(str)
    summary["success_count"] = summary["success_count"].astype(int)
    summary["failure_count"] = summary["n_events"] - summary["success_count"]
    summary["success_rate"] = summary["success_count"] / summary["n_events"]
    return summary.loc[:, columns]


def label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    # Class balance check: if outcomes are heavily skewed toward one
    # class (e.g. 95% failures), a classifier can score deceptively well
    # by just always predicting the majority class -- worth knowing before
    # trusting any accuracy number the model reports later.
    columns = ["outcome", "n_events"]
    if labels is None or labels.empty:
        return pd.DataFrame(columns=columns)
    return labels.groupby("outcome", dropna=False).size().rename("n_events").reset_index()


def _label_single_event(event: pd.Series, horizon: pd.DataFrame, z_col: str, residual_col: str) -> dict:
    # direction flips the sign check depending on which way the trade is
    # positioned: a short_spread trade wants the (positive) z-score to
    # fall, a long_spread trade wants the (negative) z-score to rise --
    # multiplying by `direction` lets both cases share one comparison
    # (`signed_z <= exit_z` / `signed_z >= stop_loss_z`) instead of two
    # separate branches.
    direction = 1.0 if event["side"] == "short_spread" else -1.0
    for holding_period, (_, row) in enumerate(horizon.iterrows(), start=1):
        z_value = float(row[z_col])
        signed_z = direction * z_value
        if signed_z <= float(event["exit_z"]):
            return _label_row(event, row, label=1, outcome="success", reason="reversion", holding_period=holding_period, z_col=z_col, residual_col=residual_col)
        if signed_z >= float(event["stop_loss_z"]):
            return _label_row(event, row, label=0, outcome="failure", reason="stop_loss", holding_period=holding_period, z_col=z_col, residual_col=residual_col)
    # neither exit nor stop-loss triggered anywhere in the holding
    # window -- ran out the clock, scored as a failure at the last
    # available observation
    last = horizon.iloc[-1]
    return _label_row(event, last, label=0, outcome="failure", reason="max_holding_period", holding_period=int(len(horizon)), z_col=z_col, residual_col=residual_col)


def _label_row(
    event: pd.Series,
    exit_row: pd.Series,
    label: int,
    outcome: str,
    reason: str,
    holding_period: int,
    z_col: str,
    residual_col: str,
) -> dict:
    return {
        "event_id": event["event_id"],
        "triplet_id": event.get("triplet_id", "single_series"),
        "method": event.get("method", "unspecified"),
        "event_date": event["event_date"],
        "exit_date": exit_row["date"],
        "side": event["side"],
        "label": int(label),
        "outcome": outcome,
        "exit_reason": reason,
        "holding_period": int(holding_period),
        "entry_z_score": float(event["entry_z_score"]),
        "entry_abs_z": float(event["entry_abs_z"]),
        "exit_z_score": float(exit_row[z_col]),
        "entry_residual": float(event["entry_residual"]),
        "exit_residual": float(exit_row[residual_col]),
        "entry_threshold": float(event["entry_threshold"]),
        "exit_z": float(event["exit_z"]),
        "stop_loss_z": float(event["stop_loss_z"]),
        "max_holding_period": int(event["max_holding_period"]),
    }


def _missing_path_label(event: pd.Series, reason: str) -> dict:
    # An event whose residual history can't be found or has no forward
    # data is labeled a failure rather than dropped silently -- dropping
    # it would quietly shrink the dataset in a way that's easy to miss;
    # marking it a failure keeps it visible (with `exit_reason` recording
    # exactly why) while still being conservative about what counts as
    # a "win."
    return {
        "event_id": event["event_id"],
        "triplet_id": event.get("triplet_id", "single_series"),
        "method": event.get("method", "unspecified"),
        "event_date": event["event_date"],
        "exit_date": pd.NaT,
        "side": event["side"],
        "label": 0,
        "outcome": "failure",
        "exit_reason": reason,
        "holding_period": 0,
        "entry_z_score": float(event["entry_z_score"]),
        "entry_abs_z": float(event["entry_abs_z"]),
        "exit_z_score": np.nan,
        "entry_residual": float(event["entry_residual"]),
        "exit_residual": np.nan,
        "entry_threshold": float(event["entry_threshold"]),
        "exit_z": float(event["exit_z"]),
        "stop_loss_z": float(event["stop_loss_z"]),
        "max_holding_period": int(event["max_holding_period"]),
    }


def _prepare_residual_frame(frame: pd.DataFrame, residual_col: str) -> pd.DataFrame:
    # Normalizes the input into a consistent shape: a "date" column
    # (pulled from the index if needed), sorted, with inf/NaN rows
    # dropped -- so every function below this point can assume a clean
    # frame rather than re-checking these conditions individually.
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", residual_col])
    if residual_col not in frame.columns:
        raise KeyError(f"missing residual column: {residual_col}")
    clean = frame.copy()
    if "date" not in clean.columns:
        if isinstance(clean.index, pd.DatetimeIndex):
            clean = clean.reset_index().rename(columns={"index": "date"})
        else:
            raise KeyError("residual frame must include a date column or DatetimeIndex")
    clean["date"] = pd.to_datetime(clean["date"])
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=["date", residual_col])
    return clean.sort_values([col for col in ["triplet_id", "method", "date"] if col in clean.columns]).reset_index(drop=True)


def _rolling_z_score(series: pd.Series, window: Optional[int], min_periods: int) -> pd.Series:
    # `.shift(1)` before the rolling/expanding window is what keeps this
    # from leaking today's residual into the baseline today's own
    # z-score is measured against -- see src/residuals.py:zscore_residuals
    # for the same pattern with a longer explanation.
    values = series.astype(float)
    if window is None:
        mean = values.expanding(min_periods=min_periods).mean().shift(1)
        std = values.expanding(min_periods=min_periods).std(ddof=1).shift(1)
    else:
        mean = values.rolling(window=window, min_periods=min_periods).mean().shift(1)
        std = values.rolling(window=window, min_periods=min_periods).std(ddof=1).shift(1)
    z = (values - mean) / std.replace(0.0, np.nan)
    return z.astype(float)


def _validate_config(config: LabelingConfig) -> None:
    if config.entry_z <= 0:
        raise ValueError("entry_z must be positive")
    if config.exit_z < 0:
        raise ValueError("exit_z must be non-negative")
    if config.exit_z >= config.entry_z:
        # exit_z must be a tighter (closer to zero) threshold than
        # entry_z, or "reversion" would never actually be achievable --
        # you'd be asking the residual to move further away to "succeed"
        raise ValueError("exit_z must be below entry_z")
    if config.stop_loss_z <= config.entry_z:
        # stop_loss_z has to be beyond entry_z, or every event would
        # immediately be stopped out at entry
        raise ValueError("stop_loss_z must exceed entry_z")
    if config.max_holding_period <= 0:
        raise ValueError("max_holding_period must be positive")
    if config.z_window is not None and config.z_window <= 1:
        raise ValueError("z_window must exceed one when provided")
    if config.min_periods <= 1:
        raise ValueError("min_periods must exceed one")


def _available_group_cols(frame: pd.DataFrame, group_cols: Sequence[str]) -> list[str]:
    return [col for col in group_cols if col in frame.columns]


def _tuple_key(value: object) -> tuple:
    if isinstance(value, tuple):
        return value
    return (value,)


def _candidate_key(event: pd.Series, groups: Sequence[str]) -> tuple:
    if not groups:
        return ()
    return tuple(event[col] for col in groups)


def _event_id(row: dict) -> str:
    # A deterministic hash of the event's identifying fields, rather than
    # an auto-incrementing counter -- means the same event always gets
    # the same ID across separate pipeline runs, which matters for
    # joining labels back to features/predictions generated in a
    # different run.
    date_value = pd.Timestamp(row["event_date"]).strftime("%Y-%m-%d")
    raw = "|".join(
        [
            str(row.get("triplet_id", "single_series")),
            str(row.get("method", "unspecified")),
            date_value,
            str(row["side"]),
            f"{float(row['entry_z_score']):.8f}",
        ]
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:16]


def _candidate_columns(metadata_cols: Iterable[str]) -> list[str]:
    base = [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "side",
        "entry_z_score",
        "entry_abs_z",
        "entry_residual",
        "entry_threshold",
        "exit_z",
        "stop_loss_z",
        "max_holding_period",
        "z_window",
        "event_row",
    ]
    insert_at = 3
    for col in metadata_cols:
        if col not in base:
            base.insert(insert_at, col)
            insert_at += 1
    return base


def _label_columns() -> list[str]:
    return [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "exit_date",
        "side",
        "label",
        "outcome",
        "exit_reason",
        "holding_period",
        "entry_z_score",
        "entry_abs_z",
        "exit_z_score",
        "entry_residual",
        "exit_residual",
        "entry_threshold",
        "exit_z",
        "stop_loss_z",
        "max_holding_period",
    ]
