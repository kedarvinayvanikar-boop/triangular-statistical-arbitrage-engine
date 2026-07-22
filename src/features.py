from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    residual_window: int = 20
    volatility_window: int = 20
    stability_window: int = 20
    correlation_window: int = 20
    moving_average_window: int = 20
    min_periods: int = 5
    market_symbol: str = "QQQ"


@dataclass(frozen=True)
class ScalingParameters:
    columns: tuple[str, ...]
    means: pd.Series
    standard_deviations: pd.Series


def build_event_feature_matrix(
    candidate_events: pd.DataFrame,
    residuals: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    labels: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    log_prices: Optional[pd.DataFrame] = None,
    volumes: Optional[pd.DataFrame] = None,
    coefficients: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Builds one row of features per candidate event -- the input table
    the logistic regression (and decision tree) are actually trained on.

    Every feature merged in here is looked up "as of" the event's own
    date via left joins keyed on (triplet_id, method, event_date), never
    a later date -- the temporal-boundary logic that keeps this lookahead
    -free lives in `_trailing_window`/`_current_value` below, both of
    which filter to `index <= event_date` before computing anything.
    `returns`, `log_prices`, `volumes`, and `coefficients` are all
    optional: if a given input isn't supplied, its features are filled
    with NaN rather than the whole function failing, so features can be
    added incrementally as more inputs become available.
    """
    cfg = config or FeatureConfig()
    _validate_config(cfg)
    if candidate_events is None or candidate_events.empty:
        return pd.DataFrame(columns=_base_feature_columns(include_label=labels is not None))
    if residuals is None or residuals.empty:
        raise ValueError("residuals are required to build event features")

    events = _prepare_events(candidate_events)
    residual_features = _residual_feature_panel(residuals, cfg)
    feature_matrix = events.merge(
        residual_features,
        left_on=["triplet_id", "method", "event_date"],
        right_on=["triplet_id", "method", "date"],
        how="left",
    )
    feature_matrix = feature_matrix.drop(columns=["date"], errors="ignore")

    if coefficients is not None and not coefficients.empty:
        coefficient_features = _coefficient_feature_panel(coefficients, cfg)
        feature_matrix = feature_matrix.merge(
            coefficient_features,
            left_on=["triplet_id", "method", "event_date"],
            right_on=["triplet_id", "method", "date"],
            how="left",
        ).drop(columns=["date"], errors="ignore")

    returns_wide = _wide_market_frame(returns, value_col="return") if returns is not None and not returns.empty else None
    prices_wide = _wide_market_frame(log_prices, value_col="log_price") if log_prices is not None and not log_prices.empty else None
    volume_wide = _wide_market_frame(volumes, value_col="volume") if volumes is not None and not volumes.empty else None

    market_features = _event_market_features(events, returns_wide, prices_wide, volume_wide, cfg)
    feature_matrix = feature_matrix.merge(market_features, on="event_id", how="left")

    if labels is not None and not labels.empty:
        # labels are attached last and only if provided -- at prediction
        # time (scoring new, still-open events) there is no label yet,
        # and this merge simply leaves that column absent rather than
        # erroring
        label_cols = ["event_id", "label", "outcome", "exit_reason", "holding_period"]
        available = [col for col in label_cols if col in labels.columns]
        feature_matrix = feature_matrix.merge(labels.loc[:, available], on="event_id", how="left")

    return _ordered_feature_matrix(feature_matrix)


def feature_summary_statistics(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    # Basic descriptive stats per feature -- the first thing to check
    # before training: does any feature have an implausible range, a
    # suspiciously narrow spread, or an obviously wrong scale?
    numeric = _numeric_feature_columns(feature_matrix)
    if not numeric:
        return pd.DataFrame(columns=["feature", "count", "mean", "std", "min", "p25", "median", "p75", "max"])
    summary = feature_matrix[numeric].describe(percentiles=[0.25, 0.5, 0.75]).T
    summary = summary.rename(columns={"25%": "p25", "50%": "median", "75%": "p75"})
    summary.index.name = "feature"
    return summary.reset_index().loc[:, ["feature", "count", "mean", "std", "min", "p25", "median", "p75", "max"]]


def feature_missingness_report(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    # How often each feature is missing -- a feature that's NaN for most
    # events (e.g. because volume data wasn't supplied) contributes very
    # little after being median-filled and is worth knowing about before
    # trusting its coefficient in the trained model.
    if feature_matrix is None or feature_matrix.empty:
        return pd.DataFrame(columns=["column", "missing_count", "missing_rate"])
    missing_count = feature_matrix.isna().sum()
    report = pd.DataFrame(
        {
            "column": missing_count.index,
            "missing_count": missing_count.to_numpy(dtype=int),
            "missing_rate": missing_count.to_numpy(dtype=float) / float(feature_matrix.shape[0]),
        }
    )
    return report.sort_values(["missing_rate", "column"], ascending=[False, True]).reset_index(drop=True)


def feature_correlation_matrix(feature_matrix: pd.DataFrame, min_non_null: int = 3) -> pd.DataFrame:
    numeric = _numeric_feature_columns(feature_matrix)
    if not numeric:
        return pd.DataFrame()
    usable = []
    for col in numeric:
        # a feature with too few non-null values or that's constant
        # across every event can't produce a meaningful correlation
        # (division by a near-zero standard deviation), so it's excluded
        # from the matrix rather than showing up as a NaN row/column
        if feature_matrix[col].notna().sum() >= min_non_null and feature_matrix[col].nunique(dropna=True) > 1:
            usable.append(col)
    if not usable:
        return pd.DataFrame()
    return feature_matrix[usable].corr()


def collinear_feature_pairs(correlation_matrix: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Flags feature pairs whose absolute correlation exceeds `threshold`.

    Two near-duplicate inputs (e.g. a stability metric and a near-identical
    variant of it) don't add information to a logistic model -- they inflate
    the standard errors of both coefficients and make sign/magnitude hard to
    trust. This does not decide what to drop; it is a diagnostic to run
    before committing a feature set, not an automatic feature-selection step.
    """
    if correlation_matrix.empty:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    cols = correlation_matrix.columns.tolist()
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            corr = correlation_matrix.loc[a, b]
            if pd.notna(corr) and abs(corr) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "correlation": float(corr), "abs_correlation": float(abs(corr))})
    result = pd.DataFrame(rows, columns=["feature_a", "feature_b", "correlation", "abs_correlation"])
    return result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def fit_feature_scaler(feature_matrix: pd.DataFrame, columns: Optional[Sequence[str]] = None) -> ScalingParameters:
    # A general-purpose standardizer, independent of the one baked into
    # fit_logistic_regression -- useful when a feature set needs to be
    # standardized once and then reused consistently across several
    # different models/analyses, not just inside one training call.
    selected = tuple(columns) if columns is not None else tuple(_numeric_feature_columns(feature_matrix))
    if not selected:
        raise ValueError("at least one numeric feature is required")
    means = feature_matrix.loc[:, selected].mean()
    standard_deviations = feature_matrix.loc[:, selected].std(ddof=0).replace(0.0, 1.0)
    return ScalingParameters(columns=selected, means=means, standard_deviations=standard_deviations)


def transform_feature_matrix(feature_matrix: pd.DataFrame, parameters: ScalingParameters) -> pd.DataFrame:
    transformed = feature_matrix.copy()
    for col in parameters.columns:
        if col not in transformed.columns:
            raise KeyError(f"missing feature column: {col}")
        transformed[col] = (transformed[col] - parameters.means[col]) / parameters.standard_deviations[col]
    return transformed


def create_feature_outputs(
    candidate_events: pd.DataFrame,
    residuals: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    labels: Optional[pd.DataFrame] = None,
    returns: Optional[pd.DataFrame] = None,
    log_prices: Optional[pd.DataFrame] = None,
    volumes: Optional[pd.DataFrame] = None,
    coefficients: Optional[pd.DataFrame] = None,
) -> dict[str, pd.DataFrame]:
    # Convenience wrapper bundling the feature matrix with its own
    # diagnostics (summary stats, missingness, correlation) in one call,
    # since these three checks are almost always wanted together right
    # after building a new feature set.
    matrix = build_event_feature_matrix(
        candidate_events=candidate_events,
        residuals=residuals,
        config=config,
        labels=labels,
        returns=returns,
        log_prices=log_prices,
        volumes=volumes,
        coefficients=coefficients,
    )
    return {
        "feature_matrix": matrix,
        "feature_summary_statistics": feature_summary_statistics(matrix),
        "feature_missingness_report": feature_missingness_report(matrix),
        "feature_correlation_matrix": feature_correlation_matrix(matrix),
    }


def _residual_feature_panel(residuals: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    # Builds a daily (not just event-day) table of residual-derived
    # features per triplet/method, which then gets left-joined onto each
    # event's own date in build_event_feature_matrix. Computed once per
    # triplet/method group (not per event) since many events can share
    # the same underlying daily residual history.
    frame = _prepare_residuals(residuals)
    rows = []
    for _, group in frame.groupby(["triplet_id", "method"], sort=False, dropna=False):
        group = group.sort_values("date").copy()
        group["residual_change"] = group["residual"].diff()
        group["residual_volatility"] = group["residual_change"].rolling(
            cfg.volatility_window,
            min_periods=cfg.min_periods,
        ).std()
        group["residual_autocorrelation"] = group["residual"].rolling(
            cfg.residual_window,
            min_periods=cfg.min_periods,
        ).apply(_lag_one_autocorrelation, raw=True)
        group["half_life_estimate"] = group["residual"].rolling(
            cfg.residual_window,
            min_periods=cfg.min_periods,
        ).apply(_half_life, raw=True)

        if {"actual_log_price", "fitted_log_price"}.issubset(group.columns):
            # rolling R^2: how well the hedge-ratio fit has been tracking
            # the target's actual price over just the last
            # `residual_window` days -- a locally-computed version of the
            # R^2 already reported once for the whole fit, useful for
            # spotting a relationship that's degrading over time even if
            # the full-sample R^2 still looks fine
            error = group["actual_log_price"] - group["fitted_log_price"]
            sse = (error**2).rolling(cfg.residual_window, min_periods=cfg.min_periods).sum()
            mean_y = group["actual_log_price"].rolling(cfg.residual_window, min_periods=cfg.min_periods).mean()
            sst = ((group["actual_log_price"] - mean_y) ** 2).rolling(cfg.residual_window, min_periods=cfg.min_periods).sum()
            group["rolling_r_squared"] = 1.0 - sse / sst.replace(0.0, np.nan)
        else:
            group["rolling_r_squared"] = np.nan

        rows.append(
            group.loc[
                :,
                [
                    "date",
                    "triplet_id",
                    "method",
                    "z_score",
                    "residual",
                    "residual_change",
                    "residual_volatility",
                    "residual_autocorrelation",
                    "half_life_estimate",
                    "rolling_r_squared",
                ],
            ]
        )
    return pd.concat(rows, ignore_index=True, sort=False)


def _coefficient_feature_panel(coefficients: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    # "Beta stability" -- how much the hedge ratios have been wobbling
    # recently -- is itself a predictive feature: a triplet whose
    # coefficients are jumping around a lot suggests the underlying
    # relationship is less trustworthy right now, independent of what the
    # residual/z-score is doing.
    frame = coefficients.copy()
    if "date" not in frame.columns:
        raise KeyError("coefficients must contain a date column")
    for col in ["triplet_id", "method", "beta_1", "beta_2"]:
        if col not in frame.columns:
            raise KeyError(f"coefficients must contain {col}")
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for _, group in frame.groupby(["triplet_id", "method"], sort=False, dropna=False):
        group = group.sort_values("date").copy()
        group["beta_1_stability"] = group["beta_1"].rolling(cfg.stability_window, min_periods=cfg.min_periods).std()
        group["beta_2_stability"] = group["beta_2"].rolling(cfg.stability_window, min_periods=cfg.min_periods).std()
        # combines both legs' wobbliness into one number (Euclidean norm)
        # rather than reporting them as two separate, harder-to-interpret
        # features
        group["beta_stability"] = np.sqrt(group["beta_1_stability"] ** 2 + group["beta_2_stability"] ** 2)
        rows.append(group.loc[:, ["date", "triplet_id", "method", "beta_1", "beta_2", "beta_1_stability", "beta_2_stability", "beta_stability"]])
    return pd.concat(rows, ignore_index=True, sort=False)


def _event_market_features(
    events: pd.DataFrame,
    returns: Optional[pd.DataFrame],
    log_prices: Optional[pd.DataFrame],
    volumes: Optional[pd.DataFrame],
    cfg: FeatureConfig,
) -> pd.DataFrame:
    # Per-event market context features -- how volatile each leg has
    # been, how correlated the legs are with each other right now, how
    # the broader market/sector moved that day, whether volume spiked.
    # Looped per-event (rather than vectorized) because each event
    # potentially references a different target/hedge symbol pair, so
    # the trailing window has to be sliced individually per event anyway.
    rows = []
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["event_date"])
        target = event.get("target_symbol")
        hedge_1 = event.get("hedge_symbol_1")
        hedge_2 = event.get("hedge_symbol_2")
        row = {"event_id": event["event_id"]}

        if returns is not None:
            ret_window = _trailing_window(returns, event_date, cfg.volatility_window)
            corr_window = _trailing_window(returns, event_date, cfg.correlation_window)
            row["target_return_volatility"] = _column_std(ret_window, target)
            row["anchor_1_return_volatility"] = _column_std(ret_window, hedge_1)
            row["anchor_2_return_volatility"] = _column_std(ret_window, hedge_2)
            row["market_return"] = _current_value(returns, event_date, cfg.market_symbol)
            row["sector_return"] = _current_value(returns, event_date, hedge_1)
            row["target_anchor_1_correlation"] = _pair_corr(corr_window, target, hedge_1)
            row["target_anchor_2_correlation"] = _pair_corr(corr_window, target, hedge_2)
            row["anchor_correlation"] = _pair_corr(corr_window, hedge_1, hedge_2)
            row["correlation_stability"] = _correlation_stability(returns, event_date, target, hedge_1, hedge_2, cfg)
        else:
            for col in [
                "target_return_volatility",
                "anchor_1_return_volatility",
                "anchor_2_return_volatility",
                "market_return",
                "sector_return",
                "target_anchor_1_correlation",
                "target_anchor_2_correlation",
                "anchor_correlation",
                "correlation_stability",
            ]:
                row[col] = np.nan

        if log_prices is not None:
            price_window = _trailing_window(log_prices, event_date, cfg.moving_average_window)
            row["recent_drawdown"] = _recent_drawdown(price_window, target)
            row["distance_from_moving_average"] = _distance_from_moving_average(price_window, target)
        else:
            row["recent_drawdown"] = np.nan
            row["distance_from_moving_average"] = np.nan

        if volumes is not None:
            volume_window = _trailing_window(volumes, event_date, cfg.moving_average_window)
            row["volume_shock"] = _volume_shock(volume_window, target)
        else:
            row["volume_shock"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    required = ["event_id", "triplet_id", "method", "event_date", "side", "entry_z_score", "entry_abs_z"]
    missing = [col for col in required if col not in events.columns]
    if missing:
        raise KeyError(f"candidate_events missing columns: {missing}")
    frame = events.copy()
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def _prepare_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    required = ["date", "triplet_id", "method", "residual"]
    missing = [col for col in required if col not in residuals.columns]
    if missing:
        raise KeyError(f"residuals missing columns: {missing}")
    frame = residuals.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["residual"] = frame["residual"].astype(float)
    if "z_score" not in frame.columns:
        # fallback: compute an expanding-window z-score here if the
        # residual table didn't already come with one (e.g. from
        # src/labeling.py's add_event_z_scores). Uses an expanding rather
        # than fixed-size rolling window since no explicit window length
        # is available in this context.
        pieces = []
        for _, group in frame.groupby(["triplet_id", "method"], sort=False, dropna=False):
            group = group.sort_values("date").copy()
            mean = group["residual"].expanding(min_periods=2).mean()
            std = group["residual"].expanding(min_periods=2).std().replace(0.0, np.nan)
            group["z_score"] = (group["residual"] - mean) / std
            pieces.append(group)
        frame = pd.concat(pieces, ignore_index=True, sort=False)
    return frame


def _wide_market_frame(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # Normalizes market data (returns, log prices, volumes) from a long
    # format (one row per symbol per date) into a wide format (one column
    # per symbol, indexed by date) -- the shape _trailing_window and
    # _current_value expect, since they slice by date across many symbols
    # at once.
    data = frame.copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        return data.sort_index()
    if {"date", "symbol", value_col}.issubset(data.columns):
        wide = data.pivot_table(index="date", columns="symbol", values=value_col, aggfunc="last")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
        return data.set_index("date").sort_index()
    raise KeyError("market frame must have a DatetimeIndex or a date column")


def _trailing_window(frame: pd.DataFrame, event_date: pd.Timestamp, window: int) -> pd.DataFrame:
    # The core anti-lookahead guard for every market feature in this
    # module: filters to dates on or before the event first, THEN takes
    # the trailing `window` rows -- so no feature can ever see data from
    # after the event it's describing.
    eligible = frame.loc[frame.index <= event_date]
    return eligible.tail(window)


def _current_value(frame: pd.DataFrame, event_date: pd.Timestamp, column: object) -> float:
    # Same `<= event_date` boundary as _trailing_window, for a single
    # point-in-time value (e.g. "the market's return on/before this
    # event's date") rather than a whole window.
    if column is None or column not in frame.columns:
        return np.nan
    eligible = frame.loc[frame.index <= event_date, column].dropna()
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1])


def _column_std(frame: pd.DataFrame, column: object) -> float:
    if column is None or column not in frame.columns:
        return np.nan
    values = frame[column].dropna()
    if values.shape[0] < 2:
        return np.nan
    return float(values.std())


def _pair_corr(frame: pd.DataFrame, left: object, right: object) -> float:
    if left is None or right is None or left not in frame.columns or right not in frame.columns:
        return np.nan
    clean = frame.loc[:, [left, right]].dropna()
    if clean.shape[0] < 3 or clean[left].nunique() <= 1 or clean[right].nunique() <= 1:
        return np.nan
    return float(clean[left].corr(clean[right]))


def _correlation_stability(
    returns: pd.DataFrame,
    event_date: pd.Timestamp,
    target: object,
    hedge_1: object,
    hedge_2: object,
    cfg: FeatureConfig,
) -> float:
    # How much the target-to-hedge-leg correlation has itself been
    # wobbling recently: computes a rolling correlation series over the
    # trailing window, then takes the standard deviation of THAT series
    # -- a "correlation of correlations" second-order feature meant to
    # capture regime instability that a single point-in-time correlation
    # number wouldn't show.
    if target is None or hedge_1 is None or hedge_2 is None:
        return np.nan
    required = [target, hedge_1, hedge_2]
    if any(col not in returns.columns for col in required):
        return np.nan
    window = _trailing_window(returns.loc[:, required], event_date, cfg.correlation_window + cfg.stability_window)
    if window.shape[0] < cfg.min_periods + 2:
        return np.nan
    corr_1 = window[target].rolling(cfg.correlation_window, min_periods=cfg.min_periods).corr(window[hedge_1])
    corr_2 = window[target].rolling(cfg.correlation_window, min_periods=cfg.min_periods).corr(window[hedge_2])
    combined = pd.concat([corr_1, corr_2], axis=1).mean(axis=1).dropna()
    if combined.shape[0] < 2:
        return np.nan
    return float(combined.tail(cfg.stability_window).std())


def _recent_drawdown(price_window: pd.DataFrame, column: object) -> float:
    # Current price vs. the highest price seen in the trailing window,
    # expressed as a return (e.g. -0.05 = 5% below the recent high) --
    # since prices here are stored in log space, exp(current - high) - 1
    # converts that log-difference back into an ordinary percentage.
    if column is None or column not in price_window.columns:
        return np.nan
    values = price_window[column].dropna()
    if values.empty:
        return np.nan
    current = float(values.iloc[-1])
    high = float(values.max())
    if high == 0.0:
        return np.nan
    return float(np.exp(current - high) - 1.0)


def _distance_from_moving_average(price_window: pd.DataFrame, column: object) -> float:
    if column is None or column not in price_window.columns:
        return np.nan
    values = price_window[column].dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1] - values.mean())


def _volume_shock(volume_window: pd.DataFrame, column: object) -> float:
    # Today's volume relative to the recent typical (median) volume --
    # median rather than mean specifically so one huge outlier day
    # earlier in the window doesn't itself distort what counts as
    # "normal" volume to compare today against.
    if column is None or column not in volume_window.columns:
        return np.nan
    values = volume_window[column].dropna()
    if values.shape[0] < 2:
        return np.nan
    current = float(values.iloc[-1])
    baseline = float(values.iloc[:-1].median())
    if baseline <= 0.0:
        return np.nan
    return current / baseline - 1.0


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    clean = pd.Series(values).dropna().to_numpy(dtype=float)
    if clean.shape[0] < 3:
        return np.nan
    left = clean[:-1]
    right = clean[1:]
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _half_life(values: np.ndarray) -> float:
    """Estimates the Ornstein-Uhlenbeck half-life of mean reversion: if
    the residual is truly mean-reverting, roughly how many days does it
    take to close half the gap back toward its average?

    Works by regressing the day-to-day change (delta) on the prior day's
    level (lagged): a residual sitting far from its mean should, on
    average, move back toward it -- so the regression slope should be
    negative, and the more negative it is, the faster the reversion.
    `slope` here is that regression coefficient, computed directly from
    the covariance/variance ratio (the one-variable OLS formula) rather
    than calling the general fit_ols machinery, since this only ever
    needs a single coefficient. The half-life itself, -ln(2)/slope,
    comes from solving the OU process's expected-decay equation for the
    time at which half the initial deviation has decayed away.

    A non-negative slope means the "mean reversion" assumption doesn't
    actually hold on this window (the residual isn't decaying back
    toward its average) -- returned as NaN rather than a nonsensical
    negative or infinite half-life.
    """
    clean = pd.Series(values).dropna().to_numpy(dtype=float)
    if clean.shape[0] < 4:
        return np.nan
    lagged = clean[:-1]
    delta = np.diff(clean)
    variance = float(np.var(lagged))
    if variance == 0.0:
        return np.nan
    slope = float(np.cov(lagged, delta, ddof=0)[0, 1] / variance)
    if slope >= 0.0:
        return np.nan
    return float(-np.log(2.0) / slope)


def _numeric_feature_columns(feature_matrix: pd.DataFrame) -> list[str]:
    if feature_matrix is None or feature_matrix.empty:
        return []
    excluded = {"label", "holding_period"}
    metadata = set(_metadata_columns()) | excluded
    return [col for col in feature_matrix.select_dtypes(include=[np.number]).columns if col not in metadata]


def _ordered_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    # Purely cosmetic: puts identifying columns first, the label/outcome
    # columns last, and everything else (the actual features) in the
    # middle -- makes the table easier to read when inspected directly,
    # doesn't affect anything downstream that selects columns by name.
    leading = [col for col in _metadata_columns() if col in frame.columns]
    labels = [col for col in ["label", "outcome", "exit_reason", "holding_period"] if col in frame.columns]
    remaining = [col for col in frame.columns if col not in leading and col not in labels]
    return frame.loc[:, [*leading, *remaining, *labels]]


def _metadata_columns() -> list[str]:
    return [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "side",
        "entry_z_score",
        "entry_abs_z",
    ]


def _base_feature_columns(include_label: bool) -> list[str]:
    columns = [
        *_metadata_columns(),
        "z_score",
        "residual",
        "residual_change",
        "residual_volatility",
        "residual_autocorrelation",
        "half_life_estimate",
        "rolling_r_squared",
        "beta_1",
        "beta_2",
        "beta_stability",
        "target_return_volatility",
        "anchor_1_return_volatility",
        "anchor_2_return_volatility",
        "market_return",
        "sector_return",
        "target_anchor_1_correlation",
        "target_anchor_2_correlation",
        "anchor_correlation",
        "correlation_stability",
        "recent_drawdown",
        "distance_from_moving_average",
        "volume_shock",
    ]
    if include_label:
        columns.extend(["label", "outcome", "exit_reason", "holding_period"])
    return columns


def _validate_config(cfg: FeatureConfig) -> None:
    for name in ["residual_window", "volatility_window", "stability_window", "correlation_window", "moving_average_window", "min_periods"]:
        value = int(getattr(cfg, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    largest_window = max(cfg.residual_window, cfg.volatility_window, cfg.stability_window, cfg.correlation_window, cfg.moving_average_window)
    if cfg.min_periods > largest_window:
        raise ValueError("min_periods cannot exceed all feature windows")
