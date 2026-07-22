from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PortfolioSummaryConfig:
    annualization_factor: int = TRADING_DAYS_PER_YEAR


def equity_curve_from_trades(
    trades: pd.DataFrame,
    pnl_col: str = "net_pnl",
    date_col: str = "pnl_date",
    strategy_col: str = "strategy",
) -> pd.DataFrame:
    # Collapses a trade-level log into a daily equity curve per strategy:
    # sum same-day PnL, take a running cumulative total (`equity`), and
    # track the running peak so far so `drawdown` (equity minus that
    # peak) is always <= 0 and reflects the worst dip below any
    # previously-reached high, not just below the starting point.
    required = {pnl_col, date_col, strategy_col}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"missing trade columns: {sorted(missing)}")
    if trades.empty:
        return pd.DataFrame(columns=["date", "strategy", "daily_pnl", "equity", "running_peak", "drawdown"])

    frame = trades.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    daily = (
        frame.groupby([strategy_col, date_col], as_index=False)[pnl_col]
        .sum()
        .rename(columns={date_col: "date", pnl_col: "daily_pnl"})
    )
    outputs = []
    for strategy, group in daily.groupby(strategy_col, sort=False):
        ordered = group.sort_values("date").copy()
        ordered["strategy"] = strategy
        ordered["equity"] = ordered["daily_pnl"].cumsum()
        ordered["running_peak"] = ordered["equity"].cummax()
        ordered["drawdown"] = ordered["equity"] - ordered["running_peak"]
        outputs.append(ordered.loc[:, ["date", "strategy", "daily_pnl", "equity", "running_peak", "drawdown"]])
    return pd.concat(outputs, ignore_index=True, sort=False)


def _calendar_filled_daily_pnl(
    dates: pd.Series,
    pnl: np.ndarray,
    calendar_start: pd.Timestamp,
    calendar_end: pd.Timestamp,
) -> np.ndarray:
    """Reindexes a strategy's realized PnL onto the full business-day calendar
    spanning the backtest window, filling non-trading days with zero.

    A strategy that closes 20 trades a year is flat on the other ~230
    trading days. Computing Sharpe from the trade-only days and annualizing
    with sqrt(252) implicitly assumes a return is realized every trading
    day, which overstates both the mean and understates how much of the
    year was spent with no position -- for a low-frequency strategy this
    can inflate the reported Sharpe by an order of magnitude. Zero-filling
    idle days before annualizing is the correction.
    """
    idx = pd.to_datetime(pd.Series(dates).to_numpy())
    series = pd.Series(pnl, index=idx).groupby(level=0).sum()
    full_index = pd.bdate_range(calendar_start, calendar_end)
    return series.reindex(full_index, fill_value=0.0).to_numpy()


def strategy_performance_summary(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame | None = None,
    pnl_col: str = "net_pnl",
    gross_pnl_col: str = "gross_pnl",
    turnover_col: str = "turnover",
    strategy_col: str = "strategy",
    date_col: str = "pnl_date",
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    required = {pnl_col, strategy_col}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"missing trade columns: {sorted(missing)}")
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "trade_count",
                "gross_pnl",
                "net_pnl",
                "average_net_pnl",
                "win_rate",
                "turnover",
                "max_drawdown",
                "sharpe",
            ]
        )

    if equity_curve is None:
        equity_curve = equity_curve_from_trades(trades, pnl_col=pnl_col, strategy_col=strategy_col)

    # Sharpe is computed against a calendar shared across every strategy in
    # this comparison, not each strategy's own first/last trade -- otherwise
    # a strategy that happens to trade over a shorter span looks artificially
    # less (or more) volatile relative to one measured over the full window.
    has_dates = date_col in trades.columns
    if has_dates:
        all_dates = pd.to_datetime(trades[date_col])
        calendar_start, calendar_end = all_dates.min(), all_dates.max()

    rows = []
    for strategy, group in trades.groupby(strategy_col, sort=False):
        pnl = group[pnl_col].astype(float).to_numpy()
        daily = equity_curve.loc[equity_curve["strategy"].eq(strategy), "daily_pnl"].astype(float).to_numpy()
        max_drawdown = equity_curve.loc[equity_curve["strategy"].eq(strategy), "drawdown"].min()

        if has_dates:
            calendar_pnl = _calendar_filled_daily_pnl(group[date_col], pnl, calendar_start, calendar_end)
            sharpe = annualized_sharpe(calendar_pnl, annualization_factor=annualization_factor)
        else:
            # no date information available -- fall back to the sparse
            # trade-day series (previous behavior), which should be treated
            # as an upper bound, not a comparable annualized figure
            sharpe = annualized_sharpe(daily, annualization_factor=annualization_factor)

        rows.append(
            {
                "strategy": strategy,
                "trade_count": int(group.shape[0]),
                "gross_pnl": float(group[gross_pnl_col].sum()) if gross_pnl_col in group.columns else np.nan,
                "net_pnl": float(group[pnl_col].sum()),
                "average_net_pnl": float(np.mean(pnl)) if pnl.size else np.nan,
                "median_net_pnl": float(np.median(pnl)) if pnl.size else np.nan,
                "win_rate": float(np.mean(pnl > 0.0)) if pnl.size else np.nan,
                "turnover": float(group[turnover_col].sum()) if turnover_col in group.columns else np.nan,
                "max_drawdown": float(max_drawdown) if pd.notna(max_drawdown) else np.nan,
                "sharpe": sharpe,
            }
        )
    return pd.DataFrame(rows)


def annualized_sharpe(values: np.ndarray | pd.Series | list[float], annualization_factor: int = TRADING_DAYS_PER_YEAR) -> float:
    # mean daily return / std of daily returns, scaled up to a yearly
    # figure by sqrt(annualization_factor) -- the square root (not a
    # straight multiply) is because standard deviation scales with the
    # square root of time for a random-walk-like return series, while
    # the mean scales linearly; sqrt(N) is what keeps the ratio's units
    # consistent when annualizing.
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    std = np.std(arr, ddof=1)
    if std == 0.0:
        # a perfectly flat (zero-variance) return series has an
        # undefined Sharpe ratio (division by zero), not an infinite one
        return np.nan
    return float(np.sqrt(annualization_factor) * np.mean(arr) / std)


def shared_leg_groups(triplet_ids: Sequence[str]) -> pd.DataFrame:
    """Groups triplets that share at least one hedge-leg symbol.

    Aggregating PnL across triplets as if each is an independent bet
    overstates diversification when several triplets are built from the
    same anchor ETFs (e.g. two names both hedged against QQQ and XLK) --
    a drawdown in one is likely to coincide with a drawdown in the other,
    since a meaningful part of both trades' risk is the same underlying
    factor. This does not compute a correlation-adjusted Sharpe; it flags
    which triplets should be treated as one correlated cluster rather than
    two independent observations when interpreting portfolio-level results.
    """
    parsed = {}
    for tid in triplet_ids:
        parts = tid.split("_")
        if len(parts) < 3:
            continue
        target, legs = parts[0], frozenset(parts[1:])
        parsed[tid] = (target, legs)

    groups: list[set[str]] = []
    for tid, (_, legs) in parsed.items():
        matched = None
        for group in groups:
            if any(legs & parsed[other][1] for other in group):
                matched = group
                break
        if matched is not None:
            matched.add(tid)
        else:
            groups.append({tid})

    # merge any groups that ended up transitively connected
    merged: list[set[str]] = []
    for group in groups:
        overlap = [m for m in merged if any(parsed[a][1] & parsed[b][1] for a in group for b in m)]
        if overlap:
            target = overlap[0]
            target |= group
            for extra in overlap[1:]:
                target |= extra
                merged.remove(extra)
        else:
            merged.append(set(group))

    rows = []
    for i, group in enumerate(merged):
        leg_sets = [parsed[t][1] for t in group]
        shared = set.intersection(*(set(s) for s in leg_sets)) if len(group) > 1 else set()
        all_legs = set().union(*(set(s) for s in leg_sets)) if len(group) > 1 else set()
        rows.append({
            "cluster_id": i,
            "triplets": sorted(group),
            "cluster_size": len(group),
            "shared_legs": sorted(shared) if shared else sorted(all_legs),
            "independent_bet": len(group) == 1,
        })
    return pd.DataFrame(rows)


def compare_strategy_tables(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "trade_count", "net_pnl", "sharpe", "turnover", "max_drawdown"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"missing summary columns: {sorted(missing)}")
    ordered = summary.copy()
    strategy_order = {"baseline_rule_based": 0, "ml_filtered": 1, "ml_probability_sized": 2}
    ordered["strategy_order"] = ordered["strategy"].map(strategy_order).fillna(99)
    return ordered.sort_values(["strategy_order", "strategy"]).drop(columns="strategy_order").reset_index(drop=True)


_Z_SCORES_BY_CONFIDENCE = {0.80: 1.2815515655, 0.90: 1.6448536269, 0.95: 1.9599639845, 0.99: 2.5758293035}


def wilson_score_interval(successes: int, n: int, confidence: float = 0.90) -> dict:
    """Wilson score interval for a binomial proportion (e.g. trade win rate).

    A percentile bootstrap is degenerate when every observed trade won (or
    every one lost): resampling with replacement from an all-1s array can
    only ever produce 1.0, so it reports a zero-width interval regardless
    of sample size -- silently implying certainty a 20-trade sample cannot
    support. The Wilson interval doesn't have this failure mode and is the
    standard correction for small-n or extreme observed proportions.

    Supports the confidence levels in `_Z_SCORES_BY_CONFIDENCE` without
    taking a SciPy dependency, consistent with the rest of this codebase
    implementing its own statistics rather than importing them.
    """
    if n <= 0:
        return {"successes": successes, "n": n, "rate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "confidence": confidence}
    if confidence not in _Z_SCORES_BY_CONFIDENCE:
        raise ValueError(f"confidence must be one of {sorted(_Z_SCORES_BY_CONFIDENCE)}, got {confidence}")
    z = _Z_SCORES_BY_CONFIDENCE[confidence]
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return {
        "successes": successes,
        "n": n,
        "rate": float(p),
        "ci_low": float(max(0.0, center - half_width)),
        "ci_high": float(min(1.0, center + half_width)),
        "confidence": confidence,
    }


def bootstrap_trade_metric_ci(
    pnl: np.ndarray | pd.Series | list[float],
    metric: str = "win_rate",
    n_resamples: int = 2000,
    confidence: float = 0.90,
    random_state: int | None = 0,
) -> dict:
    """Bootstrap confidence interval for a per-trade metric (win rate or mean PnL).

    Every headline figure elsewhere in this pipeline is a point estimate on
    13-90 trades. Reporting it without a sense of its sampling uncertainty is
    misleading precision -- a 100% win rate on 28 trades and a 100% win rate
    on 2,800 trades are not the same claim. This resamples trades with
    replacement to give an interval, not a verdict on whether the strategy
    is "really" good; with these sample sizes the interval will usually be
    wide, and that width is itself the finding.
    """
    arr = np.asarray(pnl, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 2:
        return {"metric": metric, "point_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_trades": n}

    if metric == "win_rate":
        point = float(np.mean(arr > 0.0))
        stat_fn = lambda sample: np.mean(sample > 0.0)  # noqa: E731
    elif metric == "mean_pnl":
        point = float(np.mean(arr))
        stat_fn = lambda sample: np.mean(sample)  # noqa: E731
    else:
        raise ValueError(f"unsupported metric: {metric!r}")

    rng = np.random.default_rng(random_state)
    resampled = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        resampled[i] = stat_fn(sample)

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(resampled, [alpha, 1.0 - alpha])
    return {
        "metric": metric,
        "point_estimate": point,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
        "n_trades": n,
        "n_resamples": n_resamples,
    }


def volatility_target_position_size(
    realized_vol: np.ndarray | pd.Series | float,
    target_vol: float,
    max_leverage: float = 3.0,
    min_vol_floor: float = 1e-6,
) -> np.ndarray | float:
    """Inverse-volatility position sizing: size = target_vol / realized_vol,
    capped at `max_leverage`.

    Every strategy elsewhere in this project sizes positions either at a
    flat 1.0 or by the model's predicted probability -- neither has
    anything to do with how volatile the specific trade actually is. Two
    trades with the same probability but very different residual
    volatility carry very different risk at the same nominal size; this
    scales size down for volatile setups and up (to the leverage cap) for
    quiet ones, so the *risk* contributed per trade is comparable rather
    than the *notional*.
    """
    vol = np.asarray(realized_vol, dtype=float) if not np.isscalar(realized_vol) else float(realized_vol)
    vol_floored = np.maximum(vol, min_vol_floor) if not np.isscalar(vol) else max(vol, min_vol_floor)
    size = target_vol / vol_floored
    return np.clip(size, 0.0, max_leverage) if not np.isscalar(size) else float(np.clip(size, 0.0, max_leverage))


def apply_volatility_targeting(
    trades: pd.DataFrame,
    vol_col: str,
    target_vol: float,
    max_leverage: float = 3.0,
    spread_pnl_col: str = "spread_pnl_units",
    cost_per_unit_col: str | None = None,
    flat_cost_per_trade: float = 0.0,
) -> pd.DataFrame:
    """Re-sizes an existing trade log by inverse realized volatility rather
    than its original flat/probability-based sizing, and recomputes PnL,
    cost, and turnover consistently with the new sizes. Does not mutate
    the input; returns a new frame so the original (flat or
    probability-sized) trade log is still available for comparison.
    """
    required = {vol_col, spread_pnl_col}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    frame = trades.copy()
    frame["position_size"] = volatility_target_position_size(frame[vol_col].to_numpy(), target_vol, max_leverage)
    frame["gross_pnl"] = frame["position_size"] * frame[spread_pnl_col]
    if cost_per_unit_col is not None and cost_per_unit_col in frame.columns:
        frame["transaction_cost"] = frame[cost_per_unit_col].astype(float) * frame["position_size"].abs()
    else:
        frame["transaction_cost"] = flat_cost_per_trade * frame["position_size"].abs()
    frame["net_pnl"] = frame["gross_pnl"] - frame["transaction_cost"]
    frame["turnover"] = 2.0 * frame["position_size"].abs()
    return frame


def _moving_block_bootstrap_paths(values: np.ndarray, block_size: int, n_resamples: int, rng: np.random.Generator) -> np.ndarray:
    """Generates `n_resamples` resampled paths of the same length as
    `values`, built from overlapping contiguous blocks (Kunsch 1989 moving
    block bootstrap) rather than resampling individual observations
    independently. Path-dependent statistics like drawdown are sensitive
    to the *order* of returns, and daily returns are typically
    autocorrelated (volatility clustering); an i.i.d. resample would
    scramble that structure and distort the drawdown/path-Sharpe estimate.
    """
    n = len(values)
    block_size = max(1, min(block_size, n))
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size
    paths = np.empty((n_resamples, n), dtype=float)
    for i in range(n_resamples):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        blocks = [values[s: s + block_size] for s in starts]
        paths[i] = np.concatenate(blocks)[:n]
    return paths


def _max_drawdown_from_pnl(daily_pnl: np.ndarray) -> float:
    equity = np.cumsum(daily_pnl)
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity - running_peak
    return float(drawdown.min())


def bootstrap_path_metric_ci(
    daily_pnl: np.ndarray | pd.Series | list[float],
    metric: str = "max_drawdown",
    block_size: int = 20,
    n_resamples: int = 1000,
    confidence: float = 0.90,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
    random_state: int | None = 0,
) -> dict:
    """Moving-block-bootstrap confidence interval for a path-dependent
    equity-curve metric (max drawdown or annualized Sharpe of the path),
    as distinct from `bootstrap_trade_metric_ci`, which bootstraps
    per-trade metrics (win rate, mean PnL) and is not path-dependent.
    """
    arr = np.asarray(daily_pnl, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < max(10, block_size):
        return {"metric": metric, "point_estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_obs": n}

    if metric == "max_drawdown":
        point = _max_drawdown_from_pnl(arr)
        stat_fn = _max_drawdown_from_pnl
    elif metric == "sharpe":
        point = annualized_sharpe(arr, annualization_factor=annualization_factor)
        stat_fn = lambda path: annualized_sharpe(path, annualization_factor=annualization_factor)  # noqa: E731
    else:
        raise ValueError(f"unsupported metric: {metric!r}")

    rng = np.random.default_rng(random_state)
    paths = _moving_block_bootstrap_paths(arr, block_size, n_resamples, rng)
    resampled = np.array([stat_fn(path) for path in paths])
    resampled = resampled[np.isfinite(resampled)]
    if resampled.size == 0:
        return {"metric": metric, "point_estimate": point, "ci_low": np.nan, "ci_high": np.nan, "n_obs": n}

    alpha = (1.0 - confidence) / 2.0
    ci_low, ci_high = np.quantile(resampled, [alpha, 1.0 - alpha])
    return {
        "metric": metric, "point_estimate": point,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "confidence": confidence, "n_obs": n, "block_size": block_size, "n_resamples": n_resamples,
    }


def benchmark_buy_and_hold(
    prices: pd.DataFrame,
    symbols: list[str],
    price_col: str = "adj_close",
    symbol_col: str = "symbol",
    date_col: str = "date",
    weighting: str = "equal",
) -> pd.DataFrame:
    """Equal-weight (or, if provided, custom-weighted) buy-and-hold daily
    return series across `symbols`, for comparison against the strategy
    equity curve. Every performance number elsewhere in this project is
    relative to the strategy's own baseline variant -- none of them show
    whether the whole exercise beats just holding the underlying stocks,
    which is the first thing a reviewer will ask.
    """
    required = {price_col, symbol_col, date_col}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    frame = prices.loc[prices[symbol_col].isin(symbols)].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values([symbol_col, date_col])
    frame["daily_return"] = frame.groupby(symbol_col)[price_col].pct_change()

    wide = frame.pivot_table(index=date_col, columns=symbol_col, values="daily_return")
    if weighting == "equal":
        portfolio_return = wide.mean(axis=1, skipna=True)
    else:
        raise ValueError(f"unsupported weighting: {weighting!r}")

    equity = (1.0 + portfolio_return.fillna(0.0)).cumprod()
    return pd.DataFrame({
        "date": portfolio_return.index,
        "daily_return": portfolio_return.to_numpy(),
        "cumulative_return": equity.to_numpy() - 1.0,
    }).reset_index(drop=True)
