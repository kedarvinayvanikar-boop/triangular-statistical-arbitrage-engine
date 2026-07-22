from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.portfolio import equity_curve_from_trades, strategy_performance_summary


@dataclass(frozen=True)
class MLBacktestConfig:
    probability_threshold: float = 0.6
    transaction_cost: float = 0.05
    probability_sizing_floor: float = 0.0
    probability_sizing_cap: float = 1.0


def prepare_ml_backtest_events(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    probability_col: str = "predicted_reversion_probability",
) -> pd.DataFrame:
    # Joins the labeled events (src/labeling.py's output: what actually
    # happened to each trade) with the model's predicted probabilities
    # for those same events, on event_id. `validate="one_to_one"` enforces
    # that neither table has duplicate event_ids -- a silent many-to-one
    # join here would duplicate trades in the backtest without any
    # visible error.
    label_required = {
        "event_id",
        "triplet_id",
        "event_date",
        "exit_date",
        "side",
        "label",
        "entry_z_score",
        "exit_z_score",
    }
    prediction_required = {"event_id", probability_col}
    label_missing = label_required.difference(labels.columns)
    pred_missing = prediction_required.difference(predictions.columns)
    if label_missing:
        raise KeyError(f"missing label columns: {sorted(label_missing)}")
    if pred_missing:
        raise KeyError(f"missing prediction columns: {sorted(pred_missing)}")

    pred_cols = ["event_id", probability_col]
    for optional in ["split", "predicted_label", "classification_threshold"]:
        if optional in predictions.columns:
            pred_cols.append(optional)
    frame = labels.merge(predictions.loc[:, pred_cols], on="event_id", how="inner", validate="one_to_one")
    if frame.empty:
        raise ValueError("no overlapping event_id values between labels and predictions")
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    # pnl_date is when a trade's PnL should be counted for calendar/Sharpe
    # purposes -- normally the exit date, but falls back to the entry
    # date for an event with no recorded exit (see src/labeling.py's
    # _missing_path_label)
    frame["pnl_date"] = frame["exit_date"].fillna(frame["event_date"])
    frame[probability_col] = frame[probability_col].astype(float)
    if frame[probability_col].isna().any() or ((frame[probability_col] < 0.0) | (frame[probability_col] > 1.0)).any():
        raise ValueError("predicted probabilities must be finite values between 0 and 1")
    return frame.sort_values(["event_date", "triplet_id", "event_id"]).reset_index(drop=True)


def event_spread_pnl(events: pd.DataFrame) -> pd.Series:
    """Realized PnL of one unit of the triangular spread trade, in
    z-score units (converted to dollars later by position sizing and the
    cost model). For a short_spread trade (betting the residual falls
    back down), profit is entry_z minus exit_z -- profitable if the
    residual actually dropped. For a long_spread trade (betting it rises
    back up), it's the mirror image, exit_z minus entry_z.

    Important limitation, documented directly here because it drives a
    real finding in CHANGELOG.md: this uses each event's originally
    RECORDED exit_z_score -- the price point where the residual actually
    ended up, fixed at label-generation time. It does not get recomputed
    if a hypothetical alternate exit_threshold/stop_loss/holding_period is
    later tried (see `relabel_events_for_threshold_scenario` below, which
    changes the win/loss `label` for such a scenario but never touches
    `exit_z_score`). Practically: this project's realized PnL is only
    sensitive to which events pass the entry filter, not to exit rules
    tried after the fact.
    """
    required = {"side", "entry_z_score", "exit_z_score"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"missing event columns: {sorted(missing)}")
    side = events["side"].astype(str)
    entry = events["entry_z_score"].astype(float)
    exit_ = events["exit_z_score"].astype(float)
    pnl = np.where(side.eq("short_spread"), entry - exit_, exit_ - entry)
    if not side.isin(["short_spread", "long_spread"]).all():
        raise ValueError("side must contain short_spread or long_spread")
    return pd.Series(pnl, index=events.index, name="spread_pnl_units")


def build_strategy_trades(
    events: pd.DataFrame,
    strategy: str,
    probability_threshold: float = 0.6,
    transaction_cost: float = 0.05,
    probability_col: str = "predicted_reversion_probability",
) -> pd.DataFrame:
    """Turns a universe of candidate events into an actual trade log for
    one of three strategies:
      - baseline_rule_based: take every candidate event, full size --
        the "what if we ignored the ML model entirely" comparison point.
      - ml_filtered: only take events the model predicts above
        `probability_threshold`, still at full size.
      - ml_probability_sized: same filter, but position size scales with
        the model's confidence (a 0.9-probability event gets a bigger bet
        than a 0.61-probability one).
    """
    threshold = _validate_probability_threshold(probability_threshold)
    cost = _validate_transaction_cost(transaction_cost)
    if probability_col not in events.columns:
        raise KeyError(f"missing probability column: {probability_col}")

    frame = events.copy()
    if strategy == "baseline_rule_based":
        selected = frame.copy()
        selected["position_size"] = 1.0
    elif strategy == "ml_filtered":
        selected = frame.loc[frame[probability_col].gt(threshold)].copy()
        selected["position_size"] = 1.0
    elif strategy == "ml_probability_sized":
        selected = frame.loc[frame[probability_col].gt(threshold)].copy()
        selected["position_size"] = selected[probability_col].clip(lower=0.0, upper=1.0)
    else:
        raise ValueError("strategy must be baseline_rule_based, ml_filtered, or ml_probability_sized")

    if selected.empty:
        return _empty_trade_frame()

    selected["strategy"] = strategy
    selected["spread_pnl_units"] = event_spread_pnl(selected)
    selected["gross_pnl"] = selected["position_size"] * selected["spread_pnl_units"]
    # a flat, position-scaled transaction cost here -- see
    # apply_transaction_cost_assumption below for the fuller cost model
    # (commission + spread + slippage + holding-period-scaled borrow cost)
    # used in the cost-sensitivity analysis
    selected["transaction_cost"] = cost * selected["position_size"].abs()
    selected["net_pnl"] = selected["gross_pnl"] - selected["transaction_cost"]
    selected["turnover"] = 2.0 * selected["position_size"].abs()  # 2x: one leg to enter, one to exit
    selected["accepted_by_strategy"] = True
    return selected.loc[:, _trade_columns(selected)]


def run_ml_backtest_comparison(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    probability_threshold: float = 0.6,
    transaction_cost: float = 0.05,
    probability_col: str = "predicted_reversion_probability",
) -> dict[str, pd.DataFrame]:
    # Runs all three strategies against the same event universe and
    # combines their trade logs/summaries into one comparison -- the main
    # entry point most callers actually use (including
    # scripts/run_universe_pipeline.py).
    events = prepare_ml_backtest_events(labels, predictions, probability_col=probability_col)
    trades = []
    for strategy in ["baseline_rule_based", "ml_filtered", "ml_probability_sized"]:
        trades.append(
            build_strategy_trades(
                events,
                strategy=strategy,
                probability_threshold=probability_threshold,
                transaction_cost=transaction_cost,
                probability_col=probability_col,
            )
        )
    trade_log = pd.concat(trades, ignore_index=True, sort=False)
    equity = equity_curve_from_trades(trade_log) if not trade_log.empty else pd.DataFrame()
    summary = strategy_performance_summary(trade_log, equity) if not trade_log.empty else pd.DataFrame()
    avoided = avoided_bad_trades_analysis(
        events,
        probability_threshold=probability_threshold,
        transaction_cost=transaction_cost,
        probability_col=probability_col,
    )
    return {
        "backtest_event_universe": events,
        "strategy_trade_log": trade_log,
        "strategy_equity_curve": equity,
        "strategy_summary": summary,
        "avoided_bad_trades": avoided,
    }


def avoided_bad_trades_analysis(
    events: pd.DataFrame,
    probability_threshold: float = 0.6,
    transaction_cost: float = 0.05,
    probability_col: str = "predicted_reversion_probability",
) -> pd.DataFrame:
    # Looks specifically at the events the ML filter REJECTED (probability
    # <= threshold) and asks: what would have happened if baseline had
    # taken them anyway? This is the direct evidence for whether the
    # filter is actually avoiding bad trades, rather than just reducing
    # trade count for its own sake.
    threshold = _validate_probability_threshold(probability_threshold)
    cost = _validate_transaction_cost(transaction_cost)
    if probability_col not in events.columns:
        raise KeyError(f"missing probability column: {probability_col}")
    frame = events.loc[events[probability_col].le(threshold)].copy()
    if frame.empty:
        return pd.DataFrame(
            [
                {
                    "probability_threshold": threshold,
                    "avoided_trade_count": 0,
                    "avoided_failure_count": 0,
                    "avoided_success_count": 0,
                    "avoided_failure_rate": np.nan,
                    "avoided_baseline_net_pnl": 0.0,
                    "avoided_average_probability": np.nan,
                }
            ]
        )
    frame["spread_pnl_units"] = event_spread_pnl(frame)
    frame["baseline_net_pnl"] = frame["spread_pnl_units"] - cost
    failures = int(np.sum(frame["label"].astype(int).eq(0))) if "label" in frame.columns else np.nan
    successes = int(np.sum(frame["label"].astype(int).eq(1))) if "label" in frame.columns else np.nan
    return pd.DataFrame(
        [
            {
                "probability_threshold": threshold,
                "avoided_trade_count": int(frame.shape[0]),
                "avoided_failure_count": failures,
                "avoided_success_count": successes,
                "avoided_failure_rate": float(failures / frame.shape[0]) if frame.shape[0] else np.nan,
                "avoided_baseline_net_pnl": float(frame["baseline_net_pnl"].sum()),
                "avoided_average_probability": float(frame[probability_col].mean()),
            }
        ]
    )


def trade_count_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "trade_count"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"missing summary columns: {sorted(missing)}")
    frame = summary.loc[:, ["strategy", "trade_count"]].copy()
    baseline = frame.loc[frame["strategy"].eq("baseline_rule_based"), "trade_count"]
    baseline_count = int(baseline.iloc[0]) if not baseline.empty else np.nan
    frame["baseline_trade_count"] = baseline_count
    frame["trades_avoided_vs_baseline"] = baseline_count - frame["trade_count"]
    frame["trade_count_ratio_to_baseline"] = frame["trade_count"] / baseline_count if baseline_count else np.nan
    return frame


def pnl_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"strategy", "net_pnl", "gross_pnl", "sharpe", "max_drawdown", "turnover"}
    missing = required.difference(summary.columns)
    if missing:
        raise KeyError(f"missing summary columns: {sorted(missing)}")
    columns = ["strategy", "gross_pnl", "net_pnl", "sharpe", "max_drawdown", "turnover", "win_rate"]
    return summary.loc[:, [col for col in columns if col in summary.columns]].copy()


def _trade_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "exit_date",
        "pnl_date",
        "split",
        "strategy",
        "side",
        "label",
        "outcome",
        "exit_reason",
        "predicted_reversion_probability",
        "entry_z_score",
        "exit_z_score",
        "spread_pnl_units",
        "position_size",
        "gross_pnl",
        "transaction_cost",
        "net_pnl",
        "turnover",
        "holding_period",
    ]
    return [col for col in preferred if col in frame.columns]


def _empty_trade_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "triplet_id",
            "method",
            "event_date",
            "exit_date",
            "pnl_date",
            "split",
            "strategy",
            "side",
            "label",
            "outcome",
            "exit_reason",
            "predicted_reversion_probability",
            "entry_z_score",
            "exit_z_score",
            "spread_pnl_units",
            "position_size",
            "gross_pnl",
            "transaction_cost",
            "net_pnl",
            "turnover",
            "holding_period",
        ]
    )


def _validate_probability_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("probability_threshold must be between 0 and 1")
    return threshold


def _validate_transaction_cost(value: float) -> float:
    cost = float(value)
    if cost < 0.0:
        raise ValueError("transaction_cost must be nonnegative")
    return cost


@dataclass(frozen=True)
class TransactionCostAssumption:
    scenario: str
    commission_per_trade: float = 0.0
    bid_ask_spread_proxy: float = 0.0
    slippage: float = 0.0
    borrow_cost_bps_per_day: float = 0.0

    @property
    def total_cost_per_unit(self) -> float:
        return float(self.commission_per_trade + self.bid_ask_spread_proxy + self.slippage)

    def borrow_cost_for_holding_period(self, holding_days: float) -> float:
        """Borrow cost accrues daily while a short leg is held, unlike the
        other cost components (commission/spread/slippage), which are
        one-time costs paid at entry/exit regardless of how long the
        position is open. A triangular arb trade shorts at least one leg,
        so ignoring this understates cost for anything held more than a
        few days -- the transaction_cost_sensitivity.csv scenarios prior
        to this had no holding-period-dependent cost component at all.
        """
        return float(self.borrow_cost_bps_per_day / 10_000.0 * max(0.0, holding_days))


@dataclass(frozen=True)
class RobustnessGrid:
    # The full parameter sweep used by run_threshold_sensitivity: 3 x 3 x
    # 3 x 3 = 81 combinations per strategy (243 total scenarios). See the
    # note on relabel_events_for_threshold_scenario below for what this
    # grid does and does not actually change.
    entry_thresholds: tuple[float, ...] = (1.5, 2.0, 2.5)
    exit_thresholds: tuple[float, ...] = (0.25, 0.5, 0.75)
    stop_loss_levels: tuple[float, ...] = (2.75, 3.0, 3.5)
    max_holding_periods: tuple[int, ...] = (10, 15, 20)


def apply_transaction_cost_assumption(
    trades: pd.DataFrame,
    assumption: TransactionCostAssumption | dict,
    gross_pnl_col: str = "gross_pnl",
    position_col: str = "position_size",
    holding_period_col: str = "holding_period",
) -> pd.DataFrame:
    # Re-prices an existing trade log under a different cost assumption --
    # separated from build_strategy_trades's flat cost so the same set of
    # trades can be cheaply re-costed under many scenarios (zero/low/base
    # /stress) without re-running the selection logic each time.
    cost = _coerce_cost_assumption(assumption)
    required = {gross_pnl_col, position_col}
    missing = required.difference(trades.columns)
    if missing:
        raise KeyError(f"missing trade columns: {sorted(missing)}")
    frame = trades.copy()
    frame["cost_scenario"] = cost.scenario
    frame["commission_per_trade"] = cost.commission_per_trade
    frame["bid_ask_spread_proxy"] = cost.bid_ask_spread_proxy
    frame["slippage"] = cost.slippage
    frame["total_cost_per_unit"] = cost.total_cost_per_unit
    one_time_cost = cost.total_cost_per_unit * frame[position_col].abs().astype(float)

    if cost.borrow_cost_bps_per_day > 0.0 and holding_period_col in frame.columns:
        borrow_cost = frame[holding_period_col].astype(float).apply(cost.borrow_cost_for_holding_period)
        frame["borrow_cost"] = borrow_cost * frame[position_col].abs().astype(float)
    else:
        frame["borrow_cost"] = 0.0

    frame["transaction_cost"] = one_time_cost + frame["borrow_cost"]
    frame["net_pnl"] = frame[gross_pnl_col].astype(float) - frame["transaction_cost"]
    frame["cost_drag"] = frame["transaction_cost"]
    return frame


def run_transaction_cost_sensitivity(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    cost_assumptions: Iterable[TransactionCostAssumption | dict] | None = None,
    probability_threshold: float = 0.6,
    probability_col: str = "predicted_reversion_probability",
) -> dict[str, pd.DataFrame]:
    # Builds the trade log ONCE at zero cost, then re-prices that same
    # fixed set of trades under every cost scenario -- guarantees the
    # trade SELECTION is identical across scenarios, so any difference in
    # the results is purely attributable to costs, not to a different set
    # of trades being compared.
    assumptions = _default_cost_assumptions() if cost_assumptions is None else [_coerce_cost_assumption(item) for item in cost_assumptions]
    events = prepare_ml_backtest_events(labels, predictions, probability_col=probability_col)

    zero_cost_trades = []
    for strategy in ["baseline_rule_based", "ml_filtered", "ml_probability_sized"]:
        zero_cost_trades.append(
            build_strategy_trades(
                events,
                strategy=strategy,
                probability_threshold=probability_threshold,
                transaction_cost=0.0,
                probability_col=probability_col,
            )
        )
    base_trades = pd.concat(zero_cost_trades, ignore_index=True, sort=False)

    costed_logs = []
    summary_rows = []
    for assumption in assumptions:
        costed = apply_transaction_cost_assumption(base_trades, assumption)
        costed_logs.append(costed)
        equity = equity_curve_from_trades(costed) if not costed.empty else pd.DataFrame()
        summary = strategy_performance_summary(costed, equity) if not costed.empty else pd.DataFrame()
        if not summary.empty:
            summary["cost_scenario"] = assumption.scenario
            summary["total_cost_per_unit"] = assumption.total_cost_per_unit
            summary["commission_per_trade"] = assumption.commission_per_trade
            summary["bid_ask_spread_proxy"] = assumption.bid_ask_spread_proxy
            summary["slippage"] = assumption.slippage
            summary["cost_drag"] = costed.groupby("strategy")["cost_drag"].sum().reindex(summary["strategy"]).to_numpy()
            summary["gross_to_net_pnl_delta"] = summary["gross_pnl"] - summary["net_pnl"]
            summary_rows.append(summary)

    trade_log = pd.concat(costed_logs, ignore_index=True, sort=False) if costed_logs else pd.DataFrame()
    sensitivity = pd.concat(summary_rows, ignore_index=True, sort=False) if summary_rows else pd.DataFrame()
    return {
        "transaction_cost_trade_log": trade_log,
        "transaction_cost_sensitivity": sensitivity,
    }


def relabel_events_for_threshold_scenario(
    events: pd.DataFrame,
    entry_threshold: float,
    exit_threshold: float,
    stop_loss_level: float,
    max_holding_period: int,
) -> pd.DataFrame:
    """Re-derives the win/loss `label` for a hypothetical alternative set
    of entry/exit/stop-loss/holding-period rules, without needing to
    re-run the full labeling pipeline from raw residuals.

    IMPORTANT, and worth reading if you're using this for a "robustness"
    claim: `entry_threshold` is the only one of these four parameters
    that changes which events even make it into `frame` (the filter on
    line below) -- and trade selection/realized PnL downstream (in
    `build_strategy_trades`/`event_spread_pnl`) depends only on which
    events are present and each event's fixed, already-recorded
    `exit_z_score`, never on the `label` this function computes. So
    `exit_threshold`, `stop_loss_level`, and `max_holding_period` change
    the `label` column (used elsewhere as the ML training target) but do
    NOT change net_pnl, Sharpe, or trade_count in
    `run_threshold_sensitivity` below. See CHANGELOG.md for how this was
    found and what it means for interpreting the sensitivity grid.
    """
    entry = _validate_positive_threshold(entry_threshold, "entry_threshold")
    exit_ = _validate_nonnegative_threshold(exit_threshold, "exit_threshold")
    stop = _validate_positive_threshold(stop_loss_level, "stop_loss_level")
    holding = int(max_holding_period)
    if holding <= 0:
        raise ValueError("max_holding_period must be positive")
    if stop <= entry:
        raise ValueError("stop_loss_level should be greater than entry_threshold")

    required = {"side", "entry_z_score", "exit_z_score", "holding_period"}
    missing = required.difference(events.columns)
    if missing:
        raise KeyError(f"missing event columns: {sorted(missing)}")

    frame = events.copy()
    # the only line in this function that actually changes which events
    # survive into a strategy's trade log for this scenario
    frame = frame.loc[frame["entry_z_score"].abs().ge(entry)].copy()
    if frame.empty:
        return frame.assign(
            label=pd.Series(dtype=int),
            scenario_exit_reason=pd.Series(dtype=str),
            entry_threshold=entry,
            exit_threshold=exit_,
            stop_loss_level=stop,
            max_holding_period=holding,
        )

    side = frame["side"].astype(str)
    exit_z = frame["exit_z_score"].astype(float)
    period = frame["holding_period"].astype(float)

    # relabels each event's outcome under the hypothetical rule, using
    # the direction-flip trick from src/labeling.py:_label_single_event
    # (short_spread wants exit_z to fall, long_spread wants it to rise)
    short_success = side.eq("short_spread") & exit_z.le(exit_)
    long_success = side.eq("long_spread") & exit_z.ge(-exit_)
    short_stop = side.eq("short_spread") & exit_z.ge(stop)
    long_stop = side.eq("long_spread") & exit_z.le(-stop)
    holding_fail = period.gt(holding)

    success = (short_success | long_success) & ~holding_fail & ~(short_stop | long_stop)
    stop_hit = short_stop | long_stop
    reason = np.where(success, "reversion", np.where(stop_hit, "stop_loss", np.where(holding_fail, "max_holding", "no_reversion")))

    frame["label"] = success.astype(int)
    frame["scenario_exit_reason"] = reason
    frame["entry_threshold"] = entry
    frame["exit_threshold"] = exit_
    frame["stop_loss_level"] = stop
    frame["max_holding_period"] = holding
    return frame.reset_index(drop=True)


def run_threshold_sensitivity(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    grid: RobustnessGrid | dict | None = None,
    probability_threshold: float = 0.6,
    transaction_cost: float = 0.05,
    probability_col: str = "predicted_reversion_probability",
) -> pd.DataFrame:
    # Runs every (entry, exit, stop, holding) combination in `grid`
    # through relabel + build_strategy_trades for all three strategies --
    # see relabel_events_for_threshold_scenario's docstring for what this
    # grid does and does not actually test.
    cfg = _coerce_robustness_grid(grid)
    events = prepare_ml_backtest_events(labels, predictions, probability_col=probability_col)
    rows = []
    for entry in cfg.entry_thresholds:
        for exit_ in cfg.exit_thresholds:
            for stop in cfg.stop_loss_levels:
                for holding in cfg.max_holding_periods:
                    scenario_events = relabel_events_for_threshold_scenario(events, entry, exit_, stop, holding)
                    if scenario_events.empty:
                        continue
                    for strategy in ["baseline_rule_based", "ml_filtered", "ml_probability_sized"]:
                        trades = build_strategy_trades(
                            scenario_events,
                            strategy=strategy,
                            probability_threshold=probability_threshold,
                            transaction_cost=transaction_cost,
                            probability_col=probability_col,
                        )
                        if trades.empty:
                            rows.append(_empty_threshold_summary_row(strategy, entry, exit_, stop, holding, probability_threshold, transaction_cost))
                            continue
                        equity = equity_curve_from_trades(trades)
                        summary = strategy_performance_summary(trades, equity).iloc[0].to_dict()
                        summary.update(
                            {
                                "entry_threshold": float(entry),
                                "exit_threshold": float(exit_),
                                "stop_loss_level": float(stop),
                                "max_holding_period": int(holding),
                                "probability_threshold": float(probability_threshold),
                                "transaction_cost": float(transaction_cost),
                            }
                        )
                        rows.append(summary)
    return pd.DataFrame(rows)


def run_cost_and_threshold_robustness(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    cost_assumptions: Iterable[TransactionCostAssumption | dict] | None = None,
    grid: RobustnessGrid | dict | None = None,
    probability_threshold: float = 0.6,
    probability_col: str = "predicted_reversion_probability",
) -> dict[str, pd.DataFrame]:
    # Combines both robustness checks (cost sensitivity and
    # threshold/parameter sensitivity) into one call and a single summary
    # table, using the middle cost scenario as the fixed cost level for
    # the threshold sweep (so the two checks aren't testing at completely
    # unrelated cost assumptions).
    cost_outputs = run_transaction_cost_sensitivity(
        labels,
        predictions,
        cost_assumptions=cost_assumptions,
        probability_threshold=probability_threshold,
        probability_col=probability_col,
    )
    assumptions = _default_cost_assumptions() if cost_assumptions is None else [_coerce_cost_assumption(item) for item in cost_assumptions]
    base_cost = assumptions[len(assumptions) // 2].total_cost_per_unit if assumptions else 0.0
    threshold = run_threshold_sensitivity(
        labels,
        predictions,
        grid=grid,
        probability_threshold=probability_threshold,
        transaction_cost=base_cost,
        probability_col=probability_col,
    )
    from src.metrics import robustness_summary_from_sensitivity

    robustness = robustness_summary_from_sensitivity(threshold, cost_outputs["transaction_cost_sensitivity"])
    return {
        **cost_outputs,
        "threshold_sensitivity_table": threshold,
        "robustness_summary": robustness,
    }


def _default_cost_assumptions() -> list[TransactionCostAssumption]:
    return [
        TransactionCostAssumption("zero_cost", 0.0, 0.0, 0.0),
        TransactionCostAssumption("low_cost", 0.005, 0.010, 0.005),
        TransactionCostAssumption("base_cost", 0.010, 0.025, 0.015),
        TransactionCostAssumption("high_cost", 0.020, 0.050, 0.030),
    ]


def _coerce_cost_assumption(value: TransactionCostAssumption | dict) -> TransactionCostAssumption:
    if isinstance(value, TransactionCostAssumption):
        assumption = value
    elif isinstance(value, dict):
        assumption = TransactionCostAssumption(
            scenario=str(value.get("scenario", value.get("cost_scenario", "custom"))),
            commission_per_trade=float(value.get("commission_per_trade", 0.0)),
            bid_ask_spread_proxy=float(value.get("bid_ask_spread_proxy", value.get("bid_ask_spread", 0.0))),
            slippage=float(value.get("slippage", 0.0)),
            borrow_cost_bps_per_day=float(value.get("borrow_cost_bps_per_day", 0.0)),
        )
    else:
        raise TypeError("cost assumption must be a TransactionCostAssumption or dict")
    if not assumption.scenario:
        raise ValueError("cost scenario must be non-empty")
    for field in [assumption.commission_per_trade, assumption.bid_ask_spread_proxy, assumption.slippage, assumption.borrow_cost_bps_per_day]:
        if float(field) < 0.0:
            raise ValueError("cost components must be nonnegative")
    return assumption


def _coerce_robustness_grid(value: RobustnessGrid | dict | None) -> RobustnessGrid:
    if value is None:
        return RobustnessGrid()
    if isinstance(value, RobustnessGrid):
        return value
    if not isinstance(value, dict):
        raise TypeError("grid must be a RobustnessGrid, dict, or None")
    return RobustnessGrid(
        entry_thresholds=tuple(float(x) for x in value.get("entry_thresholds", RobustnessGrid.entry_thresholds)),
        exit_thresholds=tuple(float(x) for x in value.get("exit_thresholds", RobustnessGrid.exit_thresholds)),
        stop_loss_levels=tuple(float(x) for x in value.get("stop_loss_levels", RobustnessGrid.stop_loss_levels)),
        max_holding_periods=tuple(int(x) for x in value.get("max_holding_periods", RobustnessGrid.max_holding_periods)),
    )


def _validate_positive_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if threshold <= 0.0:
        raise ValueError(f"{name} must be positive")
    return threshold


def _validate_nonnegative_threshold(value: float, name: str) -> float:
    threshold = float(value)
    if threshold < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return threshold


def _empty_threshold_summary_row(
    strategy: str,
    entry_threshold: float,
    exit_threshold: float,
    stop_loss_level: float,
    max_holding_period: int,
    probability_threshold: float,
    transaction_cost: float,
) -> dict:
    return {
        "strategy": strategy,
        "trade_count": 0,
        "gross_pnl": 0.0,
        "net_pnl": 0.0,
        "average_net_pnl": np.nan,
        "median_net_pnl": np.nan,
        "win_rate": np.nan,
        "turnover": 0.0,
        "max_drawdown": np.nan,
        "sharpe": np.nan,
        "entry_threshold": float(entry_threshold),
        "exit_threshold": float(exit_threshold),
        "stop_loss_level": float(stop_loss_level),
        "max_holding_period": int(max_holding_period),
        "probability_threshold": float(probability_threshold),
        "transaction_cost": float(transaction_cost),
    }
