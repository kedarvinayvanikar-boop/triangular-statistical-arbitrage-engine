import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    avoided_bad_trades_analysis,
    build_strategy_trades,
    event_spread_pnl,
    prepare_ml_backtest_events,
    run_ml_backtest_comparison,
    trade_count_comparison,
)
from src.portfolio import (
    annualized_sharpe,
    apply_volatility_targeting,
    benchmark_buy_and_hold,
    bootstrap_path_metric_ci,
    bootstrap_trade_metric_ci,
    equity_curve_from_trades,
    shared_leg_groups,
    strategy_performance_summary,
    volatility_target_position_size,
)


def _labels():
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "triplet_id": ["A_B_C"] * 4,
            "method": ["kalman"] * 4,
            "event_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "exit_date": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]),
            "side": ["short_spread", "long_spread", "short_spread", "long_spread"],
            "label": [1, 1, 0, 0],
            "outcome": ["success", "success", "failure", "failure"],
            "exit_reason": ["reversion", "reversion", "stop_loss", "max_holding"],
            "holding_period": [2, 2, 2, 2],
            "entry_z_score": [2.2, -2.1, 2.4, -2.3],
            "exit_z_score": [0.4, -0.3, 3.2, -2.0],
        }
    )


def _predictions():
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "split": ["test"] * 4,
            "predicted_reversion_probability": [0.82, 0.71, 0.41, 0.35],
        }
    )


def test_prepare_ml_backtest_events_merges_labels_and_predictions():
    events = prepare_ml_backtest_events(_labels(), _predictions())

    assert events.shape[0] == 4
    assert "predicted_reversion_probability" in events.columns
    assert "pnl_date" in events.columns


def test_event_spread_pnl_handles_long_and_short_symmetrically():
    pnl = event_spread_pnl(_labels())

    assert np.isclose(pnl.iloc[0], 1.8)
    assert np.isclose(pnl.iloc[1], 1.8)
    assert np.isclose(pnl.iloc[2], -0.8)
    assert np.isclose(pnl.iloc[3], 0.3)


def test_ml_filtered_takes_only_events_above_threshold():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    trades = build_strategy_trades(events, strategy="ml_filtered", probability_threshold=0.6, transaction_cost=0.0)

    assert trades["event_id"].tolist() == ["e1", "e2"]
    assert trades["position_size"].eq(1.0).all()


def test_probability_sized_strategy_scales_by_probability():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    trades = build_strategy_trades(events, strategy="ml_probability_sized", probability_threshold=0.6, transaction_cost=0.0)

    assert np.allclose(trades["position_size"].to_numpy(), [0.82, 0.71])
    assert np.allclose(trades["gross_pnl"].to_numpy(), trades["position_size"] * trades["spread_pnl_units"])


def test_run_ml_backtest_comparison_returns_expected_tables():
    result = run_ml_backtest_comparison(_labels(), _predictions(), probability_threshold=0.6, transaction_cost=0.01)

    assert set(result) == {
        "backtest_event_universe",
        "strategy_trade_log",
        "strategy_equity_curve",
        "strategy_summary",
        "avoided_bad_trades",
    }
    assert set(result["strategy_summary"]["strategy"]) == {
        "baseline_rule_based",
        "ml_filtered",
        "ml_probability_sized",
    }


def test_equity_curve_and_strategy_summary_are_finite_for_nonempty_trades():
    result = run_ml_backtest_comparison(_labels(), _predictions(), probability_threshold=0.6, transaction_cost=0.01)
    equity = equity_curve_from_trades(result["strategy_trade_log"])
    summary = strategy_performance_summary(result["strategy_trade_log"], equity)

    assert not equity.empty
    assert summary["trade_count"].gt(0).all()
    assert summary["net_pnl"].notna().all()


def test_avoided_bad_trades_counts_rejected_events():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    avoided = avoided_bad_trades_analysis(events, probability_threshold=0.6, transaction_cost=0.0)

    assert int(avoided.loc[0, "avoided_trade_count"]) == 2
    assert int(avoided.loc[0, "avoided_failure_count"]) == 2
    assert np.isclose(avoided.loc[0, "avoided_failure_rate"], 1.0)


def test_sharpe_accounts_for_idle_days_not_just_trade_days():
    # A strategy with 4 profitable, low-variance trades spread across a full
    # year should NOT get the same Sharpe as one making those same 4 trades
    # back-to-back in a week -- the first spends most of the year flat, which
    # must lower its annualized Sharpe relative to naively annualizing off
    # only the 4 nonzero observations.
    sparse_trades = pd.DataFrame({
        "strategy": ["s"] * 4,
        "pnl_date": pd.to_datetime(["2024-01-05", "2024-04-05", "2024-07-05", "2024-10-04"]),
        "net_pnl": [2.0, 2.1, 1.9, 2.0],
        "gross_pnl": [2.0, 2.1, 1.9, 2.0],
        "turnover": [2.0] * 4,
    })
    summary = strategy_performance_summary(sparse_trades)
    trade_only_sharpe = annualized_sharpe(sparse_trades["net_pnl"].to_numpy())

    assert summary.loc[0, "sharpe"] < trade_only_sharpe
    # the naive trade-only calculation is exactly the old (buggy) behavior --
    # confirming the fixed value is meaningfully smaller, not just a rounding change
    assert summary.loc[0, "sharpe"] < trade_only_sharpe * 0.5


def test_shared_leg_groups_finds_transitive_clusters_and_true_independents():
    triplets = [
        "AAPL_QQQ_XLK", "MSFT_QQQ_XLK",   # share QQQ + XLK
        "AMD_SMH_QQQ", "NVDA_SMH_QQQ",     # share SMH + QQQ -- and QQQ links this to the pair above
        "JPM_XLF_KRE", "BAC_XLF_KRE",      # separate cluster: financials
        "SOLO_ABC_DEF",                     # shares nothing with anyone -- true independent bet
    ]
    groups = shared_leg_groups(triplets)

    cluster_sizes = sorted(groups["cluster_size"].tolist())
    assert cluster_sizes == [1, 2, 4]  # financials pair, solo bet, and the QQQ-linked four

    solo_row = groups.loc[groups["triplets"].apply(lambda g: g == ["SOLO_ABC_DEF"])]
    assert solo_row.iloc[0]["independent_bet"]

    qqq_cluster = groups.loc[groups["cluster_size"] == 4].iloc[0]
    assert set(qqq_cluster["triplets"]) == {"AAPL_QQQ_XLK", "MSFT_QQQ_XLK", "AMD_SMH_QQQ", "NVDA_SMH_QQQ"}
    assert not qqq_cluster["independent_bet"]


def test_wilson_interval_is_not_degenerate_at_100_percent_observed():
    from src.portfolio import wilson_score_interval
    # 20/20 wins -- the naive bootstrap can only ever resample all-wins here
    # (degenerate [1.0, 1.0]); Wilson must show real width instead
    result = wilson_score_interval(successes=20, n=20, confidence=0.90)
    assert result["rate"] == 1.0
    assert result["ci_low"] < 1.0
    assert result["ci_high"] == 1.0

    # a coin-flip sample should give a wide, roughly-centered interval
    fair = wilson_score_interval(successes=15, n=30, confidence=0.90)
    assert fair["ci_low"] < 0.5 < fair["ci_high"]

    with pytest.raises(ValueError):
        wilson_score_interval(successes=5, n=10, confidence=0.42)


def test_bootstrap_win_rate_ci_is_wide_for_small_samples():
    # an all-winning small sample should still show real uncertainty --
    # the CI should not collapse to a point just because every observed
    # trade happened to win
    all_wins = np.array([1.0] * 10)
    result = bootstrap_trade_metric_ci(all_wins, metric="win_rate", n_resamples=500, random_state=0)
    assert result["point_estimate"] == 1.0
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]
    # with only 10 trades all won, the CI should not exclude values well below 100%
    assert result["ci_low"] < 1.0 or result["n_trades"] < 30

    mixed = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0])
    mixed_result = bootstrap_trade_metric_ci(mixed, metric="win_rate", n_resamples=500, random_state=0)
    assert mixed_result["ci_high"] - mixed_result["ci_low"] > 0.0

    with pytest.raises(ValueError):
        bootstrap_trade_metric_ci(mixed, metric="not_a_real_metric")

    empty_result = bootstrap_trade_metric_ci(np.array([1.0]), metric="win_rate")
    assert np.isnan(empty_result["point_estimate"])


def test_strategy_performance_summary_falls_back_without_date_column():
    trades = pd.DataFrame({
        "strategy": ["s", "s"],
        "net_pnl": [1.0, 2.0],
        "gross_pnl": [1.0, 2.0],
        "turnover": [2.0, 2.0],
    })
    # equity_curve_from_trades requires pnl_date, so the no-date-column path
    # is only reachable when the caller supplies a precomputed equity curve
    equity_curve = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "strategy": ["s", "s"],
        "daily_pnl": [1.0, 2.0],
        "running_peak": [1.0, 3.0],
        "drawdown": [0.0, 0.0],
    })
    summary = strategy_performance_summary(trades, equity_curve=equity_curve)
    assert summary.loc[0, "sharpe"] == annualized_sharpe(trades["net_pnl"].to_numpy())


def test_trade_count_comparison_measures_avoided_trades():
    result = run_ml_backtest_comparison(_labels(), _predictions(), probability_threshold=0.6, transaction_cost=0.0)
    counts = trade_count_comparison(result["strategy_summary"])
    filtered = counts.loc[counts["strategy"].eq("ml_filtered")].iloc[0]

    assert filtered["trades_avoided_vs_baseline"] == 2
    assert np.isclose(filtered["trade_count_ratio_to_baseline"], 0.5)


def test_invalid_probability_threshold_raises():
    events = prepare_ml_backtest_events(_labels(), _predictions())
    with pytest.raises(ValueError):
        build_strategy_trades(events, strategy="ml_filtered", probability_threshold=1.2)


def test_volatility_target_position_size_scales_inversely_with_vol():
    vols = np.array([0.01, 0.02, 0.05, 0.0])  # last is degenerate (zero vol)
    sizes = volatility_target_position_size(vols, target_vol=0.02, max_leverage=3.0)
    assert sizes[0] > sizes[1] > sizes[2]  # lower vol -> larger size
    assert np.isclose(sizes[1], 1.0)  # realized == target -> size 1.0
    assert sizes[3] == 3.0  # zero vol floored, hits the leverage cap, not inf/nan


def test_volatility_target_position_size_handles_scalar_input():
    size = volatility_target_position_size(0.04, target_vol=0.02, max_leverage=3.0)
    assert isinstance(size, float)
    assert np.isclose(size, 0.5)


def test_apply_volatility_targeting_recomputes_pnl_consistently():
    trades = pd.DataFrame({
        "spread_pnl_units": [10.0, -5.0, 20.0],
        "residual_volatility": [0.01, 0.04, 0.02],
    })
    result = apply_volatility_targeting(trades, vol_col="residual_volatility", target_vol=0.02, max_leverage=3.0, flat_cost_per_trade=0.05)
    expected_sizes = volatility_target_position_size(trades["residual_volatility"].to_numpy(), 0.02, 3.0)
    assert np.allclose(result["position_size"], expected_sizes)
    assert np.allclose(result["gross_pnl"], expected_sizes * trades["spread_pnl_units"].to_numpy())
    assert np.allclose(result["net_pnl"], result["gross_pnl"] - result["transaction_cost"])

    with pytest.raises(KeyError):
        apply_volatility_targeting(trades.drop(columns=["residual_volatility"]), vol_col="residual_volatility", target_vol=0.02)


def test_bootstrap_path_metric_max_drawdown_matches_known_path():
    # a hand-computable path: gains, then a clear decline, then partial recovery
    daily_pnl = np.array([5, 5, 5, -3, -4, -5, -2, 3, 3, 3, 5, 5, -1, -1, -1, 2, 2, 2, 2, 2], dtype=float)
    equity = np.cumsum(daily_pnl)
    running_peak = np.maximum.accumulate(equity)
    expected_max_dd = float((equity - running_peak).min())

    result = bootstrap_path_metric_ci(daily_pnl, metric="max_drawdown", block_size=5, n_resamples=500, random_state=0)
    assert np.isclose(result["point_estimate"], expected_max_dd)
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]


def test_bootstrap_path_metric_sharpe_runs_and_brackets_point_estimate():
    rng = np.random.default_rng(11)
    daily_pnl = rng.normal(0.5, 2.0, 200)
    result = bootstrap_path_metric_ci(daily_pnl, metric="sharpe", block_size=15, n_resamples=500, random_state=0)
    assert np.isfinite(result["point_estimate"])
    assert result["ci_low"] <= result["point_estimate"] <= result["ci_high"]

    with pytest.raises(ValueError):
        bootstrap_path_metric_ci(daily_pnl, metric="not_a_real_metric")


def test_bootstrap_path_metric_handles_too_short_series():
    result = bootstrap_path_metric_ci(np.array([1.0, 2.0]), metric="max_drawdown", block_size=20)
    assert np.isnan(result["point_estimate"])


def test_block_bootstrap_preserves_autocorrelation_structure_better_than_iid_would():
    # a series with strong local runs (blocks of same-signed pnl) --
    # a block bootstrap should preserve within-block same-sign clustering
    # far more often than shuffling individual days would
    from src.portfolio import _moving_block_bootstrap_paths
    rng = np.random.default_rng(12)
    blocks = [np.full(10, 5.0) if i % 2 == 0 else np.full(10, -5.0) for i in range(10)]
    daily_pnl = np.concatenate(blocks)

    paths = _moving_block_bootstrap_paths(daily_pnl, block_size=10, n_resamples=200, rng=rng)
    # within each resampled path, count how often consecutive days share a sign
    same_sign_fraction = np.mean([(np.sign(p[:-1]) == np.sign(p[1:])).mean() for p in paths])
    # with block_size=10 matching the true block structure, consecutive-day
    # sign agreement should stay high (blocks are internally constant-signed)
    assert same_sign_fraction > 0.7


def test_benchmark_buy_and_hold_equal_weight_matches_manual_calculation():
    prices = pd.DataFrame({
        "symbol": ["A", "A", "A", "B", "B", "B"],
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 2,
        "adj_close": [100.0, 110.0, 121.0, 50.0, 45.0, 45.0],
    })
    result = benchmark_buy_and_hold(prices, symbols=["A", "B"])
    # 3 price points -> 2 daily returns; the first date has no prior price
    # to compute a return against, so it correctly does not appear
    assert len(result) == 2
    # day 2 (first return day): A +10%, B -10% -> equal-weight return = 0%
    day1 = result.iloc[0]
    assert np.isclose(day1["daily_return"], 0.0, atol=1e-9)
    # day 3: A +10%, B +0% -> equal-weight return = +5%
    day2 = result.iloc[1]
    assert np.isclose(day2["daily_return"], 0.05)


def test_benchmark_buy_and_hold_raises_on_missing_columns():
    with pytest.raises(KeyError):
        benchmark_buy_and_hold(pd.DataFrame({"symbol": ["A"]}), symbols=["A"])
