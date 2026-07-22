-- Phase 16 report queries.
-- These queries summarize pipeline coverage, model outputs, and strategy audit results.

-- name: pipeline_row_counts
SELECT *
FROM v_required_table_row_counts
ORDER BY table_name;

-- name: asset_price_coverage
SELECT
    a.symbol,
    a.asset_type,
    MIN(p.date) AS first_clean_date,
    MAX(p.date) AS last_clean_date,
    COUNT(p.date) AS clean_price_rows
FROM assets a
LEFT JOIN prices_clean p ON p.symbol = a.symbol
GROUP BY a.symbol, a.asset_type
ORDER BY a.symbol;

-- name: triplet_pipeline_coverage
SELECT *
FROM v_pipeline_coverage
ORDER BY triplet_id;

-- name: residual_diagnostics_report
SELECT
    triplet_id,
    method,
    n_obs,
    residual_mean,
    residual_std,
    residual_autocorr_1,
    half_life_estimate
FROM residual_diagnostics
ORDER BY triplet_id, method;

-- name: event_success_by_triplet
SELECT
    triplet_id,
    method,
    COUNT(*) AS labeled_events,
    SUM(label) AS success_count,
    CAST(SUM(label) AS REAL) / NULLIF(COUNT(*), 0) AS success_rate
FROM event_labels
GROUP BY triplet_id, method
ORDER BY success_rate DESC;

-- name: model_prediction_summary
SELECT
    m.model_run_id,
    m.model_type,
    p.split,
    COUNT(*) AS n_predictions,
    AVG(p.predicted_reversion_probability) AS average_predicted_probability,
    AVG(p.label) AS realized_success_rate
FROM model_predictions p
JOIN model_runs m ON m.model_run_id = p.model_run_id
GROUP BY m.model_run_id, m.model_type, p.split
ORDER BY m.model_run_id, p.split;

-- name: probability_bucket_report
SELECT
    CASE
        WHEN predicted_reversion_probability < 0.2 THEN '0.00-0.20'
        WHEN predicted_reversion_probability < 0.4 THEN '0.20-0.40'
        WHEN predicted_reversion_probability < 0.6 THEN '0.40-0.60'
        WHEN predicted_reversion_probability < 0.8 THEN '0.60-0.80'
        ELSE '0.80-1.00'
    END AS probability_bucket,
    COUNT(*) AS n_predictions,
    AVG(predicted_reversion_probability) AS average_predicted_probability,
    AVG(label) AS realized_success_rate
FROM model_predictions
GROUP BY probability_bucket
ORDER BY probability_bucket;

-- name: strategy_performance_report
SELECT
    strategy,
    model_run_id,
    period_start,
    period_end,
    trade_count,
    win_rate,
    total_net_pnl,
    max_drawdown,
    sharpe,
    turnover
FROM performance_summary
ORDER BY strategy, model_run_id;

-- name: daily_pnl_report
SELECT
    date,
    strategy,
    gross_pnl,
    transaction_cost,
    net_pnl,
    equity,
    drawdown,
    turnover
FROM pnl_daily
ORDER BY strategy, date;

-- name: trade_audit_report
SELECT
    t.trade_id,
    t.event_id,
    t.strategy,
    t.entry_date,
    t.exit_date,
    t.side,
    t.predicted_reversion_probability,
    t.position_size,
    t.gross_pnl,
    t.transaction_cost,
    t.net_pnl,
    t.exit_reason
FROM trades t
ORDER BY t.strategy, t.entry_date, t.trade_id;
