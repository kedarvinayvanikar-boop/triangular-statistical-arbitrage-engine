-- Phase 16 validation queries.
-- Each query returns rows only when an integrity issue is present, except the row-count summary.

-- name: required_table_row_counts
SELECT *
FROM v_required_table_row_counts
ORDER BY table_name;

-- name: triplet_symbols_missing_from_assets
SELECT
    t.triplet_id,
    t.target_symbol,
    t.hedge_symbol_1,
    t.hedge_symbol_2
FROM triplets t
LEFT JOIN assets a0 ON a0.symbol = t.target_symbol
LEFT JOIN assets a1 ON a1.symbol = t.hedge_symbol_1
LEFT JOIN assets a2 ON a2.symbol = t.hedge_symbol_2
WHERE a0.symbol IS NULL OR a1.symbol IS NULL OR a2.symbol IS NULL;

-- name: clean_prices_without_assets
SELECT p.symbol, COUNT(*) AS row_count
FROM prices_clean p
LEFT JOIN assets a ON a.symbol = p.symbol
WHERE a.symbol IS NULL
GROUP BY p.symbol;

-- name: returns_without_clean_prices
SELECT r.symbol, r.date
FROM returns_daily r
LEFT JOIN prices_clean p
    ON p.symbol = r.symbol AND p.date = r.date
WHERE p.symbol IS NULL;

-- name: residuals_without_triplets
SELECT r.triplet_id, COUNT(*) AS row_count
FROM residuals r
LEFT JOIN triplets t ON t.triplet_id = r.triplet_id
WHERE t.triplet_id IS NULL
GROUP BY r.triplet_id;

-- name: candidate_events_without_residuals
SELECT e.event_id, e.triplet_id, e.method, e.event_date
FROM candidate_events e
LEFT JOIN residuals r
    ON r.triplet_id = e.triplet_id
   AND r.method = e.method
   AND r.date = e.event_date
WHERE r.triplet_id IS NULL;

-- name: event_features_without_candidate_events
SELECT f.event_id, f.triplet_id, f.event_date
FROM event_features f
LEFT JOIN candidate_events e ON e.event_id = f.event_id
WHERE e.event_id IS NULL;

-- name: event_labels_without_candidate_events
SELECT l.event_id, l.triplet_id, l.event_date
FROM event_labels l
LEFT JOIN candidate_events e ON e.event_id = l.event_id
WHERE e.event_id IS NULL;

-- name: model_predictions_without_model_runs
SELECT p.prediction_id, p.model_run_id
FROM model_predictions p
LEFT JOIN model_runs m ON m.model_run_id = p.model_run_id
WHERE m.model_run_id IS NULL;

-- name: model_predictions_without_candidate_events
SELECT p.prediction_id, p.event_id
FROM model_predictions p
LEFT JOIN candidate_events e ON e.event_id = p.event_id
WHERE e.event_id IS NULL;

-- name: model_prediction_probability_bounds
SELECT prediction_id, predicted_reversion_probability
FROM model_predictions
WHERE predicted_reversion_probability < 0
   OR predicted_reversion_probability > 1
   OR predicted_reversion_probability IS NULL;

-- name: trades_without_candidate_events
SELECT tr.trade_id, tr.event_id
FROM trades tr
LEFT JOIN candidate_events e ON e.event_id = tr.event_id
WHERE e.event_id IS NULL;

-- name: positions_without_trades
SELECT p.position_id, p.trade_id
FROM positions p
LEFT JOIN trades tr ON tr.trade_id = p.trade_id
WHERE tr.trade_id IS NULL;

-- name: pnl_missing_required_values
SELECT date, strategy, net_pnl, equity
FROM pnl_daily
WHERE date IS NULL
   OR strategy IS NULL
   OR net_pnl IS NULL
   OR equity IS NULL;

-- name: duplicate_prices_clean_keys
SELECT symbol, date, COUNT(*) AS row_count
FROM prices_clean
GROUP BY symbol, date
HAVING COUNT(*) > 1;

-- name: duplicate_model_predictions
SELECT model_run_id, event_id, split, COUNT(*) AS row_count
FROM model_predictions
GROUP BY model_run_id, event_id, split
HAVING COUNT(*) > 1;
