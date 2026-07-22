-- Canonical research database schema.
-- This section stores the full quant research pipeline in auditable tables.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS assets (
    symbol TEXT PRIMARY KEY,
    asset_name TEXT,
    asset_type TEXT NOT NULL DEFAULT 'equity',
    sector TEXT,
    industry TEXT,
    exchange TEXT,
    currency TEXT DEFAULT 'USD',
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prices_raw (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume REAL,
    source TEXT NOT NULL,
    source_file TEXT,
    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date, source),
    FOREIGN KEY (symbol) REFERENCES assets(symbol)
);

CREATE TABLE IF NOT EXISTS prices_clean (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    close REAL,
    adj_close REAL NOT NULL,
    volume REAL,
    quality_flag TEXT DEFAULT 'clean',
    source TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol) REFERENCES assets(symbol)
);

CREATE TABLE IF NOT EXISTS returns_daily (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    simple_return REAL,
    log_return REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (symbol, date),
    FOREIGN KEY (symbol, date) REFERENCES prices_clean(symbol, date)
);

CREATE TABLE IF NOT EXISTS triplets (
    triplet_id TEXT PRIMARY KEY,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    relationship_theme TEXT,
    active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (target_symbol) REFERENCES assets(symbol),
    FOREIGN KEY (hedge_symbol_1) REFERENCES assets(symbol),
    FOREIGN KEY (hedge_symbol_2) REFERENCES assets(symbol)
);

CREATE TABLE IF NOT EXISTS regression_results (
    regression_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta_1 REAL NOT NULL,
    beta_2 REAL NOT NULL,
    r_squared REAL,
    residual_std REAL,
    n_obs INTEGER,
    ridge_alpha REAL,
    rolling_window INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS hedge_ratios (
    triplet_id TEXT NOT NULL,
    date TEXT NOT NULL,
    method TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta_1 REAL NOT NULL,
    beta_2 REAL NOT NULL,
    regression_id TEXT,
    model_run_id TEXT,
    rolling_window INTEGER,
    ridge_alpha REAL,
    process_noise REAL,
    measurement_noise REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (triplet_id, date, method),
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id),
    FOREIGN KEY (regression_id) REFERENCES regression_results(regression_id)
);

CREATE TABLE IF NOT EXISTS residuals (
    triplet_id TEXT NOT NULL,
    date TEXT NOT NULL,
    method TEXT NOT NULL,
    actual_log_price REAL NOT NULL,
    fitted_log_price REAL NOT NULL,
    residual REAL NOT NULL,
    z_score REAL,
    model_run_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (triplet_id, date, method),
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS residual_diagnostics (
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    window_start TEXT,
    window_end TEXT,
    n_obs INTEGER,
    residual_mean REAL,
    residual_std REAL,
    residual_autocorr_1 REAL,
    half_life_estimate REAL,
    adf_statistic REAL,
    adf_p_value REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (triplet_id, method, window_start, window_end),
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS candidate_events (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    target_symbol TEXT,
    hedge_symbol_1 TEXT,
    hedge_symbol_2 TEXT,
    side TEXT NOT NULL CHECK (side IN ('long_spread', 'short_spread')),
    entry_z_score REAL NOT NULL,
    entry_abs_z REAL NOT NULL,
    entry_residual REAL,
    entry_threshold REAL NOT NULL,
    exit_z REAL NOT NULL,
    stop_loss_z REAL NOT NULL,
    max_holding_period INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS event_features (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    residual_z_score REAL,
    residual_change REAL,
    residual_volatility REAL,
    residual_autocorrelation REAL,
    half_life_estimate REAL,
    rolling_r_squared REAL,
    beta_stability REAL,
    correlation_stability REAL,
    target_return_volatility REAL,
    anchor_1_return_volatility REAL,
    anchor_2_return_volatility REAL,
    market_return REAL,
    sector_return REAL,
    volume_shock REAL,
    recent_drawdown REAL,
    distance_from_moving_average REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES candidate_events(event_id),
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS event_labels (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    exit_date TEXT,
    side TEXT NOT NULL CHECK (side IN ('long_spread', 'short_spread')),
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure')),
    exit_reason TEXT NOT NULL,
    holding_period INTEGER NOT NULL,
    entry_z_score REAL NOT NULL,
    entry_abs_z REAL NOT NULL,
    exit_z_score REAL,
    entry_residual REAL,
    exit_residual REAL,
    entry_threshold REAL NOT NULL,
    exit_z REAL NOT NULL,
    stop_loss_z REAL NOT NULL,
    max_holding_period INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES candidate_events(event_id),
    FOREIGN KEY (triplet_id) REFERENCES triplets(triplet_id)
);

CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id TEXT PRIMARY KEY,
    model_type TEXT NOT NULL,
    feature_set_name TEXT,
    label_name TEXT DEFAULT 'reversion_before_stop',
    train_start TEXT,
    train_end TEXT,
    validation_start TEXT,
    validation_end TEXT,
    test_start TEXT,
    test_end TEXT,
    data_cutoff TEXT,
    hyperparameters_json TEXT,
    random_seed INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id TEXT PRIMARY KEY,
    model_run_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    split TEXT NOT NULL,
    predicted_reversion_probability REAL NOT NULL CHECK (predicted_reversion_probability >= 0 AND predicted_reversion_probability <= 1),
    classification_threshold REAL,
    predicted_label INTEGER CHECK (predicted_label IN (0, 1)),
    label INTEGER CHECK (label IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id),
    FOREIGN KEY (event_id) REFERENCES candidate_events(event_id)
);

CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    event_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    model_run_id TEXT,
    side TEXT NOT NULL CHECK (side IN ('long_spread', 'short_spread')),
    entry_date TEXT NOT NULL,
    exit_date TEXT,
    entry_z_score REAL,
    exit_z_score REAL,
    predicted_reversion_probability REAL,
    position_size REAL NOT NULL DEFAULT 1.0,
    gross_pnl REAL,
    transaction_cost REAL,
    net_pnl REAL,
    turnover REAL,
    exit_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES candidate_events(event_id),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id)
);

CREATE TABLE IF NOT EXISTS positions (
    position_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL,
    date TEXT NOT NULL,
    target_symbol TEXT,
    hedge_symbol_1 TEXT,
    hedge_symbol_2 TEXT,
    target_quantity REAL,
    hedge_1_quantity REAL,
    hedge_2_quantity REAL,
    gross_exposure REAL,
    net_exposure REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);

CREATE TABLE IF NOT EXISTS pnl_daily (
    date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    model_run_id TEXT,
    gross_pnl REAL,
    transaction_cost REAL,
    net_pnl REAL,
    turnover REAL,
    equity REAL,
    running_peak REAL,
    drawdown REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, strategy),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id)
);

CREATE TABLE IF NOT EXISTS performance_summary (
    strategy TEXT NOT NULL,
    model_run_id TEXT,
    period_start TEXT,
    period_end TEXT,
    trade_count INTEGER,
    win_rate REAL,
    total_gross_pnl REAL,
    total_transaction_cost REAL,
    total_net_pnl REAL,
    average_net_pnl REAL,
    max_drawdown REAL,
    sharpe REAL,
    turnover REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy, model_run_id, period_start, period_end),
    FOREIGN KEY (model_run_id) REFERENCES model_runs(model_run_id)
);

CREATE INDEX IF NOT EXISTS idx_prices_raw_symbol_date ON prices_raw(symbol, date);
CREATE INDEX IF NOT EXISTS idx_prices_clean_symbol_date ON prices_clean(symbol, date);
CREATE INDEX IF NOT EXISTS idx_returns_daily_symbol_date ON returns_daily(symbol, date);
CREATE INDEX IF NOT EXISTS idx_triplets_symbols ON triplets(target_symbol, hedge_symbol_1, hedge_symbol_2);
CREATE INDEX IF NOT EXISTS idx_regression_results_triplet_method ON regression_results(triplet_id, method, train_end);
CREATE INDEX IF NOT EXISTS idx_hedge_ratios_triplet_method_date ON hedge_ratios(triplet_id, method, date);
CREATE INDEX IF NOT EXISTS idx_residuals_triplet_method_date ON residuals(triplet_id, method, date);
CREATE INDEX IF NOT EXISTS idx_residuals_z_score ON residuals(method, z_score);
CREATE INDEX IF NOT EXISTS idx_candidate_events_triplet_date ON candidate_events(triplet_id, method, event_date);
CREATE INDEX IF NOT EXISTS idx_event_features_triplet_date ON event_features(triplet_id, method, event_date);
CREATE INDEX IF NOT EXISTS idx_model_predictions_run_split ON model_predictions(model_run_id, split, predicted_reversion_probability);
CREATE INDEX IF NOT EXISTS idx_trades_strategy_date ON trades(strategy, entry_date, exit_date);
CREATE INDEX IF NOT EXISTS idx_pnl_daily_strategy_date ON pnl_daily(strategy, date);

CREATE VIEW IF NOT EXISTS v_required_table_row_counts AS
SELECT 'assets' AS table_name, COUNT(*) AS row_count FROM assets
UNION ALL SELECT 'prices_raw', COUNT(*) FROM prices_raw
UNION ALL SELECT 'prices_clean', COUNT(*) FROM prices_clean
UNION ALL SELECT 'returns_daily', COUNT(*) FROM returns_daily
UNION ALL SELECT 'triplets', COUNT(*) FROM triplets
UNION ALL SELECT 'regression_results', COUNT(*) FROM regression_results
UNION ALL SELECT 'hedge_ratios', COUNT(*) FROM hedge_ratios
UNION ALL SELECT 'residuals', COUNT(*) FROM residuals
UNION ALL SELECT 'residual_diagnostics', COUNT(*) FROM residual_diagnostics
UNION ALL SELECT 'candidate_events', COUNT(*) FROM candidate_events
UNION ALL SELECT 'event_features', COUNT(*) FROM event_features
UNION ALL SELECT 'event_labels', COUNT(*) FROM event_labels
UNION ALL SELECT 'model_runs', COUNT(*) FROM model_runs
UNION ALL SELECT 'model_predictions', COUNT(*) FROM model_predictions
UNION ALL SELECT 'trades', COUNT(*) FROM trades
UNION ALL SELECT 'positions', COUNT(*) FROM positions
UNION ALL SELECT 'pnl_daily', COUNT(*) FROM pnl_daily
UNION ALL SELECT 'performance_summary', COUNT(*) FROM performance_summary;

CREATE VIEW IF NOT EXISTS v_pipeline_coverage AS
SELECT
    t.triplet_id,
    t.target_symbol,
    t.hedge_symbol_1,
    t.hedge_symbol_2,
    COUNT(DISTINCT r.date) AS residual_days,
    COUNT(DISTINCT e.event_id) AS candidate_events,
    COUNT(DISTINCT l.event_id) AS labeled_events,
    COUNT(DISTINCT p.prediction_id) AS model_predictions,
    COUNT(DISTINCT tr.trade_id) AS trades
FROM triplets t
LEFT JOIN residuals r ON r.triplet_id = t.triplet_id
LEFT JOIN candidate_events e ON e.triplet_id = t.triplet_id
LEFT JOIN event_labels l ON l.triplet_id = t.triplet_id
LEFT JOIN model_predictions p ON p.event_id = e.event_id
LEFT JOIN trades tr ON tr.event_id = e.event_id
GROUP BY t.triplet_id, t.target_symbol, t.hedge_symbol_1, t.hedge_symbol_2;


-- Legacy stage-specific tables and views retained for backward compatibility.

CREATE TABLE IF NOT EXISTS rolling_coefficients (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta_1 REAL NOT NULL,
    beta_2 REAL NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    window INTEGER NOT NULL,
    method TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, method)
);

CREATE TABLE IF NOT EXISTS ridge_coefficients (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta_1 REAL NOT NULL,
    beta_2 REAL NOT NULL,
    train_start TEXT NOT NULL,
    train_end TEXT NOT NULL,
    window INTEGER NOT NULL,
    ridge_alpha REAL NOT NULL,
    method TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, method, ridge_alpha)
);

CREATE TABLE IF NOT EXISTS dynamic_residuals (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    method TEXT NOT NULL,
    actual_log_price REAL NOT NULL,
    fitted_log_price REAL NOT NULL,
    residual REAL NOT NULL,
    window INTEGER NOT NULL,
    ridge_alpha REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, method)
);

CREATE TABLE IF NOT EXISTS residual_method_summary (
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    n_obs INTEGER NOT NULL,
    residual_mean REAL NOT NULL,
    residual_std REAL NOT NULL,
    residual_abs_mean REAL NOT NULL,
    residual_min REAL NOT NULL,
    residual_q05 REAL NOT NULL,
    residual_median REAL NOT NULL,
    residual_q95 REAL NOT NULL,
    residual_max REAL NOT NULL,
    autocorr_1 REAL,
    static_residual_std REAL,
    static_abs_mean REAL,
    std_ratio_vs_static REAL,
    abs_mean_ratio_vs_static REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (triplet_id, method)
);

CREATE INDEX IF NOT EXISTS idx_rolling_coefficients_triplet_date
ON rolling_coefficients (triplet_id, date);

CREATE INDEX IF NOT EXISTS idx_ridge_coefficients_triplet_date
ON ridge_coefficients (triplet_id, date);

CREATE INDEX IF NOT EXISTS idx_dynamic_residuals_triplet_method_date
ON dynamic_residuals (triplet_id, method, date);

CREATE VIEW IF NOT EXISTS v_latest_residual_method_summary AS
SELECT
    triplet_id,
    method,
    n_obs,
    residual_mean,
    residual_std,
    residual_abs_mean,
    autocorr_1,
    std_ratio_vs_static,
    abs_mean_ratio_vs_static
FROM residual_method_summary;

CREATE TABLE IF NOT EXISTS kalman_states (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    alpha REAL NOT NULL,
    beta_1 REAL NOT NULL,
    beta_2 REAL NOT NULL,
    predicted_alpha REAL NOT NULL,
    predicted_beta_1 REAL NOT NULL,
    predicted_beta_2 REAL NOT NULL,
    kalman_gain_alpha REAL NOT NULL,
    kalman_gain_beta_1 REAL NOT NULL,
    kalman_gain_beta_2 REAL NOT NULL,
    state_cov_trace REAL NOT NULL,
    process_noise REAL NOT NULL,
    measurement_noise REAL NOT NULL,
    initial_window INTEGER,
    method TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, method, process_noise, measurement_noise)
);

CREATE TABLE IF NOT EXISTS kalman_residuals (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    target_symbol TEXT NOT NULL,
    hedge_symbol_1 TEXT NOT NULL,
    hedge_symbol_2 TEXT NOT NULL,
    method TEXT NOT NULL,
    actual_log_price REAL NOT NULL,
    fitted_log_price REAL NOT NULL,
    residual REAL NOT NULL,
    residual_variance REAL NOT NULL,
    process_noise REAL NOT NULL,
    measurement_noise REAL NOT NULL,
    initial_window INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, method, process_noise, measurement_noise)
);

CREATE INDEX IF NOT EXISTS idx_kalman_states_triplet_date
ON kalman_states (triplet_id, date);

CREATE INDEX IF NOT EXISTS idx_kalman_residuals_triplet_date
ON kalman_residuals (triplet_id, date);

CREATE VIEW IF NOT EXISTS v_kalman_state_summary AS
SELECT
    triplet_id,
    method,
    COUNT(*) AS n_obs,
    AVG(alpha) AS avg_alpha,
    AVG(beta_1) AS avg_beta_1,
    AVG(beta_2) AS avg_beta_2,
    AVG(state_cov_trace) AS avg_state_cov_trace,
    AVG(residual_variance) AS avg_residual_variance
FROM kalman_states
LEFT JOIN kalman_residuals USING (date, triplet_id, method, process_noise, measurement_noise)
GROUP BY triplet_id, method;

CREATE TABLE IF NOT EXISTS candidate_trade_events (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    target_symbol TEXT,
    hedge_symbol_1 TEXT,
    hedge_symbol_2 TEXT,
    side TEXT NOT NULL CHECK (side IN ('long_spread', 'short_spread')),
    entry_z_score REAL NOT NULL,
    entry_abs_z REAL NOT NULL,
    entry_residual REAL NOT NULL,
    entry_threshold REAL NOT NULL,
    exit_z REAL NOT NULL,
    stop_loss_z REAL NOT NULL,
    max_holding_period INTEGER NOT NULL,
    z_window INTEGER,
    event_row INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_labels (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    exit_date TEXT,
    side TEXT NOT NULL CHECK (side IN ('long_spread', 'short_spread')),
    label INTEGER NOT NULL CHECK (label IN (0, 1)),
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure')),
    exit_reason TEXT NOT NULL,
    holding_period INTEGER NOT NULL,
    entry_z_score REAL NOT NULL,
    entry_abs_z REAL NOT NULL,
    exit_z_score REAL,
    entry_residual REAL NOT NULL,
    exit_residual REAL,
    entry_threshold REAL NOT NULL,
    exit_z REAL NOT NULL,
    stop_loss_z REAL NOT NULL,
    max_holding_period INTEGER NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES candidate_trade_events(event_id)
);

CREATE TABLE IF NOT EXISTS event_label_summary (
    triplet_id TEXT,
    method TEXT,
    n_events INTEGER NOT NULL,
    success_count INTEGER NOT NULL,
    failure_count INTEGER NOT NULL,
    success_rate REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_candidate_trade_events_triplet_date
ON candidate_trade_events (triplet_id, method, event_date);

CREATE INDEX IF NOT EXISTS idx_event_labels_triplet_date
ON event_labels (triplet_id, method, event_date);

CREATE INDEX IF NOT EXISTS idx_event_labels_outcome
ON event_labels (method, outcome, exit_reason);

CREATE VIEW IF NOT EXISTS v_event_label_distribution AS
SELECT
    method,
    outcome,
    exit_reason,
    COUNT(*) AS n_events
FROM event_labels
GROUP BY method, outcome, exit_reason;

CREATE VIEW IF NOT EXISTS v_success_rate_by_triplet AS
SELECT
    triplet_id,
    method,
    COUNT(*) AS n_events,
    SUM(label) AS success_count,
    COUNT(*) - SUM(label) AS failure_count,
    CAST(SUM(label) AS REAL) / COUNT(*) AS success_rate
FROM event_labels
GROUP BY triplet_id, method;

CREATE TABLE IF NOT EXISTS event_feature_matrix (
    event_id TEXT PRIMARY KEY,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    target_symbol TEXT,
    hedge_symbol_1 TEXT,
    hedge_symbol_2 TEXT,
    side TEXT,
    entry_z_score REAL,
    entry_abs_z REAL,
    z_score REAL,
    residual REAL,
    residual_change REAL,
    residual_volatility REAL,
    residual_autocorrelation REAL,
    half_life_estimate REAL,
    rolling_r_squared REAL,
    beta_1 REAL,
    beta_2 REAL,
    beta_1_stability REAL,
    beta_2_stability REAL,
    beta_stability REAL,
    target_return_volatility REAL,
    anchor_1_return_volatility REAL,
    anchor_2_return_volatility REAL,
    market_return REAL,
    sector_return REAL,
    target_anchor_1_correlation REAL,
    target_anchor_2_correlation REAL,
    anchor_correlation REAL,
    correlation_stability REAL,
    recent_drawdown REAL,
    distance_from_moving_average REAL,
    volume_shock REAL,
    label INTEGER CHECK (label IN (0, 1)),
    outcome TEXT,
    exit_reason TEXT,
    holding_period INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (event_id) REFERENCES candidate_trade_events(event_id)
);

CREATE TABLE IF NOT EXISTS feature_summary_statistics (
    feature TEXT PRIMARY KEY,
    count REAL,
    mean REAL,
    std REAL,
    min REAL,
    p25 REAL,
    median REAL,
    p75 REAL,
    max REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_missingness_report (
    column TEXT PRIMARY KEY,
    missing_count INTEGER NOT NULL,
    missing_rate REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_event_feature_matrix_triplet_date
ON event_feature_matrix (triplet_id, method, event_date);

CREATE INDEX IF NOT EXISTS idx_event_feature_matrix_label
ON event_feature_matrix (method, label);

CREATE VIEW IF NOT EXISTS v_feature_label_summary AS
SELECT
    method,
    COUNT(*) AS n_events,
    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS success_count,
    SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS failure_count,
    AVG(label) AS success_rate
FROM event_feature_matrix
WHERE label IS NOT NULL
GROUP BY method;

CREATE TABLE IF NOT EXISTS logistic_model_coefficients (
    feature TEXT PRIMARY KEY,
    coefficient REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logistic_training_loss (
    iteration INTEGER PRIMARY KEY,
    loss REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logistic_event_predictions (
    event_id TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    method TEXT NOT NULL,
    event_date TEXT NOT NULL,
    split TEXT NOT NULL,
    predicted_reversion_probability REAL NOT NULL,
    classification_threshold REAL NOT NULL,
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    label INTEGER CHECK (label IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, split),
    FOREIGN KEY (event_id) REFERENCES candidate_trade_events(event_id)
);

CREATE TABLE IF NOT EXISTS logistic_validation_metrics (
    split TEXT PRIMARY KEY,
    n_obs INTEGER NOT NULL,
    positive_rate REAL,
    threshold REAL NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    log_loss REAL,
    brier_score REAL,
    roc_auc REAL,
    true_positive INTEGER,
    true_negative INTEGER,
    false_positive INTEGER,
    false_negative INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logistic_probability_calibration (
    probability_bucket TEXT NOT NULL,
    n_events INTEGER NOT NULL,
    mean_predicted_probability REAL,
    observed_success_rate REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (probability_bucket)
);

CREATE INDEX IF NOT EXISTS idx_logistic_event_predictions_triplet_date
ON logistic_event_predictions (triplet_id, method, event_date);

CREATE INDEX IF NOT EXISTS idx_logistic_event_predictions_probability
ON logistic_event_predictions (split, predicted_reversion_probability);

CREATE VIEW IF NOT EXISTS v_logistic_prediction_summary AS
SELECT
    split,
    COUNT(*) AS n_events,
    AVG(predicted_reversion_probability) AS mean_predicted_reversion_probability,
    AVG(label) AS observed_success_rate,
    AVG(predicted_label) AS predicted_positive_rate
FROM logistic_event_predictions
GROUP BY split;

CREATE TABLE IF NOT EXISTS model_evaluation_summary (
    split TEXT NOT NULL,
    n_obs INTEGER NOT NULL,
    positive_rate REAL,
    threshold REAL NOT NULL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    specificity REAL,
    negative_predictive_value REAL,
    f1 REAL,
    log_loss REAL,
    brier_score REAL,
    roc_auc REAL,
    true_positive INTEGER,
    true_negative INTEGER,
    false_positive INTEGER,
    false_negative INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (split, threshold)
);

CREATE TABLE IF NOT EXISTS model_confusion_matrix (
    split TEXT,
    actual_label INTEGER NOT NULL CHECK (actual_label IN (0, 1)),
    predicted_label INTEGER NOT NULL CHECK (predicted_label IN (0, 1)),
    count INTEGER NOT NULL,
    threshold REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (split, actual_label, predicted_label, threshold)
);

CREATE TABLE IF NOT EXISTS probability_bucket_summary (
    split TEXT,
    probability_bucket TEXT NOT NULL,
    bucket_lower REAL NOT NULL,
    bucket_upper REAL NOT NULL,
    n_events INTEGER NOT NULL,
    success_count INTEGER,
    failure_count INTEGER,
    mean_predicted_probability REAL,
    realized_success_rate REAL,
    precision REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (split, probability_bucket)
);

CREATE TABLE IF NOT EXISTS model_calibration_curve (
    split TEXT,
    probability_bucket TEXT NOT NULL,
    n_events INTEGER NOT NULL,
    mean_predicted_probability REAL,
    realized_success_rate REAL,
    calibration_error REAL,
    absolute_calibration_error REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (split, probability_bucket)
);

CREATE VIEW IF NOT EXISTS v_model_evaluation_latest AS
SELECT
    split,
    n_obs,
    positive_rate,
    threshold,
    accuracy,
    precision,
    recall,
    brier_score,
    roc_auc,
    true_positive,
    false_positive,
    true_negative,
    false_negative
FROM model_evaluation_summary;

CREATE VIEW IF NOT EXISTS v_probability_bucket_success AS
SELECT
    split,
    probability_bucket,
    n_events,
    mean_predicted_probability,
    realized_success_rate,
    precision
FROM probability_bucket_summary;

CREATE TABLE IF NOT EXISTS ml_backtest_trade_log (
    event_id TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    method TEXT,
    event_date TEXT NOT NULL,
    exit_date TEXT,
    pnl_date TEXT,
    split TEXT,
    strategy TEXT NOT NULL,
    side TEXT NOT NULL,
    label INTEGER CHECK (label IN (0, 1)),
    outcome TEXT,
    exit_reason TEXT,
    predicted_reversion_probability REAL,
    entry_z_score REAL,
    exit_z_score REAL,
    spread_pnl_units REAL,
    position_size REAL,
    gross_pnl REAL,
    transaction_cost REAL,
    net_pnl REAL,
    turnover REAL,
    holding_period INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, strategy)
);

CREATE TABLE IF NOT EXISTS ml_backtest_equity_curve (
    date TEXT NOT NULL,
    strategy TEXT NOT NULL,
    daily_pnl REAL,
    equity REAL,
    running_peak REAL,
    drawdown REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, strategy)
);

CREATE TABLE IF NOT EXISTS ml_backtest_summary (
    strategy TEXT PRIMARY KEY,
    trade_count INTEGER,
    gross_pnl REAL,
    net_pnl REAL,
    average_net_pnl REAL,
    median_net_pnl REAL,
    win_rate REAL,
    turnover REAL,
    max_drawdown REAL,
    sharpe REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS avoided_bad_trades_analysis (
    probability_threshold REAL PRIMARY KEY,
    avoided_trade_count INTEGER,
    avoided_failure_count INTEGER,
    avoided_success_count INTEGER,
    avoided_failure_rate REAL,
    avoided_baseline_net_pnl REAL,
    avoided_average_probability REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS v_ml_backtest_strategy_comparison AS
SELECT
    strategy,
    trade_count,
    gross_pnl,
    net_pnl,
    win_rate,
    turnover,
    max_drawdown,
    sharpe
FROM ml_backtest_summary;

CREATE TABLE IF NOT EXISTS decision_tree_event_predictions (
    event_id TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    method TEXT,
    event_date TEXT NOT NULL,
    split TEXT,
    model_type TEXT NOT NULL,
    predicted_reversion_probability REAL NOT NULL,
    classification_threshold REAL NOT NULL,
    predicted_label INTEGER CHECK (predicted_label IN (0, 1)),
    label INTEGER CHECK (label IN (0, 1)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (event_id, split, model_type)
);

CREATE TABLE IF NOT EXISTS decision_tree_feature_splits (
    node_id INTEGER PRIMARY KEY,
    depth INTEGER NOT NULL,
    feature TEXT,
    threshold REAL,
    information_gain REAL,
    n_samples INTEGER,
    positive_rate REAL,
    left_node_id INTEGER,
    right_node_id INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decision_tree_logistic_comparison (
    model_type TEXT PRIMARY KEY,
    n_obs INTEGER,
    positive_rate REAL,
    threshold REAL,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1 REAL,
    brier_score REAL,
    roc_auc REAL,
    true_positive INTEGER,
    false_positive INTEGER,
    true_negative INTEGER,
    false_negative INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE VIEW IF NOT EXISTS v_decision_tree_split_importance AS
SELECT
    feature,
    COUNT(*) AS split_count,
    SUM(information_gain) AS total_information_gain,
    AVG(information_gain) AS average_information_gain
FROM decision_tree_feature_splits
GROUP BY feature
ORDER BY total_information_gain DESC;

CREATE TABLE IF NOT EXISTS hmm_regime_probabilities (
    date TEXT NOT NULL,
    triplet_id TEXT NOT NULL,
    method TEXT,
    model_type TEXT NOT NULL,
    feature_column TEXT,
    feature_value REAL,
    state_0_probability REAL,
    state_1_probability REAL,
    state_2_probability REAL,
    state_0_label TEXT,
    state_1_label TEXT,
    state_2_label TEXT,
    mean_reverting_probability REAL,
    trending_probability REAL,
    volatile_breakdown_probability REAL,
    most_likely_state INTEGER,
    most_likely_regime TEXT,
    viterbi_state INTEGER,
    viterbi_regime TEXT,
    log_likelihood REAL,
    n_states INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, triplet_id, model_type, feature_column)
);

CREATE TABLE IF NOT EXISTS hmm_regime_parameters (
    triplet_id TEXT NOT NULL,
    model_type TEXT NOT NULL,
    state INTEGER NOT NULL,
    regime_label TEXT,
    mean REAL,
    variance REAL,
    start_probability REAL,
    transition_to_state_0 REAL,
    transition_to_state_1 REAL,
    transition_to_state_2 REAL,
    log_likelihood REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (triplet_id, model_type, state)
);

CREATE TABLE IF NOT EXISTS hmm_strategy_performance_by_regime (
    strategy TEXT NOT NULL,
    regime_bucket TEXT NOT NULL,
    trade_count INTEGER,
    average_mean_reverting_probability REAL,
    success_rate REAL,
    net_pnl REAL,
    average_net_pnl REAL,
    turnover REAL,
    regime_probability_threshold REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy, regime_bucket)
);

CREATE VIEW IF NOT EXISTS v_hmm_latest_regime AS
SELECT
    triplet_id,
    MAX(date) AS latest_date,
    AVG(mean_reverting_probability) AS average_mean_reverting_probability,
    AVG(volatile_breakdown_probability) AS average_volatile_breakdown_probability
FROM hmm_regime_probabilities
GROUP BY triplet_id;

CREATE TABLE IF NOT EXISTS c_kernel_validation (
    output_column TEXT NOT NULL,
    max_abs_diff REAL,
    mean_abs_diff REAL,
    n_compared INTEGER,
    backend_pair TEXT DEFAULT 'python_vs_c',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (output_column, backend_pair)
);

CREATE TABLE IF NOT EXISTS c_kernel_benchmark (
    backend TEXT NOT NULL,
    repeats INTEGER,
    mean_seconds REAL,
    min_seconds REAL,
    max_seconds REAL,
    sample_rows INTEGER,
    window INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (backend, sample_rows, window)
);

CREATE TABLE IF NOT EXISTS c_kernel_test_results (
    test_name TEXT NOT NULL PRIMARY KEY,
    status TEXT NOT NULL,
    detail TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_c_kernel_validation_column
ON c_kernel_validation (output_column);

CREATE INDEX IF NOT EXISTS idx_c_kernel_benchmark_backend
ON c_kernel_benchmark (backend);

CREATE TABLE IF NOT EXISTS transaction_cost_sensitivity (
    cost_scenario TEXT NOT NULL,
    strategy TEXT NOT NULL,
    trade_count INTEGER,
    gross_pnl REAL,
    net_pnl REAL,
    average_net_pnl REAL,
    median_net_pnl REAL,
    win_rate REAL,
    turnover REAL,
    max_drawdown REAL,
    sharpe REAL,
    total_cost_per_unit REAL,
    commission_per_trade REAL,
    bid_ask_spread_proxy REAL,
    slippage REAL,
    cost_drag REAL,
    gross_to_net_pnl_delta REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (cost_scenario, strategy)
);

CREATE TABLE IF NOT EXISTS threshold_sensitivity (
    strategy TEXT NOT NULL,
    entry_threshold REAL NOT NULL,
    exit_threshold REAL NOT NULL,
    stop_loss_level REAL NOT NULL,
    max_holding_period INTEGER NOT NULL,
    probability_threshold REAL,
    transaction_cost REAL,
    trade_count INTEGER,
    gross_pnl REAL,
    net_pnl REAL,
    average_net_pnl REAL,
    median_net_pnl REAL,
    win_rate REAL,
    turnover REAL,
    max_drawdown REAL,
    sharpe REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (strategy, entry_threshold, exit_threshold, stop_loss_level, max_holding_period)
);

CREATE TABLE IF NOT EXISTS robustness_summary (
    strategy TEXT NOT NULL PRIMARY KEY,
    threshold_scenarios INTEGER,
    average_net_pnl REAL,
    median_net_pnl REAL,
    worst_net_pnl REAL,
    best_net_pnl REAL,
    profitable_scenario_rate REAL,
    average_sharpe REAL,
    worst_drawdown REAL,
    average_turnover REAL,
    minimum_trade_count REAL,
    maximum_trade_count REAL,
    cost_scenarios INTEGER,
    worst_cost_adjusted_net_pnl REAL,
    average_cost_drag REAL,
    maximum_cost_drag REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_transaction_cost_sensitivity_strategy
ON transaction_cost_sensitivity (strategy, cost_scenario);

CREATE INDEX IF NOT EXISTS idx_threshold_sensitivity_strategy_grid
ON threshold_sensitivity (strategy, entry_threshold, exit_threshold, stop_loss_level, max_holding_period);
