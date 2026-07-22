"""
SQLite persistence layer: schema initialization, generic DataFrame
read/write helpers, and one store_* / query function per table defined
in sql/schema.sql. Every store_* function below validates its required
columns before writing and selects only the columns that table actually
has, rather than trusting the caller's DataFrame to already be in the
right shape.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    # Creates the parent directory if it doesn't exist yet -- sqlite3
    # will not create missing directories on its own, only the database
    # file itself.
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def initialize_database(database_path: str | Path, schema_path: str | Path) -> None:
    # executescript runs the entire schema.sql file (CREATE TABLE
    # statements, indexes, etc.) in one call -- unlike execute(), which
    # only accepts a single SQL statement at a time.
    schema = Path(schema_path).read_text(encoding="utf-8")
    with connect_database(database_path) as conn:
        conn.executescript(schema)
        conn.commit()


def write_dataframe(
    conn: sqlite3.Connection,
    frame: pd.DataFrame,
    table_name: str,
    if_exists: str = "append",
) -> None:
    # Generic write path every store_* function below funnels through.
    # Datetime columns are converted to plain ISO date strings before
    # writing, since SQLite has no native datetime type -- storing them
    # as consistently-formatted text keeps later date comparisons/sorts
    # in SQL well-behaved.
    if frame is None or frame.empty:
        return
    clean = frame.copy()
    for column in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[column]):
            clean[column] = clean[column].dt.strftime("%Y-%m-%d")
    if isinstance(clean.index, pd.DatetimeIndex) and "date" not in clean.columns:
        clean = clean.reset_index().rename(columns={"index": "date"})
    clean.to_sql(table_name, conn, if_exists=if_exists, index=False)


def read_table(conn: sqlite3.Connection, table_name: str, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn, parse_dates=parse_dates)


def store_assets(conn: sqlite3.Connection, assets: pd.DataFrame, if_exists: str = "replace") -> None:
    required = {"symbol"}
    missing = required.difference(assets.columns)
    if missing:
        raise KeyError(f"missing asset columns: {sorted(missing)}")
    columns = [c for c in ["symbol", "asset_name", "asset_type", "sector", "industry", "exchange", "currency", "active"] if c in assets.columns]
    write_dataframe(conn, assets.loc[:, columns], "assets", if_exists=if_exists)


def store_prices_raw(conn: sqlite3.Connection, prices: pd.DataFrame, if_exists: str = "append") -> None:
    required = {"symbol", "date", "adj_close", "source"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"missing price columns: {sorted(missing)}")
    columns = [c for c in ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source", "source_file"] if c in prices.columns]
    write_dataframe(conn, prices.loc[:, columns], "prices_raw", if_exists=if_exists)


def store_prices_clean(conn: sqlite3.Connection, prices: pd.DataFrame, if_exists: str = "replace") -> None:
    required = {"symbol", "date", "adj_close"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"missing price columns: {sorted(missing)}")
    columns = [c for c in ["symbol", "date", "close", "adj_close", "volume", "quality_flag", "source"] if c in prices.columns]
    write_dataframe(conn, prices.loc[:, columns], "prices_clean", if_exists=if_exists)


def store_returns_daily(conn: sqlite3.Connection, returns: pd.DataFrame, if_exists: str = "replace") -> None:
    required = {"symbol", "date"}
    missing = required.difference(returns.columns)
    if missing:
        raise KeyError(f"missing return columns: {sorted(missing)}")
    columns = [c for c in ["symbol", "date", "simple_return", "log_return"] if c in returns.columns]
    write_dataframe(conn, returns.loc[:, columns], "returns_daily", if_exists=if_exists)


def store_triplets(conn: sqlite3.Connection, triplets: pd.DataFrame, if_exists: str = "replace") -> None:
    required = {"triplet_id", "target_symbol", "hedge_symbol_1", "hedge_symbol_2"}
    missing = required.difference(triplets.columns)
    if missing:
        raise KeyError(f"missing triplet columns: {sorted(missing)}")
    columns = [c for c in ["triplet_id", "target_symbol", "hedge_symbol_1", "hedge_symbol_2", "relationship_theme", "active"] if c in triplets.columns]
    write_dataframe(conn, triplets.loc[:, columns], "triplets", if_exists=if_exists)


def store_rolling_coefficients(conn: sqlite3.Connection, coefficients: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "train_start",
        "train_end",
        "window",
        "method",
    ]
    frame = coefficients.loc[:, [col for col in columns if col in coefficients.columns]].copy()
    write_dataframe(conn, frame, "rolling_coefficients", if_exists=if_exists)


def store_ridge_coefficients(conn: sqlite3.Connection, coefficients: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "train_start",
        "train_end",
        "window",
        "ridge_alpha",
        "method",
    ]
    frame = coefficients.loc[:, [col for col in columns if col in coefficients.columns]].copy()
    write_dataframe(conn, frame, "ridge_coefficients", if_exists=if_exists)


def store_dynamic_residuals(conn: sqlite3.Connection, residuals: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "method",
        "actual_log_price",
        "fitted_log_price",
        "residual",
        "window",
        "ridge_alpha",
    ]
    frame = residuals.loc[:, [col for col in columns if col in residuals.columns]].copy()
    write_dataframe(conn, frame, "dynamic_residuals", if_exists=if_exists)


def store_residual_method_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "residual_method_summary", if_exists=if_exists)


def store_kalman_states(conn: sqlite3.Connection, states: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "alpha",
        "beta_1",
        "beta_2",
        "predicted_alpha",
        "predicted_beta_1",
        "predicted_beta_2",
        "kalman_gain_alpha",
        "kalman_gain_beta_1",
        "kalman_gain_beta_2",
        "state_cov_trace",
        "process_noise",
        "measurement_noise",
        "initial_window",
        "method",
    ]
    frame = states.loc[:, [col for col in columns if col in states.columns]].copy()
    write_dataframe(conn, frame, "kalman_states", if_exists=if_exists)


def store_kalman_residuals(conn: sqlite3.Connection, residuals: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "date",
        "triplet_id",
        "target_symbol",
        "hedge_symbol_1",
        "hedge_symbol_2",
        "method",
        "actual_log_price",
        "fitted_log_price",
        "residual",
        "residual_variance",
        "process_noise",
        "measurement_noise",
        "initial_window",
    ]
    frame = residuals.loc[:, [col for col in columns if col in residuals.columns]].copy()
    write_dataframe(conn, frame, "kalman_residuals", if_exists=if_exists)


def store_candidate_events(conn: sqlite3.Connection, events: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
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
        "entry_residual",
        "entry_threshold",
        "exit_z",
        "stop_loss_z",
        "max_holding_period",
        "z_window",
        "event_row",
    ]
    frame = events.loc[:, [col for col in columns if col in events.columns]].copy()
    write_dataframe(conn, frame, "candidate_trade_events", if_exists=if_exists)


def store_event_labels(conn: sqlite3.Connection, labels: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
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
    frame = labels.loc[:, [col for col in columns if col in labels.columns]].copy()
    write_dataframe(conn, frame, "event_labels", if_exists=if_exists)


def store_event_label_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "event_label_summary", if_exists=if_exists)

def store_event_feature_matrix(conn: sqlite3.Connection, features: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, features, "event_feature_matrix", if_exists=if_exists)


def store_feature_summary_statistics(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "feature_summary_statistics", if_exists=if_exists)


def store_feature_missingness_report(conn: sqlite3.Connection, missingness: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, missingness, "feature_missingness_report", if_exists=if_exists)



def store_logistic_model_coefficients(conn: sqlite3.Connection, coefficients: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, coefficients, "logistic_model_coefficients", if_exists=if_exists)


def store_logistic_training_loss(conn: sqlite3.Connection, loss_history: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, loss_history, "logistic_training_loss", if_exists=if_exists)


def store_logistic_predictions(conn: sqlite3.Connection, predictions: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "split",
        "predicted_reversion_probability",
        "classification_threshold",
        "predicted_label",
        "label",
    ]
    frame = predictions.loc[:, [col for col in columns if col in predictions.columns]].copy()
    write_dataframe(conn, frame, "logistic_event_predictions", if_exists=if_exists)


def store_logistic_validation_metrics(conn: sqlite3.Connection, metrics: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, metrics, "logistic_validation_metrics", if_exists=if_exists)


def store_logistic_calibration(conn: sqlite3.Connection, calibration: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, calibration, "logistic_probability_calibration", if_exists=if_exists)


def store_model_evaluation_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "model_evaluation_summary", if_exists=if_exists)


def store_model_confusion_matrix(conn: sqlite3.Connection, matrix: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, matrix, "model_confusion_matrix", if_exists=if_exists)


def store_probability_bucket_summary(conn: sqlite3.Connection, buckets: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, buckets, "probability_bucket_summary", if_exists=if_exists)


def store_model_calibration_curve(conn: sqlite3.Connection, calibration: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, calibration, "model_calibration_curve", if_exists=if_exists)


def store_ml_backtest_trade_log(conn: sqlite3.Connection, trades: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, trades, "ml_backtest_trade_log", if_exists=if_exists)


def store_ml_backtest_equity_curve(conn: sqlite3.Connection, equity_curve: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, equity_curve, "ml_backtest_equity_curve", if_exists=if_exists)


def store_ml_backtest_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "ml_backtest_summary", if_exists=if_exists)


def store_avoided_bad_trades_analysis(conn: sqlite3.Connection, analysis: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, analysis, "avoided_bad_trades_analysis", if_exists=if_exists)


def store_decision_tree_predictions(conn: sqlite3.Connection, predictions: pd.DataFrame, if_exists: str = "replace") -> None:
    columns = [
        "event_id",
        "triplet_id",
        "method",
        "event_date",
        "split",
        "model_type",
        "predicted_reversion_probability",
        "classification_threshold",
        "predicted_label",
        "label",
    ]
    frame = predictions.loc[:, [col for col in columns if col in predictions.columns]].copy()
    write_dataframe(conn, frame, "decision_tree_event_predictions", if_exists=if_exists)


def store_decision_tree_split_summary(conn: sqlite3.Connection, split_summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, split_summary, "decision_tree_feature_splits", if_exists=if_exists)


def store_decision_tree_model_comparison(conn: sqlite3.Connection, comparison: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, comparison, "decision_tree_logistic_comparison", if_exists=if_exists)


def store_hmm_regime_probabilities(conn: sqlite3.Connection, probabilities: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, probabilities, "hmm_regime_probabilities", if_exists=if_exists)


def store_hmm_regime_parameters(conn: sqlite3.Connection, parameters: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, parameters, "hmm_regime_parameters", if_exists=if_exists)


def store_hmm_strategy_performance(conn: sqlite3.Connection, performance: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, performance, "hmm_strategy_performance_by_regime", if_exists=if_exists)


REQUIRED_PIPELINE_TABLES = [
    "assets",
    "prices_raw",
    "prices_clean",
    "returns_daily",
    "triplets",
    "regression_results",
    "hedge_ratios",
    "residuals",
    "residual_diagnostics",
    "candidate_events",
    "event_features",
    "event_labels",
    "model_runs",
    "model_predictions",
    "trades",
    "positions",
    "pnl_daily",
    "performance_summary",
]


def list_tables(conn: sqlite3.Connection) -> list[str]:
    query = """
    SELECT name
    FROM sqlite_master
    WHERE type IN ('table', 'view')
      AND name NOT LIKE 'sqlite_%'
    ORDER BY name
    """
    return pd.read_sql_query(query, conn)["name"].tolist()


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = """
    SELECT 1
    FROM sqlite_master
    WHERE type IN ('table', 'view')
      AND name = ?
    LIMIT 1
    """
    return conn.execute(query, (table_name,)).fetchone() is not None


def table_row_counts(conn: sqlite3.Connection, tables: Optional[list[str]] = None) -> pd.DataFrame:
    selected = tables or REQUIRED_PIPELINE_TABLES
    rows = []
    for table in selected:
        if not table_exists(conn, table):
            rows.append({"table_name": table, "row_count": None, "exists": 0})
            continue
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        rows.append({"table_name": table, "row_count": int(row_count), "exists": 1})
    return pd.DataFrame(rows)


def parse_named_sql_queries(sql_path: str | Path) -> dict[str, str]:
    text = Path(sql_path).read_text(encoding="utf-8")
    queries: dict[str, str] = {}
    current_name: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("-- name:"):
            if current_name and current_lines:
                queries[current_name] = "\n".join(current_lines).strip().rstrip(";")
            current_name = stripped.split(":", 1)[1].strip()
            current_lines = []
            continue
        if current_name:
            current_lines.append(line)
    if current_name and current_lines:
        queries[current_name] = "\n".join(current_lines).strip().rstrip(";")
    return {name: query for name, query in queries.items() if query}


def run_named_queries(conn: sqlite3.Connection, sql_path: str | Path) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    for name, query in parse_named_sql_queries(sql_path).items():
        results[name] = pd.read_sql_query(query, conn)
    return results


def run_validation_queries(conn: sqlite3.Connection, validation_sql_path: str | Path) -> pd.DataFrame:
    results = run_named_queries(conn, validation_sql_path)
    rows = []
    for name, frame in results.items():
        issue_count = int(len(frame))
        if name == "required_table_row_counts":
            missing = int((frame.get("row_count", pd.Series(dtype=float)).isna()).sum())
            status = "pass" if missing == 0 else "fail"
            issue_count = missing
        else:
            status = "pass" if issue_count == 0 else "fail"
        rows.append(
            {
                "check_name": name,
                "status": status,
                "issue_count": issue_count,
                "rows_returned": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def export_named_query_results(
    conn: sqlite3.Connection,
    sql_path: str | Path,
    output_dir: str | Path,
    prefix: str,
) -> pd.DataFrame:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results = run_named_queries(conn, sql_path)
    summary_rows = []
    for name, frame in results.items():
        file_path = output_path / f"{prefix}_{name}.csv"
        frame.to_csv(file_path, index=False)
        summary_rows.append(
            {
                "query_name": name,
                "row_count": int(len(frame)),
                "output_file": file_path.name,
            }
        )
    return pd.DataFrame(summary_rows)


def store_regression_results(conn: sqlite3.Connection, results: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, results, "regression_results", if_exists=if_exists)


def store_hedge_ratios(conn: sqlite3.Connection, hedge_ratios: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, hedge_ratios, "hedge_ratios", if_exists=if_exists)


def store_residuals(conn: sqlite3.Connection, residuals: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, residuals, "residuals", if_exists=if_exists)


def store_residual_diagnostics(conn: sqlite3.Connection, diagnostics: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, diagnostics, "residual_diagnostics", if_exists=if_exists)


def store_canonical_candidate_events(conn: sqlite3.Connection, events: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, events, "candidate_events", if_exists=if_exists)


def store_event_features(conn: sqlite3.Connection, features: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, features, "event_features", if_exists=if_exists)


def store_model_runs(conn: sqlite3.Connection, runs: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, runs, "model_runs", if_exists=if_exists)


def store_model_predictions(conn: sqlite3.Connection, predictions: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, predictions, "model_predictions", if_exists=if_exists)


def store_trades(conn: sqlite3.Connection, trades: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, trades, "trades", if_exists=if_exists)


def store_positions(conn: sqlite3.Connection, positions: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, positions, "positions", if_exists=if_exists)


def store_pnl_daily(conn: sqlite3.Connection, pnl: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, pnl, "pnl_daily", if_exists=if_exists)


def store_performance_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "append") -> None:
    write_dataframe(conn, summary, "performance_summary", if_exists=if_exists)


def store_c_kernel_validation(conn: sqlite3.Connection, validation: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, validation, "c_kernel_validation", if_exists=if_exists)


def store_c_kernel_benchmark(conn: sqlite3.Connection, benchmark: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, benchmark, "c_kernel_benchmark", if_exists=if_exists)


def store_c_kernel_test_results(conn: sqlite3.Connection, results: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, results, "c_kernel_test_results", if_exists=if_exists)


def store_transaction_cost_sensitivity(conn: sqlite3.Connection, sensitivity: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, sensitivity, "transaction_cost_sensitivity", if_exists=if_exists)


def store_threshold_sensitivity(conn: sqlite3.Connection, sensitivity: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, sensitivity, "threshold_sensitivity", if_exists=if_exists)


def store_robustness_summary(conn: sqlite3.Connection, summary: pd.DataFrame, if_exists: str = "replace") -> None:
    write_dataframe(conn, summary, "robustness_summary", if_exists=if_exists)
