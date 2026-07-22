# Phase 16: SQL Database Design and Validation

This phase finalizes the database layer for the triangular stat-arb research system. The goal is not to make SQL a cosmetic storage layer. The database is the audit trail for market data, transformations, residuals, event labels, model predictions, trades, daily PnL, and performance summaries.

## Design goals

The schema is built around four requirements:

1. Store every major pipeline output in a durable table.
2. Use primary keys and foreign-key relationships to make entity links explicit.
3. Add indexes on the columns used most often in joins, filters, and time-series reports.
4. Provide validation and reporting SQL files that can be run after each pipeline refresh.

## Required table coverage

The finalized schema includes the required tables:

- assets
- prices_raw
- prices_clean
- returns_daily
- triplets
- regression_results
- hedge_ratios
- residuals
- residual_diagnostics
- candidate_events
- event_features
- event_labels
- model_runs
- model_predictions
- trades
- positions
- pnl_daily
- performance_summary

Legacy phase-specific tables are retained for backward compatibility. The canonical tables above provide the clean end-to-end research database design.

## Validation queries

`sql/validation_queries.sql` contains integrity checks for missing assets, orphaned returns, orphaned residuals, missing event links, invalid probabilities, duplicate keys, missing PnL values, and missing model-run references.

## Report queries

`sql/report_queries.sql` contains research reports for row counts, asset coverage, triplet pipeline coverage, residual diagnostics, event success rates, model prediction summaries, probability buckets, daily PnL, trade audit, and strategy performance.

## Outputs

The included Phase 16 outputs were generated from a small synthetic validation database. They verify schema mechanics and expected query formats. They are not market-data research results.

- `data/processed/phase16_row_counts_by_table.csv`
- `data/processed/phase16_validation_query_results.csv`
- `data/processed/phase16_data_integrity_checks.csv`
- `data/processed/phase16_report_query_results.csv`
- `data/processed/phase16_validation_sample.db`
- `data/processed/phase16_validation_outputs/`
- `data/processed/phase16_report_outputs/`

## Usage

Initialize the schema:

```python
from src.database import initialize_database

initialize_database("data/research.db", "sql/schema.sql")
```

Run validations:

```python
from src.database import connect_database, run_validation_queries

with connect_database("data/research.db") as conn:
    checks = run_validation_queries(conn, "sql/validation_queries.sql")
```

Export report query results:

```python
from src.database import connect_database, export_named_query_results

with connect_database("data/research.db") as conn:
    index = export_named_query_results(
        conn,
        "sql/report_queries.sql",
        "data/processed/report_outputs",
        "report",
    )
```
