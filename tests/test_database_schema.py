from pathlib import Path

import pandas as pd

from src.database import (
    REQUIRED_PIPELINE_TABLES,
    connect_database,
    initialize_database,
    parse_named_sql_queries,
    run_validation_queries,
    store_assets,
    store_triplets,
    table_row_counts,
)


def test_sql_schema_creates_required_tables(tmp_path):
    db_path = tmp_path / "research.db"
    schema_path = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"
    initialize_database(db_path, schema_path)
    with connect_database(db_path) as conn:
        counts = table_row_counts(conn)
    assert set(REQUIRED_PIPELINE_TABLES).issubset(set(counts["table_name"]))
    assert counts["exists"].eq(1).all()


def test_sql_named_query_parser_loads_validation_queries():
    sql_path = Path(__file__).resolve().parents[1] / "sql" / "validation_queries.sql"
    queries = parse_named_sql_queries(sql_path)
    assert "required_table_row_counts" in queries
    assert "model_prediction_probability_bounds" in queries
    assert all(query.lower().startswith("select") for query in queries.values())


def test_sql_validation_queries_return_status_rows(tmp_path):
    db_path = tmp_path / "research.db"
    root = Path(__file__).resolve().parents[1]
    initialize_database(db_path, root / "sql" / "schema.sql")
    with connect_database(db_path) as conn:
        results = run_validation_queries(conn, root / "sql" / "validation_queries.sql")
    assert {"check_name", "status", "issue_count", "rows_returned"}.issubset(results.columns)
    assert "required_table_row_counts" in set(results["check_name"])


def test_sql_canonical_store_functions_write_core_rows(tmp_path):
    db_path = tmp_path / "research.db"
    root = Path(__file__).resolve().parents[1]
    initialize_database(db_path, root / "sql" / "schema.sql")
    with connect_database(db_path) as conn:
        store_assets(
            conn,
            pd.DataFrame(
                [
                    {"symbol": "NVDA", "asset_name": "NVIDIA", "asset_type": "equity"},
                    {"symbol": "SMH", "asset_name": "VanEck Semiconductor ETF", "asset_type": "etf"},
                    {"symbol": "QQQ", "asset_name": "Invesco QQQ Trust", "asset_type": "etf"},
                ]
            ),
        )
        store_triplets(
            conn,
            pd.DataFrame(
                [
                    {
                        "triplet_id": "NVDA_SMH_QQQ",
                        "target_symbol": "NVDA",
                        "hedge_symbol_1": "SMH",
                        "hedge_symbol_2": "QQQ",
                        "relationship_theme": "semiconductor_vs_tech",
                    }
                ]
            ),
        )
        counts = table_row_counts(conn, ["assets", "triplets"])
    assert counts.set_index("table_name").loc["assets", "row_count"] == 3
    assert counts.set_index("table_name").loc["triplets", "row_count"] == 1
