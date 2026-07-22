import numpy as np
import pandas as pd
import pytest

from src.ingest import (
    build_assets_table,
    build_triplets_table,
    clean_prices,
    compute_daily_returns,
    coverage_report,
    unique_symbols,
)


def _triplets():
    return [
        {"triplet_id": "AAA_BBB_CCC", "target": "AAA", "hedge_1": "BBB", "hedge_2": "CCC", "theme": "test_theme"},
        {"triplet_id": "DDD_BBB_CCC", "target": "DDD", "hedge_1": "BBB", "hedge_2": "CCC", "theme": "test_theme"},
    ]


def test_unique_symbols_deduplicates_shared_hedge_legs():
    symbols = unique_symbols(_triplets())
    assert symbols == ["AAA", "BBB", "CCC", "DDD"]


def test_clean_prices_flags_invalid_and_drops_exact_duplicates():
    raw = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA", "AAA"],
        "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"],
        "adj_close": [10.0, 10.5, np.nan, -1.0],
        "source": ["yfinance"] * 4,
    })
    cleaned = clean_prices(raw)

    assert len(cleaned) == 3  # the exact duplicate date is deduped, keeping the last value
    assert cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-02"), "adj_close"].iloc[0] == 10.5
    assert cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-02"), "quality_flag"].iloc[0] == "clean"
    assert cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-03"), "quality_flag"].iloc[0] == "invalid_price"
    assert cleaned.loc[cleaned["date"] == pd.Timestamp("2024-01-04"), "quality_flag"].iloc[0] == "invalid_price"

    with pytest.raises(KeyError):
        clean_prices(pd.DataFrame({"symbol": ["AAA"]}))


def test_compute_daily_returns_excludes_invalid_rows_and_matches_pct_change():
    raw = pd.DataFrame({
        "symbol": ["AAA"] * 4,
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "adj_close": [100.0, 110.0, np.nan, 121.0],
    })
    cleaned = clean_prices(raw)
    returns = compute_daily_returns(cleaned)

    # the invalid row (nan on 01-04) must not appear, and must not be used
    # as a base for the following day's return computation
    assert len(returns) == 3
    first_return = returns.loc[returns["date"] == pd.Timestamp("2024-01-03"), "simple_return"].iloc[0]
    assert np.isclose(first_return, 0.10)
    # since the invalid row was excluded, the 01-05 return is computed
    # against 01-03 (110), not fabricated against the dropped nan row
    last_return = returns.loc[returns["date"] == pd.Timestamp("2024-01-05"), "simple_return"].iloc[0]
    assert np.isclose(last_return, 121.0 / 110.0 - 1.0)


def test_compute_daily_returns_does_not_bleed_across_symbols():
    raw = pd.DataFrame({
        "symbol": ["AAA", "BBB"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "adj_close": [100.0, 50.0],
    })
    cleaned = clean_prices(raw)
    returns = compute_daily_returns(cleaned)
    assert returns["simple_return"].isna().all()  # single observation per symbol -- no prior day to diff against


def test_build_assets_table_distinguishes_targets_from_hedges():
    assets = build_assets_table(_triplets())
    types = assets.set_index("symbol")["asset_type"]
    assert types["AAA"] == "equity"
    assert types["DDD"] == "equity"
    assert types["BBB"] == "etf"
    assert types["CCC"] == "etf"
    assert assets["active"].eq(1).all()


def test_build_triplets_table_matches_definitions():
    triplets = build_triplets_table(_triplets())
    assert set(triplets["triplet_id"]) == {"AAA_BBB_CCC", "DDD_BBB_CCC"}
    row = triplets.loc[triplets["triplet_id"] == "AAA_BBB_CCC"].iloc[0]
    assert row["target_symbol"] == "AAA"
    assert row["hedge_symbol_1"] == "BBB"
    assert row["hedge_symbol_2"] == "CCC"
    assert row["relationship_theme"] == "test_theme"


def test_coverage_report_flags_missing_symbols_as_zero():
    cleaned = pd.DataFrame({
        "symbol": ["AAA", "AAA", "AAA"],
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "adj_close": [10.0, 10.5, 11.0],
        "quality_flag": ["clean", "clean", "invalid_price"],
    })
    report = coverage_report(cleaned, symbols=["AAA", "ZZZ"])
    aaa = report.set_index("symbol").loc["AAA"]
    assert aaa["n_observations"] == 3
    assert aaa["n_clean"] == 2
    assert np.isclose(aaa["clean_ratio"], 2 / 3)

    zzz = report.set_index("symbol").loc["ZZZ"]
    assert zzz["n_observations"] == 0
    assert zzz["n_clean"] == 0
