"""
Populates the SQLite database and data/processed CSVs with real adjusted
daily prices for every symbol in the triplet universe (src/config.py).

This requires network access to Yahoo Finance and therefore cannot run
inside a fully offline or network-restricted sandbox -- if you see 403 /
"host not in allowlist" errors, that's a network policy blocking this
script, not a bug in it. Run it from a normal machine with internet access.

Usage:
    python scripts/ingest_prices.py --start 2021-01-01 --end 2026-01-01

By default this fetches ~5 years of history for all 107 underlying symbols
across the 82-triplet universe. Yahoo Finance rate-limits aggressive
polling; this script fetches one symbol at a time with a short delay
rather than in parallel, so a full run takes several minutes, not seconds.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    DATABASE_DIR,
    DEFAULT_DATABASE_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SQL_DIR,
    TRIPLET_DEFINITIONS,
)
from src.database import (
    connect_database,
    initialize_database,
    store_assets,
    store_prices_clean,
    store_prices_raw,
    store_returns_daily,
    store_triplets,
)
from src.ingest import (
    build_assets_table,
    build_triplets_table,
    clean_prices,
    compute_daily_returns,
    coverage_report,
    fetch_adjusted_prices,
    unique_symbols,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01", help="fetch start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="fetch end date (YYYY-MM-DD), default today")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between symbol fetches")
    args = parser.parse_args()

    symbols = unique_symbols(TRIPLET_DEFINITIONS)
    print(f"universe: {len(TRIPLET_DEFINITIONS)} triplets, {len(symbols)} unique symbols")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"fetching {len(symbols)} symbols from {args.start} to {args.end or 'today'} ...")
    all_raw = []
    for i, symbol in enumerate(symbols, start=1):
        try:
            chunk = fetch_adjusted_prices([symbol], start=args.start, end=args.end or _today(), source="yfinance")
            all_raw.append(chunk)
        except RuntimeError as exc:
            print(f"  [{i}/{len(symbols)}] {symbol}: FAILED -- {exc}")
        else:
            print(f"  [{i}/{len(symbols)}] {symbol}: {len(chunk)} rows")
        time.sleep(args.delay)

    if not all_raw:
        raise SystemExit("no symbols were fetched successfully -- check network access and try again")

    import pandas as pd
    raw_prices = pd.concat(all_raw, ignore_index=True)
    raw_prices.to_csv(RAW_DATA_DIR / "adjusted_prices_raw.csv", index=False)

    cleaned = clean_prices(raw_prices)
    cleaned.to_csv(PROCESSED_DATA_DIR / "adjusted_prices_clean.csv", index=False)

    returns = compute_daily_returns(cleaned)
    returns.to_csv(PROCESSED_DATA_DIR / "returns_daily.csv", index=False)

    coverage = coverage_report(cleaned, symbols)
    coverage.to_csv(PROCESSED_DATA_DIR / "price_coverage_report.csv", index=False)
    thin = coverage.loc[coverage["clean_ratio"].fillna(0) < 0.9]
    if not thin.empty:
        print(f"\nwarning: {len(thin)} symbols have <90% clean coverage -- check before using their triplets:")
        print(thin.to_string(index=False))

    assets = build_assets_table(TRIPLET_DEFINITIONS)
    triplets = build_triplets_table(TRIPLET_DEFINITIONS)

    initialize_database(DEFAULT_DATABASE_PATH, SQL_DIR / "schema.sql")
    with connect_database(DEFAULT_DATABASE_PATH) as conn:
        store_assets(conn, assets)
        store_triplets(conn, triplets)
        store_prices_raw(conn, raw_prices, if_exists="replace")
        store_prices_clean(conn, cleaned, if_exists="replace")
        store_returns_daily(conn, returns, if_exists="replace")

    print(f"\ndone. {len(cleaned)} clean price rows across {cleaned['symbol'].nunique()} symbols "
          f"written to {DEFAULT_DATABASE_PATH} and data/processed/.")
    print("Next: re-run the hedge-ratio / residual notebooks (06+) pointed at this real price data "
          "instead of their synthetic placeholder fallbacks.")


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


if __name__ == "__main__":
    main()
