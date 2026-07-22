"""
Real adjusted-price ingestion for the triplet universe in src/config.py.

This replaces the synthetic placeholder data used elsewhere in the
pipeline with actual market data. It is split deliberately into:

  - `fetch_adjusted_prices`: the one function that touches the network
    (via yfinance / Yahoo Finance). Not unit tested here, since a real
    network call has no place in a test suite -- verify it manually by
    running scripts/ingest_prices.py.
  - everything else: pure functions operating on DataFrames, fully unit
    tested in tests/test_ingest.py without any network dependency.

Adjusted close (not raw close) is used throughout downstream, since it
accounts for splits and dividends -- using raw close would introduce
artificial jumps at every split/dividend date that have nothing to do with
the triangular relationship the residual is supposed to measure.
"""
from __future__ import annotations

from datetime import date
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def unique_symbols(triplet_definitions: list[dict]) -> list[str]:
    """All distinct target/hedge symbols across a triplet universe, so
    each underlying is fetched once even though it appears in multiple
    triplets (e.g. QQQ appears in dozens of them)."""
    symbols: set[str] = set()
    for triplet in triplet_definitions:
        symbols.update([triplet["target"], triplet["hedge_1"], triplet["hedge_2"]])
    return sorted(symbols)


def fetch_adjusted_prices(
    symbols: Iterable[str],
    start: str | date,
    end: str | date,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Fetches daily adjusted OHLCV for `symbols` between `start` and `end`.

    Requires network access to Yahoo Finance -- this is the one function in
    this module that cannot run in a fully offline / sandboxed environment.
    Raises RuntimeError with the underlying cause rather than returning a
    silently empty frame, since a silent empty result here would corrupt
    every downstream table without any visible failure.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required: pip install yfinance") from exc

    symbols = list(symbols)
    frames = []
    failed = []
    for symbol in symbols:
        try:
            raw = yf.download(symbol, start=str(start), end=str(end), progress=False, auto_adjust=False)
        except Exception as exc:  # noqa: BLE001 -- network calls fail in many ways; surface all of them
            failed.append((symbol, str(exc)))
            continue
        if raw is None or raw.empty:
            failed.append((symbol, "empty response"))
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frame = raw.reset_index().rename(columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
        })
        frame["symbol"] = symbol
        frame["source"] = source
        frames.append(frame)

    if failed:
        detail = "; ".join(f"{sym}: {reason}" for sym, reason in failed[:10])
        suffix = f" (+{len(failed) - 10} more)" if len(failed) > 10 else ""
        if not frames:
            raise RuntimeError(f"failed to fetch any symbols -- {detail}{suffix}")
        print(f"warning: {len(failed)}/{len(symbols)} symbols failed to fetch -- {detail}{suffix}")

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    return combined[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]]


def clean_prices(raw_prices: pd.DataFrame, min_price: float = 0.01) -> pd.DataFrame:
    """Deduplicates, sorts, and flags rows that fail basic sanity checks
    (non-positive or missing adjusted close, non-monotonic duplicate dates).
    Rows failing the check are kept but flagged, not silently dropped --
    downstream steps that need only clean rows can filter on
    `quality_flag == 'clean'` explicitly rather than have data vanish
    with no record of why.
    """
    required = {"symbol", "date", "adj_close"}
    missing = required.difference(raw_prices.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    frame = raw_prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"], keep="last")

    valid = frame["adj_close"].notna() & (frame["adj_close"] >= min_price)
    frame["quality_flag"] = np.where(valid, "clean", "invalid_price")

    columns = [c for c in ["symbol", "date", "close", "adj_close", "volume", "quality_flag", "source"] if c in frame.columns]
    return frame.loc[:, columns].reset_index(drop=True)


def compute_daily_returns(clean_prices_frame: pd.DataFrame) -> pd.DataFrame:
    """Simple and log returns per symbol, computed only on rows flagged
    clean -- a return computed across an invalid-price gap would be
    meaningless and would silently corrupt every residual built on top of
    it, so invalid rows are excluded rather than interpolated.
    """
    required = {"symbol", "date", "adj_close"}
    missing = required.difference(clean_prices_frame.columns)
    if missing:
        raise KeyError(f"missing columns: {sorted(missing)}")

    frame = clean_prices_frame.copy()
    if "quality_flag" in frame.columns:
        frame = frame.loc[frame["quality_flag"] == "clean"].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"])

    frame["simple_return"] = frame.groupby("symbol")["adj_close"].pct_change()
    frame["log_return"] = np.log(frame["adj_close"]).groupby(frame["symbol"]).diff()

    return frame.loc[:, ["symbol", "date", "simple_return", "log_return"]].reset_index(drop=True)


def build_assets_table(triplet_definitions: list[dict], sector_by_symbol: Optional[dict[str, str]] = None) -> pd.DataFrame:
    """Minimal `assets` table rows for every symbol in the universe.
    `sector_by_symbol` is optional metadata (e.g. from the `theme` field
    already on each triplet) -- without it, sector is left null rather than
    guessed, since a hedge ETF doesn't have a single GICS sector the way a
    target stock does.
    """
    symbols = unique_symbols(triplet_definitions)
    rows = []
    theme_by_target = {t["target"]: t.get("theme") for t in triplet_definitions}
    for symbol in symbols:
        rows.append({
            "symbol": symbol,
            "asset_type": "equity" if symbol in theme_by_target else "etf",
            "sector": (sector_by_symbol or {}).get(symbol, theme_by_target.get(symbol)),
            "active": 1,
        })
    return pd.DataFrame(rows)


def build_triplets_table(triplet_definitions: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "triplet_id": t["triplet_id"],
            "target_symbol": t["target"],
            "hedge_symbol_1": t["hedge_1"],
            "hedge_symbol_2": t["hedge_2"],
            "relationship_theme": t.get("theme"),
            "active": 1,
        }
        for t in triplet_definitions
    ]
    return pd.DataFrame(rows)


def coverage_report(clean_prices_frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Per-symbol row counts and clean-row ratio -- run this before trusting
    any downstream hedge-ratio fit. A triplet built on a symbol with sparse
    or mostly-invalid coverage will produce a residual series that looks
    fine but is fit on too little real information.
    """
    if clean_prices_frame.empty:
        return pd.DataFrame({"symbol": symbols, "n_observations": 0, "n_clean": 0, "clean_ratio": np.nan})

    grouped = clean_prices_frame.groupby("symbol").agg(
        n_observations=("date", "count"),
        n_clean=("quality_flag", lambda s: (s == "clean").sum()) if "quality_flag" in clean_prices_frame.columns else ("date", "count"),
    )
    grouped["clean_ratio"] = grouped["n_clean"] / grouped["n_observations"]
    report = grouped.reindex(symbols).reset_index().rename(columns={"index": "symbol"})
    report[["n_observations", "n_clean"]] = report[["n_observations", "n_clean"]].fillna(0).astype(int)
    return report
