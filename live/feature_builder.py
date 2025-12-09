"""
Feature builder for live inference.

Responsibilities:
- Maintain/aggregate OHLCV bars from 1m base data into derived intervals.
- Reuse research feature engineering to match training schema.
- Produce a single-row feature DataFrame aligned to the model's expected columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from common.utils import fetch_intraday
from common.features.vortex import add_delta_vortex_features


# Paths and schema helpers
FEATURE_TABLE_PATH = Path("data/processed/feature_table.parquet")
DEFAULT_EXCLUDE = {"target", "timestamp", "Open", "High", "Low", "Close", "Volume"}

# Map config interval strings to pandas offsets
PANDAS_FREQ_MAP = {
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "5m": "5min",
    "10m": "10min",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def fetch_intraday_df(client, ticker: str, start, end) -> pd.DataFrame:
    """
    Fetch 1m bars using Polygon v3 aggregates and return normalized DataFrame.
    """
    df = fetch_intraday(client, ticker, start=start, end=end)
    if df is None or df.empty:
        print(f"No intraday bars available for {start} -> {end}, skipping feature build.")
        return pd.DataFrame()
    return df.sort_values("timestamp")


def load_feature_columns(
    feature_table_path: Path = FEATURE_TABLE_PATH,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    """Load training feature column order, excluding non-feature columns."""
    exclude_set = set(exclude) if exclude else set(DEFAULT_EXCLUDE)
    df = pd.read_parquet(feature_table_path)
    cols = [c for c in df.columns if c not in exclude_set]
    return cols


def merge_new_bars(
    existing: pd.DataFrame | None, new_bars: pd.DataFrame, max_rows: int = 5000
) -> pd.DataFrame:
    """
    Append and de-duplicate bars by timestamp, keeping the most recent rows.
    Expects columns: timestamp, Open, High, Low, Close, Volume.
    """
    new_bars = new_bars.copy()
    new_bars["timestamp"] = pd.to_datetime(new_bars["timestamp"], utc=True)

    if existing is None or existing.empty:
        merged = new_bars
    else:
        existing = existing.copy()
        existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True)
        merged = (
            pd.concat([existing, new_bars], ignore_index=True)
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp")
        )
    if len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].copy()
    return merged.reset_index(drop=True)


def aggregate_bars(bars_df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Aggregate 1m bars into the requested interval."""
    freq = PANDAS_FREQ_MAP.get(interval, interval)
    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    agg = df.resample(freq, label="right", closed="right").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    )
    agg = agg.dropna(subset=["Open", "High", "Low", "Close"])
    agg = agg.reset_index()
    return agg


def build_features_for_symbol(
    symbol: str,
    interval: str,
    bars_df: pd.DataFrame,
    feature_columns: list[str],
    lookback_buffer: int = 150,
) -> pd.DataFrame | None:
    """
    Build a single-row feature DataFrame for the latest bar of a symbol/interval.

    Args:
        symbol: ticker
        interval: interval string (e.g., 1m, 2m, 4h)
        bars_df: OHLCV bars for this interval
        feature_columns: ordered list matching training features (no OHLCV/timestamp)
        lookback_buffer: minimum rows to ensure rolling features are valid
    """
    if bars_df is None or len(bars_df) < lookback_buffer:
        return None

    df = bars_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = add_delta_vortex_features(df)

    if df.empty:
        return None

    latest = df.iloc[[-1]].copy()

    # Align columns to training schema; add missing with NaN
    for col in feature_columns:
        if col not in latest.columns:
            latest[col] = np.nan
    latest = latest[feature_columns]

    latest["symbol"] = symbol
    latest["interval"] = interval
    return latest

