from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from common.features.vortex import add_delta_vortex_features
from live.feature_builder import build_features_for_symbol, load_feature_columns


def load_sample_df() -> pd.DataFrame:
    candidates = [
        Path("data/processed/spy/spy_1m.parquet"),
        Path("data/processed/qqq/qqq_1m.parquet"),
        Path("data/raw/sample_generated.csv"),
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path, parse_dates=["timestamp"])
            else:
                df = pd.read_parquet(path)
            # ensure required columns
            rename_map = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            for k, v in rename_map.items():
                if k in df.columns and v not in df.columns:
                    df[v] = df[k]
            if "symbol" not in df.columns:
                df["symbol"] = "SPY"
            return df
    return pd.DataFrame()


def test_research_vs_live_features_match():
    df = load_sample_df()
    if df.empty or len(df) < 200:
        pytest.skip("No sufficient sample data available for feature consistency test.")

    df = df.sort_values("timestamp").reset_index(drop=True)
    # Use last 200 rows to satisfy lookback
    df_slice = df.tail(400).copy()
    feature_cols = load_feature_columns()

    research_feat = add_delta_vortex_features(df_slice)
    # Use live builder to produce latest row
    live_feat_row = build_features_for_symbol(
        symbol=str(df_slice["symbol"].iloc[0]),
        interval="1m",
        bars_df=df_slice[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy(),
        feature_columns=feature_cols,
        lookback_buffer=150,
    )
    if live_feat_row is None or live_feat_row.empty:
        pytest.skip("Live feature builder returned empty.")

    research_last = research_feat.iloc[[-1]].copy()
    # Align columns intersection
    cols = sorted(set(live_feat_row.columns) & set(research_last.columns))
    pd.testing.assert_frame_equal(
        live_feat_row[cols].reset_index(drop=True),
        research_last[cols].reset_index(drop=True),
        check_dtype=False,
        atol=1e-6,
        rtol=1e-6,
    )

