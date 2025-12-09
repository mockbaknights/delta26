from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from common.features.vortex import add_delta_vortex_features as add_common
from research.features.vortex import add_delta_vortex_features as add_research


def load_sample_df() -> pd.DataFrame:
    candidates = [
        pathlib.Path("data/raw/sample_generated.csv"),
        pathlib.Path("data/processed/qqq/qqq_1m.parquet"),
        pathlib.Path("data/processed/spy/spy_1m.parquet"),
    ]
    for path in candidates:
        if path.exists():
            if path.suffix == ".csv":
                df = pd.read_csv(path, parse_dates=["timestamp"])
            else:
                df = pd.read_parquet(path)
            # Normalize column names (capitalized OHLCV)
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
            return df
    return pd.DataFrame()


def test_common_vs_research_feature_parity():
    df = load_sample_df()
    if df.empty:
        pytest.skip("No sample data available for feature consistency test.")

    base = df.copy()
    common_df = add_common(base)
    research_df = add_research(base)

    # Align columns intersection
    cols = sorted(set(common_df.columns) & set(research_df.columns))
    common_sub = common_df[cols].reset_index(drop=True)
    research_sub = research_df[cols].reset_index(drop=True)

    assert len(common_sub) == len(research_sub)
    pd.testing.assert_frame_equal(
        common_sub,
        research_sub,
        check_dtype=False,
        atol=1e-6,
        rtol=1e-6,
    )

