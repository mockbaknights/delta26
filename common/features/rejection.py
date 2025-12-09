"""
Wick rejection and range percentage features.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def add_rejection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add bullish/bearish rejection ratios and normalized wick sizes.
    """
    df = df.copy()
    range_len = (df["High"] - df["Low"]).replace(0, np.nan)
    df["bear_rejection_ratio"] = (df["High"] - df["Close"]) / range_len
    df["bull_rejection_ratio"] = (df["Close"] - df["Low"]) / range_len
    df["range_pct"] = range_len / df["Close"]
    return df.dropna()

