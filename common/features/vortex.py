"""
Delta Vortex feature engineering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ta = None


def add_delta_vortex_features(
    df: pd.DataFrame, lookback: int = 10, sensitivity: float = 0.5
) -> pd.DataFrame:
    """
    Compute Delta Vortex derived features.
    """
    df = df.copy()

    df["uptop"] = df["High"] - df["Close"]
    df["upbot"] = df["Open"] - df["Low"]
    df["dntop"] = df["High"] - df["Open"]
    df["dnbot"] = df["Close"] - df["Low"]

    df["up_delta"] = df["upbot"] - df["uptop"]
    df["down_delta"] = df["dntop"] - df["dnbot"]
    df["delta"] = np.where(df["up_delta"] != 0, df["up_delta"], df["down_delta"])

    pos_series = df["delta"].clip(lower=0)
    neg_series = df["delta"].clip(upper=0)

    df["pos_std_dev"] = pos_series.rolling(lookback).std() * sensitivity
    df["neg_std_dev"] = neg_series.rolling(lookback).std() * sensitivity

    df["pos_std_dev"] = df["pos_std_dev"].replace(0, 1e-10).fillna(1e-10)
    df["neg_std_dev"] = df["neg_std_dev"].replace(0, 1e-10).fillna(1e-10)

    df["strength"] = np.where(
        df["delta"] >= 0,
        df["delta"] / df["pos_std_dev"],
        (df["delta"] / df["neg_std_dev"]) * -1.0,
    )

    range_len = (df["High"] - df["Low"]).replace(0, 1e-10)
    df["bear_rejection_ratio"] = (df["High"] - df["Close"]) / range_len
    df["bull_rejection_ratio"] = (df["Close"] - df["Low"]) / range_len

    if "Volume" in df.columns:
        df["vwap"] = _vwap(df)
        df["vwap"] = df["vwap"].fillna(df["Close"])
        dist = df["Close"] - df["vwap"]
        stretch_len = 120
        rolling_mean = dist.rolling(stretch_len).mean()
        rolling_std = dist.rolling(stretch_len).std().replace(0, 1e-10)
        df["stretch_z"] = (dist - rolling_mean) / rolling_std

    df["rsi"] = _rsi(df["Close"], length=13)
    df["atr"] = _atr(df, period=14)
    return df.dropna()


def _vwap(df: pd.DataFrame) -> pd.Series:
    if ta:
        return ta.vwap(df["High"], df["Low"], df["Close"], df["Volume"])
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cumulative_tp_vol = (typical_price * df["Volume"]).cumsum()
    cumulative_vol = df["Volume"].replace(0, np.nan).cumsum()
    return cumulative_tp_vol / cumulative_vol


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if ta:
        return ta.rsi(close, length=length)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean().replace(0, 1e-10)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

