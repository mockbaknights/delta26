"""
Simple risk utilities for live trading decisions.
"""

from __future__ import annotations


def compute_position_size(
    equity: float, risk_per_trade_pct: float, stop_distance: float
) -> int:
    """
    Basic fixed-fractional position sizing.

    Args:
        equity: account equity
        risk_per_trade_pct: fraction of equity to risk per trade (e.g., 0.005 = 0.5%)
        stop_distance: price distance between entry and stop
    """
    if equity <= 0 or risk_per_trade_pct <= 0 or stop_distance <= 0:
        return 0
    dollar_risk = equity * risk_per_trade_pct
    size = dollar_risk / stop_distance
    return max(int(size), 0)


def should_trade(signal: str, mode: str) -> bool:
    """
    Gate trades based on mode and signal strength.
    """
    strong = {"STRONG_BUY", "STRONG_SELL"}
    moderate = {"BUY", "SELL"}

    if signal in strong:
        return True
    if mode == "intraday" and signal in moderate:
        return True
    return False

