"""
Common utilities for data fetching and helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from polygon import RESTClient

logger = logging.getLogger(__name__)


def get_most_recent_trading_day(client: RESTClient, ticker: str = "SPY") -> Optional[str]:
    """
    Use Polygon previous-close (v2) to infer the most recent trading day.
    """
    try:
        resp = client.get_previous_close(ticker=ticker, adjusted=True)
        rows = None
        if hasattr(resp, "results"):
            rows = resp.results
        elif isinstance(resp, dict):
            rows = resp.get("results")
        if rows:
            ts = rows[0].get("t") if isinstance(rows[0], dict) else getattr(rows[0], "t", None)
            if ts:
                dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                return dt.date().isoformat()
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("get_most_recent_trading_day failed: %s", e)
    return None


def _normalize_aggs_to_df(resp) -> pd.DataFrame:
    """
    Convert Polygon aggs response to normalized OHLCV DataFrame.
    """
    # Handle empty list responses
    if isinstance(resp, list):
        return pd.DataFrame()

    rows = None
    if hasattr(resp, "results"):
        rows = resp.results
    elif isinstance(resp, dict):
        rows = resp.get("results")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.rename(
        columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume",
            "t": "timestamp",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]


def fetch_intraday(
    client: RESTClient,
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    lookback_minutes: Optional[int] = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch 1-minute aggregates via Polygon v3 API with required params and safe empty handling.

    If the requested date has no data (empty 404 / missing results), automatically
    fallback to the most recent trading day based on previous close.
    """
    # Derive dates from lookback if provided
    if lookback_minutes is not None:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(minutes=lookback_minutes)
        start = start_dt.date().isoformat()
        end = end_dt.date().isoformat()

    # If no dates provided, use most recent trading day
    if not start or not end:
        fallback = get_most_recent_trading_day(client, ticker="SPY")
        start = end = fallback or datetime.now(timezone.utc).date().isoformat()

    target_date = start

    def _call(date_str: str):
        return client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=date_str,
            to=date_str,
            adjusted=True,
            sort="asc",
            limit=5000,
        )

    try:
        resp = _call(target_date)
        df = _normalize_aggs_to_df(resp)
        if df.empty:
            logger.warning("Polygon returned empty 404/NoData for %s", target_date)
            fallback_date = get_most_recent_trading_day(client, ticker="SPY")
            if fallback_date and fallback_date != target_date:
                logger.warning("Retrying with fallback date %s", fallback_date)
                resp = _call(fallback_date)
                df = _normalize_aggs_to_df(resp)
        logger.debug("Polygon v3 response rows=%s for %s", len(df) if df is not None else 0, target_date)
        return df
    except Exception as e:
        logger.warning("Polygon intraday fetch failed for %s: %s", target_date, e)
        return None
