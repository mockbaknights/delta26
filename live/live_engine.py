"""
Polling-based live engine using Polygon REST API.
"""

from __future__ import annotations

import os
import time
import logging
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple, Any

# Ensure repo root on sys.path for absolute imports
ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import pandas as pd
import yaml
from dotenv import load_dotenv
from polygon import RESTClient

from live import alerts, risk
from live.feature_builder import (
    aggregate_bars,
    build_features_for_symbol,
    load_feature_columns,
    merge_new_bars,
)
from live.strategy_loader import classify_signal, load_model, predict_proba, resolve_model_from_registry
from common.utils import fetch_intraday

# Load environment (for POLYGON_API_KEY)
load_dotenv()

CONFIG_PATH = Path("live/config.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def init_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def weighted_prob(prob: float, interval: str, weights: dict[str, float]) -> float:
    w = weights.get(interval, 1.0)
    return prob * w


def apply_hysteresis(prob: float, last_signal: str | None, thresholds: dict[str, float], buffer: float = 0.05) -> str:
    """
    Prevent rapid flip-flop by requiring margin before switching sides.
    """
    signal = classify_signal(prob, thresholds)
    if last_signal is None:
        return signal

    bullish = {"BUY", "STRONG_BUY"}
    bearish = {"SELL", "STRONG_SELL"}

    if last_signal in bullish and signal in bearish:
        # require stronger opposite evidence
        if prob >= thresholds.get("sell", 0.4) + buffer:
            return signal
        return last_signal
    if last_signal in bearish and signal in bullish:
        if prob <= thresholds.get("buy", 0.6) - buffer:
            return signal
        return last_signal
    return signal


def risk_filters_pass(
    bars_df,
    feat_row,
    mode_cfg: dict[str, Any],
    interval: str,
) -> bool:
    # Volatility filter
    vol_cfg = mode_cfg.get("volatility", {})
    vol_window = vol_cfg.get("window", 50)
    vol_max = vol_cfg.get("std_max", None)
    if vol_max is not None and len(bars_df) >= vol_window:
        std = bars_df["Close"].tail(vol_window).std()
        if std > vol_max:
            return False

    # ATR regime
    atr_cfg = mode_cfg.get("atr", {})
    atr_max = atr_cfg.get("max", None)
    atr_min = atr_cfg.get("min", None)
    if "atr" in feat_row.columns:
        atr_val = float(feat_row["atr"].iloc[-1])
        if atr_max is not None and atr_val > atr_max:
            return False
        if atr_min is not None and atr_val < atr_min:
            return False

    # Time-of-day block (UTC hours)
    tod_cfg = mode_cfg.get("time_blocks", {})
    blocked = set(tod_cfg.get("blocked_hours", []))
    if blocked:
        ts = bars_df["timestamp"].iloc[-1]
        hour = ts.hour if hasattr(ts, "hour") else pd.to_datetime(ts).hour
        if hour in blocked:
            return False

    return True


def stream_signal(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def fetch_recent_1m(
    client: RESTClient, symbol: str, lookback_minutes: int = 400
):
    """
    Fetch recent 1m bars using Polygon v3 aggregates.
    """
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(minutes=lookback_minutes)
    df = fetch_intraday(
        client,
        ticker=symbol,
        start=start_dt.date().isoformat(),
        end=end_dt.date().isoformat(),
    )
    if df is None or df.empty:
        logging.warning("No intraday data for %s between %s and %s", symbol, start_dt.date(), end_dt.date())
        return pd.DataFrame()
    df = df[df["timestamp"] >= start_dt]
    return df.sort_values("timestamp")


def run_loop(config: dict) -> None:
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        raise RuntimeError("POLYGON_API_KEY is not set. Check your .env or environment.")

    symbols = config.get("symbols", [])
    modes = config.get("modes", {})
    log_level = config.get("logging", {}).get("level", "INFO")
    signal_stream_path = Path(config.get("logging", {}).get("signal_stream_path", "data/live_stream.jsonl"))
    weights = config.get("signal_weights", {})  # interval weighting
    hyst_buffer = config.get("hysteresis_buffer", 0.05)
    init_logging(log_level)

    model_cfg = config.get("model", {}) if config else {}
    registry_path = Path(model_cfg.get("registry_path", "research/models/model_registry.json"))
    override_version = model_cfg.get("active_version")
    model_path, metrics_path = resolve_model_from_registry(registry_path, override_version)

    model = load_model(model_path)
    feature_columns = load_feature_columns()
    client = RESTClient(api_key)

    bars_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    last_run: Dict[str, float] = {}

    logging.info("Starting live engine for symbols=%s", symbols)

    last_signals: Dict[Tuple[str, str], str] = {}
    while True:
        now_ts = time.time()
        for mode_name, mode_cfg in modes.items():
            if not mode_cfg.get("enabled", False):
                continue
            poll_seconds = mode_cfg.get("poll_seconds", 60)
            if now_ts - last_run.get(mode_name, 0) < poll_seconds:
                continue

            last_run[mode_name] = now_ts
            base_interval = mode_cfg.get("base_interval", "1m")
            derived_intervals = mode_cfg.get("derived_intervals", [])
            thresholds = mode_cfg.get("prob_thresholds", {})

            for symbol in symbols:
                try:
                    recent = fetch_recent_1m(
                        client, symbol, lookback_minutes=800 if mode_name == "swing" else 400
                    )
                    if recent is None or recent.empty:
                        logging.warning("No data fetched for %s; skipping cycle", symbol)
                        continue

                    cache_key = (symbol, base_interval)
                    cached = bars_cache.get(cache_key)
                    merged = merge_new_bars(cached, recent)
                    bars_cache[cache_key] = merged

                    for interval in derived_intervals:
                        bars_for_interval = (
                            merged if interval == base_interval else aggregate_bars(merged, interval)
                        )
                        if bars_for_interval.empty:
                            continue

                        feat_row = build_features_for_symbol(
                            symbol,
                            interval,
                            bars_for_interval,
                            feature_columns,
                        )
                        if feat_row is None or feat_row.empty:
                            continue

                        prob_raw = predict_proba(model, feat_row)
                        prob = weighted_prob(prob_raw, interval, weights)
                        last_key = (symbol, interval)
                        last_sig = last_signals.get(last_key)
                        signal = apply_hysteresis(prob, last_sig, thresholds, buffer=hyst_buffer)
                        last_signals[last_key] = signal
                        price = float(bars_for_interval["Close"].iloc[-1])

                        if not risk_filters_pass(bars_for_interval, feat_row, mode_cfg, interval):
                            logging.debug("Risk filter blocked %s %s %s", symbol, interval, signal)
                            continue

                        if risk.should_trade(signal, mode_name):
                            alerts.log_signal(
                                symbol,
                                interval,
                                signal,
                                prob,
                                price,
                                mode_name,
                            )
                            payload = {
                                "timestamp": bars_for_interval["timestamp"].iloc[-1].isoformat(),
                                "symbol": symbol,
                                "interval": interval,
                                "mode": mode_name,
                                "signal": signal,
                                "prob": prob,
                                "price": price,
                                "version": model_path.name,
                            }
                            stream_signal(payload, signal_stream_path)
                        else:
                            logging.debug(
                                "Filtered signal %s %s %s prob=%.3f",
                                symbol,
                                interval,
                                signal,
                                prob,
                            )
                except Exception as e:  # pragma: no cover - resilience in live loop
                    logging.exception("Error processing %s in mode %s: %s", symbol, mode_name, e)

        time.sleep(1)


def main() -> None:
    config = load_config()
    run_loop(config)


if __name__ == "__main__":
    main()

