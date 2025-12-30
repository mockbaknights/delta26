"""
Simple alert logging to console and file.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import os
import logging

SIGNAL_LOG_PATH = Path("data/live_signals.log")
MODE = (os.getenv("DELTA26_MODE") or "live").lower()


def log_signal(
    symbol: str,
    interval: str,
    signal: str,
    prob: float,
    price: float,
    mode: str,
    extra: dict[str, Any] | None = None,
    log_path: Path = SIGNAL_LOG_PATH,
) -> None:
    """
    Log a signal to stdout and append to a JSONL file.
    """
    if MODE != "live":
        logging.info("[SHADOW] Alert suppressed for %s %s %s", symbol, interval, signal)
        return
    ts = datetime.now(timezone.utc).isoformat()
    payload = {
        "timestamp": ts,
        "symbol": symbol,
        "interval": interval,
        "mode": mode,
        "signal": signal,
        "probability": prob,
        "price": price,
    }
    if extra:
        payload.update(extra)

    line = (
        f"[{ts}] {mode.upper()} {symbol} {interval} -> {signal} "
        f"(prob={prob:.3f}, price={price})"
    )
    print(line)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(payload) + "\n")




