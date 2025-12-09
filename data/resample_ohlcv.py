"""
Resample QQQ and SPY 1-minute parquet files to multiple timeframes.
"""

from __future__ import annotations

import logging
from pathlib import Path
import json
import time

import pandas as pd

SOURCE_FILES = {
    "QQQ": Path("data/processed/qqq/qqq_1m.parquet"),
    "SPY": Path("data/processed/spy/spy_1m.parquet"),
}

FREQS = ["2min", "3min", "5min", "10min", "15min", "30min", "1H", "4H", "1D"]

AGG = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}

REQUIRED_COLS = {"timestamp", "Open", "High", "Low", "Close", "Volume", "symbol"}
OPS_DIR = Path("data/ops/tmp")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def validate_schema(df: pd.DataFrame, path: Path) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Ensure extraction produced the expected schema.")


def resample_symbol(symbol: str, src_path: Path) -> dict:
    if not src_path.exists():
        logging.warning("Source not found for %s: %s", symbol, src_path)
        return {}

    logging.info("Loading %s", src_path)
    df = pd.read_parquet(src_path)
    if df.empty:
        logging.warning("Empty source for %s", symbol)
        return {}

    validate_schema(df, src_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")

    dest_dir = src_path.parent
    stats = {}
    for freq in FREQS:
        resampled = df.resample(freq).agg(AGG).dropna(subset=["Open", "High", "Low", "Close"])
        resampled = resampled.reset_index()
        resampled["symbol"] = symbol
        dest = dest_dir / f"{symbol.lower()}_{freq}.parquet"
        resampled.to_parquet(dest, index=False)
        logging.info("Wrote %s rows to %s", len(resampled), dest)
        stats[freq] = len(resampled)

    # summary
    ts_min, ts_max = df.index.min(), df.index.max()
    logging.info(
        "%s resample done. Source rows=%d window=%s -> %s outputs in %s",
        symbol,
        len(df),
        ts_min,
        ts_max,
        dest_dir,
    )
    return {
        "source_rows": len(df),
        "ts_min": str(ts_min),
        "ts_max": str(ts_max),
        "freq_counts": stats,
    }


def main() -> None:
    configure_logging()
    t0 = time.time()
    summary = {}
    for symbol, path in SOURCE_FILES.items():
        summary[symbol] = resample_symbol(symbol, path)

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "symbols": summary,
        "elapsed_seconds": time.time() - t0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with (OPS_DIR / "resample_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

