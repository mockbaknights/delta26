"""
Extract QQQ and SPY minute bars from Massive flatfiles and write parquet outputs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List
import json
import time

import pandas as pd

TICKERS = ["QQQ", "SPY"]
SOURCE_GLOB = "data/raw/flatfiles/*/*.csv.gz"
DEST_DIR = Path("data/processed")
DEST_DIR_QQQ = DEST_DIR / "qqq"
DEST_DIR_SPY = DEST_DIR / "spy"
OPS_DIR = Path("data/ops/tmp")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lowercase mapping for flexible schemas
    cols_lower = {c.lower(): c for c in df.columns}
    required = ["ticker", "volume", "open", "close", "high", "low"]
    for col in required:
        if col not in cols_lower:
            raise ValueError(f"Missing required column '{col}' in file columns {df.columns.tolist()}")

    # window_start variants
    ts_col = None
    for candidate in ("window_start", "windowstart", "window-start"):
        if candidate in cols_lower:
            ts_col = cols_lower[candidate]
            break
    if ts_col is None:
        # attempt camel/pascal
        for c in df.columns:
            if c.lower().replace("_", "").replace("-", "") == "windowstart":
                ts_col = c
                break
    if ts_col is None:
        raise ValueError(f"Missing window_start column in {df.columns.tolist()}")

    df = df.rename(
        columns={
            ts_col: "timestamp",
            cols_lower["open"]: "open",
            cols_lower["high"]: "high",
            cols_lower["low"]: "low",
            cols_lower["close"]: "close",
            cols_lower["volume"]: "volume",
            cols_lower["ticker"]: "ticker",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    return df[["timestamp", "ticker", "open", "high", "low", "close", "volume"]]


def process_files() -> pd.DataFrame:
    files = sorted(Path().glob(SOURCE_GLOB))
    logging.info("Found %d files", len(files))
    frames: List[pd.DataFrame] = []
    processed = 0
    file_count = 0
    for fp in files:
        try:
            logging.info("Reading %s", fp)
            df = pd.read_csv(fp, compression="gzip", low_memory=False)
            df = df[df["ticker"].str.upper().isin(TICKERS)]
            if df.empty:
                logging.info("No target tickers in %s", fp)
                continue
            df = normalize_columns(df)
            frames.append(df)
            processed += 1
            file_count += 1
        except Exception as e:  # pragma: no cover
            logging.warning("Failed to process %s: %s", fp, e)
            continue

    if not frames:
        logging.warning("No data collected.")
        return pd.DataFrame(columns=["timestamp", "ticker", "open", "high", "low", "close", "volume"])

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["timestamp", "ticker"])
    all_df = all_df.sort_values("timestamp")
    logging.info("Processed %d files; total rows=%d", processed, len(all_df))
    return all_df


def summarize(df: pd.DataFrame, ticker: str, path: Path) -> None:
    if df.empty:
        logging.info("%s: no rows", ticker)
        return
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    size_mb = Path(path).stat().st_size / (1024 * 1024) if path.exists() else 0
    logging.info(
        "%s 1-min: %d rows from %s -> %s (%.2f MB) written to %s",
        ticker,
        len(df),
        ts_min,
        ts_max,
        size_mb,
        path,
    )


def write_parquet(df: pd.DataFrame, ticker: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{ticker.lower()}_1m.parquet"
    sub = df[df["ticker"].str.upper() == ticker].copy()
    if sub.empty:
        logging.warning("No rows for %s; skipping write to %s", ticker, dest)
        return
    sub.to_parquet(dest, index=False)
    summarize(sub, ticker, dest)


def main() -> None:
    configure_logging()
    t0 = time.time()
    df = process_files()
    if df.empty:
        logging.warning("No data to write.")
        return

    write_parquet(df, "QQQ", DEST_DIR_QQQ)
    write_parquet(df, "SPY", DEST_DIR_SPY)

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "rows_total": len(df),
        "rows_qqq": int((df["ticker"].str.upper() == "QQQ").sum()) if not df.empty else 0,
        "rows_spy": int((df["ticker"].str.upper() == "SPY").sum()) if not df.empty else 0,
        "ts_min": str(df["timestamp"].min()) if not df.empty else None,
        "ts_max": str(df["timestamp"].max()) if not df.empty else None,
        "elapsed_seconds": time.time() - t0,
    }
    with (OPS_DIR / "extract_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

