"""
Aggregate ingestion run metadata and emit summary artifacts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

import pandas as pd

OPS_DIR = Path("data/ops")
TMP_DIR = OPS_DIR / "tmp"
LAST_JSON = OPS_DIR / "last_run.json"
LAST_TXT = OPS_DIR / "last_run.txt"


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_meta(filename: str) -> dict | None:
    path = TMP_DIR / filename
    if not path.exists():
        logging.warning("Meta file missing: %s", path)
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover
        logging.warning("Failed to load %s: %s", path, e)
        return None


def collect_download_meta() -> dict:
    meta = load_meta("download_meta.json") or {}
    return meta


def collect_extract_meta() -> dict:
    meta = load_meta("extract_meta.json") or {}
    return meta


def collect_resample_meta() -> dict:
    meta = load_meta("resample_meta.json") or {}
    return meta


def summarize_files() -> dict:
    raw_files = list(Path("data/raw/flatfiles").rglob("*.csv.gz"))
    processed_files = list(Path("data/processed").rglob("*.parquet"))
    return {
        "raw_files": len(raw_files),
        "processed_files": len(processed_files),
    }


def format_table(summary: Dict[str, Any]) -> str:
    lines = []
    dl = summary.get("download", {})
    lines.append("Downloads:")
    lines.append(
        f"  downloaded={dl.get('downloaded',0)} skipped={dl.get('skipped',0)} missing={dl.get('missing',0)} error={dl.get('error',0)} range={dl.get('start_date')}..{dl.get('end_date')} elapsed={dl.get('elapsed_seconds','?'):.2f}s"
        if dl
        else "  (no download meta)"
    )

    ext = summary.get("extract", {})
    lines.append("Extraction:")
    if ext:
        lines.append(
            f"  total_rows={ext.get('rows_total',0)} ts={ext.get('ts_min')} -> {ext.get('ts_max')} elapsed={ext.get('elapsed_seconds','?'):.2f}s"
        )
        lines.append(
            f"  QQQ rows={ext.get('rows_qqq',0)} | SPY rows={ext.get('rows_spy',0)}"
        )
    else:
        lines.append("  (no extract meta)")

    res = summary.get("resample", {})
    lines.append("Resample:")
    if res:
        for sym, meta in res.get("symbols", {}).items():
            if not meta:
                lines.append(f"  {sym}: (no data)")
                continue
            lines.append(
                f"  {sym}: source_rows={meta.get('source_rows',0)} window={meta.get('ts_min')} -> {meta.get('ts_max')}"
            )
            freq_counts = meta.get("freq_counts", {})
            if freq_counts:
                freq_str = ", ".join(f"{k}:{v}" for k, v in freq_counts.items())
                lines.append(f"    {freq_str}")
        lines.append(
            f"  elapsed={res.get('elapsed_seconds','?'):.2f}s at {res.get('timestamp')}"
        )
    else:
        lines.append("  (no resample meta)")

    files = summary.get("files", {})
    lines.append(f"Files: raw={files.get('raw_files',0)} processed={files.get('processed_files',0)}")
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    OPS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "download": collect_download_meta(),
        "extract": collect_extract_meta(),
        "resample": collect_resample_meta(),
        "files": summarize_files(),
    }

    text = format_table(summary)
    with LAST_TXT.open("w") as f:
        f.write(text + "\n")
    with LAST_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Run summary written to %s and %s", LAST_JSON, LAST_TXT)
    logging.info("\n%s", text)


if __name__ == "__main__":
    main()

