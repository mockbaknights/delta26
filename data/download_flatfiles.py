"""
Download daily minute aggregate flatfiles from Massive and store locally.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import boto3
from botocore.exceptions import ClientError

BUCKET = "flatfiles"
ENDPOINT_URL = "https://files.massive.com"
PROFILE = "massive"
PREFIX_ROOT = "us_stocks_sip/minute_aggs_v1"

# Date range (inclusive)
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

DEST_ROOT = Path("data/raw/flatfiles")
MAX_WORKERS = 8
OPS_DIR = Path("data/ops/tmp")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def daterange(start: date, end: date) -> List[date]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def download_day(s3_client, dt: date) -> str:
    date_str = dt.isoformat()
    key = f"{PREFIX_ROOT}/{dt.year}/{dt.month:02d}/{date_str}.csv.gz"
    dest_dir = DEST_ROOT / str(dt.year)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{date_str}.csv.gz"

    if dest_path.exists():
        logging.info("File exists, skipping %s", date_str)
        return "skipped"

    try:
        s3_client.download_file(BUCKET, key, str(dest_path))
        logging.info("Downloaded %s", date_str)
        return "downloaded"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            logging.warning("No file found %s (404)", date_str)
            return "missing"
        logging.error("Failed to download %s: %s", key, e)
        return "error"
    except Exception as e:  # pragma: no cover - defensive
        logging.error("Unexpected error downloading %s: %s", key, e)
        return "error"


def main() -> None:
    configure_logging()

    session = boto3.session.Session(profile_name=PROFILE)
    s3_client = session.client("s3", endpoint_url=ENDPOINT_URL)

    start_dt = datetime.fromisoformat(START_DATE).date()
    end_dt = datetime.fromisoformat(END_DATE).date()
    dates = daterange(start_dt, end_dt)

    stats: Dict[str, int] = {"downloaded": 0, "skipped": 0, "missing": 0, "error": 0}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_day, s3_client, d): d for d in dates}
        for fut in as_completed(futures):
            res = fut.result()
            stats[res] = stats.get(res, 0) + 1

    elapsed = time.time() - t0
    logging.info(
        "Done. downloaded=%s skipped=%s missing=%s error=%s total=%s in %.2fs",
        stats["downloaded"],
        stats["skipped"],
        stats["missing"],
        stats["error"],
        len(dates),
        elapsed,
    )

    OPS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "start_date": START_DATE,
        "end_date": END_DATE,
        "downloaded": stats["downloaded"],
        "skipped": stats["skipped"],
        "missing": stats["missing"],
        "error": stats["error"],
        "total_dates": len(dates),
        "elapsed_seconds": elapsed,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    with (OPS_DIR / "download_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

