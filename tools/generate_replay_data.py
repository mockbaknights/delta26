import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import yaml
import pandas as pd

# Ensure repo root on sys.path for absolute imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from common.utils import fetch_intraday
from dotenv import load_dotenv


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def daterange_days(days: int):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start, end


def main():
    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)
    load_dotenv()

    cfg = load_config(Path("live/config.yaml"))
    symbols = cfg.get("symbols", [])
    days = int(os.getenv("REPLAY_DAYS", "5"))
    start, end = daterange_days(days)

    out_dir = Path("data/replay")
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        df = fetch_intraday(
            client=None,
            ticker=symbol,
            start=start.isoformat(),
            end=end.isoformat(),
        )
        if df is None or df.empty:
            print(f"[WARN] No data for {symbol} in range {start}..{end}")
            continue
        df = df.sort_values("timestamp")
        out_path = out_dir / f"{symbol}.parquet"
        df.to_parquet(out_path)
        print(f"[OK] Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()

