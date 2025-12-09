"""
Health check script for live trading environment.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure repo root on sys.path before local imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv
from polygon import RESTClient

import joblib
import pandas as pd

from common.utils import fetch_intraday

MODEL_PATH = Path("research/models/delta_vortex_v1_xgb.joblib")
FEATURE_TABLE_PATH = Path("data/processed/feature_table.parquet")


def check_env_key() -> bool:
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        print("ERROR: POLYGON_API_KEY not set")
        return False
    return True


def check_model() -> bool:
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        return False
    try:
        joblib.load(MODEL_PATH)
        return True
    except Exception as e:  # pragma: no cover - defensive
        print(f"ERROR: Failed to load model: {e}")
        return False


def check_feature_table() -> bool:
    if not FEATURE_TABLE_PATH.exists():
        print(f"ERROR: Feature table missing at {FEATURE_TABLE_PATH}")
        return False
    try:
        df = pd.read_parquet(FEATURE_TABLE_PATH)
        print(f"Feature table columns: {list(df.columns)}")
        return True
    except Exception as e:  # pragma: no cover
        print(f"ERROR: Failed to read feature table: {e}")
        return False


def check_polygon(api_key: str, symbol: str = "SPY") -> bool:
    try:
        client = RESTClient(api_key)
        today = datetime.now(timezone.utc).date().isoformat()
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()

        df = fetch_intraday(client, symbol, start=today, end=today)
        if df is None or df.empty:
            df = fetch_intraday(client, symbol, start=yesterday, end=yesterday)

        if df is None:
            print("Polygon check failed: no response")
            return False

        if df.empty:
            print("Polygon OK (no minute data for requested dates)")
            return True

        print("Polygon OK")
        return True
    except Exception as e:  # pragma: no cover
        print(f"Polygon check failed: {e}")
        return False


def main() -> None:
    load_dotenv()
    ok = True

    key_ok = check_env_key()
    ok = ok and key_ok

    model_ok = check_model()
    ok = ok and model_ok

    feature_ok = check_feature_table()
    ok = ok and feature_ok

    if key_ok:
        polygon_ok = check_polygon(os.getenv("POLYGON_API_KEY"))
        ok = ok and polygon_ok
    else:
        polygon_ok = False

    print(
        f"Summary -> env={key_ok}, model={model_ok}, features={feature_ok}, polygon={polygon_ok}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

