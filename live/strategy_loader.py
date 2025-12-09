"""
Model loading and signal classification for live trading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import json

import joblib
import numpy as np
import pandas as pd

from live.feature_builder import load_feature_columns

MODEL_PATH = Path("research/models/delta_vortex_v1_xgb.joblib")
METRICS_PATH = Path("research/models/delta_vortex_v1_metrics.json")

# Load feature columns once for alignment
FEATURE_COLUMNS = load_feature_columns()


def _load_registry(registry_path: Path) -> dict:
    if not registry_path.exists():
        return {}
    try:
        return json.loads(registry_path.read_text())
    except Exception:
        return {}


def resolve_model_from_registry(registry_path: Path, override_version: str | None = None) -> tuple[Path, Path | None]:
    registry = _load_registry(registry_path)
    versions = {v["version"]: v for v in registry.get("versions", []) if "version" in v}
    version = override_version or registry.get("active_version")
    if version and version in versions:
        info = versions[version]
        model_path = Path(info["model_path"])
        metrics_path = Path(info.get("metrics_path")) if info.get("metrics_path") else None
        return model_path, metrics_path
    # fallback to legacy paths
    return MODEL_PATH, METRICS_PATH


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained XGBoost model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def load_metrics(metrics_path: Path) -> dict | None:
    """Load metrics JSON if present."""
    if metrics_path is None or not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        return json.load(f)


def predict_proba(model: Any, features_df: pd.DataFrame) -> float:
    """
    Predict probability of class 1 given a single-row feature DataFrame.
    """
    if features_df.empty:
        raise ValueError("features_df is empty")

    # Align columns to training order
    for col in FEATURE_COLUMNS:
        if col not in features_df.columns:
            features_df[col] = np.nan
    X = features_df[FEATURE_COLUMNS]
    proba = model.predict_proba(X)[:, 1][0]
    return float(proba)


def classify_signal(prob: float, thresholds: dict[str, float]) -> str:
    """
    Map probability to discrete signal buckets.
    """
    strong_buy = thresholds.get("strong_buy", 0.8)
    buy = thresholds.get("buy", 0.6)
    sell = thresholds.get("sell", 0.4)
    strong_sell = thresholds.get("strong_sell", 0.2)

    if prob >= strong_buy:
        return "STRONG_BUY"
    if prob >= buy:
        return "BUY"
    if prob <= strong_sell:
        return "STRONG_SELL"
    if prob <= sell:
        return "SELL"
    return "NEUTRAL"

