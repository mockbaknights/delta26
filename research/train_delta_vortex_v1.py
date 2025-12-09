"""
Runner for Delta Vortex ML Strategy v1 (frozen).

Loads sample OHLCV data, applies Delta Vortex features, labels with the
triple-barrier method, prepares training data, trains XGBoost, evaluates,
and saves the model artifact.
"""

from __future__ import annotations

import sys
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure repository root is on path for research imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from research.features.vortex import add_delta_vortex_features
from research.train_v1 import (
    evaluate_model,
    prepare_training_data,
    train_xgboost_model,
)
from research.run_monte_carlo import run as run_monte_carlo

REGISTRY_PATH = Path("research/models/model_registry.json")


def generate_sample_data(rows: int = 400) -> pd.DataFrame:
    """Create a small synthetic OHLCV dataset for demo/training."""
    rng = pd.date_range("2023-01-01", periods=rows, freq="H")
    base = 100 + np.cumsum(np.random.normal(0, 0.4, size=rows))

    opens = base + np.random.normal(0, 0.1, size=rows)
    closes = base + np.random.normal(0, 0.1, size=rows)
    highs = np.maximum(opens, closes) + np.abs(np.random.normal(0.2, 0.1, size=rows))
    lows = np.minimum(opens, closes) - np.abs(np.random.normal(0.2, 0.1, size=rows))
    volume = np.random.randint(1_000, 5_000, size=rows)

    return pd.DataFrame(
        {
            "timestamp": rng,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volume,
        }
    )


def ensure_data(data_dir: Path) -> list[Path]:
    """Ensure raw data exists; if absent, generate a synthetic sample."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        sample_path = data_dir / "sample_generated.csv"
        generate_sample_data().to_csv(sample_path, index=False)
        csv_files = [sample_path]
    return csv_files


def load_dataframes(csv_files: list[Path]) -> pd.DataFrame:
    """Load and concatenate CSVs, ensuring timestamp order."""
    frames = []
    for path in csv_files:
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp")
    return combined.reset_index(drop=True)


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    take_profit: float = 0.006,
    stop_loss: float = 0.003,
    time_horizon: int = 15,
) -> pd.DataFrame:
    """
    Vectorized triple-barrier labeling (percentage thresholds).

    Matches the v1 behavior: TP=+0.6%, SL=-0.3%, horizon=15 bars.
    """
    df = df.copy()
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    closes = df["Close"].values
    highs = df["High"].values if "High" in df.columns else closes
    lows = df["Low"].values if "Low" in df.columns else closes

    n = len(df)
    targets = np.zeros(n, dtype=int)

    for i in range(n - time_horizon):
        current_price = closes[i]
        tp_price = current_price * (1 + take_profit)
        sl_price = current_price * (1 - stop_loss)

        window_start = i + 1
        window_end = min(i + time_horizon + 1, n)
        future_highs = highs[window_start:window_end]
        future_lows = lows[window_start:window_end]

        tp_cross = np.where(future_highs >= tp_price)[0]
        sl_cross = np.where(future_lows <= sl_price)[0]

        if len(tp_cross) > 0 and len(sl_cross) > 0:
            targets[i] = int(tp_cross[0] < sl_cross[0])
        elif len(tp_cross) > 0:
            targets[i] = 1
        elif len(sl_cross) > 0:
            targets[i] = 0
        else:
            targets[i] = 0

    # Last horizon rows have insufficient lookahead; keep as 0
    targets[-(time_horizon + 1) :] = 0
    df["target"] = targets
    return df


def main() -> None:
    research_dir = Path(__file__).resolve().parent
    data_dir = ROOT / "data" / "raw"
    processed_dir = ROOT / "data" / "processed"
    models_root = research_dir / "models"
    reports_dir = research_dir / "reports"
    models_root.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_files = ensure_data(data_dir)
    df_raw = load_dataframes(csv_files)

    # Feature engineering
    df_features = add_delta_vortex_features(df_raw)
    feature_path = processed_dir / "feature_table.parquet"
    df_features.to_parquet(feature_path, index=False)

    # Labeling
    df_labeled = apply_triple_barrier_labels(df_features)

    # Training data prep
    X, y = prepare_training_data(df_labeled, drop_lookahead_rows=15)
    if len(X) < 20:
        raise ValueError("Not enough rows after feature prep to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = train_xgboost_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, threshold=0.5)
    train_acc = model.score(X_train, y_train)
    val_acc = model.score(X_test, y_test)

    # Simple profit factor estimate using precision/(1-precision) as proxy
    report_dict = results["classification_report"]
    precision = report_dict.get("1", {}).get("precision", 0.0)
    profit_factor_est = float("inf") if precision >= 1 else precision / max(
        1 - precision, 1e-12
    )

    # Versioned model registry
    registry = {"active_version": None, "versions": []}
    if REGISTRY_PATH.exists():
        try:
            registry = json.loads(REGISTRY_PATH.read_text())
        except Exception:
            registry = {"active_version": None, "versions": []}

    existing = [
        v["version"] for v in registry.get("versions", []) if "version" in v
    ]
    next_idx = (
        max([int(v.lstrip("v")) for v in existing if v.startswith("v") and v[1:].isdigit()], default=0)
        + 1
    )
    version = f"v{next_idx}"
    version_dir = models_root / version
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.joblib"
    joblib.dump(model, model_path)

    # Persist metrics
    metrics_payload = {
        "train_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "profit_factor_est": profit_factor_est,
        "confusion_matrix": results["confusion_matrix"].tolist(),
        "classification_report": report_dict,
    }
    metrics_path = version_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Report assets
    # Confusion matrix heatmap
    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    cm_path = version_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    # Feature importance
    importances = model.feature_importances_
    importance_fig, ax2 = plt.subplots(figsize=(6, 4))
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(indices))
    top_idx = indices[:top_n]
    ax2.bar(range(top_n), importances[top_idx])
    feature_names = list(X.columns)
    ax2.set_xticks(range(top_n))
    ax2.set_xticklabels([feature_names[i] for i in top_idx], rotation=45, ha="right")
    ax2.set_title("Feature Importance")
    ax2.set_ylabel("Gain")
    importance_fig.tight_layout()
    fi_path = version_dir / "feature_importance.png"
    importance_fig.savefig(fi_path)
    plt.close(importance_fig)

    # Monte Carlo simulation
    mc_summary = run_monte_carlo(model_path, feature_path, version_dir)
    mc_path = version_dir / "monte_carlo_summary.json"

    # Update registry
    feature_list = list(X.columns)
    entry = {
        "version": version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metrics": metrics_payload,
        "features": feature_list,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "feature_importance_path": str(fi_path),
        "confusion_matrix_path": str(cm_path),
        "monte_carlo_path": str(mc_path),
        "monte_carlo": mc_summary,
    }
    registry.setdefault("versions", []).append(entry)
    registry["active_version"] = version
    with REGISTRY_PATH.open("w") as f:
        json.dump(registry, f, indent=2)

    print("Saved model to:", model_path)
    print("Saved metrics to:", metrics_path)
    print("Saved feature table to:", feature_path)
    print("Report assets:", {"confusion_matrix": str(cm_path), "feature_importance": str(fi_path)})
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", pd.DataFrame(report_dict))
    print(
        f"Summary -> train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, profit_factor_est: {profit_factor_est:.4f}"
    )
    print(f"Registry updated. Active version: {version}")


if __name__ == "__main__":
    main()

