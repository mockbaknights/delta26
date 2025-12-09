"""
Monte Carlo simulation runner for model performance variability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load

from research.train_v1 import prepare_training_data, evaluate_model
from research.features.vortex import add_delta_vortex_features


def simulate_pnl(probas: np.ndarray, y_true: np.ndarray, threshold: float = 0.5, n_runs: int = 500, sample_frac: float = 0.5):
    rng = np.random.default_rng(42)
    pf_list = []
    win_loss_list = []
    dd_list = []
    n = len(probas)
    for _ in range(n_runs):
        idx = rng.choice(n, size=max(10, int(n * sample_frac)), replace=True)
        p = probas[idx]
        y = y_true[idx]
        preds = (p >= threshold).astype(int)
        wins = (preds == 1) & (y == 1)
        losses = (preds == 1) & (y == 0)
        win_count = wins.sum()
        loss_count = losses.sum()
        pf = (win_count + 1e-9) / max(loss_count, 1e-9)
        pf_list.append(pf)
        win_loss_list.append((win_count, loss_count))

        # crude drawdown proxy using cumulative PnL (+1 for win, -1 for loss)
        pnl = (wins.astype(int) - losses.astype(int)).cumsum()
        peak = np.maximum.accumulate(pnl)
        dd = (pnl - peak).min() if len(pnl) else 0
        dd_list.append(dd)
    return np.array(pf_list), np.array(win_loss_list), np.array(dd_list)


def plot_distribution(data: np.ndarray, title: str, path: Path, bins: int = 30):
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, alpha=0.7)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def run(model_path: Path, parquet_path: Path, out_dir: Path) -> dict:
    model = load(model_path)
    df = pd.read_parquet(parquet_path)
    df = add_delta_vortex_features(df)
    # Expect target present; otherwise skip
    if "target" not in df.columns:
        raise ValueError("Parquet data must include 'target' column for Monte Carlo simulation.")
    X, y = prepare_training_data(df, drop_lookahead_rows=0)
    probas = model.predict_proba(X)[:, 1]
    pf, wl, dd = simulate_pnl(probas, y.values)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_distribution(dd, "Drawdown Distribution", out_dir / "drawdown_distribution.png")
    plot_distribution(pf, "Profit Factor Distribution", out_dir / "pf_distribution.png")
    plot_distribution(wl[:, 0] - wl[:, 1], "Win-Loss Count Distribution", out_dir / "win_loss_distribution.png")

    summary = {
        "pf": {
            "mean": float(pf.mean()),
            "p05": float(np.percentile(pf, 5)),
            "p50": float(np.percentile(pf, 50)),
            "p95": float(np.percentile(pf, 95)),
        },
        "drawdown": {
            "mean": float(dd.mean()),
            "p05": float(np.percentile(dd, 5)),
            "p50": float(np.percentile(dd, 50)),
            "p95": float(np.percentile(dd, 95)),
        },
        "win_loss_diff": {
            "mean": float((wl[:, 0] - wl[:, 1]).mean()),
        },
    }
    with (out_dir / "monte_carlo_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--parquet", required=True, help="Path to 1m parquet with target column")
    parser.add_argument("--out_dir", required=True, help="Output directory for MC artifacts")
    args = parser.parse_args()
    summary = run(Path(args.model), Path(args.parquet), Path(args.out_dir))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

