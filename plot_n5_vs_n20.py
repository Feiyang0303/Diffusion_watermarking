#!/usr/bin/env python3
"""
Plot n=5 vs n=20 metrics as grouped bar charts for the paper.

Reads metrics_n5.csv and metrics_n20.csv; saves a figure with 4 subplots:
AUC, TPR @ 1% FPR, TPR @ 5% FPR, Best Accuracy — each with attacks on x-axis
and two bars per attack (n=5 vs n=20).

Usage:
  python plot_n5_vs_n20.py --dir outputs_tree_ring_sd_eval
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_tree_ring_sd_eval", help="Directory containing metrics_n5.csv and metrics_n20.csv")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: <dir>/metrics_n5_vs_n20.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    d = Path(args.dir)
    path_n5 = d / "metrics_n5.csv"
    path_n20 = d / "metrics_n20.csv"
    for p in (path_n5, path_n20):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}. Run compute_sd_eval_metrics.py with --out_prefix metrics_n5 and metrics_n20 first.")

    df5 = pd.read_csv(path_n5)
    df20 = pd.read_csv(path_n20)
    # Exclude random baseline for the bar chart
    df5 = df5[df5["attack"] != "random_baseline"].reset_index(drop=True)
    df20 = df20[df20["attack"] != "random_baseline"].reset_index(drop=True)

    attacks = df5["attack"].tolist()
    x = np.arange(len(attacks))
    width = 0.35

    metrics = [
        ("auc", "AUC"),
        ("tpr_at_1pct_fpr", "TPR @ 1% FPR"),
        ("tpr_at_5pct_fpr", "TPR @ 5% FPR"),
        ("best_accuracy", "Best Accuracy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        v5 = df5[col].values
        v20 = df20[col].values
        bars1 = ax.bar(x - width / 2, v5, width, label="n=5", color="steelblue", alpha=0.9)
        bars2 = ax.bar(x + width / 2, v20, width, label="n=20", color="coral", alpha=0.9)
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Tree-Ring detection: n=5 vs n=20 samples per condition", fontsize=12)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else d / "metrics_n5_vs_n20.png"
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
