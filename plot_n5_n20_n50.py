#!/usr/bin/env python3
"""
Plot n=5, n=20, and n=50 metrics as grouped bar charts in one figure.

Reads metrics_n5.csv, metrics_n20.csv, and (optional) metrics_n50.csv; saves
a figure with 4 subplots (AUC, TPR @ 1% FPR, TPR @ 5% FPR, Best Accuracy),
each with three bars per attack (n=5, n=20, n=50). If metrics_n50.csv is
missing, only n=5 and n=20 are drawn.

Usage:
  python plot_n5_n20_n50.py --dir outputs_tree_ring_sd_eval
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_tree_ring_sd_eval", help="Directory containing metrics_n5.csv, metrics_n20.csv, [metrics_n50.csv]")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path (default: <dir>/metrics_n5_n20_n50.png)")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    d = Path(args.dir)
    path_n5 = d / "metrics_n5.csv"
    path_n20 = d / "metrics_n20.csv"
    path_n50 = d / "metrics_n50.csv"

    if not path_n5.exists():
        raise FileNotFoundError(f"Missing: {path_n5}")
    if not path_n20.exists():
        raise FileNotFoundError(f"Missing: {path_n20}")

    dfs = []
    labels = []
    colors = ["steelblue", "coral", "seagreen"]
    for path, label in [(path_n5, "n=5"), (path_n20, "n=20"), (path_n50, "n=50")]:
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["attack"] != "random_baseline"].reset_index(drop=True)
            dfs.append((label, df))
        else:
            if label == "n=50":
                print(f"Optional {path} not found; plotting only n=5 and n=20.")
            break

    if len(dfs) < 2:
        raise FileNotFoundError("Need at least metrics_n5.csv and metrics_n20.csv.")

    attacks = dfs[0][1]["attack"].tolist()
    n_bars = len(dfs)
    x = np.arange(len(attacks))
    total_width = 0.8
    width = total_width / n_bars
    offsets = np.linspace(-total_width / 2 + width / 2, total_width / 2 - width / 2, n_bars)

    metrics = [
        ("auc", "AUC"),
        ("tpr_at_1pct_fpr", "TPR @ 1% FPR"),
        ("tpr_at_5pct_fpr", "TPR @ 5% FPR"),
        ("best_accuracy", "Best Accuracy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        for i, (label, df) in enumerate(dfs):
            ax.bar(x + offsets[i], df[col].values, width, label=label, color=colors[i], alpha=0.9)
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)

    n_label = "n=5, n=20, n=50" if len(dfs) == 3 else "n=5, n=20"
    fig.suptitle(f"Tree-Ring detection: {n_label} samples per condition", fontsize=12)
    plt.tight_layout()

    out_path = Path(args.out) if args.out else d / "metrics_n5_n20_n50.png"
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
