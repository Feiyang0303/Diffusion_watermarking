#!/usr/bin/env python3
"""
Single-panel ROC curve for SD eval CSV (watermarked vs clean).
Generates "ROC - Tree-Ring detection (image-level)" with actual curve from distance threshold sweep.

Usage:
  python plot_sd_eval_roc.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv --attack none
  python plot_sd_eval_roc.py --csv outputs_tree_ring_sd_eval/sd_eval_merged.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roc_from_distances(wm_distances: np.ndarray, clean_distances: np.ndarray, num_thresholds: int = 200):
    """TPR/FPR by sweeping threshold; predict positive when distance < threshold."""
    wm = np.asarray(wm_distances, dtype=float)
    clean = np.asarray(clean_distances, dtype=float)
    if len(wm) == 0 or len(clean) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    scores = np.concatenate([wm, clean])
    labels = np.concatenate([np.ones(len(wm)), np.zeros(len(clean))])
    th_min, th_max = scores.min(), scores.max()
    if th_min >= th_max:
        thresholds = np.array([th_min])
    else:
        thresholds = np.linspace(th_min, th_max, num_thresholds)
    tprs, fprs = [], []
    for th in thresholds:
        pred = (scores < th).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
    return np.array([0.0] + fprs + [1.0]), np.array([0.0] + tprs + [1.0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="SD eval CSV with columns: type, distance")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same dir as CSV, sd_eval_roc.png)")
    parser.add_argument("--attack", default=None, help="If CSV has 'attack' column, filter to this attack (e.g. none)")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    out_path = Path(args.out) if args.out else path.parent / "sd_eval_roc.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    if args.attack and "attack" in df.columns:
        df = df[df["attack"] == args.attack]
    wm = df[df["type"] == "watermarked"]["distance"].dropna().to_numpy()
    clean = df[df["type"] == "clean"]["distance"].dropna().to_numpy()

    if len(wm) == 0 or len(clean) == 0:
        raise SystemExit("Need both watermarked and clean rows in CSV.")

    fpr, tpr = roc_from_distances(wm, clean)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, "b-", linewidth=2, label="Tree-Ring (image-level)")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC – Tree-Ring detection (image-level)")
    plt.legend(loc="lower right")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
