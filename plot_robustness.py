#!/usr/bin/env python3
"""
Plot robustness under attacks: ROC curves and distance histograms per attack.

Expects a CSV from run_tree_ring_sd_eval.py with --attacks none,jpeg,resize,crop
(i.e. columns: sample_idx, type, attack, attack_param, distance, ...).

Usage:
  python plot_robustness.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv
  python plot_robustness.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv --out_dir outputs_tree_ring_sd_eval
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def roc_from_distances(wm_distances: np.ndarray, clean_distances: np.ndarray, num_thresholds: int = 200):
    """
    Compute TPR/FPR by sweeping threshold on distance.
    Label = watermarked (1) vs clean (0). Predict positive when distance < threshold
    (lower distance => more likely watermarked).
    """
    scores = np.concatenate([wm_distances, clean_distances])
    labels = np.concatenate([np.ones(len(wm_distances)), np.zeros(len(clean_distances))])
    th_min, th_max = scores.min(), scores.max()
    if th_min >= th_max:
        thresholds = np.array([th_min])
    else:
        thresholds = np.linspace(th_min, th_max, num_thresholds)
    tprs = []
    fprs = []
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
    # Ensure (0,0) and (1,1) for closed curve
    return np.array([0.0] + fprs + [1.0]), np.array([0.0] + tprs + [1.0])


def main():
    parser = argparse.ArgumentParser(description="Plot Tree-Ring robustness: ROC per attack")
    parser.add_argument("--csv", type=str, required=True, help="Path to SD eval CSV (with attack column)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for output plots (default: same as CSV)")
    parser.add_argument("--prefix", type=str, default="robustness", help="Filename prefix for saved figures")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "attack" not in df.columns:
        raise ValueError("CSV must contain an 'attack' column (run with --attacks none,jpeg,resize,crop)")

    attacks = sorted(df["attack"].unique(), key=lambda a: (0 if a == "none" else 1, a))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(attacks), 1)))
    attack_color = {a: colors[i] for i, a in enumerate(attacks)}

    # ----- ROC curves (one figure) -----
    plt.figure(figsize=(6, 5))
    for atk in attacks:
        sub = df[df["attack"] == atk]
        wm = sub[sub["type"] == "watermarked"]["distance"].to_numpy()
        clean = sub[sub["type"] == "clean"]["distance"].to_numpy()
        if len(wm) == 0 or len(clean) == 0:
            continue
        fpr, tpr = roc_from_distances(wm, clean)
        label = f"{atk}"
        if (sub["attack_param"] != "").any():
            param = sub["attack_param"].dropna().iloc[0]
            if str(param):
                label += f" ({param})"
        plt.plot(fpr, tpr, color=attack_color[atk], label=label, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Tree-Ring detection robustness under attacks (ROC)")
    plt.legend(loc="lower right")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = out_dir / f"{args.prefix}_roc.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"Saved ROC: {roc_path}")

    # ----- Distance histograms per attack (subplots) -----
    n_attacks = len(attacks)
    ncols = 2
    nrows = (n_attacks + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n_attacks == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for idx, atk in enumerate(attacks):
        ax = axes[idx]
        sub = df[df["attack"] == atk]
        wm = sub[sub["type"] == "watermarked"]["distance"].to_numpy()
        clean = sub[sub["type"] == "clean"]["distance"].to_numpy()
        if len(wm):
            ax.hist(wm, bins=min(20, max(len(wm) // 2, 5)), alpha=0.7, label="watermarked", density=True, color="C0")
        if len(clean):
            ax.hist(clean, bins=min(20, max(len(clean) // 2, 5)), alpha=0.7, label="clean", density=True, color="C1")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Density")
        ax.set_title(f"Attack: {atk}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    for j in range(len(attacks), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    hist_path = out_dir / f"{args.prefix}_dist_hist.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved distance histograms: {hist_path}")


if __name__ == "__main__":
    main()
