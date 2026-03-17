#!/usr/bin/env python3
"""
Single-panel distance histogram for SD eval CSV (watermarked vs clean).
Uses data range for x-axis so histogram is visible (image-level distances are ~5–25, not 0–1).

Usage:
  python plot_sd_eval_dist.py --csv outputs_tree_ring_sd_eval/sd_eval_merged.csv
  python plot_sd_eval_dist.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv  # uses first attack only, or pass --attack
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="SD eval CSV with columns: type, distance")
    parser.add_argument("--out", default=None, help="Output PNG path (default: same dir as CSV, name sd_eval_dist_hist.png)")
    parser.add_argument("--attack", default=None, help="If CSV has 'attack' column, filter to this attack (e.g. none)")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)
    out_path = Path(args.out) if args.out else path.parent / "sd_eval_dist_hist.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path)
    if args.attack and "attack" in df.columns:
        df = df[df["attack"] == args.attack]
    wm = df[df["type"] == "watermarked"]["distance"].dropna()
    clean = df[df["type"] == "clean"]["distance"].dropna()

    plt.figure(figsize=(6, 4))
    if len(wm):
        plt.hist(wm, bins=min(25, max(len(wm) // 2, 5)), alpha=0.7, label="watermarked", density=True, color="C0")
    if len(clean):
        plt.hist(clean, bins=min(25, max(len(clean) // 2, 5)), alpha=0.7, label="clean", density=True, color="C1")
    plt.xlabel("Tree-Ring distance")
    plt.ylabel("density")
    plt.title("Tree-Ring distances (image-level, SD eval)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Do NOT set xlim(0,1) — image-level distances are typically 5–25
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
