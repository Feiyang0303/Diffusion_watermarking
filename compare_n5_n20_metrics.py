#!/usr/bin/env python3
"""
Build a comparison table: Tree-Ring metrics with n=5 vs n=20 samples.

Run after:
  compute_sd_eval_metrics.py --csv sd_eval_attacks.csv          --out_prefix metrics_n5
  compute_sd_eval_metrics.py --csv sd_eval_attacks_n20.csv      --out_prefix metrics_n20

Usage:
  python compare_n5_n20_metrics.py --dir outputs_tree_ring_sd_eval
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_tree_ring_sd_eval", help="Directory containing metrics_n5.csv and metrics_n20.csv")
    parser.add_argument("--out", type=str, default=None, help="Output markdown path (default: <dir>/metrics_compare_n5_vs_n20.md)")
    args = parser.parse_args()

    d = Path(args.dir)
    path_n5 = d / "metrics_n5.csv"
    path_n20 = d / "metrics_n20.csv"
    for p in (path_n5, path_n20):
        if not p.exists():
            raise FileNotFoundError(f"Run compute_sd_eval_metrics.py with --out_prefix metrics_n5 and metrics_n20 first. Missing: {p}")

    df5 = pd.read_csv(path_n5)
    df20 = pd.read_csv(path_n20)
    out_path = Path(args.out) if args.out else d / "metrics_compare_n5_vs_n20.md"

    with open(out_path, "w") as f:
        f.write("# Tree-Ring detection: n=5 vs n=20 samples\n\n")
        f.write("Contrast small vs larger sample size. Same setup (SD v1.5, attacks: none, jpeg, resize, crop).\n\n")

        f.write("## n = 5 per condition\n\n")
        f.write("| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |\n")
        f.write("|--------|-------|-----|--------------|--------------|----------|\n")
        for _, r in df5.iterrows():
            if r["attack"] == "random_baseline":
                f.write("| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |\n")
            else:
                p = r["attack_param"] if pd.notna(r["attack_param"]) and str(r["attack_param"]).strip() else "—"
                f.write(f"| {r['attack']} | {p} | {r['auc']:.2f} | {r['tpr_at_1pct_fpr']:.2f} | {r['tpr_at_5pct_fpr']:.2f} | {r['best_accuracy']:.2f} |\n")
        f.write("\n")

        f.write("## n = 20 per condition\n\n")
        f.write("| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |\n")
        f.write("|--------|-------|-----|--------------|--------------|----------|\n")
        for _, r in df20.iterrows():
            if r["attack"] == "random_baseline":
                f.write("| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |\n")
            else:
                p = r["attack_param"] if pd.notna(r["attack_param"]) and str(r["attack_param"]).strip() else "—"
                f.write(f"| {r['attack']} | {p} | {r['auc']:.2f} | {r['tpr_at_1pct_fpr']:.2f} | {r['tpr_at_5pct_fpr']:.2f} | {r['best_accuracy']:.2f} |\n")
        f.write("\n")

        f.write("## Summary\n\n")
        f.write("- With **n=5**, metrics can be optimistic (e.g. perfect AUC for none/jpeg); with **n=20**, estimates are more stable.\n")
        f.write("- Use the **n=20** table as the main results; show **n=5** as a quick sanity check or ablation.\n")
        f.write("- Caption suggestion: *Left: 5 watermarked + 5 clean per attack. Right: 20 per condition.*\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
