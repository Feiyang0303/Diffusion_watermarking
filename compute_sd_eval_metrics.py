#!/usr/bin/env python3
"""
Compute detection metrics from SD eval CSV (AUC, TPR@FPR, accuracy) for the experiments section.

Reads the same CSV as plot_robustness.py; outputs:
  - sd_eval_metrics.csv: one row per attack + random baseline
  - sd_eval_metrics_table.md: markdown table for the paper

Usage:
  python compute_sd_eval_metrics.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv
  python compute_sd_eval_metrics.py --csv outputs_tree_ring_sd_eval/sd_eval_attacks.csv --out_dir outputs_tree_ring_sd_eval
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def roc_from_distances(wm_distances: np.ndarray, clean_distances: np.ndarray, num_thresholds: int = 500):
    """Return (fpr, tpr) arrays; predict positive when distance < threshold."""
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


def auc_trapezoidal(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under ROC curve (trapezoidal rule)."""
    order = np.argsort(fpr)
    fpr = fpr[order]
    tpr = tpr[order]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    return float(np.trapz(tpr, fpr))  # NumPy < 2.0


def tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
    """Max TPR such that FPR <= target_fpr."""
    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def main():
    parser = argparse.ArgumentParser(description="Compute Tree-Ring detection metrics from SD eval CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to SD eval CSV (with attack column)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory for outputs (default: same as CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if "attack" not in df.columns:
        raise ValueError("CSV must have an 'attack' column (run eval with --attacks none,jpeg,resize,crop)")

    attacks = sorted(df["attack"].unique(), key=lambda a: (0 if a == "none" else 1, a))
    rows = []

    for atk in attacks:
        sub = df[df["attack"] == atk]
        wm = sub[sub["type"] == "watermarked"]["distance"].dropna().to_numpy()
        clean = sub[sub["type"] == "clean"]["distance"].dropna().to_numpy()
        if len(wm) == 0 or len(clean) == 0:
            continue
        fpr, tpr = roc_from_distances(wm, clean)
        auc = auc_trapezoidal(fpr, tpr)
        tpr_at_1 = tpr_at_fpr(fpr, tpr, 0.01)
        tpr_at_5 = tpr_at_fpr(fpr, tpr, 0.05)
        # Best threshold: maximize Youden (TPR - FPR); then accuracy at that point
        youden = tpr - fpr
        idx = np.argmax(youden)
        best_acc = (tpr[idx] * len(wm) + (1 - fpr[idx]) * len(clean)) / (len(wm) + len(clean))
        attack_param = ""
        if "attack_param" in sub.columns:
            params = sub["attack_param"].dropna()
            if len(params) > 0 and str(params.iloc[0]).strip():
                attack_param = str(params.iloc[0]).strip()
        rows.append({
            "attack": atk,
            "attack_param": attack_param,
            "n_wm": len(wm),
            "n_clean": len(clean),
            "auc": round(auc, 4),
            "tpr_at_1pct_fpr": round(tpr_at_1, 4),
            "tpr_at_5pct_fpr": round(tpr_at_5, 4),
            "best_accuracy": round(best_acc, 4),
        })

    # Random baseline
    n_total = df.groupby("attack").size()
    n_wm = len(df[(df["type"] == "watermarked") & (df["attack"] == attacks[0])]) if attacks else 0
    n_clean = len(df[(df["type"] == "clean") & (df["attack"] == attacks[0])]) if attacks else 0
    rows.append({
        "attack": "random_baseline",
        "attack_param": "",
        "n_wm": n_wm,
        "n_clean": n_clean,
        "auc": 0.5,
        "tpr_at_1pct_fpr": 0.01,
        "tpr_at_5pct_fpr": 0.05,
        "best_accuracy": 0.5,
    })

    metrics_df = pd.DataFrame(rows)
    metrics_csv = out_dir / "sd_eval_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics CSV: {metrics_csv}")

    # Markdown table for paper
    md_path = out_dir / "sd_eval_metrics_table.md"
    with open(md_path, "w") as f:
        f.write("# Tree-Ring detection metrics (image-level SD eval)\n\n")
        f.write("Lower distance = more likely watermarked. Threshold swept to compute ROC.\n\n")
        f.write("| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |\n")
        f.write("|--------|-------|-----|--------------|--------------|----------|\n")
        for _, r in metrics_df.iterrows():
            atk = r["attack"]
            if atk == "random_baseline":
                f.write("| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |\n")
            else:
                param = r["attack_param"] if pd.notna(r["attack_param"]) else "—"
                f.write(f"| {atk} | {param} | {r['auc']:.2f} | {r['tpr_at_1pct_fpr']:.2f} | {r['tpr_at_5pct_fpr']:.2f} | {r['best_accuracy']:.2f} |\n")
        f.write("\n")
    print(f"Saved markdown table: {md_path}")

    # Print summary
    print("\n--- Metrics summary ---")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
