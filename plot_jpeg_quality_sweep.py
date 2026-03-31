#!/usr/bin/env python3
"""
Plot JPEG quality sweep results: AUC and Best Accuracy vs JPEG quality for each radius.

Reads metrics CSVs from experiments/jpeg_defense/runs/jpeg_quality_sweep_n50/.

Usage (from repo root):
  python plot_jpeg_quality_sweep.py
  python plot_jpeg_quality_sweep.py --out results/jpeg_quality_sweep.png
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RADII = [8, 10, 12]
QUALITIES = [10, 15, 25, 50, 75]
METRICS_DIR = "experiments/jpeg_defense/runs/jpeg_quality_sweep_n50"


def _parse_metrics(path: Path) -> dict[str, float] | None:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("attack") or "").strip() == "jpeg":
                return {k: float(v) for k, v in row.items() if k != "attack" and k != "attack_param"}
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/jpeg_quality_sweep.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    metrics_dir = root / METRICS_DIR
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[int, dict[int, dict[str, float]]] = {}
    for r in RADII:
        data[r] = {}
        for q in QUALITIES:
            p = metrics_dir / f"metrics_min_dist_r{r}_q{q}_n50.csv"
            if p.is_file():
                parsed = _parse_metrics(p)
                if parsed:
                    data[r][q] = parsed

    colors = {8: "#e74c3c", 10: "#2980b9", 12: "#27ae60"}
    markers = {8: "o", 10: "s", 12: "D"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Min-dist detector vs JPEG compression strength (n=50)",
        fontsize=16, fontweight="bold", y=0.97,
    )

    metric_keys = [
        ("auc", "AUC"),
        ("best_accuracy", "Best Accuracy"),
        ("tpr_at_5pct_fpr", "TPR @ 5% FPR"),
        ("tpr_at_1pct_fpr", "TPR @ 1% FPR"),
    ]

    for ax, (key, label) in zip(axes.flat, metric_keys):
        for r in RADII:
            qs = sorted(data[r].keys())
            vals = [data[r][q][key] for q in qs]
            ax.plot(qs, vals, color=colors[r], marker=markers[r],
                    linewidth=2.2, markersize=8, label=f"r={r}")
            for q_val, v in zip(qs, vals):
                ax.annotate(f"{v:.2f}", (q_val, v), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=8, color=colors[r])

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random")
        ax.set_xlabel("JPEG Quality (lower = more compression)", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks(QUALITIES)
        ax.set_xlim(5, 80)
        if key == "auc" or key == "best_accuracy":
            ax.set_ylim(0.45, 1.02)
        else:
            ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
