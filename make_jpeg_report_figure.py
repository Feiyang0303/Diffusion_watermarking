#!/usr/bin/env python3
"""
One-page figure: sample images (clean / watermarked / JPEG) + JPEG metrics table.
Run from repo root: python make_jpeg_report_figure.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg
from matplotlib.table import Table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="results/jpeg_report_summary.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Folder with sample_clean.png, sample_watermarked.png, sample_watermarked_jpeg_q25.png",
    )
    parser.add_argument(
        "--median_n_column",
        type=str,
        default="50",
        choices=("20", "50"),
        help="Sample count for median row if CSV missing (default 50 after WatGPU n=50 run).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    median_csv = root / "experiments/jpeg_defense/runs/median_r10_n50/metrics_jpeg_median_r10_n50.csv"
    median_n = args.median_n_column
    median_cells = ["0.62", "0.30", "0.35", "0.65"]
    if median_csv.is_file():
        with open(median_csv, newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("attack") or "").strip() == "jpeg":
                    median_n = str(int(float(row["n_wm"])))
                    median_cells = [
                        f"{float(row['auc']):.2f}",
                        f"{float(row['tpr_at_1pct_fpr']):.2f}",
                        f"{float(row['tpr_at_5pct_fpr']):.2f}",
                        f"{float(row['best_accuracy']):.2f}",
                    ]
                    break
    res = root / args.results_dir
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Tree-Ring + Stable Diffusion v1.5 — JPEG compression (Q=25) detection study",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --- Images row ---
    titles = ("Clean", "Watermarked", "Watermarked + JPEG (Q25)")
    names = ("sample_clean.png", "sample_watermarked.png", "sample_watermarked_jpeg_q25.png")
    for i, (fname, title) in enumerate(zip(names, titles)):
        ax = fig.add_axes([0.06 + i * 0.29, 0.58, 0.27, 0.32])
        p = res / fname
        if not p.exists():
            ax.text(0.5, 0.5, f"Missing:\n{p.name}", ha="center", va="center")
            ax.set_axis_off()
            continue
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_axis_off()

    # --- Metrics table (numbers from your WatGPU / snapshot runs) ---
    col_labels = ["Setting", "n", "AUC", "TPR @ 1%", "TPR @ 5%", "Best acc"]
    cell_text = [
        ["First channel (baseline)", "50", "0.75", "0.10", "0.30", "0.69"],
        ["Mean agg., key_scale=1.0", "50", "0.75", "0.06", "0.10", "0.70"],
        ["Mean agg., key_scale=1.12", "50", "0.75", "0.04", "0.08", "0.70"],
        ["Median agg., key_scale=1.0", median_n, *median_cells],
        ["Min-dist channel, key_scale=1.0", "50", "0.90", "0.26", "0.58", "0.85"],
    ]

    ax_tbl = fig.add_axes([0.08, 0.10, 0.84, 0.40])
    ax_tbl.set_axis_off()
    table: Table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.95)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f8f9fa" if row % 2 else "white")

    median_note = (
        "Median: r=10 from runs/median_r10_n50/metrics CSV when present (else --median_n_column + legacy n=20 numbers). "
        "Min-dist row: r=12 n=50 snapshot (best-of-4 channels; may affect FPR).\n"
        "Other rows: n=50 baselines / mean runs."
    )
    fig.text(
        0.08,
        0.06,
        median_note
        + "\nLower distance ⇒ more likely watermarked. Metrics from compute_sd_eval_metrics.py (threshold sweep).",
        fontsize=8,
        color="#444",
        verticalalignment="bottom",
    )

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
