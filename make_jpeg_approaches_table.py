#!/usr/bin/env python3
"""
Render JPEG-defense approach tables as a single PNG (slides / reports).
Run from repo root:
  MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_approaches_table.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=str,
        default="results/jpeg_approaches_table.png",
        help="Output PNG path (relative to repo root)",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    out_path = root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 10), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Tree-Ring + SD v1.5 — JPEG (Q=25): approaches compared",
        fontsize=16,
        fontweight="bold",
        y=0.97,
    )

    # --- Table A: detector / key (r=10) ---
    hdr_a = ["Approach", "n", "AUC", "TPR @ 1%", "TPR @ 5%", "Best acc"]
    rows_a = [
        ["First channel (baseline)", "50", "0.75", "0.10", "0.30", "0.69"],
        ["Mean across latent ch.", "50", "0.75", "0.06", "0.10", "0.70"],
        ["Mean + key ×1.12", "50", "0.75", "0.04", "0.08", "0.70"],
        ["Median across latent ch.", "20", "0.62", "0.30", "0.35", "0.65"],
    ]
    ax_a = fig.add_axes([0.05, 0.52, 0.9, 0.38])
    ax_a.set_axis_off()
    ax_a.set_title("A. Detector & key (Fourier mask radius r = 10)", fontsize=12, fontweight="bold", loc="left", pad=12)
    tbl_a = ax_a.table(
        cellText=rows_a,
        colLabels=hdr_a,
        loc="upper center",
        cellLoc="center",
    )
    tbl_a.auto_set_font_size(False)
    tbl_a.set_fontsize(11)
    tbl_a.scale(1.0, 2.4)
    for (r, c), cell in tbl_a.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#eaf2f8" if r % 2 else "white")

    # --- Table B: radius ---
    hdr_b = ["Mask radius r", "n", "AUC", "TPR @ 1%", "TPR @ 5%", "Best acc"]
    rows_b = [
        ["8 (mean, key ×1.0)", "20", "0.79", "0.10", "0.40", "0.78"],
        ["10", "20", "0.73", "0.00", "0.30", "0.72"],
        ["12", "20", "0.80", "0.20", "0.30", "0.78"],
    ]
    ax_b = fig.add_axes([0.05, 0.10, 0.9, 0.34])
    ax_b.set_axis_off()
    ax_b.set_title("B. Fourier mask radius (mean, key ×1.0, JPEG only)", fontsize=12, fontweight="bold", loc="left", pad=12)
    tbl_b = ax_b.table(
        cellText=rows_b,
        colLabels=hdr_b,
        loc="upper center",
        cellLoc="center",
    )
    tbl_b.auto_set_font_size(False)
    tbl_b.set_fontsize(11)
    tbl_b.scale(1.0, 2.6)
    for (r, c), cell in tbl_b.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#eaf2f8" if r % 2 else "white")

    fig.text(
        0.05,
        0.02,
        "Metrics: compute_sd_eval_metrics.py (distance threshold sweep). "
        "Section A: n=50 except median n=20. Section B: preliminary n=20 — confirm r at n=50.",
        fontsize=9,
        color="#333",
        verticalalignment="bottom",
    )

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
