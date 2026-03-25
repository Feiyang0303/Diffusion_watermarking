#!/usr/bin/env python3
"""
Render JPEG-defense approaches as one combined table PNG (slides / reports).
Run from repo root:
  MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_approaches_table.py

Optional CSV dirs under experiments/jpeg_defense/runs/:
  - min_dist_radius_n50/metrics_jpeg_min_dist_r{R}_n50.csv
  - mean_radius_n50/metrics_jpeg_radius{R}_n50.csv  (from run_jpeg_radius_ablation.sh, NUM_SAMPLES=50)
  - median_r10_n50/metrics_jpeg_median_r10_n50.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fallback when runs/min_dist_radius_n50/*.csv are missing (mixed n=20 / n=50).
MIN_DIST_RADIUS_ROWS: dict[int, tuple[str, str, str, str, str]] = {
    8: ("50", "0.89", "0.26", "0.62", "0.83"),
    10: ("50", "0.90", "0.30", "0.58", "0.83"),
    12: ("50", "0.90", "0.26", "0.58", "0.85"),
}

# Mean × radius: n=20 WatGPU ablation numbers; n column forced to 50 until mean_radius_n50/*.csv overwrites.
MEAN_RADIUS_ROWS_N50_DISPLAY: dict[int, tuple[str, str, str, str, str]] = {
    8: ("50", "0.79", "0.10", "0.40", "0.78"),
    10: ("50", "0.73", "0.00", "0.30", "0.72"),
    12: ("50", "0.80", "0.20", "0.30", "0.78"),
}


def _parse_jpeg_metrics_csv(path: Path) -> tuple[str, str, str, str, str] | None:
    """Return (n_wm, auc, tpr1, tpr5, best_acc) from the jpeg row, or None."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("attack") or "").strip() != "jpeg":
                continue
            n_wm = int(float(row["n_wm"]))
            return (
                str(n_wm),
                f"{float(row['auc']):.2f}",
                f"{float(row['tpr_at_1pct_fpr']):.2f}",
                f"{float(row['tpr_at_5pct_fpr']):.2f}",
                f"{float(row['best_accuracy']):.2f}",
            )
    return None


def _merge_min_dist_rows(root: Path) -> tuple[dict[int, tuple[str, str, str, str, str]], str]:
    """Overlay runs/min_dist_radius_n50/metrics_jpeg_min_dist_r{R}_n50.csv onto fallback (per radius)."""
    d = root / "experiments/jpeg_defense/runs/min_dist_radius_n50"
    rows = dict(MIN_DIST_RADIUS_ROWS)
    loaded: list[str] = []
    for r in (8, 10, 12):
        p = d / f"metrics_jpeg_min_dist_r{r}_n50.csv"
        if not p.is_file():
            continue
        parsed = _parse_jpeg_metrics_csv(p)
        if parsed is None:
            continue
        rows[r] = parsed
        loaded.append(f"r{r}=file")
    if not loaded:
        note = "fallback MIN_DIST_RADIUS_ROWS (scp metrics CSVs → runs/min_dist_radius_n50/; do not quote {8,10,12})"
    elif len(loaded) == 3:
        note = "min-dist: all r∈{8,10,12} from runs/min_dist_radius_n50/*.csv"
    else:
        note = "min-dist: " + ", ".join(loaded) + "; missing radii use script fallback"
    return rows, note


def _load_median_r10_n50_row(root: Path) -> tuple[tuple[str, str, str, str, str], str]:
    """Median, r=10, k=1.0 from runs/median_r10_n50/metrics_jpeg_median_r10_n50.csv if present."""
    p = root / "experiments/jpeg_defense/runs/median_r10_n50/metrics_jpeg_median_r10_n50.csv"
    fallback = ("20", "0.62", "0.30", "0.35", "0.65")
    if not p.is_file():
        return fallback, "median r=10: fallback n=20 slice"
    parsed = _parse_jpeg_metrics_csv(p)
    if parsed is None:
        return fallback, "median r=10: parse failed, fallback n=20"
    n_wm, a, b, c, d = parsed
    n_wm = str(int(float(n_wm)))
    return (n_wm, a, b, c, d), f"median r=10: n={n_wm} from runs/median_r10_n50/*.csv"


def _merge_mean_radius_rows(root: Path) -> tuple[dict[int, tuple[str, str, str, str, str]], str]:
    """runs/mean_radius_n50/metrics_jpeg_radius{R}_n50.csv (per run_jpeg_radius_ablation.sh)."""
    d = root / "experiments/jpeg_defense/runs/mean_radius_n50"
    rows = dict(MEAN_RADIUS_ROWS_N50_DISPLAY)
    loaded: list[str] = []
    for r in (8, 10, 12):
        p = d / f"metrics_jpeg_radius{r}_n50.csv"
        if not p.is_file():
            continue
        parsed = _parse_jpeg_metrics_csv(p)
        if parsed is None:
            continue
        rows[r] = parsed
        loaded.append(f"r{r}=file")
    if not loaded:
        note = "mean×r: n=50 column; metrics still n=20 ablation — add runs/mean_radius_n50/metrics_jpeg_radius{R}_n50.csv"
    elif len(loaded) == 3:
        note = "mean×r: all r∈{8,10,12} from runs/mean_radius_n50/*.csv"
    else:
        note = "mean×r: " + ", ".join(loaded) + "; missing radii use n=20 ablation numbers @ n=50 label"
    return rows, note


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

    min_dist_rows, min_dist_source = _merge_min_dist_rows(root)
    median_row_vals, median_source = _load_median_r10_n50_row(root)
    mean_rows, mean_source = _merge_mean_radius_rows(root)

    hdr = ["Setting", "n", "AUC", "TPR @ 1%", "TPR @ 5%", "Best acc"]
    rows: list[list[str]] = [
        # r=10 default mask; detector / key tweaks
        ["First channel (baseline), r=10, k=1.0", "50", "0.75", "0.10", "0.30", "0.69"],
        ["Mean latent ch., r=10, k=1.0", "50", "0.75", "0.06", "0.10", "0.70"],
        ["Mean latent ch., r=10, k=1.12", "50", "0.75", "0.04", "0.08", "0.70"],
        ["Median latent ch., r=10, k=1.0", *median_row_vals],
        # Mean + radius (JPEG); prefer n=50 metrics CSV, else n=20 ablation @ n=50 column
        ["Mean, r=8, k=1.0", *mean_rows[8]],
        ["Mean, r=10, k=1.0", *mean_rows[10]],
        ["Mean, r=12, k=1.0", *mean_rows[12]],
        # Min-dist + radius (JPEG)
        [
            "Min-dist (best ch.), r=8, k=1.0",
            *min_dist_rows[8],
        ],
        [
            "Min-dist (best ch.), r=10, k=1.0",
            *min_dist_rows[10],
        ],
        [
            "Min-dist (best ch.), r=12, k=1.0",
            *min_dist_rows[12],
        ],
    ]

    nrows = len(rows)
    fig_h = max(11.0, 4.5 + 0.42 * nrows)
    fig = plt.figure(figsize=(17.5, fig_h), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Tree-Ring + SD v1.5 — JPEG (Q=25): all approaches (one table)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.set_axis_off()

    tbl = ax.table(
        cellText=rows,
        colLabels=hdr,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.15)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a5276")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#eaf2f8" if r % 2 else "white")

    fig.text(
        0.04,
        0.015,
        "Metrics: compute_sd_eval_metrics.py (distance threshold sweep). "
        "Top block: detector rows @ r=10. "
        f"{mean_source}. "
        f"{median_source}. "
        f"Min-dist × radius: {min_dist_source}.",
        fontsize=8.5,
        color="#333",
        verticalalignment="bottom",
    )

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}")
    print(mean_source)
    print(median_source)
    print(min_dist_source)


if __name__ == "__main__":
    main()
