#!/usr/bin/env python3
"""
Render JPEG-defense approaches as one combined table PNG (slides / reports).
Run from repo root:
  MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_approaches_table.py

Optional: copy metrics_jpeg_min_dist_r{8,10,12}_n50.csv into
experiments/jpeg_defense/runs/min_dist_radius_n50/ — table uses them automatically.
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
    8: ("20", "0.86", "0.20", "0.60", "0.80"),
    10: ("20", "0.83", "0.30", "0.30", "0.82"),
    12: ("50", "0.90", "0.26", "0.58", "0.85"),
}


def _parse_min_dist_metrics_csv(path: Path) -> tuple[str, str, str, str, str] | None:
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
        parsed = _parse_min_dist_metrics_csv(p)
        if parsed is None:
            continue
        rows[r] = parsed
        loaded.append(f"r{r}=file")
    if not loaded:
        note = "fallback MIN_DIST_RADIUS_ROWS (scp metrics CSVs → runs/min_dist_radius_n50/; do not quote {8,10,12})"
    else:
        note = "min-dist: " + ", ".join(loaded) + "; other radii fallback"
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

    hdr = ["Setting", "n", "AUC", "TPR @ 1%", "TPR @ 5%", "Best acc"]
    rows: list[list[str]] = [
        # r=10 default mask; detector / key tweaks
        ["First channel (baseline), r=10, k=1.0", "50", "0.75", "0.10", "0.30", "0.69"],
        ["Mean latent ch., r=10, k=1.0", "50", "0.75", "0.06", "0.10", "0.70"],
        ["Mean latent ch., r=10, k=1.12", "50", "0.75", "0.04", "0.08", "0.70"],
        ["Median latent ch., r=10, k=1.0", "20", "0.62", "0.30", "0.35", "0.65"],
        # Mean + radius (JPEG only, n=20)
        ["Mean, r=8, k=1.0", "20", "0.79", "0.10", "0.40", "0.78"],
        ["Mean, r=10, k=1.0", "20", "0.73", "0.00", "0.30", "0.72"],
        ["Mean, r=12, k=1.0", "20", "0.80", "0.20", "0.30", "0.78"],
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
        "n=50 where noted; else n=20 (preliminary). "
        "Mean r=10 row (n=20) is the radius ablation slice — not the same run as Mean r=10 (n=50). "
        f"Min-dist × radius: {min_dist_source}.",
        fontsize=8.5,
        color="#333",
        verticalalignment="bottom",
    )

    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out_path}")
    print(min_dist_source)


if __name__ == "__main__":
    main()
