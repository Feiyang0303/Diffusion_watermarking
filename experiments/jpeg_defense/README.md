# JPEG (Q=25) robustness experiments — Tree-Ring + SD v1.5

All attempts to **improve detection after JPEG compression** (defensive rate / separability), in one place.

**Detailed narrative:** [`WHAT_I_TRIED.md`](WHAT_I_TRIED.md) · **Slide tables:** [`SLIDES.md`](SLIDES.md) · **Table as PNG:** [`results/jpeg_approaches_table.png`](../results/jpeg_approaches_table.png) — one combined table (`make_jpeg_approaches_table.py`; min-dist **r=12 n=50** filled from WatGPU)

## Quick index

| # | Attempt | Where artifacts live | Result (summary) |
|---|---------|----------------------|------------------|
| 1 | **Baseline** — first channel, `radius=10`, `key_scale=1.0`, n=50 | [`results/metrics_snapshot_n50.md`](../../results/metrics_snapshot_n50.md) (JPEG row) | AUC ~0.75, TPR@1% 0.10 |
| 2 | **Mean** channels, k=1.0, n=50 | WatGPU: `outputs_tree_ring_sd_eval_jpeg_updated/sd_eval_jpeg_mean_k1.csv` | AUC ~flat; best acc +0.01; **TPR@low FPR worse** |
| 3 | **Mean** channels, k=1.12, n=50 | WatGPU: `…/sd_eval_jpeg_mean_k112.csv` | Similar to k=1.0; strict FPR still weak |
| 4 | **Median** channels, k=1.0, n=20 | **[`runs/median_n20/`](runs/median_n20/)** (local copy) | AUC ~0.62 @ n=20 — not better |
| 5 | **Fourier mask radius** 8 / 10 / 12 — mean, k=1.0, n=20 | WatGPU: `outputs_tree_ring_sd_eval_jpeg_radius/` | **r=8 & 12** best AUC (~0.79–0.80); **r=10** weak on this run |
| 6 | **Min-dist** × r, k=1.0 | **`runs/min_dist_n20/`** + `outputs_tree_ring_sd_eval_jpeg_min_dist_radius/` | **r=12, n=50:** AUC **~0.90**, TPR@1% **0.26**; r=8/10 still **n=20** in table until you paste n=50 |
| — | **Figures / narrative** | [`results/jpeg_report_summary.png`](../../results/jpeg_report_summary.png), [`results/jpeg_q25_detector_comparison.md`](../../results/jpeg_q25_detector_comparison.md) | One-pager + baseline vs mean writeup |

## Folder layout

```
experiments/jpeg_defense/
├── README.md              ← you are here
├── ATTEMPTS.md            ← full metrics table + reproduction commands
└── runs/
    ├── median_n20/        ← median-channel n=20
    └── min_dist_n20/      ← min-dist channel n=20 (JPEG)
```

**Large GPU outputs** stay under the repo root (gitignored) so clones stay small:

- `outputs_tree_ring_sd_eval_jpeg_updated/` — mean / median / k112 runs  
- `outputs_tree_ring_sd_eval_jpeg_radius/` — radius ablation (CSV, metrics, ROC PNGs)

Copy what you need into `runs/<name>/` for archiving or `scp` to your laptop.

## Reproduce

- **Radius ablation (n=20 or 50):**  
  [`scripts/run_jpeg_radius_ablation.sh`](../../scripts/run_jpeg_radius_ablation.sh)  
  `NUM_SAMPLES=20 bash scripts/run_jpeg_radius_ablation.sh`

- **Min-dist + radius (r ∈ {8,10,12}):**  
  [`scripts/run_jpeg_min_dist_radius_ablation.sh`](../../scripts/run_jpeg_min_dist_radius_ablation.sh)  
  `NUM_SAMPLES=20 bash scripts/run_jpeg_min_dist_radius_ablation.sh`  
  Then refresh `MIN_DIST_RADIUS_ROWS` in [`make_jpeg_approaches_table.py`](../../make_jpeg_approaches_table.py) and regenerate the PNG.

- **Single JPEG eval:**  
  `run_tree_ring_sd_eval.py --attack jpeg --jpeg_quality 25 …`  
  See [`ATTEMPTS.md`](ATTEMPTS.md).

- **Summary figure (images + table PNG):** from repo root,  
  `MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_report_figure.py`

## Takeaway

- **Channel tricks** (mean / median / k-scale) did **not** clearly beat the **first-channel baseline** on strict FPR at n=50.  
- **Min-dist, r=12, n=50:** **AUC ~0.90** vs. baseline **~0.75** — strongest JPEG result so far; still mind **FPR** (best-of-4).  
- **Radius ablation** at n=20 suggests **r=8 or 12** may beat **r=10** under JPEG — **confirm with n=50** on WatGPU.
