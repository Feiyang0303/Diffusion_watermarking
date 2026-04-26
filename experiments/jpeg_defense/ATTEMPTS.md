# JPEG defense attempts — metrics log

**Protocol:** Stable Diffusion v1.5, DDIM 50 steps, Tree-Ring `rings`, attack **JPEG quality 25**, metrics from `compute_sd_eval_metrics.py` (distance threshold sweep).

## A. Detector / embedding knobs (fixed radius 10 unless noted)

| Attempt | n | Key settings | AUC | TPR @ 1% | TPR @ 5% | Best acc | Notes |
|---------|---|--------------|-----|----------|----------|----------|--------|
| First channel (baseline) | 50 | `first`, k=1.0 | 0.75 | 0.10 | 0.30 | 0.69 | From multi-attack snapshot |
| Mean | 50 | `mean`, k=1.0 | 0.749 | 0.06 | 0.10 | 0.70 | JPEG-only CSV |
| Mean | 50 | `mean`, k=1.12 | 0.747 | 0.04 | 0.08 | 0.70 | JPEG-only CSV |
| Median, r=10 | 20 | `median`, k=1.0 | 0.62 | 0.30 | 0.35 | 0.65 | [`runs/median_n20/`](runs/median_n20/) |
| Median, r=10 | 50 | `median`, k=1.0 | 0.65 | 0.24 | 0.32 | 0.65 | AUC 0.653 raw — [`runs/median_r10_n50/`](runs/median_r10_n50/) |
| Median × r (full sweep) | 50 | `median`, k=1.0, r∈{8,10,12} | — | — | — | — | `run_jpeg_median_radius_ablation.sh`; see §D |
| Min-dist, r=10 | 20 | `min_dist`, k=1.0 | 0.83 | 0.30 | 0.30 | 0.82 | [`runs/min_dist_n20/`](runs/min_dist_n20/) |

## B. Fourier mask radius (JPEG only, mean, k=1.0)

| radius | n | AUC | TPR @ 1% | TPR @ 5% | Best acc |
|--------|---|-----|----------|----------|----------|
| 8 | 20 | 0.79 | 0.10 | 0.40 | 0.78 |
| 10 | 20 | 0.73 | 0.00 | 0.30 | 0.72 |
| 12 | 20 | 0.80 | 0.20 | 0.30 | 0.78 |

**WatGPU folder:** `outputs_tree_ring_sd_eval_jpeg_radius/`  
**Confirm at n=50** for r=8 and/or r=12 before strong claims. For the combined approaches figure, `scp` `metrics_jpeg_radius{R}_n50.csv` into [`runs/mean_radius_n50/`](runs/mean_radius_n50/) (see README there); until then the PNG uses **n=20 ablation numbers** with **n=50** in the sample-size column.

## C. Min-dist + Fourier radius (JPEG only, k=1.0), **all n=50**

WatGPU files: `metrics_jpeg_min_dist_r8_n50.csv`, `metrics_jpeg_min_dist_r10_n50.csv`, `metrics_jpeg_min_dist_r12_n50.csv` (same folder). **All three radii** are produced by `scripts/run_jpeg_min_dist_radius_ablation.sh`.

**In-repo:** [`runs/min_dist_radius_n50/`](runs/min_dist_radius_n50/) — `metrics_jpeg_min_dist_r{8,10,12}_n50.csv`. **r=8:** AUC **0.89** (0.8928 raw), TPR@1% **0.26**, TPR@5% **0.62**, best acc **0.83**. **r=10:** AUC **0.90** (0.8966 raw), TPR@1% **0.30**, TPR@5% **0.58**, best acc **0.83**. **r=12:** AUC **0.90** (0.9026 raw), TPR@1% **0.26**, TPR@5% **0.58**, best acc **0.85**.

**WatGPU folder:** `outputs_tree_ring_sd_eval_jpeg_min_dist_radius/` (gitignored).

## D. Median + Fourier radius (JPEG only, k=1.0)

| radius | n | AUC | TPR @ 1% | TPR @ 5% | Best acc | Notes |
|--------|---|-----|----------|----------|----------|--------|
| 8 | 50 | — | — | — | — | `metrics_jpeg_median_r8_n50.csv` |
| 10 | 50 | 0.65 | 0.24 | 0.32 | 0.65 | Matches [`runs/median_r10_n50/metrics_jpeg_median_r10_n50.csv`](runs/median_r10_n50/metrics_jpeg_median_r10_n50.csv) (AUC 0.653 raw) |
| 12 | 50 | — | — | — | — | `metrics_jpeg_median_r12_n50.csv` |

**WatGPU folder:** `outputs_tree_ring_sd_eval_jpeg_median_radius/` (gitignored).  
**Script:** `NUM_SAMPLES=50 bash scripts/run_jpeg_median_radius_ablation.sh`

## E. Min-dist × JPEG quality sweep (k=1.0, n=50)

How does `min_dist` detection hold up under varying JPEG compression strength?
Sweeps **Q ∈ {10, 15, 25, 50, 75}** × **R ∈ {6, 8, 10, 12, 14}**. Lower **Q** = stronger lossy compression (PIL `Image.save(..., format="JPEG", quality=Q)` in `run_tree_ring_sd_eval.py`).

**Script:** `NUM_SAMPLES=50 bash scripts/run_jpeg_quality_sweep.sh` (r=8,10,12); extended radii: `scripts/run_jpeg_quality_sweep_r6_r14.sh`  
**In-repo:** [`runs/jpeg_quality_sweep_n50/`](runs/jpeg_quality_sweep_n50/) — one metrics CSV per (radius, quality): `metrics_min_dist_r{R}_q{Q}_n50.csv`  
**Figure:** [`results/jpeg_quality_sweep.png`](../../results/jpeg_quality_sweep.png)

### JPEG quality levels (compression conditions tried)

| Quality `Q` | Compression (qualitative) | Role in this study |
|-------------|---------------------------|--------------------|
| **10** | Very strong | Stress test; heavy blocking / HF loss |
| **15** | Strong | Between harsh and paper baseline |
| **25** | Strong (paper-style) | Matches Tree-Ring paper / rest of this doc |
| **50** | Moderate | Typical “web-ish” save |
| **75** | Mild | Light compression; near-lossless visually |

### Metrics by radius × quality (columns = compression level)

| Radius | Q=10 | Q=15 | Q=25 | Q=50 | Q=75 |
|--------|------|------|------|------|------|
| **AUC** | | | | | |
| r=6 | 0.71 | 0.79 | 0.88 | 0.92 | 0.94 |
| r=8 | 0.76 | 0.83 | 0.89 | 0.94 | 0.97 |
| r=10 | 0.77 | 0.86 | 0.90 | 0.94 | 0.96 |
| r=12 | 0.80 | 0.88 | 0.90 | 0.93 | 0.96 |
| r=14 | 0.84 | 0.88 | 0.90 | 0.93 | 0.96 |
| **TPR @ 1% FPR** | | | | | |
| r=6 | 0.08 | 0.26 | 0.22 | 0.26 | 0.24 |
| r=8 | 0.18 | 0.36 | 0.26 | 0.20 | 0.30 |
| r=10 | 0.26 | 0.30 | 0.30 | 0.20 | 0.36 |
| r=12 | 0.40 | 0.42 | 0.26 | 0.26 | 0.28 |
| r=14 | 0.38 | 0.42 | 0.28 | 0.20 | 0.24 |
| **TPR @ 5% FPR** | | | | | |
| r=6 | 0.28 | 0.34 | 0.66 | 0.52 | 0.52 |
| r=8 | 0.42 | 0.44 | 0.62 | 0.58 | 0.74 |
| r=10 | 0.44 | 0.46 | 0.58 | 0.54 | 0.74 |
| r=12 | 0.46 | 0.60 | 0.58 | 0.60 | 0.82 |
| r=14 | 0.50 | 0.58 | 0.64 | 0.68 | 0.84 |
| **Best accuracy** | | | | | |
| r=6 | 0.66 | 0.75 | 0.81 | 0.89 | 0.90 |
| r=8 | 0.72 | 0.80 | 0.83 | 0.90 | 0.93 |
| r=10 | 0.74 | 0.80 | 0.83 | 0.91 | 0.94 |
| r=12 | 0.75 | 0.81 | 0.85 | 0.91 | 0.94 |
| r=14 | 0.79 | 0.83 | 0.85 | 0.92 | 0.94 |

**Key findings:**
- Detection degrades as compression strengthens (moving left in the table): AUC at Q10 is much lower than at Q75 for every radius.
- **Wider mask (larger r)** helps most when compression is harsh (Q10–15); **r=14** leads there, **r=6** lags.
- Q25 numbers for r=8/10/12 match Section C (e.g. r=10 AUC 0.90), confirming reproducibility.
- Under mild compression (Q50+), radii converge to similar AUC (~0.92–0.97).

## Example commands

```bash
# Mean, k=1.0, n=50
.venv/bin/python run_tree_ring_sd_eval.py --num_samples 50 --attack jpeg --jpeg_quality 25 \
  --detect_channel_agg mean --key_scale 1.0 --out_csv outputs_tree_ring_sd_eval_jpeg_updated/sd_eval_jpeg_mean_k1.csv \
  --out_dir outputs_tree_ring_sd_eval_jpeg_updated --save_images 0

# Radius=12, n=50
.venv/bin/python run_tree_ring_sd_eval.py --num_samples 50 --attack jpeg --jpeg_quality 25 \
  --detect_channel_agg mean --key_scale 1.0 --radius 12 \
  --out_csv outputs_tree_ring_sd_eval_jpeg_radius/sd_eval_jpeg_radius12_n50.csv \
  --out_dir outputs_tree_ring_sd_eval_jpeg_radius --save_images 0

# Min-dist × radius r∈{8,10,12}
NUM_SAMPLES=50 bash scripts/run_jpeg_min_dist_radius_ablation.sh

# Median × radius r∈{8,10,12} (JPEG Q25, n=50 per radius)
NUM_SAMPLES=50 bash scripts/run_jpeg_median_radius_ablation.sh

# Min-dist × JPEG quality sweep — Q∈{10,15,25,50,75}, n=50
NUM_SAMPLES=50 bash scripts/run_jpeg_quality_sweep.sh          # R∈{8,10,12}
NUM_SAMPLES=50 bash scripts/run_jpeg_quality_sweep_r6_r14.sh    # R∈{6,14}
```

Then always: `compute_sd_eval_metrics.py --csv <path> --out_dir <dir> --out_prefix <name>`.
