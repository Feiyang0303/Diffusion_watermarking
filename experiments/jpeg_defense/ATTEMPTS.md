# JPEG defense attempts — metrics log

**Protocol:** Stable Diffusion v1.5, DDIM 50 steps, Tree-Ring `rings`, attack **JPEG quality 25**, metrics from `compute_sd_eval_metrics.py` (distance threshold sweep).

## A. Detector / embedding knobs (fixed radius 10 unless noted)

| Attempt | n | Key settings | AUC | TPR @ 1% | TPR @ 5% | Best acc | Notes |
|---------|---|--------------|-----|----------|----------|----------|--------|
| First channel (baseline) | 50 | `first`, k=1.0 | 0.75 | 0.10 | 0.30 | 0.69 | From multi-attack snapshot |
| Mean | 50 | `mean`, k=1.0 | 0.749 | 0.06 | 0.10 | 0.70 | JPEG-only CSV |
| Mean | 50 | `mean`, k=1.12 | 0.747 | 0.04 | 0.08 | 0.70 | JPEG-only CSV |
| Median | 20 | `median`, k=1.0 | 0.62 | 0.30 | 0.35 | 0.65 | Preliminary; high variance |
| Min-dist, r=10 | 20 | `min_dist`, k=1.0 | 0.83 | 0.30 | 0.30 | 0.82 | [`runs/min_dist_n20/`](runs/min_dist_n20/) |

## B. Fourier mask radius (JPEG only, mean, k=1.0)

| radius | n | AUC | TPR @ 1% | TPR @ 5% | Best acc |
|--------|---|-----|----------|----------|----------|
| 8 | 20 | 0.79 | 0.10 | 0.40 | 0.78 |
| 10 | 20 | 0.73 | 0.00 | 0.30 | 0.72 |
| 12 | 20 | 0.80 | 0.20 | 0.30 | 0.78 |

**WatGPU folder:** `outputs_tree_ring_sd_eval_jpeg_radius/`  
**Confirm at n=50** for r=8 and/or r=12 before strong claims.

## C. Min-dist + Fourier radius (JPEG only, k=1.0)

| radius | n | AUC | TPR @ 1% | TPR @ 5% | Best acc | Notes |
|--------|---|-----|----------|----------|----------|--------|
| 8 | 20 | 0.86 | 0.20 | 0.60 | 0.80 | Preliminary |
| 10 | 20 | 0.83 | 0.30 | 0.30 | 0.82 | Same as §A min-dist row |
| 12 | 50 | 0.90 | 0.26 | 0.58 | 0.85 | WatGPU `metrics_jpeg_min_dist_r12_n50` (AUC 0.9026 raw) |

**WatGPU folder:** `outputs_tree_ring_sd_eval_jpeg_min_dist_radius/` (gitignored).  
**Script:** `scripts/run_jpeg_min_dist_radius_ablation.sh`. If §C r=12 row is wrong radius, check which `metrics_jpeg_min_dist_r*_n50` file the paste came from.

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
```

Then always: `compute_sd_eval_metrics.py --csv <path> --out_dir <dir> --out_prefix <name>`.
