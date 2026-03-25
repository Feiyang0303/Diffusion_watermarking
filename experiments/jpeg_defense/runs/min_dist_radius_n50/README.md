# Min-dist × radius, n=50 metrics (for `make_jpeg_approaches_table.py`)

Drop `metrics_jpeg_min_dist_r{8,10,12}_n50.csv` here. The figure script **merges per radius**: any file present overrides the fallback in `make_jpeg_approaches_table.py`.

**Committed in git:** `metrics_jpeg_min_dist_r12_n50.csv` (WatGPU n=50 run). **r=8 and r=10:** copy from WatGPU after `NUM_SAMPLES=50 bash scripts/run_jpeg_min_dist_radius_ablation.sh`.

## scp (zsh / bash) — do **not** put quotes around `{8,10,12}`

Single quotes prevent **local** brace expansion, so the remote sees a literal `r{8,10,12}` and fails.

```bash
# From repo root — unquoted brace expansion → three remote paths
mkdir -p experiments/jpeg_defense/runs/min_dist_radius_n50
scp f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_min_dist_radius/metrics_jpeg_min_dist_r{8,10,12}_n50.csv \
  experiments/jpeg_defense/runs/min_dist_radius_n50/

MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_approaches_table.py
```

Or three explicit paths (works if braces are awkward):

```bash
scp \
  f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_min_dist_radius/metrics_jpeg_min_dist_r8_n50.csv \
  f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_min_dist_radius/metrics_jpeg_min_dist_r10_n50.csv \
  f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_min_dist_radius/metrics_jpeg_min_dist_r12_n50.csv \
  experiments/jpeg_defense/runs/min_dist_radius_n50/
```
