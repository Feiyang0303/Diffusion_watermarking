# Local copies of JPEG experiment outputs

| Subfolder | Contents |
|-----------|----------|
| **`median_n20/`** | Median-channel ablation at n=20 (CSV + metrics table). |
| **`min_dist_n20/`** | Min-dist channel detector, JPEG Q25, n=20 (CSV + metrics). |
| **`min_dist_radius_n50/`** | Drop `metrics_jpeg_min_dist_r{8,10,12}_n50.csv` here → auto-fill [`make_jpeg_approaches_table.py`](../../make_jpeg_approaches_table.py). |
| **`median_r10_n50/`** | Median channel, r=10, JPEG n=50 metrics CSV → approaches table + report figure. |
| **`mean_radius_n50/`** | `metrics_jpeg_radius{8,10,12}_n50.csv` from `run_jpeg_radius_ablation.sh` → mean×r rows in approaches table. |
| **`k112_smoke/`** | Leftover smoke-test folder (can delete if empty). |

**Full radius ablation + mean/k112 CSVs** (large) stay on WatGPU / gitignored dirs:

- `outputs_tree_ring_sd_eval_jpeg_radius/`
- `outputs_tree_ring_sd_eval_jpeg_median_radius/` — median × r (n=50 script)
- `outputs_tree_ring_sd_eval_jpeg_updated/`

Copy into a new subfolder here after `scp`, e.g.:

```bash
scp -r f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_radius/ \
  experiments/jpeg_defense/runs/radius_ablation_n20/
```
