# JPEG (Q=25) detection: baseline vs. updated detector

Side-by-side metrics from **`compute_sd_eval_metrics.py`** on **n=50** watermarked / **n=50** clean pairs, **same attack** (JPEG quality 25), Stable Diffusion v1.5, Tree-Ring `rings`, radius 10, base seed 42.

| Setting | Detector | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best accuracy |
|---------|----------|-----|--------------|--------------|---------------|
| **Baseline** | First latent channel only (`channel_agg=first`), `key_scale=1.0` | **0.75** | **0.10** | **0.30** | **0.69** |
| **Updated** | Mean over latent channels (`channel_agg=mean`), `key_scale=1.0` | **0.749** | **0.06** | **0.10** | **0.70** |
| Δ (updated − baseline) | | −0.001 | −0.04 | −0.20 | +0.01 |

### How to read this

- **AUC** is essentially unchanged (~0.75): overall separability of distances is similar.
- **Best accuracy** improves slightly (**0.69 → 0.70**) at the Youden-optimal threshold.
- **TPR at low FPR** is **lower** in the updated run (**0.10 → 0.06** at 1% FPR; **0.30 → 0.10** at 5% FPR), meaning under a **strict** false-positive budget the mean-channel detector **underperforms** the first-channel baseline on this JPEG-only CSV.

So the update is **not** a uniform win on JPEG for every metric: channel averaging changes the **distance distribution** and moves the best operating point. Try **`--key_scale` > 1** (e.g. 1.12) on a fresh eval if the goal is higher TPR at fixed FPR.

### Reproduce

**Baseline row:** from the multi-attack n=50 run (`sd_eval_n50.csv`); JPEG row in [`metrics_snapshot_n50.md`](metrics_snapshot_n50.md).

**Updated row:** WatGPU run  
`run_tree_ring_sd_eval.py --num_samples 50 --attack jpeg --jpeg_quality 25 --detect_channel_agg mean --key_scale 1.0 …`  
→ `compute_sd_eval_metrics.py --csv …/sd_eval_jpeg_mean_k1.csv --out_prefix metrics_jpeg_mean_k1`

*Numbers rounded for display; baseline AUC/TPR from frozen snapshot table; updated row from `metrics_jpeg_mean_k1` summary (AUC 0.7494).*
