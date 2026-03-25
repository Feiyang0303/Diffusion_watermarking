# Mean channel × Fourier radius, JPEG Q25, n=50

WatGPU:

```bash
NUM_SAMPLES=50 bash scripts/run_jpeg_radius_ablation.sh
```

Produces `outputs_tree_ring_sd_eval_jpeg_radius/metrics_jpeg_radius{8,10,12}_n50.csv`.

Copy here so `make_jpeg_approaches_table.py` uses **real n=50** cells (otherwise the figure uses n=20 ablation metrics with an n=50 column label):

```bash
scp f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_radius/metrics_jpeg_radius{8,10,12}_n50.csv \
  experiments/jpeg_defense/runs/mean_radius_n50/
```
