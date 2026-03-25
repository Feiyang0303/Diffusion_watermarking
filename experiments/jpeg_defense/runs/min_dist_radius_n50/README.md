# Min-dist × radius, n=50 (WatGPU metrics)

Copy the three metrics CSVs from WatGPU, then regenerate the big table PNG:

```bash
# From your laptop (repo root)
mkdir -p experiments/jpeg_defense/runs/min_dist_radius_n50
scp 'f82xu@watgpu.cs.uwaterloo.ca:~/diffusion_watermarking/outputs_tree_ring_sd_eval_jpeg_min_dist_radius/metrics_jpeg_min_dist_r{8,10,12}_n50.csv' \
  experiments/jpeg_defense/runs/min_dist_radius_n50/

MPLCONFIGDIR=$PWD/.mplconfig python3 make_jpeg_approaches_table.py
```

If those three files are present, `make_jpeg_approaches_table.py` fills the **min-dist r=8 / 10 / 12** rows from them (all **n=50**). Otherwise it falls back to hardcoded values in the script.
