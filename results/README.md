# Results (curated)

This folder holds **figures and tables** meant for the GitHub landing page. Large experiment dumps stay in `outputs_*` (gitignored); update these files when you want the public view to reflect a new run.

| File | Description |
|------|-------------|
| `tree_ring_sd_n50_roc.png` | ROC curves per attack (n=50, SD eval). |
| `tree_ring_sd_n50_distances.png` | Distance distributions (watermarked vs clean) per attack. |
| `montage_sample0_all_attacks.png` | One sample: watermarked image under each attack (visual sanity check). |
| `sample_clean.png` / `sample_watermarked.png` / `sample_watermarked_jpeg_q25.png` | Qualitative comparison (same prompt & seed family). |
| `metrics_snapshot_n50.md` | Markdown table of AUC / TPR@FPR / accuracy (copy of metrics at snapshot time). |
| `jpeg_q25_detector_comparison.md` | Baseline vs. updated JPEG detector (first vs. mean channel, n=50). |
| `jpeg_report_summary.png` | One-page figure: samples + JPEG metrics table (`make_jpeg_report_figure.py`). |
| `jpeg_approaches_table.png` | Two-panel table: detector/key tweaks + radius ablation (`make_jpeg_approaches_table.py`). |

**Full log of JPEG-defense experiments (mean / median / radius / k-scale):** [`../experiments/jpeg_defense/README.md`](../experiments/jpeg_defense/README.md).

## Reproduce

```bash
# Full attack suite → CSV
PYTHONPATH=.. python run_tree_ring_sd_eval.py \
  --num_samples 50 \
  --attacks none,jpeg,crop,rotation,blur,noise,color_jitter \
  --out_dir outputs_tree_ring_sd_eval_paper \
  --out_csv outputs_tree_ring_sd_eval_paper/sd_eval_n50.csv

# Metrics table
PYTHONPATH=.. python compute_sd_eval_metrics.py \
  --csv outputs_tree_ring_sd_eval_paper/sd_eval_n50.csv \
  --out_dir outputs_tree_ring_sd_eval_paper \
  --out_prefix metrics_n50

# Plots
PYTHONPATH=.. python plot_robustness.py \
  --csv outputs_tree_ring_sd_eval_paper/sd_eval_n50.csv \
  --out_dir outputs_tree_ring_sd_eval_paper \
  --prefix robustness_paper_n50
```

Then copy chosen PNGs and refresh `metrics_snapshot_n50.md` into `results/` if you want the repo view to match.
