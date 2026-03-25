#!/usr/bin/env bash
# JPEG-only Tree-Ring eval: min_dist detector, radius in {8,10,12}.
# Run on GPU after: salloc, cd repo, source .venv, export PYTHONPATH=..
# Usage:
#   NUM_SAMPLES=20 bash scripts/run_jpeg_min_dist_radius_ablation.sh
#   NUM_SAMPLES=50 bash scripts/run_jpeg_min_dist_radius_ablation.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
NUM_SAMPLES="${NUM_SAMPLES:-20}"
OUT_BASE="${OUT_BASE:-outputs_tree_ring_sd_eval_jpeg_min_dist_radius}"
PY="${PY:-.venv/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

for R in 8 10 12; do
  echo "========== min_dist  radius=$R  num_samples=$NUM_SAMPLES =========="
  "$PY" run_tree_ring_sd_eval.py \
    --num_samples "$NUM_SAMPLES" \
    --attack jpeg \
    --jpeg_quality 25 \
    --detect_channel_agg min_dist \
    --key_scale 1.0 \
    --radius "$R" \
    --out_dir "$OUT_BASE" \
    --out_csv "$OUT_BASE/sd_eval_jpeg_min_dist_r${R}_n${NUM_SAMPLES}.csv" \
    --save_images 0

  "$PY" compute_sd_eval_metrics.py \
    --csv "$OUT_BASE/sd_eval_jpeg_min_dist_r${R}_n${NUM_SAMPLES}.csv" \
    --out_dir "$OUT_BASE" \
    --out_prefix "metrics_jpeg_min_dist_r${R}_n${NUM_SAMPLES}"
done

echo "Done. Output under $OUT_BASE"
