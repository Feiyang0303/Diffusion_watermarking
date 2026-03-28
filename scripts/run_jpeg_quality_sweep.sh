#!/usr/bin/env bash
# JPEG quality sweep: min_dist detector × radius × JPEG quality.
# Extends Section C of ATTEMPTS.md (min_dist r∈{8,10,12} at Q25) to multiple
# compression strengths so we can see where min_dist breaks down.
#
# Usage:
#   NUM_SAMPLES=20 bash scripts/run_jpeg_quality_sweep.sh        # quick
#   NUM_SAMPLES=50 bash scripts/run_jpeg_quality_sweep.sh        # paper-grade
#
# Override defaults via env:
#   QUALITIES="10 25 75"  RADII="10"  NUM_SAMPLES=20  bash scripts/run_jpeg_quality_sweep.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

NUM_SAMPLES="${NUM_SAMPLES:-20}"
QUALITIES="${QUALITIES:-10 15 25 50 75}"
RADII="${RADII:-8 10 12}"
OUT_BASE="${OUT_BASE:-outputs_jpeg_quality_sweep}"
PY="${PY:-.venv/bin/python}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

echo "JPEG quality sweep (min_dist)"
echo "  Q ∈ {$QUALITIES}"
echo "  R ∈ {$RADII}"
echo "  n = $NUM_SAMPLES"
echo "  Output: $OUT_BASE"
echo ""

for R in $RADII; do
  for Q in $QUALITIES; do
    TAG="min_dist_r${R}_q${Q}_n${NUM_SAMPLES}"
    echo "========== radius=$R  quality=$Q  n=$NUM_SAMPLES =========="
    "$PY" run_tree_ring_sd_eval.py \
      --num_samples "$NUM_SAMPLES" \
      --attack jpeg \
      --jpeg_quality "$Q" \
      --detect_channel_agg min_dist \
      --key_scale 1.0 \
      --radius "$R" \
      --out_dir "$OUT_BASE" \
      --out_csv "$OUT_BASE/sd_eval_${TAG}.csv" \
      --save_images 0

    "$PY" compute_sd_eval_metrics.py \
      --csv "$OUT_BASE/sd_eval_${TAG}.csv" \
      --out_dir "$OUT_BASE" \
      --out_prefix "metrics_${TAG}"
  done
done

echo ""
echo "========== Summary =========="
echo ""
printf "%-8s %-8s %-8s %-14s %-14s %-10s\n" "Radius" "Quality" "AUC" "TPR@1%FPR" "TPR@5%FPR" "BestAcc"
printf "%-8s %-8s %-8s %-14s %-14s %-10s\n" "------" "-------" "---" "---------" "---------" "-------"
for R in $RADII; do
  for Q in $QUALITIES; do
    TAG="min_dist_r${R}_q${Q}_n${NUM_SAMPLES}"
    METRICS="$OUT_BASE/metrics_${TAG}.csv"
    if [ -f "$METRICS" ]; then
      # Parse the first data row (skip header and random_baseline)
      ROW=$(awk -F, 'NR==2{print}' "$METRICS")
      AUC=$(echo "$ROW" | cut -d, -f5)
      TPR1=$(echo "$ROW" | cut -d, -f6)
      TPR5=$(echo "$ROW" | cut -d, -f7)
      ACC=$(echo "$ROW" | cut -d, -f8)
      printf "%-8s %-8s %-8s %-14s %-14s %-10s\n" "$R" "$Q" "$AUC" "$TPR1" "$TPR5" "$ACC"
    fi
  done
done
echo ""
echo "All outputs under $OUT_BASE/"
echo "Done."
