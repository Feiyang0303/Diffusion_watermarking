#!/bin/bash
# Run Tree-Ring SD eval on GPU (n=5, n=20, n=50), then compute metrics and plot.
# Usage (activate venv first): ./run_eval_gpu.sh   or   bash run_eval_gpu.sh
# To use a specific GPU: CUDA_VISIBLE_DEVICES=1 ./run_eval_gpu.sh

set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
OUT_DIR="${OUT_DIR:-outputs_tree_ring_sd_eval}"
PAPER_ATTACKS="none,jpeg,crop,rotation,blur,noise,color_jitter"
PY="${PY:-python}"   # use .venv: PY=.venv/bin/python ./run_eval_gpu.sh

# Use GPU (optional: set CUDA_VISIBLE_DEVICES if you want a specific GPU)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Tree-Ring SD eval on GPU (out_dir=$OUT_DIR) ==="

for N in 5 20 50; do
  CSV="$OUT_DIR/sd_eval_n${N}.csv"
  if [ -f "$CSV" ]; then
    echo "Skipping n=$N (already exists: $CSV)"
  else
    echo "--- n=$N ---"
    PYTHONPATH=.. "$PY" run_tree_ring_sd_eval.py \
      --num_samples "$N" \
      --attacks "$PAPER_ATTACKS" \
      --out_dir "$OUT_DIR" \
      --out_csv "$CSV"
  fi
done

echo "=== Computing metrics (metrics_n5, metrics_n20, metrics_n50) ==="
for N in 5 20 50; do
  CSV="$OUT_DIR/sd_eval_n${N}.csv"
  PYTHONPATH=.. "$PY" compute_sd_eval_metrics.py \
    --csv "$CSV" \
    --out_dir "$OUT_DIR" \
    --out_prefix "metrics_n${N}"
done

echo "=== Plotting ==="
PYTHONPATH=.. "$PY" plot_n5_n20_n50.py --dir "$OUT_DIR"
echo "Done. Plot: $OUT_DIR/metrics_n5_n20_n50.png"
