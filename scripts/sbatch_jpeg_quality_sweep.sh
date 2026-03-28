#!/usr/bin/env bash
#SBATCH --job-name=jpeg_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=jpeg_sweep_%j.log

set -euo pipefail

REPO_URL="https://github.com/Feiyang0303/Diffusion_watermarking.git"
REPO_DIR="$HOME/diffusion_watermarking"

# Clone or update repo (compute nodes have network access)
if [ -d "$REPO_DIR/.git" ]; then
  echo "Repo exists, pulling latest..."
  cd "$REPO_DIR"
  git pull origin main
else
  echo "Cloning repo..."
  git clone "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

# Set up venv if needed
if [ ! -f .venv/bin/activate ]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

export PYTHONPATH="$REPO_DIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

echo ""
echo "=== Starting JPEG quality sweep (n=50) ==="
echo ""

NUM_SAMPLES=50 bash scripts/run_jpeg_quality_sweep.sh

echo ""
echo "=== Job complete ==="
nvidia-smi
