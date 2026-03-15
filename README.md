# Diffusion Model Watermarking

Implementations of two approaches from the literature:

1. **Tree-Ring Watermarks** (Wen et al., arXiv:2305.20030) – training-free, invisible fingerprints in the Fourier space of the initial noise.
2. **WatermarkDM-style pipelines** (Zhao et al., arXiv:2303.10137) – binary watermark encoder/decoder for unconditional DMs, and trigger-prompt watermarking for text-to-image DMs.

## Papers

- **2305.20030** – *Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*  
  Watermark is embedded in the initial noise vector’s Fourier space (circular mask). Detection via DDIM inversion + key matching. No training; works as a plug-in for any diffusion model.

- **2303.10137** – *A Recipe for Watermarking Diffusion Models*  
  - *Unconditional/class-conditional:* Train encoder \(E_\phi\) and decoder \(D_\phi\) to embed a binary string in data; train the DM on watermarked data; decode with \(D_\phi\) from generated images.  
  - *Text-to-image:* Finetune a pretrained DM with a (trigger prompt, watermark image) pair and weight-constrained regularization \(\lambda \|\theta - \hat\theta\|_1\).

## Structure

```
diffusion_watermarking/
├── tree_ring.py          # Tree-Ring key construction, injection, detection (numpy/scipy)
├── watermark_dm.py       # WatermarkDM: encoder/decoder nets + text-to-image loss helpers
├── run_tree_ring_demo.py # Demo Tree-Ring with numpy only (no SD)
├── run_tree_ring_sd.py   # Full Tree-Ring + Stable Diffusion (generate & detect)
├── requirements.txt
└── README.md
```

## Quick Start

### Tree-Ring (no GPU required for demo)

```bash
cd research/code/diffusion_watermarking
pip install numpy scipy
python run_tree_ring_demo.py
```

### Tree-Ring with Stable Diffusion

```bash
pip install -r requirements.txt
python run_tree_ring_sd.py --mode both --key rings --prompt "A cat on a sofa"
# Outputs: outputs_tree_ring/watermarked.png, outputs_tree_ring/clean.png, and detection result
```

Options: `--key zeros|rand|rings`, `--radius`, `--seed`, `--steps`.

### Viewing actual results (images, metrics, paper comparison)

To get **visible outputs** (images, detection results, training curves) for analysis and comparison with the papers:

**Tree-Ring (Stable Diffusion):**  
Generates watermarked and clean images, runs detection, and saves a summary.

```bash
PYTHONPATH=.. python run_tree_ring_sd.py --mode both --key rings --prompt "A cat on a sofa" --out_dir outputs_tree_ring
```

- **outputs_tree_ring/watermarked.png** – image generated with Tree-Ring watermark  
- **outputs_tree_ring/clean.png** – same prompt, no watermark (for visual comparison)  
- **outputs_tree_ring/detection_result.txt** – distance, p_value, is_watermarked (for Table/Fig comparison)

**WatermarkDM (training + metrics + samples):**  
Trains encoder/decoder and writes metrics and sample images.

```bash
PYTHONPATH=.. python run_train_watermark_dm.py --epochs 20 --out_dir outputs_watermark_dm --save outputs_watermark_dm/checkpoints
```

- **outputs_watermark_dm/training_metrics.csv** – epoch, loss, bit_accuracy (for training curves)  
- **outputs_watermark_dm/samples_original.png**, **samples_watermarked.png** – grid of originals vs watermarked  
- **outputs_watermark_dm/training_curves.png** – loss and bit accuracy vs epoch (if matplotlib installed)  
- **outputs_watermark_dm/checkpoints/** – encoder.pt, decoder.pt

**One command for both demos:**

```bash
PYTHONPATH=.. python run_demos.py --all
```

Use `--tree_ring_prompt` and `--watermark_dm_epochs` to customize. Output directories are listed at the end.

### Run tests

From the `diffusion_watermarking` directory. On macOS with Homebrew Python, use a virtual environment first:

```bash
cd diffusion_watermarking
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_tests.py
```

(To run only Tree-Ring tests without PyTorch: `pip install numpy scipy` then `python run_tests.py`.)

**WatermarkDM recipe tests** (`tests/test_watermark_dm_recipe.py`): follow the paper recipe (Pipeline 1)—train encoder/decoder with Eq. (2), build watermarked data E(x,w), and assert decode bit accuracy. Optional full recipe with a minimal diffusion model trained on watermarked data: `RUN_WATERMARKDM_FULL_RECIPE=1 python run_tests.py`.

Or with pytest: `PYTHONPATH=.. pytest tests/ -v`

### Train WatermarkDM encoder/decoder on GPU

Requires PyTorch (and a CUDA-capable GPU for faster training). Uses synthetic data by default; replace with your dataset for real training.

```bash
pip install -r requirements.txt
cd research/diffusion_watermarking
PYTHONPATH=.. python run_train_watermark_dm.py --device cuda --epochs 20 --save checkpoints/watermark_dm
```

Options: `--device cuda|cpu`, `--epochs`, `--batch_size`, `--lr`, `--gamma`, `--bit_length`, `--image_size`, `--num_samples`, `--save DIR`.

**Train on Tiny ImageNet 200:** point `--data_dir` to the dataset root (train split: `train/<class>/images/*.JPEG`). Requires `torchvision` and `Pillow`.

```bash
PYTHONPATH=.. python run_train_watermark_dm.py \
  --data_dir /path/to/tiny-imagenet-200 \
  --image_size 64 \
  --epochs 30 \
  --batch_size 32 \
  --device cuda \
  --save checkpoints/watermark_dm_tinyimagenet
```

### WatermarkDM encoder/decoder (library)

Use `watermark_dm.py` for the binary encoder/decoder (PyTorch). Training a full DM on watermarked data requires a separate training loop (e.g. with EDM or a small diffusion model); the module provides the networks and the loss (Eq. 2). For text-to-image trigger watermarking, the loss in Eq. (5) is implemented as `text_to_image_watermark_loss` and `get_weight_penalty_l1`; actual finetuning of Stable Diffusion is left to your training setup.

## References

- Wen et al., *Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust*, arXiv:2305.20030.
- Zhao et al., *A Recipe for Watermarking Diffusion Models*, arXiv:2303.10137.

---

## Path on this machine

Project folder (absolute path):

```
/Users/feiyangxu/Downloads/Feiyang Xu/Code/research/diffusion_watermarking
```

In Terminal: `cd "/Users/feiyangxu/Downloads/Feiyang Xu/Code/research/diffusion_watermarking"`

## GitHub

This repo is a Git project. To push to GitHub:

1. **Create a new repo on GitHub** (e.g. `diffusion_watermarking`), leave it empty (no README).

2. **From this folder:**
   ```bash
   cd "/Users/feiyangxu/Downloads/Feiyang Xu/Code/research/diffusion_watermarking"
   git add .
   git commit -m "Initial commit: Tree-Ring + WatermarkDM"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/diffusion_watermarking.git
   git push -u origin main
   ```
   Replace `YOUR_USERNAME` with your GitHub username (or use the SSH URL from GitHub).

## Running on watgpu (Linux)

On Linux the repo directory name is case-sensitive. Python expects the package `diffusion_watermarking` (lowercase). After cloning you’ll have `Diffusion_watermarking`; use one of these:

**Option A – Rename the folder (simplest)**  
From your home directory:
```bash
cd ~
mv Diffusion_watermarking diffusion_watermarking
cd diffusion_watermarking
source .venv/bin/activate
python run_tests.py
```

**Option B – Symlink (keep folder name)**  
From your home directory:
```bash
cd ~
ln -s Diffusion_watermarking diffusion_watermarking
cd Diffusion_watermarking
source .venv/bin/activate
python run_tests.py
```

Then install deps and run tests as in “Run tests” above (e.g. `pip install -r requirements.txt` then `python run_tests.py`).
