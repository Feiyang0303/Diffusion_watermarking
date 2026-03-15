#!/usr/bin/env python3
"""
Run demos that produce visible outputs for comparison with the papers.
Outputs: images, detection results, training metrics and curves.
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Run demos to get actual outputs (images, metrics)")
    parser.add_argument("--tree_ring", action="store_true", help="Run Tree-Ring: generate watermarked + clean images, run detection")
    parser.add_argument("--watermark_dm", action="store_true", help="Run WatermarkDM training with metrics and sample images")
    parser.add_argument("--all", action="store_true", help="Run both demos")
    parser.add_argument("--tree_ring_prompt", type=str, default="A photo of an astronaut riding a horse on Mars")
    parser.add_argument("--tree_ring_out", type=str, default="outputs_tree_ring")
    parser.add_argument("--watermark_dm_epochs", type=int, default=15)
    parser.add_argument("--watermark_dm_out", type=str, default="outputs_watermark_dm")
    args = parser.parse_args()

    if not (args.tree_ring or args.watermark_dm or args.all):
        print("Choose --tree_ring, --watermark_dm, or --all")
        print()
        print("Outputs:")
        print("  Tree-Ring:    <out_dir>/watermarked.png, clean.png, detection_result.txt")
        print("  WatermarkDM:  <out_dir>/training_metrics.csv, samples_*.png, training_curves.png")
        sys.exit(1)

    run_both = args.all or (args.tree_ring and args.watermark_dm)
    if args.all:
        args.tree_ring = args.watermark_dm = True

    # Ensure we can import the package (parent on path)
    parent = ROOT.parent
    env = {**__import__("os").environ, "PYTHONPATH": str(parent)}

    if args.tree_ring:
        print("=" * 60)
        print("Tree-Ring demo (generates watermarked.png, clean.png, detection_result.txt)")
        print("=" * 60)
        out = Path(args.tree_ring_out)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "run_tree_ring_sd.py"),
            "--mode", "both",
            "--key", "rings",
            "--prompt", args.tree_ring_prompt,
            "--out_dir", str(out),
        ]
        subprocess.run(cmd, env=env, cwd=str(ROOT))
        print("Tree-Ring outputs in:", out.absolute())
        print("  - watermarked.png, clean.png: compare visually (watermark is invisible)")
        print("  - detection_result.txt: distance, p_value, is_watermarked")
        print()

    if args.watermark_dm:
        print("=" * 60)
        print("WatermarkDM demo (training + metrics + sample images)")
        print("=" * 60)
        out = Path(args.watermark_dm_out)
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "run_train_watermark_dm.py"),
            "--epochs", str(args.watermark_dm_epochs),
            "--num_samples", "200",
            "--batch_size", "16",
            "--out_dir", str(out),
            "--save", str(out / "checkpoints"),
        ]
        subprocess.run(cmd, env=env, cwd=str(ROOT))
        print("WatermarkDM outputs in:", out.absolute())
        print("  - training_metrics.csv: epoch, loss, bit_accuracy")
        print("  - samples_original.png, samples_watermarked.png: visual comparison")
        print("  - training_curves.png: loss and bit accuracy vs epoch (if matplotlib installed)")
        print("  - checkpoints/: encoder.pt, decoder.pt")
        print()

    print("Done. Open the output directories to view results and compare with the papers.")


if __name__ == "__main__":
    main()
