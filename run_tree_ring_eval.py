#!/usr/bin/env python3
"""
Tree-Ring latent-level evaluation script.

Runs many trials of:
- watermarked noise vs. random (unwatermarked) noise
- computes detection distance, p-value, and is_watermarked flag

Outputs a CSV you can use to reproduce paper-style plots:
- distance distributions
- p-value histograms
- ROC curves (by thresholding distance or p-value)
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from diffusion_watermarking.tree_ring import (
    inject_watermark_noise_latent,
    detect_tree_ring,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-Ring latent-level evaluation")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of random trials")
    parser.add_argument(
        "--latent_shape",
        type=str,
        default="4,64,64",
        help="Latent shape as C,H,W (default: 4,64,64 for SD v1.5)",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="rings",
        choices=["zeros", "rand", "rings"],
        help="Tree-Ring key type",
    )
    parser.add_argument("--radius", type=int, default=10, help="Tree-Ring radius")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--wm_perturb_std",
        type=float,
        default=0.0,
        help="Std of additive Gaussian noise applied to WATERMARKED latents before detection "
        "(mimics inversion/attack error; set e.g. 0.05).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="outputs_tree_ring/latent_eval.csv",
        help="Path to CSV file for logging results",
    )
    args = parser.parse_args()

    c, h, w = [int(x) for x in args.latent_shape.split(",")]
    latent_shape = (c, h, w)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Tree-Ring latent-level evaluation")
    print("---------------------------------")
    print(f"key={args.key}, radius={args.radius}, seed={args.seed}")
    if args.wm_perturb_std > 0:
        print(f"watermarked perturb std={args.wm_perturb_std}")
    print(f"latent_shape={latent_shape}, num_samples={args.num_samples}")
    print(f"Writing CSV to {out_path}")

    rng = np.random.default_rng(args.seed)

    n_wm_detected = 0
    n_rand_detected = 0
    wm_distances = []
    rand_distances = []

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_idx",
                "type",  # watermarked or random
                "distance",
                "eta",
                "sigma_sq",
                "p_value",
                "is_watermarked",
            ]
        )

        for i in range(args.num_samples):
            # Watermarked latent
            wm_noise = inject_watermark_noise_latent(
                latent_shape,
                key_type=args.key,
                radius=args.radius,
                seed=args.seed,  # fixed key
                noise_seed=args.seed + 1000 + i,  # vary base noise per-sample (paper-style)
            )
            if args.wm_perturb_std > 0:
                wm_noise = wm_noise + rng.standard_normal(latent_shape).astype(np.float32) * args.wm_perturb_std
            res_wm = detect_tree_ring(
                wm_noise,
                key_type=args.key,
                radius=args.radius,
                seed=args.seed,
                return_p_value=True,
            )
            writer.writerow(
                [
                    i,
                    "watermarked",
                    res_wm["distance"],
                    res_wm["eta"],
                    res_wm["sigma_sq"],
                    res_wm["p_value"],
                    int(res_wm["is_watermarked"]),
                ]
            )
            wm_distances.append(res_wm["distance"])
            if res_wm["is_watermarked"]:
                n_wm_detected += 1

            # Random (unwatermarked) latent
            rand_noise = rng.standard_normal(latent_shape).astype(np.float32)
            res_rand = detect_tree_ring(
                rand_noise,
                key_type=args.key,
                radius=args.radius,
                seed=args.seed,
                return_p_value=True,
            )
            writer.writerow(
                [
                    i,
                    "random",
                    res_rand["distance"],
                    res_rand["eta"],
                    res_rand["sigma_sq"],
                    res_rand["p_value"],
                    int(res_rand["is_watermarked"]),
                ]
            )
            rand_distances.append(res_rand["distance"])
            if res_rand["is_watermarked"]:
                n_rand_detected += 1

            if (i + 1) % max(1, args.num_samples // 10) == 0:
                print(f"  completed {i + 1}/{args.num_samples} samples")

    wm_distances = np.array(wm_distances)
    rand_distances = np.array(rand_distances)

    print("\nSummary:")
    print(f"  watermarked mean distance    = {wm_distances.mean():.4f}")
    print(f"  random mean distance         = {rand_distances.mean():.4f}")
    print(f"  watermarked detected fraction= {n_wm_detected / args.num_samples:.3f}")
    print(f"  random detected fraction     = {n_rand_detected / args.num_samples:.3f}")
    print(f"\nSaved per-sample results to {out_path}")
    print("You can now make distance histograms and ROC curves based on this CSV.")


if __name__ == "__main__":
    main()

