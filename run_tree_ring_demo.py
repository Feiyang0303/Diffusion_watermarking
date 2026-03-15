#!/usr/bin/env python3
"""
Minimal Tree-Ring demo using only numpy/scipy (no diffusion model).
Shows key construction, watermarked noise generation, and detection metric.
For full pipeline with Stable Diffusion use run_tree_ring_sd.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_watermarking.tree_ring import (
    inject_watermark_noise_latent,
    detect_tree_ring,
    build_key_for_detection,
)
import numpy as np


def main():
    print("Tree-Ring Watermark demo (numpy only)")
    print("-------------------------------------")
    latent_shape = (4, 64, 64)  # e.g. SD latent
    radius = 10
    seed = 42

    for key_type in ("zeros", "rand", "rings"):
        print("\nKey type:", key_type)
        # Generate watermarked noise
        noise = inject_watermark_noise_latent(
            latent_shape,
            key_type=key_type,
            radius=radius,
            seed=seed,
        )
        # Simulate "inverted" noise (in real pipeline this comes from DDIM inversion of generated image)
        # Here we use the same watermarked noise to verify detection
        result = detect_tree_ring(
            noise,
            key_type=key_type,
            radius=radius,
            seed=seed,
            return_p_value=True,
        )
        print("  distance = {:.4f}, p_value = {:.2e}, is_watermarked = {}".format(
            result["distance"], result["p_value"], result["is_watermarked"]
        ))

    # Non-watermarked noise should have high p_value
    print("\nNon-watermarked (random) noise:")
    random_noise = np.random.default_rng(99).standard_normal(latent_shape).astype(np.float32)
    result_rand = detect_tree_ring(
        random_noise,
        key_type="rings",
        radius=radius,
        seed=seed,
        return_p_value=True,
    )
    print("  distance = {:.4f}, p_value = {:.2e}, is_watermarked = {}".format(
        result_rand["distance"], result_rand["p_value"], result_rand["is_watermarked"]
    ))
    print("\nDone. Install diffusers/torch and run run_tree_ring_sd.py for full SD pipeline.")


if __name__ == "__main__":
    main()
