"""Test cases for Tree-Ring watermarking."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import unittest
import numpy as np
from diffusion_watermarking.tree_ring import (
    inject_watermark_noise_latent,
    detect_tree_ring,
    build_key_for_detection,
)


class TestTreeRing(unittest.TestCase):
    def test_inject_and_detect_zeros(self):
        """Watermarked noise with key_type=zeros should be detected."""
        latent_shape = (4, 64, 64)
        noise = inject_watermark_noise_latent(
            latent_shape, key_type="zeros", radius=10, seed=42
        )
        result = detect_tree_ring(noise, key_type="zeros", radius=10, seed=42, return_p_value=True)
        self.assertTrue(result["is_watermarked"], f"Expected watermarked, got {result}")

    def test_inject_and_detect_rand(self):
        """Watermarked noise with key_type=rand yields valid detection result (distance < random)."""
        latent_shape = (4, 64, 64)
        noise = inject_watermark_noise_latent(
            latent_shape, key_type="rand", radius=10, seed=42
        )
        result = detect_tree_ring(noise, key_type="rand", radius=10, seed=42, return_p_value=True)
        rng = np.random.default_rng(99)
        random_noise = rng.standard_normal(latent_shape).astype(np.float32)
        result_rand = detect_tree_ring(random_noise, key_type="rand", radius=10, seed=42, return_p_value=True)
        self.assertLess(result["distance"], result_rand["distance"], "Watermarked should have smaller distance than random")

    def test_inject_and_detect_rings(self):
        """Watermarked noise with key_type=rings yields valid detection (distance < random)."""
        latent_shape = (4, 64, 64)
        noise = inject_watermark_noise_latent(
            latent_shape, key_type="rings", radius=10, seed=42
        )
        result = detect_tree_ring(noise, key_type="rings", radius=10, seed=42, return_p_value=True)
        rng = np.random.default_rng(99)
        random_noise = rng.standard_normal(latent_shape).astype(np.float32)
        result_rand = detect_tree_ring(random_noise, key_type="rings", radius=10, seed=42, return_p_value=True)
        self.assertLess(result["distance"], result_rand["distance"], "Watermarked should have smaller distance than random")

    def test_random_noise_not_detected(self):
        """Plain random noise should have much larger distance from key than watermarked noise."""
        latent_shape = (4, 64, 64)
        # Watermarked noise: close to key -> small distance
        watermarked = inject_watermark_noise_latent(
            latent_shape, key_type="rings", radius=10, seed=42
        )
        res_w = detect_tree_ring(watermarked, key_type="rings", radius=10, seed=42, return_p_value=True)
        # Random noise: far from key -> large distance
        rng = np.random.default_rng(99)
        random_noise = rng.standard_normal(latent_shape).astype(np.float32)
        result = detect_tree_ring(
            random_noise, key_type="rings", radius=10, seed=42, return_p_value=True
        )
        self.assertGreater(
            result["distance"], res_w["distance"] * 5,
            "Random noise should have much larger distance than watermarked"
        )

    def test_build_key_shape(self):
        """Key shape should match (h, w) for 2D key."""
        h, w, r = 64, 64, 10
        key, mask = build_key_for_detection((h, w), "rings", r, seed=42)
        self.assertEqual(key.shape, (h, w))
        self.assertEqual(mask.shape, (h, w))

    def test_annulus_inject_detect_rings(self):
        """Annulus mask (radius_inner > 0) should still inject/detect consistently."""
        latent_shape = (4, 64, 64)
        noise = inject_watermark_noise_latent(
            latent_shape, key_type="rings", radius=10, radius_inner=4, seed=42
        )
        result = detect_tree_ring(
            noise, key_type="rings", radius=10, radius_inner=4, seed=42, return_p_value=True
        )
        rng = np.random.default_rng(99)
        random_noise = rng.standard_normal(latent_shape).astype(np.float32)
        result_rand = detect_tree_ring(
            random_noise,
            key_type="rings",
            radius=10,
            radius_inner=4,
            seed=42,
            return_p_value=True,
        )
        self.assertLess(
            result["distance"],
            result_rand["distance"],
            "Annulus watermarked noise should be closer to key than random",
        )

    def test_empty_annulus_raises(self):
        with self.assertRaises(ValueError):
            inject_watermark_noise_latent(
                (4, 64, 64), key_type="rings", radius=5, radius_inner=5, seed=0
            )

    def test_detect_median_min_dist_rings(self):
        """Median and min_dist aggregation should still detect injected rings noise."""
        latent_shape = (4, 64, 64)
        noise = inject_watermark_noise_latent(
            latent_shape, key_type="rings", radius=10, seed=42
        )
        for agg in ("median", "min_dist"):
            result = detect_tree_ring(
                noise, key_type="rings", radius=10, seed=42, return_p_value=True, channel_agg=agg
            )
            rng = np.random.default_rng(99)
            random_noise = rng.standard_normal(latent_shape).astype(np.float32)
            result_rand = detect_tree_ring(
                random_noise, key_type="rings", radius=10, seed=42, return_p_value=True, channel_agg=agg
            )
            self.assertLess(
                result["distance"],
                result_rand["distance"],
                f"Watermarked should beat random with channel_agg={agg}",
            )
