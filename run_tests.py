#!/usr/bin/env python3
"""Run test cases (no pytest required)."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
# Parent must be on path so "diffusion_watermarking" package is found.
# On Linux (e.g. watgpu) the repo folder must be named "diffusion_watermarking" (lowercase).
if root.name != "diffusion_watermarking":
    print("Warning: repo folder is '%s'. On Linux, rename to 'diffusion_watermarking' (lowercase) so tests can import the package." % root.name)
    print("  cd .. && mv %s diffusion_watermarking && cd diffusion_watermarking" % root.name)
research = root.parent
sys.path.insert(0, str(research))
sys.path.insert(0, str(root))

import unittest

# Import test modules (they add parent to path for diffusion_watermarking)
from tests import test_tree_ring, test_watermark_dm, test_watermark_dm_recipe


if __name__ == "__main__":
    # unittest.Runner doesn't exist - use TextTestRunner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromModule(test_tree_ring))
    suite.addTests(loader.loadTestsFromModule(test_watermark_dm))
    suite.addTests(loader.loadTestsFromModule(test_watermark_dm_recipe))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
