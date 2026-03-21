# Tree-Ring + Stable Diffusion — snapshot metrics (n=50)

**Setup:** SD v1.5, DDIM, Tree-Ring `rings` key, image-level eval with attacks (paper-style).  
**Detection:** DDIM inversion → Tree-Ring distance in Fourier mask (lower ⇒ more likely watermarked).  
**Metrics:** threshold sweep on distance → AUC, TPR at fixed FPR, best accuracy.

| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |
|--------|-------|-----|--------------|--------------|----------|
| none | — | 0.97 | 0.70 | 0.88 | 0.94 |
| blur | 8.0 | 0.72 | 0.28 | 0.28 | 0.75 |
| color_jitter | 6.0 | 0.91 | 0.04 | 0.10 | 0.93 |
| crop | 0.75 | 0.89 | 0.14 | 0.36 | 0.83 |
| jpeg | 25.0 | 0.75 | 0.10 | 0.30 | 0.69 |
| noise | 0.1 | 0.86 | 0.16 | 0.34 | 0.80 |
| rotation | 75.0 | 0.85 | 0.12 | 0.36 | 0.78 |
| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |

*This file is a frozen snapshot for the README; regenerate with `compute_sd_eval_metrics.py` after new eval runs.*
