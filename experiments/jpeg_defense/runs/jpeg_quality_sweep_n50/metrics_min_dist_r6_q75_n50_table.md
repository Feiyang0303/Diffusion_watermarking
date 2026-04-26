# Tree-Ring detection metrics (image-level SD eval)

Lower distance = more likely watermarked. Threshold swept to compute ROC.

| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |
|--------|-------|-----|--------------|--------------|----------|
| jpeg | 75 | 0.94 | 0.24 | 0.52 | 0.90 |
| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |

