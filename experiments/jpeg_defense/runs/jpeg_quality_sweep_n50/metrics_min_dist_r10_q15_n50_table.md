# Tree-Ring detection metrics (image-level SD eval)

Lower distance = more likely watermarked. Threshold swept to compute ROC.

| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |
|--------|-------|-----|--------------|--------------|----------|
| jpeg | 15 | 0.86 | 0.30 | 0.46 | 0.80 |
| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |

