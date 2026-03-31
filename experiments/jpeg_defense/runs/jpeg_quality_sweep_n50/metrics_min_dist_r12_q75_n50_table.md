# Tree-Ring detection metrics (image-level SD eval)

Lower distance = more likely watermarked. Threshold swept to compute ROC.

| Attack | Param | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best Acc |
|--------|-------|-----|--------------|--------------|----------|
| jpeg | 75 | 0.96 | 0.28 | 0.82 | 0.94 |
| *Random* | — | 0.50 | 0.01 | 0.05 | 0.50 |

