# Slide-ready tables + one-line explanations (JPEG Q=25)

Copy tables into PowerPoint / Google Slides / Keynote. **n** differs between rows — note on slide: *“Radius rows: n=20 (preliminary); detector rows: n=50 unless noted.”*

---

## Table 1 — Detector & key tweaks (Fourier mask radius **r = 10**)

| Approach | n | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best acc |
|----------|---|-----|--------------|--------------|----------|
| First channel only (baseline) | 50 | 0.75 | 0.10 | 0.30 | 0.69 |
| Mean across latent channels | 50 | 0.75 | 0.06 | 0.10 | 0.70 |
| Mean + stronger key (×1.12) | 50 | 0.75 | 0.04 | 0.08 | 0.70 |
| Median across latent channels | 20 | 0.62 | 0.30 | 0.35 | 0.65 |

---

## Table 2 — Fourier mask radius (mean channels, key ×1.0)

| Mask radius r | n | AUC | TPR @ 1% FPR | TPR @ 5% FPR | Best acc |
|----------------|---|-----|--------------|--------------|----------|
| 8 | 20 | 0.79 | 0.10 | 0.40 | 0.78 |
| 10 | 20 | 0.73 | 0.00 | 0.30 | 0.72 |
| 12 | 20 | 0.80 | 0.20 | 0.30 | 0.78 |

---

## LaTeX (Beamer / paper snippet)

```latex
\begin{tabular}{lcccccc}
\toprule
Approach & $n$ & AUC & TPR@1\% & TPR@5\% & Best acc \\
\midrule
First channel & 50 & 0.75 & 0.10 & 0.30 & 0.69 \\
Mean, $k{=}1.0$ & 50 & 0.75 & 0.06 & 0.10 & 0.70 \\
Mean, $k{=}1.12$ & 50 & 0.75 & 0.04 & 0.08 & 0.70 \\
Median, $k{=}1.0$ & 20 & 0.62 & 0.30 & 0.35 & 0.65 \\
\midrule
$r{=}8$ (mean, $k{=}1.0$) & 20 & 0.79 & 0.10 & 0.40 & 0.78 \\
$r{=}10$ & 20 & 0.73 & 0.00 & 0.30 & 0.72 \\
$r{=}12$ & 20 & 0.80 & 0.20 & 0.30 & 0.78 \\
\bottomrule
\end{tabular}
```

---

## What each approach means (speaker notes)

1. **First channel (baseline)**  
   After DDIM inversion you get 4 latent noise maps. Use **only channel 0**, FFT, then match the Tree-Ring key in a **low-frequency disk** (radius *r*). This matches the original “single-channel” style usage.

2. **Mean across channels**  
   **Average** the four channels **in space** (same pixel across channels), then one FFT. Inversion noise is partly independent per channel; averaging can **reduce variance** in the estimate before the spectral test.

3. **Mean + stronger key (key_scale 1.12)**  
   Same as mean, but the Fourier key in the mask is **scaled by 1.12** at **both** generation and detection. Idea: **larger signature** vs. JPEG noise (tradeoff: possible artifacts).

4. **Median across channels**  
   **Per-pixel median** over the four channels, then FFT. Idea: if JPEG/inversion **breaks one channel**, median is **less pulled** by that outlier than the mean.

5. **Fourier mask radius *r* (8 / 10 / 12)**  
   Tree-Ring writes the key in a **disk of radius *r*** in the noise FFT. **Larger *r*** = more frequency bins carry the key (more capacity / possibly more visible); **smaller *r*** = tighter low-frequency pattern. Rows use **mean** + **k=1.0**, JPEG Q25 only.

6. **Min-dist channel** (implemented, not in table)  
   FFT **each** channel separately, compute distance to key, take the **minimum** (best channel). Not JPEG-evaluated in your numbers yet.

**Metric reminder:** **Lower** Tree-Ring **distance** ⇒ more likely watermarked. **AUC** = overall separability; **TPR @ 1% FPR** = hit rate when false alarms are capped at 1%.
