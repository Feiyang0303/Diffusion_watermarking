# What we tried: JPEG (Q=25) robustness for Tree-Ring + Stable Diffusion

This document explains **in detail** every defense-oriented change we explored: what the pipeline is, what each knob does, how we evaluated it, and what we observed.

---

## 1. Background: Tree-Ring on Stable Diffusion

**Embedding:** The initial latent noise (4×64×64 for SD 1.5) is built so that, in the **2D FFT** of each channel, coefficients inside a **circular mask** (center = low frequencies) match a **secret key** (`rings` pattern from a fixed seed). Other frequencies stay random so the image looks like a normal sample.

**Detection:** Given a **generated image**, we VAE-encode, run **DDIM inversion** (50 steps in your setup) to approximate the initial noise, take the latent tensor, apply the detector (see below), FFT, and measure **L1 distance** between masked Fourier coefficients and the same key. **Lower distance** ⇒ closer to “watermarked.”

**JPEG attack:** Save the image as **JPEG quality 25**, then detect. JPEG is **lossy and nonlinear** (DCT blocks, quantization, chroma subsampling), so it perturbs the pixel → VAE → inversion chain and usually **hurts** frequency-domain fingerprints.

**Metrics:** From paired **watermarked vs. clean** images, `compute_sd_eval_metrics.py` sweeps a threshold on **distance** and reports **AUC**, **TPR at 1% / 5% FPR**, and **best accuracy** (Youden-optimal threshold). For deployment, **TPR @ low FPR** is often as important as AUC.

---

## 2. Baseline: first latent channel only (`channel_agg=first`)

**What we did:** After inversion, use **only channel 0** as the 2D field, FFT, then Tree-Ring distance and p-value. Fourier mask **radius = 10** (paper-style default). **key_scale = 1.0** (nominal key strength).

**Why it’s the baseline:** Matches the simplest “read one channel” story and the historical comparison in your multi-attack **n=50** snapshot.

**JPEG numbers (n=50, from snapshot / JPEG row):** AUC **~0.75**, TPR@1% **0.10**, TPR@5% **0.30**, best acc **~0.69**.

---

## 3. Mean aggregation across latent channels (`channel_agg=mean`)

**What we did:** After inversion, **average** the four latent channels **in space** (same (h,w) across channels), then **one** FFT and the same masked L1 test. **key_scale = 1.0**. Still **r = 10**.

**Why try it:** Inversion is approximate; noise in the estimated latent differs by channel. **Averaging** is a linear denoising step before FFT: if errors are partly independent across channels, the mean can **stabilize** the field before matching the key.

**JPEG numbers (n=50, dedicated JPEG CSV):** AUC **~0.749** (essentially flat vs. baseline), best acc **0.70** (**+0.01** vs. baseline), but **TPR @ 1%** **0.06** and **TPR @ 5%** **0.10** (**worse** than baseline **0.10 / 0.30**). So: **slightly better** at one summary threshold, **worse** under strict false-positive control.

---

## 4. Stronger Fourier key (`key_scale=1.12`) with mean aggregation

**What we did:** Same as mean aggregation, but when **injecting** noise at generation we write **`key_scale × key`** in the mask, and at detection we compare to the **same scaled key**. You used **1.12**.

**Why try it:** JPEG reduces reliable high-frequency detail; a **stronger** constrained pattern in the masked band might survive compression **relatively** better (at the cost of possible **visible artifacts** or statistical mismatch if scale is too large).

**JPEG numbers (n=50):** AUC **~0.747**, TPR@1% **0.04**, TPR@5% **0.08**, best acc **0.70**. **Strict-FPR TPR** fell **further** vs. baseline; **no** clear JPEG win.

---

## 5. Median aggregation across channels (`channel_agg=median`)

**What we did:** Per spatial location, **median** of the four latent channels, then FFT and Tree-Ring test. **key_scale = 1.0**, **r = 10**. Evaluated at **n = 20** (time budget).

**Why try it:** If JPEG or inversion **corrupts one channel** badly, the **mean** is pulled toward that outlier; the **median** is more **robust** to a single bad channel.

**JPEG numbers (n=20):** AUC **~0.62**, best acc **0.65**, TPR@1% **0.30**, TPR@5% **0.35** on that small run. **AUC and best acc** were **worse** than n=50 baselines; **low-FPR TPR** looks higher but **n=20** makes ROC operating points **very noisy**—not comparable to n=50 without a full rerun.

---

## 6. Min-distance channel (`channel_agg=min_dist`)

**What it does:** FFT **each** channel separately, compute Tree-Ring **distance** to the key on each; take the **minimum** distance and the **η / p-value** from that channel.

**Why try it:** If JPEG hurts channels **unevenly**, the “best” channel might still carry the signal.

**JPEG numbers:** **r = 10, n=20:** AUC **~0.83**, TPR@1% **0.30**, best acc **0.82** — [`runs/min_dist_n20/`](runs/min_dist_n20/). **Radius ablation n=20:** **r = 8** AUC **~0.86**, TPR@5% **0.60**; **r = 12** AUC **~0.87**, TPR@1% **0.30**. **r = 12, n=50 (WatGPU):** AUC **~0.90** (0.9026 raw), TPR@1% **0.26**, TPR@5% **0.58**, best acc **0.85** — much stronger than first-channel baseline (~0.75). **Best-of-four** scoring can still affect **FPR** calibration; interpret TPR@FPR alongside AUC.

---

## 7. Fourier mask radius ablation (`radius ∈ {8, 10, 12}`)

**What we did:** Fix **mean** aggregation and **key_scale = 1.0**, **JPEG Q25 only**, vary **only** the **radius** of the circular mask in the **noise FFT** (same at embed and detect). Evaluated at **n = 20** per radius on WatGPU.

**Why try it:** The key occupies a **disk in frequency**. **Smaller r** = fewer bins, tighter low-frequency pattern; **larger r** = more bins, more redundant signal but also more coupling to distortions. **r = 10** was the default; **8** and **12** test sensitivity to this design choice under JPEG.

**Results (n=20):**

| r | AUC | TPR@1% | TPR@5% | Best acc |
|---|-----|--------|--------|----------|
| 8 | 0.79 | 0.10 | 0.40 | 0.78 |
| 10 | 0.73 | 0.00 | 0.30 | 0.72 |
| 12 | 0.80 | 0.20 | 0.30 | 0.78 |

On this **preliminary** slice, **r = 8** and **r = 12** look **better** than **r = 10** on AUC and best acc; **r = 12** best TPR@1%. **Confirm at n = 50** before strong claims.

---

## 8. What we did *not* try (but could)

- **JPEG-aware training** (e.g. WatermarkDM with JPEG augmentations).
- **More inversion steps** or scheduler tuning.
- **Magnitude-only** or other spectral statistics (phase is fragile after JPEG).
- **Fusing p-values** across channels (Fisher / Bonferroni).
- **`min_dist`** JPEG at **n=50** for **r=12** logged; **r=8 / r=10** min-dist **n=50** optional to match.

---

## 9. Bottom line

| Direction | Verdict on your runs |
|-----------|----------------------|
| Mean vs. first channel | Tiny **best-acc** gain; **worse TPR @ low FPR** |
| key_scale 1.12 + mean | **No** improvement on strict metrics |
| Median (n=20) | **Worse AUC** vs. n=50 settings; noisy |
| Radius 8 vs 10 vs 12 (n=20) | **Promising**; needs **n=50** confirmation |
| Min-dist + r=12 (n=50) | **Strong** AUC **~0.90** vs. baseline **~0.75**; still check **FPR** (best-of-4) |

**Honest story for a report:** We systematically tried **detector pooling** (first / mean / median / **min-dist**), **key strength**, and **mask radius** under **JPEG Q25**. **Mean/median + key scaling** did not clearly improve **JPEG defense** under **strict FPR** at n=50. **Min-dist** with **r=12** at **n=50** is the clearest win so far on **AUC**; **mean × radius** and **min-dist × radius** at n=20 suggested **r** matters — confirm **min-dist r=8/10 at n=50** if you need a full radius sweep at fixed n.

---

See also: [`SLIDES.md`](SLIDES.md) (compact tables), [`ATTEMPTS.md`](ATTEMPTS.md) (raw numbers).
