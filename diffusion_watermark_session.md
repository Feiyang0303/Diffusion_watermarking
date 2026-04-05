# Session: Boosting Watermark Robustness vs JPEG Compression
**Project:** Diffusion Model Provenance
**Goal:** Fix the 70% recovery bottleneck when images are compressed.

**Feiyang:** I’m having some difficulties. The latent watermarks are mostly surviving crops and Gaussian noise, but JPEG compression seems to have huge damage. I think the high-frequency coefficients are getting nuked during quantization. How can we hide the signal more effectively during the encoding process?

**Agent:** We could try increasing the global scaling factor of the watermark embedding to make the signal stronger.

**Feiyang:** No, I tried that. It creates "checkerboard" artifacts in the flat areas of the generated image. It has to stay invisible. What if we actually simulate the JPEG DCT quantization inside the training loop as a differentiable layer? That way, the encoder learns to embed the signal in the coefficients that JPEG typically preserves.

**Agent:** That’s a solid idea. We can use a differentiable approximation of the rounding function, like a straight-through estimator (STE), to keep the gradients flowing. Here’s a basic implementation of a JPEG-simulator layer. [Revised code snippet]

**Feiyang:** This simulator is a bit too perfect. Real JPEG libraries use specific quantization tables and chroma subsampling. If we don't account for the YCbCr conversion loss, the 85% accuracy won't hold up in the real world. Let’s adjust the layer to use the standard IJG quantization tables.

**Agent:** [Generates revised layer code with standard tables]

**Feiyang:** [Reviewing code] Okay, this looks much more realistic. I’m going to pair this with a perceptual loss (LPIPS) so we don't lose image quality. Let’s run a test batch of sample size 50.

**Result:** After 50 epochs with the JPEG-aware layer, recovery accuracy jumped from 70% to 85% on Quality 50 JPEGs with zero perceptible loss in FID scores.
