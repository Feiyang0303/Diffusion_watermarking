# Introduction

This report concludes the paper readings and experiments conducted during the Winter 2026 term. The research is based on the Tree-Ring method for watermark hiding and embedding, used to add invisible, robust fingerprints to diffusion-generated images, as proposed by Wen et al. in *Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust* (arXiv:2305.20030).

The work focused on reproducing that pipeline with Stable Diffusion version 1.5, measuring how well watermarked images remain detectable under several distortion attacks, and exploring ways to increase the model's defence rate against JPEG compression, which proved to be the hardest attack in our default configuration.

Figures 1 and 2 break down the performance of the DDIM inversion detector across a test set of 50 watermarked and 50 clean images per attack. These charts effectively drive the motivation for the remainder of the report. The data shows that while the watermark survives geometric changes and mild visual edits, JPEG compression severely degrades the ability to tell watermarked and clean images apart. To fix this, the following sections detail the experiments aimed at making the detector more robust against high JPEG compression attacks.

![Figure 1. Empirical distributions of the Tree-Ring detection distance on the Fourier mask for watermarked versus clean images, stratified by attack. Greater overlap between the two classes under a given attack indicates weaker detectability at a fixed threshold; clear separation corresponds to more reliable identification.](results/tree_ring_sd_n50_distances.png)

*Figure 1.* Empirical distributions of the Tree-Ring detection distance on the Fourier mask for watermarked versus clean images, stratified by attack. Greater overlap between the two classes under a given attack indicates weaker detectability at a fixed threshold; clear separation corresponds to more reliable identification.

![Figure 2. Receiver operating characteristic curves obtained by sweeping a threshold on the same detection distance, with one curve per attack. The extent to which each curve lies above the diagonal encodes discriminability between watermarked and non-watermarked samples; the area under the curve provides a scalar summary comparable across attacks.](results/tree_ring_sd_n50_roc.png)

*Figure 2.* Receiver operating characteristic curves obtained by sweeping a threshold on the same detection distance, with one curve per attack. The extent to which each curve lies above the diagonal encodes discriminability between watermarked and non-watermarked samples; the area under the curve provides a scalar summary comparable across attacks.


# Background
Diffusion models generate images by gradually clearing a canvas of random noise. Since the final image is built directly on top of this initial noise, controlling that starting point gives a clever way to insert a watermark right from the beginning.

The Tree-Ring method, developed by Wen et al., takes advantage of this dynamic without requiring any extra model training. Before the model starts generating, we look at the frequencies of the initial noise (its Fourier transform) and embed a secret "key" into a specific circular pattern. Everything outside this ring remains completely random. Since the overall noise still looks perfectly normal to the model, it goes ahead and generates a high-quality image as usual.

To detect the watermark later, we run the image generation process in reverse, using a technique called DDIM inversion, to estimate the original noise canvas. By checking its frequencies, we find if our secret key pattern is sitting there in the masked ring, the watermark is verified using a statistical distance test. Since this approach doesn't rely on training a separate encoder network, it works straight out of the box with existing pre-trained models like Stable Diffusion 1.5, which is the setup used throughout this report.

