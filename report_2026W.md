# Introduction
This report concludes the paper readings and experiments conducted during the Winter 2026 term. The research is based on the Tree-Ring method for watermark hiding and embedding, utilized as a mechanism to add invisible and robust fingerprints to diffusion-generated images, as proposed by Wen et al. in Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust (arXiv:2305.20030).

Our work focused on reproducing that pipeline with Stable Diffusion version 1.5, measuring how well watermarked images stay detectable under several distortions, and exploring ways to raise the model defence rate against JPEG compression, which proved to be the hardest attack in our default configuration.
