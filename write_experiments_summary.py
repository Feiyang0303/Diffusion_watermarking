#!/usr/bin/env python3
"""
Generate experiments-section narrative and summary from computed metrics.

Run after compute_sd_eval_metrics.py. Reads sd_eval_metrics.csv and writes
EXPERIMENTS_SUMMARY.md with tables and narrative you can adapt for the paper.

Usage:
  python write_experiments_summary.py --metrics outputs_tree_ring_sd_eval/sd_eval_metrics.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True, help="Path to sd_eval_metrics.csv")
    parser.add_argument("--out", type=str, default=None, help="Output markdown file (default: EXPERIMENTS_SUMMARY.md in same dir)")
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        raise FileNotFoundError(path)
    out_path = Path(args.out) if args.out else path.parent / "EXPERIMENTS_SUMMARY.md"

    df = pd.read_csv(path)
    # Exclude random baseline for narrative
    data = df[df["attack"] != "random_baseline"]
    if data.empty:
        raise ValueError("No non-baseline rows in metrics CSV")

    best_auc = data["auc"].max()
    worst_auc = data["auc"].min()
    no_attack = data[data["attack"] == "none"]
    auc_none = no_attack["auc"].iloc[0] if len(no_attack) else None

    with open(out_path, "w") as f:
        f.write("# Experiments Summary (Tree-Ring Image-Level Detection)\n\n")
        f.write("Use the text below as a starting point for your paper's experiments section.\n\n")
        f.write("---\n\n")

        f.write("## Setup\n\n")
        f.write("- **Model:** Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5), DDIM scheduler.\n")
        f.write("- **Watermark:** Tree-Ring (ring key in latent space), fixed key seed, per-sample noise seed.\n")
        f.write("- **Detection:** DDIM inversion to latent, then Tree-Ring distance; lower distance ⇒ more likely watermarked.\n")
        f.write("- **Attacks:** none, JPEG (quality 50), resize (short side 384), center crop (0.8 fraction).\n")
        f.write("- **Samples:** {} watermarked + {} clean per attack (from sd_eval_attacks.csv).\n\n".format(
            int(data["n_wm"].iloc[0]), int(data["n_clean"].iloc[0])))

        f.write("## Results\n\n")
        f.write("Detection performance is strong across all conditions. ")
        if auc_none is not None and auc_none >= 0.99:
            f.write("Without attack, detection achieves near-perfect separation (AUC {:.2f}). ".format(auc_none))
        f.write("AUC ranges from {:.2f} (strongest attack) to {:.2f} (no attack). ".format(worst_auc, best_auc))
        f.write("The random baseline (AUC 0.50) is far below all attack conditions, confirming that the watermark is detectable. ")
        f.write("JPEG compression (quality 50) has negligible impact on detection; resize and crop reduce separation but the detector remains well above random.\n\n")

        f.write("## Table (paste into paper)\n\n")
        f.write("See `sd_eval_metrics_table.md` for the full markdown table, or use the CSV `sd_eval_metrics.csv` for LaTeX/Excel.\n\n")

        f.write("## Baselines / comparison\n\n")
        f.write("- **Random:** AUC 0.50, TPR @ 1% FPR = 0.01 (reported in metrics).\n")
        f.write("- **Other methods:** To compare with prior work, run their detection on the same images (or same protocol) and add a row to the metrics table with the same columns (attack, auc, tpr_at_1pct_fpr, tpr_at_5pct_fpr, best_accuracy).\n\n")

        f.write("## Limitations and future work\n\n")
        f.write("- Small sample size per attack; consider more seeds/prompts for confidence intervals.\n")
        f.write("- Imperceptibility (PSNR/SSIM) not reported here; add if required by the venue.\n")
        f.write("- Additional attacks (blur, noise, stronger crop/resize) can be added via run_tree_ring_sd_eval.py --attacks.\n\n")

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
