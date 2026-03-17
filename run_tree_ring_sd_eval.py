#!/usr/bin/env python3
"""
Tree-Ring image-level evaluation (Stable Diffusion).

For N samples:
- generate a clean image and a watermarked image (same prompt, different seeds)
- optionally apply an image-space attack (jpeg / resize / crop)
- run DDIM inversion + Tree-Ring detection on both
- write per-sample metrics to a CSV for paper-style plots/ROC curves
"""

# Disable tqdm progress bars so terminal keypresses don't pollute output (e.g. in tmux)
import os
os.environ["TQDM_DISABLE"] = "1"

import argparse
import csv
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _apply_attack(
    img,
    attack: str,
    jpeg_quality: int,
    resize_short: int,
    crop_frac: float,
):
    from PIL import Image

    if attack == "none":
        return img

    if attack == "jpeg":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    if attack == "resize":
        w, h = img.size
        short = min(w, h)
        if resize_short <= 0:
            raise ValueError("--resize_short must be > 0")
        scale = resize_short / float(short)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img2 = img.resize((new_w, new_h), resample=Image.BICUBIC)
        return img2.resize((w, h), resample=Image.BICUBIC)

    if attack == "crop":
        if not (0.0 < crop_frac <= 1.0):
            raise ValueError("--crop_frac must be in (0, 1]")
        w, h = img.size
        new_w = max(1, int(round(w * crop_frac)))
        new_h = max(1, int(round(h * crop_frac)))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        img2 = img.crop((left, top, left + new_w, top + new_h))
        return img2.resize((w, h), resample=Image.BICUBIC)

    raise ValueError(f"Unknown attack: {attack}")


@dataclass
class DetectResult:
    distance: float
    eta: float
    sigma_sq: float
    p_value: float
    is_watermarked: bool


def _detect_tree_ring_from_pil(
    pipe,
    pil_image,
    *,
    device,
    steps: int,
    key: str,
    radius: int,
    seed: int,
):
    import torch

    from diffusion_watermarking.tree_ring import detect_tree_ring

    # Encode image to latent x0
    with torch.no_grad():
        pixel = pipe.image_processor.preprocess(pil_image).to(device).to(pipe.vae.dtype)
        # VAE expects (B, C, H, W). Normalize from any preprocess shape (diffusers version-dependent).
        pixel = pixel.squeeze()
        if pixel.dim() == 3:
            pixel = pixel.unsqueeze(0)
        elif pixel.dim() > 4:
            pixel = pixel.reshape(-1, pixel.shape[-3], pixel.shape[-2], pixel.shape[-1])
        pixel = (pixel / 2 + 0.5).clamp(0, 1)
        latent_0 = pipe.vae.encode(pixel).latent_dist.sample() * pipe.vae.config.scaling_factor

    # DDIM inversion loop (forward noising)
    pipe.scheduler.set_timesteps(steps)
    timesteps = pipe.scheduler.timesteps
    latent_inv = latent_0.clone()

    prompt_embeds = pipe.encode_prompt(
        [""],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    while hasattr(prompt_embeds, "__len__") and not hasattr(prompt_embeds, "shape"):
        prompt_embeds = prompt_embeds[0]

    n_steps = len(timesteps)
    for step_idx, t in enumerate(timesteps):
        t_batch = torch.full((1,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_inv,
                t_batch,
                encoder_hidden_states=prompt_embeds,
            ).sample
        latent_inv = pipe.scheduler.step(noise_pred, t, latent_inv).prev_sample
        if (step_idx + 1) % 10 == 0 or step_idx == 0:
            print(f"  inversion step {step_idx + 1}/{n_steps}", flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    inverted_noise = latent_inv.detach().cpu().float().numpy()[0]
    res = detect_tree_ring(
        inverted_noise,
        key_type=key,
        radius=radius,
        seed=seed,
        return_p_value=True,
    )
    return DetectResult(
        distance=float(res["distance"]),
        eta=float(res["eta"]),
        sigma_sq=float(res["sigma_sq"]),
        p_value=float(res["p_value"]),
        is_watermarked=bool(res["is_watermarked"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-Ring SD evaluation")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--prompt", type=str, default="A photo of an astronaut riding a horse on Mars")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--key", choices=["zeros", "rand", "rings"], default="rings")
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Base seed; sample i uses seed+ i")
    parser.add_argument("--out_dir", type=str, default="outputs_tree_ring_sd_eval")
    parser.add_argument("--out_csv", type=str, default="outputs_tree_ring_sd_eval/sd_eval.csv")
    parser.add_argument("--save_images", type=int, default=3, help="Save first K clean/wm (+attacked) images")

    parser.add_argument("--attack", choices=["none", "jpeg", "resize", "crop"], default="none")
    parser.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="Comma-separated list of attacks to evaluate in one run, e.g. none,jpeg,resize,crop. "
        "Overrides --attack; each sample is generated once then each attack is applied and detected.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=50)
    parser.add_argument("--resize_short", type=int, default=384, help="Short side for resize attack")
    parser.add_argument("--crop_frac", type=float, default=0.8, help="Center crop fraction for crop attack")

    args = parser.parse_args()

    # Support multi-attack run: --attacks none,jpeg,resize,crop
    if args.attacks:
        attack_list = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]
        for a in attack_list:
            if a not in ("none", "jpeg", "resize", "crop"):
                raise ValueError(f"Unknown attack in --attacks: {a!r}")
        args._attack_list = attack_list
    else:
        args._attack_list = [args.attack]

    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    from diffusion_watermarking.tree_ring import inject_watermark_noise_latent

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    latent_shape = (1, 4, 64, 64)
    h, w = 64, 64

    print("Running SD eval")
    print("----------------")
    print(f"num_samples={args.num_samples}, steps={args.steps}, prompt={args.prompt!r}")
    print(f"key={args.key}, radius={args.radius}, base_seed={args.seed}")
    print(f"attacks={args._attack_list}")
    print(f"  jpeg_quality={args.jpeg_quality}, resize_short={args.resize_short}, crop_frac={args.crop_frac}")
    print(f"writing CSV -> {out_csv}")

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_idx",
                "type",  # clean / watermarked
                "seed",
                "prompt",
                "attack",
                "attack_param",
                "distance",
                "eta",
                "sigma_sq",
                "p_value",
                "is_watermarked",
                "seconds",
            ]
        )

        for i in range(args.num_samples):
            print(f"Sample {i + 1}/{args.num_samples} (seed={args.seed + i}) ...", flush=True)
            seed_i = args.seed + i
            t0 = time.time()

            # 1) Generate watermarked initial noise in latent space
            latents_wm = inject_watermark_noise_latent(
                (4, h, w),
                key_type=args.key,
                radius=args.radius,
                seed=args.seed,  # key seed is fixed across all samples (as in paper)
                noise_seed=seed_i,  # vary base noise per sample
            )
            latents_wm = torch.from_numpy(latents_wm).unsqueeze(0).to(device).to(pipe.unet.dtype)
            latents_wm = latents_wm * pipe.scheduler.init_noise_sigma

            # 2) Generate a clean latent
            gen_clean = torch.Generator(device=device).manual_seed(seed_i + 1234)
            latents_clean = torch.randn(latent_shape, device=device, dtype=pipe.unet.dtype, generator=gen_clean)
            latents_clean = latents_clean * pipe.scheduler.init_noise_sigma

            # 3) Generate images
            print("  generating watermarked image ...", flush=True)
            gen_img = torch.Generator(device=device).manual_seed(seed_i + 1)
            out_wm = pipe(
                prompt=args.prompt,
                latents=latents_wm,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=gen_img,
            )
            img_wm = out_wm.images[0]
            print("  generating clean image ...", flush=True)
            out_clean = pipe(
                prompt=args.prompt,
                latents=latents_clean,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device=device).manual_seed(seed_i + 2),
            )
            img_clean = out_clean.images[0]

            # 4) For each attack: apply, detect both, write rows
            for atk in args._attack_list:
                img_wm_a = _apply_attack(img_wm, atk, args.jpeg_quality, args.resize_short, args.crop_frac)
                img_clean_a = _apply_attack(img_clean, atk, args.jpeg_quality, args.resize_short, args.crop_frac)

                print(f"  attack={atk}: inverting watermarked ...", flush=True)
                res_wm = _detect_tree_ring_from_pil(
                    pipe,
                    img_wm_a,
                    device=device,
                    steps=args.steps,
                    key=args.key,
                    radius=args.radius,
                    seed=args.seed,
                )
                print(f"  attack={atk}: inverting clean ...", flush=True)
                res_clean = _detect_tree_ring_from_pil(
                    pipe,
                    img_clean_a,
                    device=device,
                    steps=args.steps,
                    key=args.key,
                    radius=args.radius,
                    seed=args.seed,
                )

                dt = time.time() - t0
                attack_param = ""
                if atk == "jpeg":
                    attack_param = str(args.jpeg_quality)
                elif atk == "resize":
                    attack_param = str(args.resize_short)
                elif atk == "crop":
                    attack_param = str(args.crop_frac)

                writer.writerow(
                    [
                        i,
                        "watermarked",
                        seed_i,
                        args.prompt,
                        atk,
                        attack_param,
                        res_wm.distance,
                        res_wm.eta,
                        res_wm.sigma_sq,
                        res_wm.p_value,
                        int(res_wm.is_watermarked),
                        dt,
                    ]
                )
                writer.writerow(
                    [
                        i,
                        "clean",
                        seed_i,
                        args.prompt,
                        atk,
                        attack_param,
                        res_clean.distance,
                        res_clean.eta,
                        res_clean.sigma_sq,
                        res_clean.p_value,
                        int(res_clean.is_watermarked),
                        dt,
                    ]
                )
                f.flush()

                if i < args.save_images and atk != "none":
                    img_clean_a.save(out_dir / f"{i:04d}_clean_{atk}.png")
                    img_wm_a.save(out_dir / f"{i:04d}_watermarked_{atk}.png")

                print(
                    f"  [{atk}] wm_dist={res_wm.distance:.3f} clean_dist={res_clean.distance:.3f}",
                    flush=True,
                )

            if i < args.save_images:
                img_clean.save(out_dir / f"{i:04d}_clean.png")
                img_wm.save(out_dir / f"{i:04d}_watermarked.png")

            dt = time.time() - t0
            print(f"[{i+1}/{args.num_samples}] seed={seed_i} ({dt:.1f}s)", flush=True)

    print("\nDone.")
    print(f"CSV: {out_csv}")
    print(f"Images (first {args.save_images}): {out_dir}")


if __name__ == "__main__":
    main()

