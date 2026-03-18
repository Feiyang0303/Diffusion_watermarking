#!/usr/bin/env python3
"""
Tree-Ring image-level evaluation (Stable Diffusion).

For N samples:
- generate a clean image and a watermarked image (same prompt, different seeds)
- optionally apply an image-space attack (paper Sec 4.3: rotation, jpeg, crop+scale, blur, noise, color_jitter)
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

import numpy as np


# All supported attacks (paper Table 2 / Sec 4.3)
ATTACK_CHOICES = (
    "none",
    "jpeg",
    "resize",
    "crop",        # random crop + rescale (paper: 75% crop)
    "rotation",    # paper: 75°
    "blur",        # paper: Gaussian 8×8
    "noise",       # paper: Gaussian σ=0.1
    "color_jitter", # paper: brightness factor in [0, 6]
)


def _apply_attack(
    img,
    attack: str,
    attack_seed: int,
    jpeg_quality: int = 25,
    resize_short: int = 384,
    crop_frac: float = 0.75,
    rotation_deg: float = 75.0,
    blur_size: int = 8,
    noise_std: float = 0.1,
    brightness_max: float = 6.0,
):
    """Apply a single attack. attack_seed ensures same random choices when applied to wm and clean."""
    from PIL import Image, ImageFilter, ImageEnhance

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
        # Paper: random cropping and scaling (75% crop). Random top-left, then resize back.
        if not (0.0 < crop_frac <= 1.0):
            raise ValueError("--crop_frac must be in (0, 1]")
        w, h = img.size
        new_w = max(1, int(round(w * crop_frac)))
        new_h = max(1, int(round(h * crop_frac)))
        rng = np.random.default_rng(attack_seed)
        left = rng.integers(0, w - new_w + 1) if w > new_w else 0
        top = rng.integers(0, h - new_h + 1) if h > new_h else 0
        img2 = img.crop((left, top, left + new_w, top + new_h))
        return img2.resize((w, h), resample=Image.BICUBIC)

    if attack == "rotation":
        # Paper: 75° rotation. Expand=False keeps original canvas size.
        angle = float(rotation_deg)
        return img.rotate(-angle, resample=Image.BICUBIC, expand=False)

    if attack == "blur":
        # Paper: Gaussian blur with 8×8 filter. PIL GaussianBlur(radius) ~ sigma.
        # For kernel size 8, use radius = (size-1)/2 ≈ 3.5 or 4.
        radius = max(0.5, (blur_size - 1) / 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if attack == "noise":
        # Paper: Gaussian noise σ=0.1. Image is 0–255; scale so σ=0.1 in normalized [0,1] → 25.5 in 0–255.
        arr = np.array(img, dtype=np.float64)
        rng = np.random.default_rng(attack_seed)
        arr = arr + rng.standard_normal(arr.shape) * (noise_std * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if attack == "color_jitter":
        # Paper: brightness factor uniformly sampled in [0, 6].
        rng = np.random.default_rng(attack_seed)
        factor = float(rng.uniform(0.0, brightness_max))
        # PIL: 1.0 = no change; 0 = black; 6 = very bright. Clamp to avoid overflow.
        factor = max(0.01, min(10.0, factor))
        return ImageEnhance.Brightness(img).enhance(factor)

    raise ValueError(f"Unknown attack: {attack!r}. Choose from {ATTACK_CHOICES}")


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

    parser.add_argument("--attack", choices=ATTACK_CHOICES, default="none")
    parser.add_argument(
        "--attacks",
        type=str,
        default=None,
        help="Comma-separated list of attacks (paper Table 2: none,jpeg,crop,rotation,blur,noise,color_jitter). "
        "Overrides --attack; each sample is generated once then each attack is applied and detected.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=25, help="JPEG quality (paper: 25)")
    parser.add_argument("--resize_short", type=int, default=384, help="Short side for resize attack")
    parser.add_argument("--crop_frac", type=float, default=0.75, help="Random crop fraction (paper: 0.75)")
    parser.add_argument("--rotation_deg", type=float, default=75.0, help="Rotation angle in degrees (paper: 75)")
    parser.add_argument("--blur_size", type=int, default=8, help="Gaussian blur kernel size (paper: 8)")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Gaussian noise std in [0,1] scale (paper: 0.1)")
    parser.add_argument("--brightness_max", type=float, default=6.0, help="Color jitter brightness factor max (paper: 6)")

    args = parser.parse_args()

    # Support multi-attack run: --attacks none,jpeg,crop,rotation,blur,noise,color_jitter
    if args.attacks:
        attack_list = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]
        for a in attack_list:
            if a not in ATTACK_CHOICES:
                raise ValueError(f"Unknown attack in --attacks: {a!r}. Choose from {ATTACK_CHOICES}")
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
    print(f"  rotation_deg={args.rotation_deg}, blur_size={args.blur_size}, noise_std={args.noise_std}, brightness_max={args.brightness_max}")
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
            for atk_idx, atk in enumerate(args._attack_list):
                attack_seed = seed_i * 1000 + atk_idx  # same random attack for wm and clean
                img_wm_a = _apply_attack(
                    img_wm,
                    atk,
                    attack_seed,
                    jpeg_quality=args.jpeg_quality,
                    resize_short=args.resize_short,
                    crop_frac=args.crop_frac,
                    rotation_deg=args.rotation_deg,
                    blur_size=args.blur_size,
                    noise_std=args.noise_std,
                    brightness_max=args.brightness_max,
                )
                img_clean_a = _apply_attack(
                    img_clean,
                    atk,
                    attack_seed,
                    jpeg_quality=args.jpeg_quality,
                    resize_short=args.resize_short,
                    crop_frac=args.crop_frac,
                    rotation_deg=args.rotation_deg,
                    blur_size=args.blur_size,
                    noise_std=args.noise_std,
                    brightness_max=args.brightness_max,
                )

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
                elif atk == "rotation":
                    attack_param = str(args.rotation_deg)
                elif atk == "blur":
                    attack_param = str(args.blur_size)
                elif atk == "noise":
                    attack_param = str(args.noise_std)
                elif atk == "color_jitter":
                    attack_param = str(args.brightness_max)

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

