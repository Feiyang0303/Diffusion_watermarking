#!/usr/bin/env python3
"""
Train WatermarkDM encoder/decoder on GPU.
Supports synthetic data (default) or Tiny ImageNet 200 via --data_dir.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusion_watermarking.watermark_dm import (
    WatermarkEncoder,
    WatermarkDecoder,
    train_encoder_decoder,
)


def main():
    try:
        import torch
    except ImportError:
        print("PyTorch is required. Install with: pip install -r requirements.txt")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train WatermarkDM encoder/decoder on GPU")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, cpu, or cuda:0 etc. Default: cuda if available")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Reconstruction weight in loss (Eq. 2)")
    parser.add_argument("--bit_length", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=256,
                        help="Number of synthetic samples (used only if --data_dir is not set)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset root (e.g. tiny-imagenet-200). If set, uses train split instead of synthetic.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"],
                        help="Split to use when --data_dir is set (default: train)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader num_workers (use 0 for Windows/small data)")
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save encoder/decoder state dicts")
    parser.add_argument("--out_dir", type=str, default="outputs_watermark_dm",
                        help="Directory for metrics CSV, sample images, and plots (default: outputs_watermark_dm)")
    parser.add_argument("--log_every", type=int, default=1,
                        help="Log and compute validation bit accuracy every N epochs (default: 1)")
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Device:", device)
    if device.type == "cuda":
        print("  GPU:", torch.cuda.get_device_name(device))

    C, H, W = 3, args.image_size, args.image_size

    if args.data_dir:
        from diffusion_watermarking.datasets import get_tiny_imagenet_200_dataset
        data_path = Path(args.data_dir).expanduser().resolve()
        if not data_path.exists():
            print(f"Error: --data_dir does not exist: {data_path}")
            sys.exit(1)
        print("Using dataset:", data_path, f"split={args.split}")
        try:
            dataset = get_tiny_imagenet_200_dataset(
                str(data_path),
                split=args.split,
                image_size=args.image_size,
            )
        except ImportError as e:
            print("Tiny ImageNet requires torchvision and Pillow. Install with:")
            print("  pip install torchvision Pillow")
            sys.exit(1)
        print("  Number of images:", len(dataset))
    else:
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples, C, H, W, seed=42):
                self.num_samples = num_samples
                self.C, self.H, self.W = C, H, W
                self.rng = torch.Generator().manual_seed(seed)

            def __len__(self):
                return self.num_samples

            def __getitem__(self, i):
                x = torch.randn(self.C, self.H, self.W, generator=self.rng)
                return {"image": x}

        dataset = SyntheticDataset(args.num_samples, C, H, W)
        print("Using synthetic data, samples:", len(dataset))

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    encoder = WatermarkEncoder(
        in_channels=C,
        bit_length=args.bit_length,
        base_channels=64,
        num_blocks=4,
    ).to(device)
    decoder = WatermarkDecoder(
        in_channels=C,
        bit_length=args.bit_length,
        base_channels=64,
    ).to(device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []

    # Fixed batch for validation bit-accuracy (same each epoch for comparable curve)
    fixed_batch = next(iter(loader))
    if isinstance(fixed_batch, dict):
        x_val = fixed_batch["image"].to(device)
    else:
        x_val = fixed_batch[0].to(device)
    B_val = min(8, x_val.size(0))
    x_val, w_val = x_val[:B_val], torch.randint(0, 2, (B_val, args.bit_length), device=device, dtype=torch.float32)

    def log_cb(epoch, d):
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            xw = encoder(x_val, w_val)
            logits = decoder(xw)
            pred = (torch.sigmoid(logits) >= 0.5).float()
            bit_acc = (pred == w_val).float().mean().item()
        d["bit_accuracy"] = bit_acc
        encoder.train()
        decoder.train()
        metrics_rows.append({"epoch": epoch, **d})
        if args.log_every and epoch % args.log_every == 0:
            print("  Epoch %d  loss=%.4f  bit_acc=%.2f%%" % (epoch, d["loss"], bit_acc * 100))

    print("Training encoder/decoder for", args.epochs, "epochs (metrics -> %s) ..." % out_dir)
    train_encoder_decoder(
        encoder,
        decoder,
        loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
        log_callback=log_cb,
    )
    print("Training done.")

    # Save metrics CSV
    import csv
    metrics_file = out_dir / "training_metrics.csv"
    if metrics_rows:
        with open(metrics_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "loss", "bit_accuracy"])
            w.writeheader()
            w.writerows(metrics_rows)
        print("Metrics saved to", metrics_file)

    # Save sample images: original, watermarked (and optional grid)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x_sample = x_val[:4]
        w_sample = torch.randint(0, 2, (4, args.bit_length), device=device, dtype=torch.float32)
        xw_sample = encoder(x_sample, w_sample)
    try:
        from torchvision.utils import save_image
        # Images are [0,1] from ToTensor or synthetic; clamp for safety
        save_image(x_sample.cpu(), out_dir / "samples_original.png", nrow=2, padding=2)
        save_image(xw_sample.cpu().clamp(0, 1), out_dir / "samples_watermarked.png", nrow=2, padding=2)
        print("Sample images saved to", out_dir / "samples_original.png", "and", out_dir / "samples_watermarked.png")
    except Exception as e:
        print("Could not save sample images:", e)

    # Optional: simple loss curve (matplotlib if available)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs = [r["epoch"] for r in metrics_rows]
        losses = [r["loss"] for r in metrics_rows]
        accs = [r["bit_accuracy"] * 100 for r in metrics_rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, losses, "b-")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training loss (Eq. 2)")
        ax2.plot(epochs, accs, "g-")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Bit accuracy (%)")
        ax2.set_title("Validation bit accuracy")
        plt.tight_layout()
        plt.savefig(out_dir / "training_curves.png", dpi=120)
        plt.close()
        print("Training curves saved to", out_dir / "training_curves.png")
    except ImportError:
        pass

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), out / "encoder.pt")
        torch.save(decoder.state_dict(), out / "decoder.pt")
        print("Checkpoints saved to", out)


if __name__ == "__main__":
    main()
