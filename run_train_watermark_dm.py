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

    print("Training encoder/decoder for", args.epochs, "epochs ...")
    train_encoder_decoder(
        encoder,
        decoder,
        loader,
        device,
        num_epochs=args.epochs,
        lr=args.lr,
        gamma=args.gamma,
    )
    print("Training done.")

    if args.save:
        out = Path(args.save)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(encoder.state_dict(), out / "encoder.pt")
        torch.save(decoder.state_dict(), out / "decoder.pt")
        print("Saved to", out)


if __name__ == "__main__":
    main()
