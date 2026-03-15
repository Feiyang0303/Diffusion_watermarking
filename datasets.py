"""
Dataset loaders for WatermarkDM training.
Supports synthetic data and Tiny ImageNet 200.
"""

from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_tiny_imagenet_200_dataset(root: str, split: str = "train", image_size: int = 64):
    """
    Tiny ImageNet 200 dataset. Expected layout:
      root/train/<class_id>/images/<class_id>_*.JPEG
      root/val/images/val_*.JPEG
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for datasets")
    try:
        from PIL import Image
        import torchvision.transforms as T
    except ImportError as e:
        raise ImportError("PIL and torchvision required. pip install torchvision Pillow") from e

    root = Path(root)
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])

    if split == "train":
        train_dir = root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Missing {train_dir}")
        image_paths = []
        for c_dir in sorted(train_dir.iterdir()):
            if not c_dir.is_dir():
                continue
            img_dir = c_dir / "images"
            if not img_dir.exists():
                img_dir = c_dir
            for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG"):
                image_paths.extend(img_dir.glob(ext))
        image_paths = [str(p) for p in image_paths]
        if not image_paths:
            raise FileNotFoundError(f"No images under {root}/train/")
    elif split == "val":
        val_dir = root / "val" / "images"
        if not val_dir.exists():
            val_dir = root / "val"
        if not val_dir.exists():
            raise FileNotFoundError(f"Missing {root}/val/")
        image_paths = []
        for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.JPG"):
            image_paths.extend(val_dir.glob(ext))
        image_paths = [str(p) for p in image_paths]
        if not image_paths:
            raise FileNotFoundError(f"No images under {root}/val/")
    else:
        raise ValueError("split must be 'train' or 'val'")

    class _TinyImageNetDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(image_paths)

        def __getitem__(self, i):
            img = Image.open(image_paths[i]).convert("RGB")
            x = transform(img)
            return {"image": x}

    return _TinyImageNetDataset()
