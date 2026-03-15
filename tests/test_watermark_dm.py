"""Test cases for WatermarkDM encoder/decoder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import unittest

from diffusion_watermarking import watermark_dm

TORCH_AVAILABLE = getattr(watermark_dm, "TORCH_AVAILABLE", False)


@unittest.skipIf(not TORCH_AVAILABLE, "PyTorch required")
class TestWatermarkDM(unittest.TestCase):
    def test_encoder_decoder_forward(self):
        """Encoder and decoder forward pass."""
        import torch
        from diffusion_watermarking.watermark_dm import WatermarkEncoder, WatermarkDecoder

        B, C, H, W = 2, 3, 32, 32
        bit_length = 64
        encoder = WatermarkEncoder(in_channels=C, bit_length=bit_length)
        decoder = WatermarkDecoder(in_channels=C, bit_length=bit_length)

        x = torch.randn(B, C, H, W)
        w = torch.randint(0, 2, (B, bit_length), dtype=torch.float32)
        xw = encoder(x, w)
        self.assertEqual(xw.shape, x.shape)

        logits = decoder(xw)
        self.assertEqual(logits.shape, (B, bit_length))

    def test_train_encoder_decoder_one_batch(self):
        """One training step on GPU if available."""
        import torch
        from diffusion_watermarking.watermark_dm import (
            WatermarkEncoder,
            WatermarkDecoder,
            train_encoder_decoder,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = WatermarkEncoder(in_channels=3, bit_length=32).to(device)
        decoder = WatermarkDecoder(in_channels=3, bit_length=32).to(device)

        class FakeDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 8

            def __getitem__(self, i):
                return {"image": torch.randn(3, 32, 32)}

        loader = torch.utils.data.DataLoader(
            FakeDataset(), batch_size=4, shuffle=True
        )
        train_encoder_decoder(
            encoder, decoder, loader, device,
            num_epochs=2, lr=1e-3, gamma=1.0,
        )
    # No assertion beyond “no exception”; training runs on device
        self.assertEqual(next(encoder.parameters()).device.type, device.type)
