"""Simplified watermark embedder/extractor for fast training."""

import torch
import torch.nn as nn


class WatermarkEmbedder(nn.Module):
    """Encode carrier + secret into a visually similar watermarked image."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.dwt = self._make_dwt()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _make_dwt(self) -> nn.Conv2d:
        dwt = nn.Conv2d(3, 3, 3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            dwt.weight.zero_()
            dwt.weight[:, :, 1, 1] = 1.0
            dwt.weight[:, :, 0, 1] = -0.25
            dwt.weight[:, :, 2, 1] = -0.25
            dwt.weight[:, :, 1, 0] = -0.25
            dwt.weight[:, :, 1, 2] = -0.25
        return dwt

    def forward(self, carrier: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        if carrier.shape != secret.shape:
            raise ValueError("carrier and secret must have the same shape.")
        carrier_hp = self.dwt(carrier)
        secret_hp = self.dwt(secret)
        x = torch.cat([carrier + 0.1 * carrier_hp, secret + 0.1 * secret_hp], dim=1)
        residual = self.encoder(x)
        watermarked = carrier + self.alpha * residual
        return torch.clamp(watermarked, 0.0, 1.0)


class WatermarkExtractor(nn.Module):
    """Extract secret watermark from enhanced attacked image and diff features."""

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, enhanced_attacked: torch.Tensor, diff_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([enhanced_attacked, diff_features], dim=1)
        return self.decoder(x)


# Backward-compatible aliases used by existing scripts.
class WatermarkGenerator(WatermarkEmbedder):
    pass
