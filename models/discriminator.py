"""Discriminator for watermark detectability at 128x128."""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU(0.2) -> MaxPool2d(2)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Discriminator(nn.Module):
    """Binary discriminator: outputs logits per image."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Input must be (B,3,H,W).")
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


# Backward-compatible alias for existing scripts.
SteganalysisDiscriminator = Discriminator
