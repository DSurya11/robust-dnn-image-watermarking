"""Differential feature extractor for 128x128 watermark recovery."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentialFeatureExtractor(nn.Module):
    """Dense-difference extractor with channel progression 3->67->131->195."""

    def __init__(self, in_channels: int = 3, growth_channels: int = 64) -> None:
        super().__init__()
        if in_channels != 3 or growth_channels != 64:
            # Architecture is fixed to the requested channel design.
            pass
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(67, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(131, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(195, 64, 3, padding=1)
        self.proj = nn.Conv2d(64, 3, 1)

    def forward(self, watermarked: torch.Tensor, attacked: torch.Tensor) -> torch.Tensor:
        if watermarked.shape != attacked.shape:
            raise ValueError("watermarked and attacked must have the same shape.")
        if watermarked.dim() != 4 or watermarked.size(1) != 3:
            raise ValueError("Inputs must be (B,3,H,W).")

        diff = watermarked - attacked
        x1 = F.relu(self.conv1(diff), inplace=True)
        x2 = F.relu(self.conv2(torch.cat([diff, x1], dim=1)), inplace=True)
        x3 = F.relu(self.conv3(torch.cat([diff, x1, x2], dim=1)), inplace=True)
        x4 = F.relu(self.conv4(torch.cat([diff, x1, x2, x3], dim=1)), inplace=True)
        return self.proj(x4)
