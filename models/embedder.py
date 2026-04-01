"""Watermark embedder network for 128x128 images."""

import torch
import torch.nn as nn


class WatermarkEmbedder(nn.Module):
    """Embed secret into carrier using residual learning for high PSNR-C."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(0.1)
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, carrier: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        if carrier.shape != secret.shape:
            raise ValueError("carrier and secret must have the same shape.")
        if carrier.dim() != 4 or carrier.size(1) != 3:
            raise ValueError("Inputs must be (B,3,128,128).")

        x = torch.cat([carrier, secret], dim=1)
        x = self.drop(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop(torch.relu(self.bn5(self.conv5(x))))
        x = self.tanh(self.conv6(x))
        watermarked = carrier + self.alpha * x
        return torch.clamp(watermarked, 0.0, 1.0)
