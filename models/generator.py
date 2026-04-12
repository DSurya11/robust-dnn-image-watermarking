"""Simple verified invertible-style steganography network."""

import torch
import torch.nn as nn


class SimpleISN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def embed(self, carrier: torch.Tensor, secret: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([carrier, secret], dim=1)
        perturbation = self.encoder(combined) * 0.1
        return torch.clamp(carrier + perturbation, 0.0, 1.0)

    def extract(self, watermarked: torch.Tensor) -> torch.Tensor:
        return self.decoder(watermarked)


class WatermarkGenerator(SimpleISN):
    """Backward-compatible alias for older training scripts."""

