"""Watermark extractor network for 128x128 images."""

import torch
import torch.nn as nn


class WatermarkExtractor(nn.Module):
    """Recover secret from enhanced attacked image and differential features."""

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
        self.out_act = nn.Sigmoid()

    def forward(self, enhanced_attacked: torch.Tensor, diff_feat: torch.Tensor) -> torch.Tensor:
        if enhanced_attacked.shape != diff_feat.shape:
            raise ValueError("enhanced_attacked and diff_feat must have same shape.")
        if enhanced_attacked.dim() != 4 or enhanced_attacked.size(1) != 3:
            raise ValueError("Inputs must be (B,3,128,128).")

        x = torch.cat([enhanced_attacked, diff_feat], dim=1)
        x = self.drop(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop(torch.relu(self.bn5(self.conv5(x))))
        x = self.out_act(self.conv6(x))
        return x
