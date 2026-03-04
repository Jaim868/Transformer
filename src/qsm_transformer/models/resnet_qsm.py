from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResNetQSM(nn.Module):
    def __init__(self, in_channels: int = 1, width: int = 48, depth: int = 8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.stem(x)
        return self.head(self.body(features))
