"""
baselines.py

Purpose:
--------
This file implements baseline models for comparison with RepVGG.

Objective:
----------
Provide fair and simple baseline architectures that can be trained using
the same pipeline as the RepVGG model.

Implemented baselines:
----------------------
1. PlainCNN
   - A VGG-style convolutional neural network without residual connections
   - Used to compare against RepVGG's plain inference-style design

2. ResNet18
   - A standard ResNet-18 model adapted for CIFAR-10 classification

Why this file matters:
----------------------
RepVGG should not be evaluated alone. To understand whether RepVGG is
effective, we compare it against:
- a plain CNN baseline
- a residual network baseline

This helps make the experimental section of the project more meaningful.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Simple convolutional block:
    Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PlainCNN(nn.Module):
    """
    Plain VGG-style CNN for CIFAR-10.

    Architecture:
    -------------
    - Stacked Conv-BN-ReLU blocks
    - Strided convolution for downsampling
    - Global average pooling
    - Linear classifier

    Notes:
    ------
    This model is intentionally simple and does not include:
    - residual connections
    - structural re-parameterization
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, stride=1),
            ConvBlock(32, 32, stride=1),

            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64, stride=1),

            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128, stride=1),

            ConvBlock(128, 256, stride=2),
            ConvBlock(256, 256, stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def create_plain_cnn(num_classes: int = 10) -> PlainCNN:
    """
    Factory function for PlainCNN.
    """
    return PlainCNN(num_classes=num_classes)


def create_resnet18(num_classes: int = 10) -> nn.Module:
    """
    Create a ResNet-18 model for CIFAR-10 classification.

    Adjustments:
    ------------
    - Replace the final fully connected layer
    - Keep the rest of the standard torchvision ResNet-18 structure
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model