"""
baselines.py

Baseline model architectures for CIFAR-10 experiments.

This file provides baseline convolutional neural network models used
for comparison against RepVGG variants.

Main tasks performed:

1. Load standard torchvision ResNet architectures.

2. Modify ImageNet-style ResNet models for CIFAR-10 by:
   - Replacing the first convolution layer with a smaller 3x3 layer
   - Removing the initial max-pooling layer
   - Replacing the final classifier layer

3. Provide ready-to-use baseline models:
   - ResNet-18
   - ResNet-34

Used for benchmark comparison, fairness evaluation,
and reproduction experiments.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def _adapt_resnet_for_cifar(
    model: nn.Module,
    num_classes: int,
) -> nn.Module:
    """
    Modify a standard ResNet architecture for CIFAR-10 input size.

    Changes applied:
    - Replace first 7x7 stride-2 convolution with 3x3 stride-1
    - Remove max-pooling layer
    - Replace final fully connected classifier
    """
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def create_resnet18(
    num_classes: int,
) -> nn.Module:
    """
    Create CIFAR-10 adapted ResNet-18 model.
    """
    model = models.resnet18(weights=None)

    return _adapt_resnet_for_cifar(
        model=model,
        num_classes=num_classes,
    )


def create_resnet34(
    num_classes: int,
) -> nn.Module:
    """
    Create CIFAR-10 adapted ResNet-34 model.
    """
    model = models.resnet34(weights=None)

    return _adapt_resnet_for_cifar(
        model=model,
        num_classes=num_classes,
    )