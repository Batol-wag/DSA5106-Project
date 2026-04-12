"""
baselines.py

Baseline architectures for CIFAR-10 experiments.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def _adapt_resnet_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_resnet18(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    return _adapt_resnet_for_cifar(model, num_classes)


def create_resnet34(num_classes: int) -> nn.Module:
    model = models.resnet34(weights=None)
    return _adapt_resnet_for_cifar(model, num_classes)