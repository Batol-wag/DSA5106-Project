"""
baselines.py

Baseline architectures for both CIFAR-10 and Tiny ImageNet experiments.
"""

from __future__ import annotations

import torch.nn as nn
import torchvision.models as models


def _adapt_resnet_for_cifar(model: nn.Module, num_classes: int) -> nn.Module:
    """
    CIFAR-10 ResNet stem adaptation:
    - 3x3 stride-1 conv
    - remove maxpool
    """
    model.conv1 = nn.Conv2d(
        3,
        64,
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


def _adapt_resnet_for_tiny_imagenet(
    model: nn.Module,
    num_classes: int,
) -> nn.Module:
    """
    Tiny ImageNet uses ImageNet-style stem:
    keep original ResNet stem.
    """
    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


def create_resnet18(
    dataset: str,
    num_classes: int,
) -> nn.Module:
    dataset = dataset.lower()

    model = models.resnet18(weights=None)

    if dataset == "cifar10":
        return _adapt_resnet_for_cifar(
            model,
            num_classes,
        )

    if dataset == "tiny_imagenet":
        return _adapt_resnet_for_tiny_imagenet(
            model,
            num_classes,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def create_resnet34(
    dataset: str,
    num_classes: int,
) -> nn.Module:
    dataset = dataset.lower()

    model = models.resnet34(weights=None)

    if dataset == "cifar10":
        return _adapt_resnet_for_cifar(
            model,
            num_classes,
        )

    if dataset == "tiny_imagenet":
        return _adapt_resnet_for_tiny_imagenet(
            model,
            num_classes,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")