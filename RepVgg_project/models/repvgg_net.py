"""
repvgg_net.py

Purpose:
--------
This file implements a small RepVGG-style network built from RepVGG blocks.

Objective:
----------
Provide a full image classification model that:
- uses RepVGGBlock as the core building unit
- supports training-time multi-branch structure
- can be converted fully into deploy mode

This version is designed for small-scale experiments such as CIFAR-10.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.repvgg_block import RepVGGBlock


class RepVGG(nn.Module):
    """
    Small RepVGG-style classifier.

    Architecture:
    -------------
    - Stem block
    - 4 stages of RepVGG blocks
    - Global average pooling
    - Fully connected classifier

    Notes:
    ------
    This version is simplified for reproduction experiments on CIFAR-10.
    """

    def __init__(
        self,
        num_blocks: list[int],
        width_multiplier: list[int],
        num_classes: int = 10,
        deploy: bool = False,
    ) -> None:
        super().__init__()

        if len(num_blocks) != 4:
            raise ValueError("num_blocks must have length 4")
        if len(width_multiplier) != 4:
            raise ValueError("width_multiplier must have length 4")

        self.deploy = deploy
        self.in_channels = min(64, 32 * width_multiplier[0])

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_channels,
            stride=1,
            deploy=deploy,
        )

        self.stage1 = self._make_stage(
            out_channels=32 * width_multiplier[0],
            num_blocks=num_blocks[0],
            stride=1,
        )

        self.stage2 = self._make_stage(
            out_channels=64 * width_multiplier[1],
            num_blocks=num_blocks[1],
            stride=2,
        )

        self.stage3 = self._make_stage(
            out_channels=128 * width_multiplier[2],
            num_blocks=num_blocks[2],
            stride=2,
        )

        self.stage4 = self._make_stage(
            out_channels=256 * width_multiplier[3],
            num_blocks=num_blocks[3],
            stride=2,
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(256 * width_multiplier[3], num_classes)

    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Build one stage of RepVGG blocks.

        The first block may downsample using the provided stride.
        Remaining blocks use stride=1.
        """
        blocks = []

        strides = [stride] + [1] * (num_blocks - 1)
        for current_stride in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=current_stride,
                    deploy=self.deploy,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.gap(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)

        return x

    def switch_to_deploy(self) -> None:
        """
        Convert all RepVGG blocks in the model to deploy mode.
        """
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()


def create_repvgg_small(num_classes: int = 10, deploy: bool = False) -> RepVGG:
    """
    Factory function for a small RepVGG suitable for CIFAR-10.
    """
    return RepVGG(
        num_blocks=[2, 2, 2, 2],
        width_multiplier=[1, 1, 1, 1],
        num_classes=num_classes,
        deploy=deploy,
    )