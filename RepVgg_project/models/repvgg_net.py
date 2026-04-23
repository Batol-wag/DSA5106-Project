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
        width_multiplier: list[float],
        num_classes: int = 10,
        deploy: bool = False,
        stage_base_channels: list[int] | None = None,
        stem_channels: int | None = None,
    ) -> None:
        super().__init__()

        if len(num_blocks) != 4:
            raise ValueError("num_blocks must have length 4")
        if len(width_multiplier) != 4:
            raise ValueError("width_multiplier must have length 4")
        if stage_base_channels is None:
            stage_base_channels = [32, 64, 128, 256]
        if len(stage_base_channels) != 4:
            raise ValueError("stage_base_channels must have length 4")

        self.deploy = deploy
        self.width_multiplier = [float(multiplier) for multiplier in width_multiplier]
        stage_channels = [
            int(base * multiplier)
            for base, multiplier in zip(stage_base_channels, self.width_multiplier)
        ]
        self.in_channels = (
            stem_channels if stem_channels is not None else min(64, stage_channels[0])
        )

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_channels,
            stride=1,
            deploy=deploy,
        )

        self.stage1 = self._make_stage(
            out_channels=stage_channels[0],
            num_blocks=num_blocks[0],
            stride=1,
        )

        self.stage2 = self._make_stage(
            out_channels=stage_channels[1],
            num_blocks=num_blocks[1],
            stride=2,
        )

        self.stage3 = self._make_stage(
            out_channels=stage_channels[2],
            num_blocks=num_blocks[2],
            stride=2,
        )

        self.stage4 = self._make_stage(
            out_channels=stage_channels[3],
            num_blocks=num_blocks[3],
            stride=2,
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(stage_channels[3], num_classes)

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


REPVGG_PRESETS = {
    "A0": {
        "num_blocks": [2, 4, 14, 1],
        "width_multiplier": [0.75, 0.75, 0.75, 2.5],
    },
    "A1": {
        "num_blocks": [2, 4, 14, 1],
        "width_multiplier": [1.0, 1.0, 1.0, 2.5],
    },
    "A2": {
        "num_blocks": [2, 4, 14, 1],
        "width_multiplier": [1.5, 1.5, 1.5, 2.75],
    },
    "B0": {
        "num_blocks": [4, 6, 16, 1],
        "width_multiplier": [1.0, 1.0, 1.0, 2.5],
    },
    "B1": {
        "num_blocks": [4, 6, 16, 1],
        "width_multiplier": [2.0, 2.0, 2.0, 4.0],
    },
}


def create_repvgg(
    *,
    num_blocks: list[int],
    width_multiplier: list[float],
    num_classes: int = 10,
    deploy: bool = False,
    stage_base_channels: list[int] | None = None,
    stem_channels: int | None = None,
) -> RepVGG:
    """
    Create a RepVGG model with the provided preset definition.
    """
    return RepVGG(
        num_blocks=num_blocks,
        width_multiplier=width_multiplier,
        num_classes=num_classes,
        deploy=deploy,
        stage_base_channels=stage_base_channels,
        stem_channels=stem_channels,
    )


def create_repvgg_small(num_classes: int = 10, deploy: bool = False) -> RepVGG:
    """
    Factory function for a small RepVGG suitable for CIFAR-10.
    """
    return create_repvgg(
        num_blocks=[2, 2, 2, 2],
        width_multiplier=[1, 1, 1, 1],
        num_classes=num_classes,
        deploy=deploy,
        stage_base_channels=[32, 64, 128, 256],
        stem_channels=32,
    )


def create_repvgg_from_preset(
    preset: str,
    num_classes: int = 10,
    deploy: bool = False,
) -> RepVGG:
    """
    Create a RepVGG model from one of the larger baseline presets.
    """
    preset = preset.upper()
    if preset not in REPVGG_PRESETS:
        raise ValueError(f"Unsupported RepVGG preset: {preset}")

    config = REPVGG_PRESETS[preset]
    return create_repvgg(
        num_blocks=config["num_blocks"],
        width_multiplier=config["width_multiplier"],
        num_classes=num_classes,
        deploy=deploy,
        stage_base_channels=[64, 128, 256, 512],
        stem_channels=min(64, int(64 * config["width_multiplier"][0])),
    )
