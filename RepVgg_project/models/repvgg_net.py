"""
repvgg_net.py

RepVGG model definitions for CIFAR-10 experiments only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.repvgg_block import RepVGGBlock


@dataclass(frozen=True)
class RepVGGConfig:
    stage_channels: list[int]
    num_blocks: list[int]
    stem_stride: int
    stage1_stride: int
    num_classes: int = 10


class RepVGG(nn.Module):
    def __init__(
        self,
        config: RepVGGConfig,
        deploy: bool = False,
    ) -> None:
        super().__init__()

        if len(config.stage_channels) != 5:
            raise ValueError("stage_channels must have length 5")
        if len(config.num_blocks) != 4:
            raise ValueError("num_blocks must have length 4")

        self.deploy = deploy
        self.config = config
        self.in_channels = config.stage_channels[0]

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=config.stage_channels[0],
            stride=config.stem_stride,
            deploy=deploy,
        )

        self.stage1 = self._make_stage(
            out_channels=config.stage_channels[1],
            num_blocks=config.num_blocks[0],
            stride=config.stage1_stride,
        )

        self.stage2 = self._make_stage(
            out_channels=config.stage_channels[2],
            num_blocks=config.num_blocks[1],
            stride=2,
        )

        self.stage3 = self._make_stage(
            out_channels=config.stage_channels[3],
            num_blocks=config.num_blocks[2],
            stride=2,
        )

        self.stage4 = self._make_stage(
            out_channels=config.stage_channels[4],
            num_blocks=config.num_blocks[3],
            stride=2,
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(config.stage_channels[4], config.num_classes)

    def _make_stage(
        self,
        out_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = []

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
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()


def _paper_style_stage_channels(
    a_multiplier: float,
    b_multiplier: float,
) -> list[int]:
    stem = min(64, int(round(64 * a_multiplier)))
    stage1 = int(round(64 * a_multiplier))
    stage2 = int(round(128 * a_multiplier))
    stage3 = int(round(256 * a_multiplier))
    stage4 = int(round(512 * b_multiplier))
    return [stem, stage1, stage2, stage3, stage4]


def create_repvgg_small(num_classes: int = 10, deploy: bool = False) -> RepVGG:
    config = RepVGGConfig(
        stage_channels=[32, 32, 64, 128, 256],
        num_blocks=[2, 2, 2, 2],
        stem_stride=1,
        stage1_stride=1,
        num_classes=num_classes,
    )
    return RepVGG(config=config, deploy=deploy)


def create_repvgg_cifar_a(
    num_classes: int = 10,
    deploy: bool = False,
    a_multiplier: float = 1.0,
    b_multiplier: float = 2.5,
) -> RepVGG:
    config = RepVGGConfig(
        stage_channels=_paper_style_stage_channels(a_multiplier, b_multiplier),
        num_blocks=[2, 4, 14, 1],
        stem_stride=1,
        stage1_stride=1,
        num_classes=num_classes,
    )
    return RepVGG(config=config, deploy=deploy)


def create_repvgg_cifar_b(
    num_classes: int = 10,
    deploy: bool = False,
    a_multiplier: float = 1.0,
    b_multiplier: float = 2.5,
) -> RepVGG:
    config = RepVGGConfig(
        stage_channels=_paper_style_stage_channels(a_multiplier, b_multiplier),
        num_blocks=[4, 6, 16, 1],
        stem_stride=1,
        stage1_stride=1,
        num_classes=num_classes,
    )
    return RepVGG(config=config, deploy=deploy)


def build_repvgg(
    variant: str,
    num_classes: int,
    deploy: bool = False,
    a_multiplier: float = 1.0,
    b_multiplier: float = 2.5,
) -> RepVGG:
    variant = variant.lower()

    if variant == "small":
        return create_repvgg_small(num_classes=num_classes, deploy=deploy)

    if variant == "a":
        return create_repvgg_cifar_a(
            num_classes=num_classes,
            deploy=deploy,
            a_multiplier=a_multiplier,
            b_multiplier=b_multiplier,
        )

    if variant == "b":
        return create_repvgg_cifar_b(
            num_classes=num_classes,
            deploy=deploy,
            a_multiplier=a_multiplier,
            b_multiplier=b_multiplier,
        )

    raise ValueError("RepVGG variant must be 'small', 'a', or 'b'")