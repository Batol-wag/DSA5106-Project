"""
repvgg_with_branches.py

Purpose:
--------
Extended RepVGG model that exposes branch outputs for branch-level knowledge distillation.

Objective:
----------
Support the Branch-Level Knowledge Distillation extension by:
- Exposing individual branch outputs (3×3, 1×1, identity)
- Enabling per-branch supervision during training
- Maintaining inference-time efficiency after re-parameterization

This module implements:
- RepVGGBlockWithBranches: A RepVGG block that returns branch outputs
- RepVGGWithBranches: Full model that can return nested branch outputs
- A factory function to create the distillation-enabled model
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.repvgg_block import RepVGGBlock
from utils.fusion import (
    fuse_conv_bn,
    pad_1x1_to_3x3,
    get_identity_kernel_bias,
)


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    """
    Create a Conv2d + BatchNorm2d block (no bias in conv).
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
    )


class RepVGGBlockWithBranches(nn.Module):
    """
    RepVGG Block with exposed branch outputs for distillation.

    Returns:
    --------
    (combined_output, branch_outputs) where:
    - combined_output: The final block output after ReLU
    - branch_outputs: Dict with keys '3x3', '1x1', 'identity' (if applicable)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.deploy = deploy

        self.nonlinearity = nn.ReLU(inplace=True)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            # 3×3 branch
            self.rbr_dense = conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            )

            # 1×1 branch
            self.rbr_1x1 = conv_bn(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

            # Identity branch
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | torch.Tensor:
        """
        Forward pass with branch outputs.

        If deploy=False, returns (final_output, branch_outputs_dict)
        If deploy=True, returns final_output only (no branching)
        """
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))

        # Training mode: compute and collect branch outputs
        out_3x3 = self.rbr_dense(x)
        out_1x1 = self.rbr_1x1(x)
        out_identity = None

        if self.rbr_identity is not None:
            out_identity = self.rbr_identity(x)

        # Combine branches
        out = out_3x3 + out_1x1
        if out_identity is not None:
            out = out + out_identity

        out = self.nonlinearity(out)

        # Pack branch outputs for distillation
        branch_outputs = {
            "3x3": self.nonlinearity(out_3x3),
            "1x1": self.nonlinearity(out_1x1),
        }
        if out_identity is not None:
            branch_outputs["identity"] = self.nonlinearity(out_identity)

        return out, branch_outputs

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel3x3, bias3x3 = fuse_conv_bn(self.rbr_dense)

        kernel1x1, bias1x1 = fuse_conv_bn(self.rbr_1x1)
        kernel1x1 = pad_1x1_to_3x3(kernel1x1)

        if self.rbr_identity is not None:
            kernelid, biasid = get_identity_kernel_bias(
                self.rbr_identity,
                self.out_channels,
                self.groups
            )
        else:
            kernelid = 0
            biasid = 0

        kernel = kernel3x3 + kernel1x1 + kernelid
        bias = bias3x3 + bias1x1 + biasid

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel.to(self.rbr_reparam.weight.device)
        self.rbr_reparam.bias.data = bias.to(self.rbr_reparam.bias.device)

        del self.rbr_dense
        del self.rbr_1x1
        if self.rbr_identity is not None:
            del self.rbr_identity

        self.deploy = True

class RepVGGWithBranches(nn.Module):
    """
    Small RepVGG-style classifier with branch output support.

    Can operate in two modes:
    - Training-time: Returns (final_output, nested_branch_structures) for distillation
    - Inference-time (deploy): Returns final_output only
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

        self.stage0 = RepVGGBlockWithBranches(
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
        Build one stage of RepVGG blocks with branch outputs.
        """
        blocks = []

        strides = [stride] + [1] * (num_blocks - 1)
        for current_stride in strides:
            blocks.append(
                RepVGGBlockWithBranches(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=current_stride,
                    deploy=self.deploy,
                )
            )
            self.in_channels = out_channels

        return nn.Sequential(*blocks)

    def forward(
        self, x: torch.Tensor, return_branches: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        return_branches : bool
            If True, return (logits, branch_outputs) for distillation.
            If False, return logits only.

        Returns
        -------
        logits : torch.Tensor
            Classification logits
        (logits, branch_data) : tuple (only if return_branches=True in training)
            Also return structurally nested branch outputs for supervision.
        """
        if self.deploy:
            # Deploy mode: no branching support
            x = self.stage0(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.gap(x)
            x = torch.flatten(x, start_dim=1)
            x = self.linear(x)
            return x

        # Training mode with optional branch tracking
        out_stage0, branch0 = self.stage0(x)

        branch_data = {}

        # Forward through stages, collecting branch outputs if needed
        out = out_stage0
        if return_branches:
            branch_data["stage0"] = branch0

        for stage_idx, stage in enumerate([self.stage1, self.stage2, self.stage3, self.stage4], start=1):
            stage_branches = {}
            for block_idx, block in enumerate(stage):
                out, block_branches = block(out)
                stage_branches[f"block_{block_idx}"] = block_branches
            if return_branches:
                branch_data[f"stage{stage_idx}"] = stage_branches

        out = self.gap(out)
        out = torch.flatten(out, start_dim=1)
        logits = self.linear(out)

        if return_branches:
            return logits, branch_data
        else:
            return logits

    # def switch_to_deploy(self):
    #     self.deploy = True
    #     for module in self.modules():
    #         if isinstance(module, RepVGGBlockWithBranches):
    #             module.switch_to_deploy()

    def switch_to_deploy(self):
        if self.deploy:
            return

        for module in self.modules():
            if isinstance(module, RepVGGBlockWithBranches):
                module.switch_to_deploy()

        self.deploy = True


def create_repvgg_with_branches(
    num_classes: int = 10, deploy: bool = False
) -> RepVGGWithBranches:
    """
    Factory function for RepVGG with branch output support.

    Suitable for branch-level knowledge distillation on CIFAR-10.
    """
    return RepVGGWithBranches(
        num_blocks=[2, 2, 2, 2],
        width_multiplier=[1, 1, 1, 1],
        num_classes=num_classes,
        deploy=deploy,
    )