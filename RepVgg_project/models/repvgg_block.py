"""
repvgg_block.py

Purpose:
--------
This file implements the core building block of the RepVGG architecture.

The RepVGG block uses a multi-branch structure during training:
- 3×3 convolution + BatchNorm
- 1×1 convolution + BatchNorm
- Identity branch + BatchNorm (only when applicable)

These branches are summed and followed by a ReLU activation.

Objective:
----------
Enable structural re-parameterization:
the multi-branch training-time structure can be fused into a single
3×3 convolution for efficient inference (deploy mode).

This file implements:
- Training-time RepVGG block
- Conversion to deploy (single conv)

Dependencies:
-------------
- Uses fusion functions from: utils/fusion.py
- Used by: repvgg_net.py
"""

import torch
import torch.nn as nn

# NOTE: This will be implemented in the next file
# (leave it for now; do not worry if import fails temporarily)
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


class RepVGGBlock(nn.Module):
    """
    RepVGG Block (training-time version + deploy conversion).

    Structure (training):
    ---------------------
    - 3×3 Conv + BN
    - 1×1 Conv + BN
    - Identity + BN (if stride=1 and channels match)

    These are summed and passed through ReLU.

    Structure (deploy):
    -------------------
    - Single 3×3 Conv (with bias) + ReLU

    Methods:
    --------
    - get_equivalent_kernel_bias(): compute fused kernel
    - switch_to_deploy(): convert block to inference form
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
            # Inference-time single conv
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
            # 3x3 branch
            self.rbr_dense = conv_bn(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            )

            # 1x1 branch
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        """
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(x))

        out = self.rbr_dense(x) + self.rbr_1x1(x)

        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)

        return self.nonlinearity(out)

    def get_equivalent_kernel_bias(self):
        """
        Compute the fused 3×3 kernel and bias by combining:
        - 3×3 branch
        - 1×1 branch (padded to 3×3)
        - identity branch
        """

        # 3x3 branch
        kernel3x3, bias3x3 = fuse_conv_bn(self.rbr_dense)

        # 1x1 branch → pad to 3x3
        kernel1x1, bias1x1 = fuse_conv_bn(self.rbr_1x1)
        kernel1x1 = pad_1x1_to_3x3(kernel1x1)

        # identity branch
        if self.rbr_identity is not None:
            kernel_id, bias_id = get_identity_kernel_bias(
                self.rbr_identity,
                self.in_channels,
                self.groups,
            )
        else:
            kernel_id = 0
            bias_id = 0

        # sum all
        kernel = kernel3x3 + kernel1x1 + kernel_id
        bias = bias3x3 + bias1x1 + bias_id

        return kernel, bias

    def switch_to_deploy(self):
        """
        Convert the block to deploy mode:
        replace all branches with a single fused 3×3 conv.
        """
        if self.deploy:
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Delete unnecessary branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")

        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True