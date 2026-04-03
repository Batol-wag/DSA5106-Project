"""
fusion.py

Purpose:
--------
This file implements the core fusion utilities required for RepVGG
structural re-parameterization.

Objective:
----------
Convert the training-time multi-branch structure into an equivalent
single 3×3 convolution for deployment.

Implemented utilities:
----------------------
1. fuse_conv_bn:
   Fuse a Conv2d + BatchNorm2d sequence into an equivalent kernel and bias.

2. pad_1x1_to_3x3:
   Convert a 1×1 convolution kernel into a 3×3 kernel by zero-padding.

3. get_identity_kernel_bias:
   Convert an identity + BatchNorm branch into an equivalent 3×3 kernel and bias.

These functions will be used by RepVGGBlock to create a deploy-time
single 3×3 convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def fuse_conv_bn(branch: nn.Sequential) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse a Conv2d + BatchNorm2d branch into an equivalent convolution kernel and bias.

    Parameters
    ----------
    branch : nn.Sequential
        A sequential module containing:
        [0] Conv2d (bias=False)
        [1] BatchNorm2d

    Returns
    -------
    kernel : torch.Tensor
        Fused convolution kernel.
    bias : torch.Tensor
        Fused convolution bias.
    """
    if branch is None:
        raise ValueError("branch must not be None")

    if not isinstance(branch, nn.Sequential) or len(branch) != 2:
        raise TypeError("branch must be nn.Sequential with [Conv2d, BatchNorm2d]")

    conv = branch[0]
    bn = branch[1]

    if not isinstance(conv, nn.Conv2d):
        raise TypeError("branch[0] must be nn.Conv2d")
    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("branch[1] must be nn.BatchNorm2d")

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)

    # reshape for broadcasting over kernel dimensions
    t = (gamma / std).reshape(-1, 1, 1, 1)

    fused_kernel = kernel * t
    fused_bias = beta - running_mean * gamma / std

    return fused_kernel, fused_bias


def pad_1x1_to_3x3(kernel_1x1: torch.Tensor) -> torch.Tensor:
    """
    Pad a 1×1 convolution kernel to a 3×3 kernel.

    The original 1×1 values are placed in the center of the 3×3 kernel.

    Parameters
    ----------
    kernel_1x1 : torch.Tensor
        Kernel of shape [out_channels, in_channels/groups, 1, 1]

    Returns
    -------
    torch.Tensor
        Kernel of shape [out_channels, in_channels/groups, 3, 3]
    """
    if kernel_1x1 is None:
        raise ValueError("kernel_1x1 must not be None")

    if kernel_1x1.size(2) != 1 or kernel_1x1.size(3) != 1:
        raise ValueError("Input kernel must have spatial size 1x1")

    return F.pad(kernel_1x1, [1, 1, 1, 1])


def get_identity_kernel_bias(
    bn: nn.BatchNorm2d,
    in_channels: int,
    groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an identity + BatchNorm branch into an equivalent 3×3 convolution kernel and bias.

    Parameters
    ----------
    bn : nn.BatchNorm2d
        BatchNorm layer applied on the identity branch.
    in_channels : int
        Number of input channels.
    groups : int, default=1
        Number of convolution groups.

    Returns
    -------
    kernel : torch.Tensor
        Equivalent identity kernel of shape [C_out, C_in/groups, 3, 3]
    bias : torch.Tensor
        Equivalent bias vector.
    """
    if bn is None:
        raise ValueError("bn must not be None")

    if not isinstance(bn, nn.BatchNorm2d):
        raise TypeError("bn must be nn.BatchNorm2d")

    if in_channels % groups != 0:
        raise ValueError("in_channels must be divisible by groups")

    input_dim = in_channels // groups
    kernel_value = torch.zeros(
        (in_channels, input_dim, 3, 3),
        dtype=bn.weight.dtype,
        device=bn.weight.device,
    )

    for i in range(in_channels):
        kernel_value[i, i % input_dim, 1, 1] = 1.0

    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)

    fused_kernel = kernel_value * t
    fused_bias = beta - running_mean * gamma / std

    return fused_kernel, fused_bias