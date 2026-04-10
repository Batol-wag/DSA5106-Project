"""
metrics.py

Purpose:
--------
Utility functions for measuring model size, computational cost, and inference speed.

Functions:
----------
- count_parameters: total trainable parameter count
- compute_flops: GFLOPs via thop (graceful fallback if not installed)
- measure_inference_latency: mean forward-pass latency in milliseconds per batch
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    """
    Return the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 32, 32),
    device: torch.device | None = None,
) -> float | None:
    """
    Estimate model GFLOPs using thop.profile().

    Parameters
    ----------
    model : nn.Module
        Model to profile (will be set to eval mode).
    input_size : tuple
        Input tensor shape as (N, C, H, W). Default is CIFAR-10 single image.
    device : torch.device or None
        Device for the dummy input. Defaults to the model's first parameter device.

    Returns
    -------
    float or None
        GFLOPs (1e9 FLOPs), or None if thop is not installed.
    """
    try:
        from thop import profile  # type: ignore
    except ImportError:
        return None

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dummy_input = torch.zeros(input_size, device=device)

    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy_input,), verbose=False)

    return flops / 1e9


def measure_inference_latency(
    model: nn.Module,
    input_size: tuple[int, int, int, int] = (1, 3, 32, 32),
    device: torch.device | None = None,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> float:
    """
    Measure mean forward-pass latency in milliseconds per batch.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark (will be set to eval mode).
    input_size : tuple
        Input tensor shape as (N, C, H, W).
    device : torch.device or None
        Device to run on. Defaults to the model's first parameter device.
    n_warmup : int
        Number of warmup forward passes before timing.
    n_runs : int
        Number of timed forward passes.

    Returns
    -------
    float
        Mean latency in milliseconds per batch.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    model.eval()
    dummy_input = torch.zeros(input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

    mean_ms = (end - start) / n_runs * 1000
    return mean_ms
