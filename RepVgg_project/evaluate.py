"""
evaluate.py

Purpose:
--------
This file contains evaluation utilities for the RepVGG reproduction project.

Objective:
----------
Provide reusable functions for:
- standard model evaluation on a dataset
- validating RepVGG accuracy before and after deploy conversion

Why this file matters:
----------------------
Evaluation is needed in multiple places in the project:
- during training
- for final testing
- for validating structural re-parameterization at the dataset level

Keeping these functions in one file makes the project easier to follow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.repvgg_net import create_repvgg_small


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Evaluation",
) -> tuple[float, float]:
    """
    Evaluate a model on a dataset.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : DataLoader
        Data loader for evaluation data.
    device : torch.device
        Device to run evaluation on.
    desc : str
        Progress bar description.

    Returns
    -------
    avg_loss : float
        Average loss over the dataset.
    accuracy : float
        Accuracy in percentage.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=desc, leave=True)
    non_blocking = device.type == "cuda"

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.2f}%",
        )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate_repvgg_deploy(
    test_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str | Path = "checkpoints/repvgg_best.pth",
) -> tuple[float, float]:
    """
    Validate RepVGG performance before and after deploy conversion.

    Parameters
    ----------
    test_loader : DataLoader
        Data loader for CIFAR-10 test data.
    device : torch.device
        Device for evaluation.
    checkpoint_path : str or Path
        Path to the saved RepVGG checkpoint.

    Returns
    -------
    acc_before : float
        Accuracy before deploy conversion.
    acc_after : float
        Accuracy after deploy conversion.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load RepVGG in training-time form
    model = create_repvgg_small(num_classes=10, deploy=False).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    print("\nEvaluating RepVGG BEFORE deploy conversion...", flush=True)
    _, acc_before = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        desc="Before Deploy",
    )

    print("\nSwitching RepVGG to deploy mode...", flush=True)
    model.switch_to_deploy()
    model = model.to(device)

    print("\nEvaluating RepVGG AFTER deploy conversion...", flush=True)
    _, acc_after = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        desc="After Deploy",
    )

    print("\n" + "=" * 60)
    print("RepVGG Deploy Validation Results")
    print("=" * 60)
    print(f"Before deploy accuracy: {acc_before:.2f}%")
    print(f"After deploy accuracy : {acc_after:.2f}%")
    print(f"Absolute difference   : {abs(acc_before - acc_after):.6f}%")
    print("=" * 60)

    return acc_before, acc_after
