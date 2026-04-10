"""
evaluate.py

Evaluation utilities shared by training and comparison scripts.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.repvgg_net import build_repvgg
from utils.metrics import count_parameters, compute_flops


def print_model_summary(
    model: nn.Module,
    input_size: tuple[int, int, int, int],
) -> None:
    """
    Print trainable parameter count and GFLOPs.
    """
    params = count_parameters(model)
    flops = compute_flops(model, input_size=input_size)
    flops_str = f"{flops:.4f} GFLOPs" if flops is not None else "N/A (install thop)"
    print(f"  Parameters : {params:,}", flush=True)
    print(f"  GFLOPs     : {flops_str}", flush=True)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    desc: str = "Evaluation",
) -> tuple[float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=desc, leave=True)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, predicted = outputs.max(dim=1)
        correct_top1 += predicted.eq(labels).sum().item()

        top_k = min(5, outputs.size(1))
        _, top5_preds = outputs.topk(top_k, dim=1, largest=True, sorted=True)
        correct_top5 += top5_preds.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

        current_loss = running_loss / total
        current_top1 = 100.0 * correct_top1 / total
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", top1=f"{current_top1:.2f}%")

    avg_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total

    return avg_loss, top1_acc, top5_acc


def validate_repvgg_deploy(
    test_loader: DataLoader,
    device: torch.device,
    dataset: str,
    model_variant: str,
    num_classes: int,
    checkpoint_path: str | Path,
    a_multiplier: float = 1.0,
    b_multiplier: float = 2.5,
) -> tuple[float, float]:
    """
    Validate RepVGG before and after deploy conversion.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_repvgg(
        dataset=dataset,
        variant=model_variant,
        num_classes=num_classes,
        deploy=False,
        a_multiplier=a_multiplier,
        b_multiplier=b_multiplier,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    print("\nEvaluating RepVGG BEFORE deploy conversion...", flush=True)
    _, acc_before, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        desc="Before Deploy",
    )

    print("\nSwitching RepVGG to deploy mode...", flush=True)
    model.switch_to_deploy()
    model = model.to(device)

    print("\nEvaluating RepVGG AFTER deploy conversion...", flush=True)
    _, acc_after, _ = evaluate_model(
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
