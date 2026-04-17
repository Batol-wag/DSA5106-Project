"""
deploy_equivalence_eval.py

Evaluate RepVGG checkpoints before and after deploy conversion.

This script is used to verify structural re-parameterization by checking
that validation accuracy remains nearly identical after converting a
training-time RepVGG model into deploy mode.

Main tasks performed:

1. Load CIFAR-10 validation data.

2. Load saved RepVGG checkpoints for selected presets:
   - A0
   - A1
   - A2
   - B0
   - B1

3. Evaluate each checkpoint before deploy conversion.

4. Convert the model using switch_to_deploy().

5. Evaluate the converted model again.

6. Compare Top-1 and Top-5 accuracy before vs after deploy.

7. Save all results into CSV format.

Used in the final report to demonstrate that deploy conversion preserves
model behavior and performance.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from evaluate import evaluate_model
from train import apply_repvgg_preset
from models.repvgg_net import build_repvgg


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for deploy equivalence evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Run deploy equivalence for all RepVGG checkpoints"
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action="store_true", default=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="recommended",
        choices=["simple", "recommended"],
    )

    parser.add_argument(
        "--save_csv",
        type=str,
        default="results/deploy_equivalence.csv",
    )

    return parser.parse_args()


def build_cifar10_val_transform() -> transforms.Compose:
    """
    Build validation preprocessing pipeline for CIFAR-10.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


def select_subset(
    dataset,
    fraction: float,
    max_samples: int | None,
    seed: int,
):
    """
    Select a reproducible subset of the dataset.
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError("subset_fraction must satisfy 0 < subset_fraction <= 1")

    total_samples = len(dataset)
    target_samples = max(1, int(total_samples * fraction))

    if max_samples is not None:
        target_samples = min(target_samples, max_samples)

    if target_samples >= total_samples:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_samples, generator=generator)[:target_samples].tolist()

    return Subset(dataset, indices)


def get_val_loader(args: argparse.Namespace) -> DataLoader:
    """
    Create CIFAR-10 validation dataloader.
    """
    val_transform = build_cifar10_val_transform()

    val_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_transform,
    )

    val_dataset = select_subset(
        val_dataset,
        args.subset_fraction,
        args.max_val_samples,
        args.seed + 1,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )

    return val_loader


def load_clean_state_dict(
    checkpoint_path: Path,
    device: torch.device,
):
    """
    Load checkpoint weights and remove profiling keys if present.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not (
            key.endswith("total_ops")
            or key.endswith("total_params")
        )
    }

    return filtered_state_dict


def evaluate_checkpoint(
    args: argparse.Namespace,
    preset: str,
    device: torch.device,
    val_loader: DataLoader,
):
    """
    Evaluate one RepVGG preset before and after deploy conversion.
    """
    local_args = argparse.Namespace(**vars(args))
    local_args.repvgg_preset = preset
    local_args = apply_repvgg_preset(local_args)

    checkpoint_path = (
        Path(args.results_dir)
        / f"cifar10_repvgg_{preset}_{args.augmentation_mode}_subset{args.subset_fraction:g}_best.pth"
    )

    if not checkpoint_path.exists():
        print(f"[SKIP] Missing checkpoint: {checkpoint_path}")
        return None

    model = build_repvgg(
        variant=local_args.repvgg_variant,
        num_classes=10,
        deploy=False,
        a_multiplier=local_args.a_multiplier,
        b_multiplier=local_args.b_multiplier,
    ).to(device)

    state_dict = load_clean_state_dict(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)

    _, top1_before, top5_before = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        desc=f"{preset} before deploy",
    )

    model.switch_to_deploy()
    model = model.to(device)

    _, top1_after, top5_after = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        desc=f"{preset} after deploy",
    )

    return {
        "model": preset,
        "top1_before": top1_before,
        "top1_after": top1_after,
        "top1_diff": abs(top1_before - top1_after),
        "top5_before": top5_before,
        "top5_after": top5_after,
        "top5_diff": abs(top5_before - top5_after),
    }


def save_csv(rows, save_path: Path) -> None:
    """
    Save deploy equivalence results to CSV file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "top1_before",
                "top1_after",
                "top1_diff",
                "top5_before",
                "top5_after",
                "top5_diff",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
    Run deploy equivalence evaluation for all RepVGG presets.
    """
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    val_loader = get_val_loader(args)

    presets = ["A0", "A1", "A2", "B0", "B1"]
    rows = []

    for preset in presets:
        result = evaluate_checkpoint(
            args,
            preset,
            device,
            val_loader,
        )

        if result is not None:
            rows.append(result)

    print("\n" + "=" * 88)
    print("Deploy Equivalence Summary")
    print("=" * 88)

    print(
        f"| {'Model':<8} | {'Top1 Before':>12} | {'Top1 After':>11} | {'Top1 Diff':>10} "
        f"| {'Top5 Before':>12} | {'Top5 After':>11} | {'Top5 Diff':>10} |"
    )

    print("-" * 88)

    for row in rows:
        print(
            f"| {row['model']:<8} "
            f"| {row['top1_before']:>12.4f} "
            f"| {row['top1_after']:>11.4f} "
            f"| {row['top1_diff']:>10.6f} "
            f"| {row['top5_before']:>12.4f} "
            f"| {row['top5_after']:>11.4f} "
            f"| {row['top5_diff']:>10.6f} |"
        )

    save_path = Path(args.save_csv)
    save_csv(rows, save_path)

    print(f"\nSaved CSV to: {save_path}")


if __name__ == "__main__":
    main()