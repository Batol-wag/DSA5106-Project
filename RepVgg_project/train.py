from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from evaluate import evaluate_model, print_model_summary
from models.baselines import create_resnet18, create_resnet34
from models.repvgg_net import build_repvgg


def log(message: str) -> None:
    print(message, flush=True)


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        height = img.size(1)
        width = img.size(2)
        mask = torch.ones((height, width), dtype=img.dtype, device=img.device)

        for _ in range(self.n_holes):
            center_y = torch.randint(0, height, (1,)).item()
            center_x = torch.randint(0, width, (1,)).item()

            y1 = max(0, center_y - self.length // 2)
            y2 = min(height, center_y + self.length // 2)
            x1 = max(0, center_x - self.length // 2)
            x2 = min(width, center_x + self.length // 2)

            mask[y1:y2, x1:x2] = 0

        return img * mask.expand_as(img)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RepVGG/ResNet on CIFAR-10"
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument(
        "--model",
        type=str,
        default="repvgg",
        choices=["repvgg", "resnet18", "resnet34"],
    )

    parser.add_argument(
        "--repvgg_variant",
        type=str,
        default="a",
        choices=["small", "a", "b"],
    )
    parser.add_argument(
        "--repvgg_preset",
        type=str,
        default=None,
        choices=["A0", "A1", "A2", "B0", "B1"],
    )

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--a_multiplier", type=float, default=1.0)
    parser.add_argument("--b_multiplier", type=float, default=2.5)

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="recommended",
        choices=["simple", "recommended"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mixup_alpha", type=float, default=0.2)
    parser.add_argument("--cutout_holes", type=int, default=1)
    parser.add_argument("--cutout_length", type=int, default=16)

    parser.add_argument("--save_dir", type=str, default="./results")

    return parser.parse_args()


def apply_repvgg_preset(args: argparse.Namespace) -> argparse.Namespace:
    if args.repvgg_preset is None:
        return args

    preset_map = {
        "A0": {"variant": "a", "a": 0.75, "b": 2.5},
        "A1": {"variant": "a", "a": 1.0, "b": 2.5},
        "A2": {"variant": "a", "a": 1.5, "b": 2.75},
        "B0": {"variant": "b", "a": 1.0, "b": 2.5},
        "B1": {"variant": "b", "a": 2.0, "b": 4.0},
    }

    config = preset_map[args.repvgg_preset]
    args.repvgg_variant = config["variant"]
    args.a_multiplier = config["a"]
    args.b_multiplier = config["b"]
    return args


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.augmentation_mode == "simple":
        args.label_smoothing = 0.0
        args.mixup_alpha = 0.0
    else:
        if args.label_smoothing == 0.0:
            args.label_smoothing = 0.1
        if args.mixup_alpha == 0.0:
            args.mixup_alpha = 0.2

    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_input_size() -> tuple[int, int, int, int]:
    return (1, 3, 32, 32)


def _select_subset(
    dataset: Dataset,
    fraction: float,
    max_samples: int | None,
    seed: int,
) -> Dataset:
    if fraction <= 0 or fraction > 1:
        raise ValueError("subset_fraction must satisfy 0 < subset_fraction <= 1")

    total = len(dataset)
    target = max(1, int(total * fraction))

    if max_samples is not None:
        target = min(target, max_samples)

    if target >= total:
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator)[:target].tolist()
    return Subset(dataset, indices)


def build_cifar10_train_transform(args: argparse.Namespace) -> transforms.Compose:
    tf_list: list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ]

    if args.augmentation_mode == "recommended":
        tf_list.append(
            Cutout(
                n_holes=args.cutout_holes,
                length=args.cutout_length,
            )
        )

    return transforms.Compose(tf_list)


def build_cifar10_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


def get_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_tf = build_cifar10_train_transform(args)
    val_tf = build_cifar10_val_transform()

    train_ds = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )
    val_ds = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_tf,
    )

    train_ds = _select_subset(
        train_ds,
        args.subset_fraction,
        args.max_train_samples,
        args.seed,
    )
    val_ds = _select_subset(
        val_ds,
        args.subset_fraction,
        args.max_val_samples,
        args.seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
    )
    return train_loader, val_loader


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, float(lam)


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "repvgg":
        return build_repvgg(
            variant=args.repvgg_variant,
            a_multiplier=args.a_multiplier,
            b_multiplier=args.b_multiplier,
            num_classes=args.num_classes,
            deploy=False,
        )

    if args.model == "resnet18":
        return create_resnet18(num_classes=args.num_classes)

    if args.model == "resnet34":
        return create_resnet34(num_classes=args.num_classes)

    raise ValueError(f"Unsupported model: {args.model}")


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> optim.Optimizer:
    return optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


def build_scheduler(
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler._LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )


def build_run_name(args: argparse.Namespace) -> str:
    if args.model == "repvgg":
        if args.repvgg_preset is not None:
            model_tag = f"repvgg_{args.repvgg_preset}"
        else:
            model_tag = (
                f"repvgg_{args.repvgg_variant}"
                f"_a{args.a_multiplier:g}"
                f"_b{args.b_multiplier:g}"
            )
    else:
        model_tag = args.model

    return (
        f"cifar10_"
        f"{model_tag}_"
        f"{args.augmentation_mode}_"
        f"subset{args.subset_fraction:g}"
    )


def get_save_paths(args: argparse.Namespace) -> dict[str, Path]:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = build_run_name(args)

    return {
        "checkpoint": save_dir / f"{run_name}_best.pth",
        "history": save_dir / f"{run_name}_history.json",
        "final": save_dir / f"{run_name}_last.pth",
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val_top1: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_top1": best_val_top1,
            "args": vars(args),
        },
        path,
    )


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    args: argparse.Namespace,
    epoch: int,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    correct_top1 = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=True)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed_images, y_a, y_b, lam = mixup_data(
            images,
            labels,
            alpha=args.mixup_alpha,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(mixed_images)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, predicted = outputs.max(dim=1)
        correct_top1 += predicted.eq(labels).sum().item()

        avg_loss = running_loss / total
        top1 = 100.0 * correct_top1 / total
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}", top1=f"{top1:.2f}%")

    avg_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    return avg_loss, top1_acc


def main() -> None:
    args = parse_args()
    args = apply_repvgg_preset(args)
    args = resolve_defaults(args)

    set_seed(args.seed)
    device = get_device()

    log(f"Using device: {device}")
    log(f"Dataset: CIFAR-10")
    log(f"Model: {args.model}")
    if args.model == "repvgg":
        log(f"RepVGG preset: {args.repvgg_preset}")
        log(f"RepVGG variant: {args.repvgg_variant}")
        log(f"A multiplier: {args.a_multiplier}")
        log(f"B multiplier: {args.b_multiplier}")
    log(f"Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"Scheduler: {args.scheduler}")
    log(f"Augmentation mode: {args.augmentation_mode}")
    log(f"Label smoothing: {args.label_smoothing}")
    log(f"Mixup alpha: {args.mixup_alpha}")

    train_loader, val_loader = get_dataloaders(args)

    model = build_model(args).to(device)
    print_model_summary(model, input_size=get_input_size())

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    save_paths = get_save_paths(args)

    history: dict[str, list[float] | float | str | int] = {
        "dataset": "CIFAR-10",
        "model": args.model,
        "repvgg_preset": args.repvgg_preset,
        "repvgg_variant": args.repvgg_variant,
        "a_multiplier": args.a_multiplier,
        "b_multiplier": args.b_multiplier,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "momentum": args.momentum,
        "scheduler": args.scheduler,
        "augmentation_mode": args.augmentation_mode,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "subset_fraction": args.subset_fraction,
        "train_loss": [],
        "train_top1": [],
        "val_loss": [],
        "val_top1": [],
        "val_top5": [],
        "lr_history": [],
        "epoch_time_sec": [],
        "best_val_top1": 0.0,
    }

    best_val_top1 = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_top1 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            args=args,
            epoch=epoch,
        )

        val_loss, val_top1, val_top5 = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch} [Val]",
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_top1"].append(train_top1)
        history["val_loss"].append(val_loss)
        history["val_top1"].append(val_top1)
        history["val_top5"].append(val_top5)
        history["lr_history"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        log(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | train_top1={train_top1:.2f}% | "
            f"val_loss={val_loss:.4f} | val_top1={val_top1:.2f}% | "
            f"val_top5={val_top5:.2f}% | lr={current_lr:.6f}"
        )

        if val_top1 > best_val_top1:
            best_val_top1 = val_top1
            history["best_val_top1"] = best_val_top1

            save_checkpoint(
                path=save_paths["checkpoint"],
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_top1=best_val_top1,
                args=args,
            )
            log(f"[Saved best checkpoint] {save_paths['checkpoint']}")

        with open(save_paths["history"], "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    save_checkpoint(
        path=save_paths["final"],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs,
        best_val_top1=best_val_top1,
        args=args,
    )
    log(f"[Saved final checkpoint] {save_paths['final']}")
    log(f"[Saved history] {save_paths['history']}")
    log(f"Best validation Top-1: {best_val_top1:.2f}%")


if __name__ == "__main__":
    main()