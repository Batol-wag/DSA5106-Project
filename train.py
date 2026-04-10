from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from tqdm import tqdm
from datasets import load_dataset
from PIL import Image

from evaluate import evaluate_model, print_model_summary
from models.baselines import create_resnet18, create_resnet34
from models.repvgg_net import build_repvgg


def log(message: str) -> None:
    print(message, flush=True)


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


##############################################
# ARGUMENTS
##############################################
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RepVGG/ResNet on CIFAR-10 or Tiny ImageNet"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "tiny_imagenet"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="repvgg",
        choices=["repvgg", "resnet18", "resnet34"],
    )

    parser.add_argument(
        "--repvgg_variant",
        type=str,
        default="small",
        choices=["small", "a", "b"],
    )
    parser.add_argument(
        "--repvgg_preset",
        type=str,
        default=None,
        choices=["A0", "A1", "A2", "B0", "B1"],
    )

    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["step", "cosine"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--imagenet_image_size", type=int, default=64)

    parser.add_argument("--a_multiplier", type=float, default=1.0)
    parser.add_argument("--b_multiplier", type=float, default=2.5)

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="simple",
        choices=["simple", "strong"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--autoaugment", type=str2bool, default=False)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)

    return parser.parse_args()


##############################################
# PRESET SYSTEM
##############################################
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


##############################################
# DEFAULTS
##############################################
def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.num_classes is None:
        args.num_classes = 10 if args.dataset == "cifar10" else 200

    if args.epochs is None:
        args.epochs = 20 if args.dataset == "cifar10" else 120

    if args.lr is None:
        args.lr = 0.01 if args.dataset == "cifar10" else 0.1

    if args.weight_decay is None:
        args.weight_decay = 5e-4 if args.dataset == "cifar10" else 1e-4

    if args.scheduler is None:
        args.scheduler = "step" if args.dataset == "cifar10" else "cosine"

    if args.dataset == "cifar10":
        args.repvgg_variant = "small"

    if args.dataset == "tiny_imagenet" and args.repvgg_variant == "small":
        args.repvgg_variant = "a"

    if args.dataset == "tiny_imagenet" and args.num_classes != 200:
        raise ValueError("Tiny ImageNet requires num_classes=200.")

    if args.dataset == "cifar10" and args.repvgg_preset is not None:
        raise ValueError("RepVGG presets are only for tiny_imagenet.")

    if args.augmentation_mode == "strong":
        if args.label_smoothing == 0.0:
            args.label_smoothing = 0.1
        if not args.autoaugment:
            args.autoaugment = True
        if args.mixup_alpha == 0.0:
            args.mixup_alpha = 0.2

    return args


##############################################
# UTILITIES
##############################################
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_input_size(args: argparse.Namespace) -> tuple[int, int, int, int]:
    if args.dataset == "cifar10":
        return (1, 3, 32, 32)
    return (1, 3, args.imagenet_image_size, args.imagenet_image_size)


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
    tf_list = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if args.autoaugment:
        tf_list.append(AutoAugment(policy=AutoAugmentPolicy.CIFAR10))

    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    return transforms.Compose(tf_list)


def build_cifar10_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])


def build_tiny_imagenet_train_transform(args: argparse.Namespace) -> transforms.Compose:
    tf_list = [
        transforms.RandomResizedCrop(args.imagenet_image_size),
        transforms.RandomHorizontalFlip(),
    ]

    if args.autoaugment:
        tf_list.append(AutoAugment(policy=AutoAugmentPolicy.IMAGENET))

    tf_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    return transforms.Compose(tf_list)


def build_tiny_imagenet_val_transform(args: argparse.Namespace) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(args.imagenet_image_size / 0.875)),
        transforms.CenterCrop(args.imagenet_image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])


##############################################
# DATASET LOADING
##############################################
def get_cifar10_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_tf = build_cifar10_train_transform(args)
    val_tf = build_cifar10_val_transform()

    train_ds = datasets.CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )

    val_ds = datasets.CIFAR10(
        args.data_dir,
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

    return (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        ),
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        ),
    )


def prepare_tiny_imagenet_from_hf(data_dir: str) -> str:
    root = Path(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if train_dir.exists() and val_dir.exists():
        log(f"Tiny ImageNet already prepared at: {root}")
        return str(root)

    log("Downloading Tiny ImageNet from Hugging Face...")
    ds = load_dataset("zh-plus/tiny-imagenet")

    root.mkdir(parents=True, exist_ok=True)

    for hf_split, out_split in {"train": "train", "valid": "val"}.items():
        split_dir = root / out_split
        split_dir.mkdir(parents=True, exist_ok=True)

        log(f"Processing split: {hf_split} -> {out_split}")

        for idx, example in enumerate(ds[hf_split]):
            img = example["image"]
            label = ds[hf_split].features["label"].int2str(example["label"])

            cls_dir = split_dir / label
            cls_dir.mkdir(parents=True, exist_ok=True)

            img.save(cls_dir / f"{idx:06d}.png")

    log(f"Tiny ImageNet prepared successfully at: {root}")
    return str(root)


def get_tiny_imagenet_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    prepare_tiny_imagenet_from_hf(args.data_dir)

    train_tf = build_tiny_imagenet_train_transform(args)
    val_tf = build_tiny_imagenet_val_transform(args)

    train_ds = datasets.ImageFolder(
        Path(args.data_dir) / "train",
        transform=train_tf,
    )
    val_ds = datasets.ImageFolder(
        Path(args.data_dir) / "val",
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

    return (
        DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        ),
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and torch.cuda.is_available(),
        ),
    )


def get_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.dataset == "cifar10":
        return get_cifar10_dataloaders(args)
    return get_tiny_imagenet_dataloaders(args)


##############################################
# MODEL
##############################################
def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "repvgg":
        return build_repvgg(
            dataset=args.dataset,
            variant=args.repvgg_variant,
            num_classes=args.num_classes,
            deploy=False,
            a_multiplier=args.a_multiplier,
            b_multiplier=args.b_multiplier,
        )

    if args.model == "resnet18":
        return create_resnet18(args.dataset, args.num_classes)

    return create_resnet34(args.dataset, args.num_classes)

def apply_mixup(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
):
    if alpha <= 0.0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam

def mixup_loss(
    criterion: nn.Module,
    outputs: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
):
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    amp_enabled: bool,
    mixup_alpha: float = 0.0,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        use_mixup = mixup_alpha > 0.0

        if use_mixup:
            images, labels_a, labels_b, lam = apply_mixup(
                images, labels, mixup_alpha, device
            )

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)

            if use_mixup:
                loss = mixup_loss(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)

        pred = outputs.argmax(dim=1)

        if use_mixup:
            correct += (
                lam * pred.eq(labels_a).sum().item()
                + (1 - lam) * pred.eq(labels_b).sum().item()
            )
        else:
            correct += pred.eq(labels).sum().item()

        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

    return total_loss / total, 100.0 * correct / total


##############################################
# SCHEDULER
##############################################
def make_scheduler(
    optimizer: optim.Optimizer,
    args: argparse.Namespace,
):
    if args.scheduler == "cosine":
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=0.0,
        )
    else:
        step_size = max(1, args.epochs // 2)
        main_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=0.1,
        )

    if args.warmup_epochs <= 0:
        return main_scheduler

    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=args.warmup_epochs,
    )

    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_epochs],
    )

def save_checkpoint(model: nn.Module, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def save_training_history(history: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

def main() -> None:
    args = parse_args()
    args = apply_repvgg_preset(args)
    args = resolve_defaults(args)

    set_seed(args.seed)

    device = get_device()
    amp_enabled = bool(args.amp and device.type == "cuda")

    model_tag = args.model
    if args.model == "repvgg":
        if args.repvgg_preset is not None:
            model_tag += f"_{args.repvgg_preset}"
        else:
            model_tag += f"_{args.repvgg_variant}_a{args.a_multiplier:g}_b{args.b_multiplier:g}"

    run_tag = (
        f"{args.dataset}_"
        f"{model_tag}_"
        f"img{args.imagenet_image_size}_"
        f"{args.augmentation_mode}_"
        f"subset{args.subset_fraction:g}"
    )

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    best_model_path = checkpoint_dir / f"{run_tag}_best.pth"
    last_model_path = checkpoint_dir / f"{run_tag}_last.pth"
    history_path = results_dir / f"{run_tag}_history.json"

    log("Starting training script...")
    log(f"Dataset: {args.dataset}")
    log(f"Data dir: {args.data_dir}")
    log(f"Model: {args.model}")

    if args.model == "repvgg":
        if args.repvgg_preset is not None:
            log(f"RepVGG preset: {args.repvgg_preset}")
        log(f"RepVGG variant: {args.repvgg_variant}")
        if args.dataset == "tiny_imagenet":
            log(f"Width multipliers: a={args.a_multiplier}, b={args.b_multiplier}")

    log(f"Augmentation mode: {args.augmentation_mode}")
    log(f"AutoAugment: {args.autoaugment}")
    log(f"Label smoothing: {args.label_smoothing}")
    log(f"Mixup alpha: {args.mixup_alpha}")

    log(f"Using device: {device}")
    log(f"AMP enabled: {amp_enabled}")
    log(f"Epochs: {args.epochs}")
    log(f"Batch size: {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log(f"Weight decay: {args.weight_decay}")
    log(f"Scheduler: {args.scheduler}")
    log(f"Warmup epochs: {args.warmup_epochs}")
    log(f"Subset fraction: {args.subset_fraction}")

    train_loader, val_loader = get_dataloaders(args)

    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches: {len(val_loader)}")

    model = build_model(args).to(device)

    log("Model summary:")
    print_model_summary(model, input_size=get_input_size(args))

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(optimizer, args)
    scaler = GradScaler(device=device.type, enabled=amp_enabled)

    history = {
        "dataset": args.dataset,
        "data_dir": args.data_dir,
        "model_name": args.model,
        "repvgg_variant": args.repvgg_variant if args.model == "repvgg" else None,
        "repvgg_preset": args.repvgg_preset if args.model == "repvgg" else None,
        "augmentation_mode": args.augmentation_mode,
        "autoaugment": args.autoaugment,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "subset_fraction": args.subset_fraction,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5_acc": [],
        "learning_rate": [],
        "epoch_time_sec": [],
    }

    best_val_acc = 0.0

    log("Training starts now.\n")

    for epoch in range(1, args.epochs + 1):
        log(f"Starting epoch {epoch}/{args.epochs}...")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp_enabled=amp_enabled,
            mixup_alpha=args.mixup_alpha,
        )

        log(f"Finished training epoch {epoch}, starting evaluation...")

        val_loss, val_acc, val_top5_acc = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=f"Epoch {epoch}/{args.epochs} [Val]",
        )

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_top5_acc"].append(val_top5_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        log(
            f"\nEpoch [{epoch}/{args.epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc (Top-1): {val_acc:.2f}% | "
            f"Val Acc (Top-5): {val_top5_acc:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_model_path)
            log(f"Best model saved with validation accuracy: {best_val_acc:.2f}%")

        save_checkpoint(model, last_model_path)
        save_training_history(history, history_path)

        scheduler.step()
        log("")

    log(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    log(f"Best checkpoint: {best_model_path}")
    log(f"Last checkpoint: {last_model_path}")
    log(f"History file: {history_path}")


if __name__ == "__main__":
    main()