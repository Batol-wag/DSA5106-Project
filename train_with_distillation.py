"""
train_with_distillation.py

Purpose:
--------
Train a student RepVGG model using branch-level knowledge distillation.

Objective:
----------
Implement the extension proposed in the project:
- Train a student RepVGG model
- Use a teacher model (standard RepVGG or larger architecture)
- Apply branch-level distillation loss
- Evaluate accuracy preservation and potential improvements

This script is used in Phase 2.2 of the project: Branch-Level Distillation Extension.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import chain
from pathlib import Path

REPVGG_PROJECT_ROOT = Path(__file__).resolve().parent / "RepVgg_project"
if str(REPVGG_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPVGG_PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from repvgg_with_branches import create_repvgg_with_branches
from models.repvgg_net import create_repvgg_small, create_repvgg_from_preset
from models.baselines import create_resnet18, create_resnet34, create_plain_cnn
from distillation_loss import create_distillation_loss
from training_utils import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    CUTOUT_LENGTH,
    LABEL_SMOOTHING,
    MIXUP_ALPHA,
    get_cifar10_transforms,
    mixup_batch,
    mixup_correct_predictions,
    resolve_device,
)


DISTILLATION_CONFIGS = {
    "default": {
        "temperature": 4.0,
        "alpha_ce": 1.0,
        "alpha_output_kl": 2.0,
        "alpha_branch_kl": 1.0,
    },
    "strong_branch": {
        "temperature": 5.0,
        "alpha_ce": 1.0,
        "alpha_output_kl": 1.5,
        "alpha_branch_kl": 2.0,
    },
    "light_distill": {
        "temperature": 3.0,
        "alpha_ce": 2.0,
        "alpha_output_kl": 1.0,
        "alpha_branch_kl": 0.5,
    },
}

LATEST_BASELINE_CHECKPOINTS = {
    "repvgg_a0": Path("results_new/cifar10_repvgg_A0_recommended_subset1_best.pth"),
    "repvgg_a1": Path("results_new/cifar10_repvgg_A1_recommended_subset1_best.pth"),
    "repvgg_a2": Path("results_new/cifar10_repvgg_A2_recommended_subset1_best.pth"),
    "repvgg_b0": Path("results_new/cifar10_repvgg_B0_recommended_subset1_best.pth"),
    "repvgg_b1": Path("results_new/cifar10_repvgg_B1_recommended_subset1_best.pth"),
    "resnet18_latest": Path("results_new/cifar10_resnet18_recommended_subset1_best.pth"),
    "resnet34": Path("results_new/cifar10_resnet34_recommended_subset1_best.pth"),
}


def log(message: str) -> None:
    """Print a message immediately to the terminal."""
    print(message, flush=True)


def get_device(device_preference: str = "auto") -> torch.device:
    """Return GPU if available, otherwise CPU."""
    return resolve_device(device_preference)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RepVGG with branch-level knowledge distillation"
    )

    parser.add_argument(
        "--student_model",
        type=str,
        default="repvgg",
        choices=["repvgg"],
        help="Student model architecture",
    )
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="repvgg",
        choices=[
            "repvgg",
            "repvgg_a0",
            "repvgg_a1",
            "repvgg_a2",
            "repvgg_b0",
            "repvgg_b1",
            "resnet18",
            "resnet18_latest",
            "resnet34",
            "plaincnn",
        ],
        help="Teacher model architecture",
    )
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        default=None,
        help="Path to pre-trained teacher checkpoint. Defaults to the baseline best checkpoint for the selected teacher model.",
    )
    parser.add_argument(
        "--distillation_config",
        type=str,
        default="default",
        choices=[*DISTILLATION_CONFIGS.keys(), "custom"],
        help="Named distillation hyperparameter preset. Use 'custom' to keep explicit temperature/alpha values.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=4.0,
        help="Temperature for knowledge distillation",
    )
    parser.add_argument(
        "--alpha_ce",
        type=float,
        default=1.0,
        help="Weight for cross-entropy loss",
    )
    parser.add_argument(
        "--alpha_output_kl",
        type=float,
        default=2.0,
        help="Weight for output-level KL loss",
    )
    parser.add_argument(
        "--alpha_branch_kl",
        type=float,
        default=1.0,
        help="Weight for branch-level KL loss",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to run on. 'auto' prefers CUDA, then MPS, then CPU.",
    )

    return parser.parse_args()


def resolve_teacher_checkpoint(
    teacher_model_name: str,
    teacher_checkpoint: str | None,
) -> Path:
    """Resolve the teacher checkpoint path from args or baseline outputs."""
    if teacher_checkpoint is not None:
        return Path(teacher_checkpoint)

    if teacher_model_name in LATEST_BASELINE_CHECKPOINTS:
        return Path(LATEST_BASELINE_CHECKPOINTS[teacher_model_name])

    return REPVGG_PROJECT_ROOT / "checkpoints" / f"{teacher_model_name}_best.pth"


def resolve_distillation_hyperparameters(
    args: argparse.Namespace,
) -> dict[str, float]:
    """Use a recommended preset unless custom values were explicitly requested."""
    if args.distillation_config == "custom":
        return {
            "temperature": args.temperature,
            "alpha_ce": args.alpha_ce,
            "alpha_output_kl": args.alpha_output_kl,
            "alpha_branch_kl": args.alpha_branch_kl,
        }

    return DISTILLATION_CONFIGS[args.distillation_config].copy()


def get_dataloaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test dataloaders."""
    log("Preparing CIFAR-10 transforms...")

    train_transform, test_transform = get_cifar10_transforms()

    log("Loading CIFAR-10 training set...")
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform,
    )

    log("Loading CIFAR-10 test set...")
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=test_transform,
    )

    log("Creating dataloaders...")
    pin_memory = device is not None and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    log("Dataloaders are ready.")
    return train_loader, test_loader


def build_student_model(num_classes: int = 10) -> nn.Module:
    """Build student model with branch output support."""
    return create_repvgg_with_branches(num_classes=num_classes, deploy=False)


def build_teacher_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Build teacher model."""
    if model_name == "repvgg":
        return create_repvgg_small(num_classes=num_classes, deploy=False)
    elif model_name == "repvgg_a0":
        return create_repvgg_from_preset("A0", num_classes=num_classes, deploy=False)
    elif model_name == "repvgg_a1":
        return create_repvgg_from_preset("A1", num_classes=num_classes, deploy=False)
    elif model_name == "repvgg_a2":
        return create_repvgg_from_preset("A2", num_classes=num_classes, deploy=False)
    elif model_name == "repvgg_b0":
        return create_repvgg_from_preset("B0", num_classes=num_classes, deploy=False)
    elif model_name == "repvgg_b1":
        return create_repvgg_from_preset("B1", num_classes=num_classes, deploy=False)
    elif model_name == "resnet18" or model_name == "resnet18_latest":
        return create_resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        return create_resnet34(num_classes=num_classes)
    elif model_name == "plaincnn":
        return create_plain_cnn(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported teacher model: {model_name}")


def load_teacher_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load pre-trained teacher weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain a valid state dict."
        )

    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("total_ops") and not key.endswith("total_params")
    }

    model.load_state_dict(filtered_state_dict)
    log(f"Teacher model loaded from: {checkpoint_path}")


def train_one_epoch_with_distillation(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    distillation_loss_fn,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float, dict[str, float]]:
    """Train the student model for one epoch with distillation."""
    student.train()
    teacher.eval()

    running_loss = 0.0
    running_component_losses = {
        "ce_loss": 0.0,
        "output_kl": 0.0,
        "branch_kl": 0.0,
    }
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=True,
        file=sys.stdout,
    )
    non_blocking = device.type == "cuda"

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        images, target_probs, primary_labels, secondary_labels, lam = mixup_batch(
            images,
            labels,
            alpha=MIXUP_ALPHA,
            num_classes=10,
            label_smoothing=LABEL_SMOOTHING,
        )

        # Student forward (with branch tracking)
        student_logits, student_branches = student(
            images, return_branches=True
        )

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_logits = teacher(images)

        # Compute distillation loss
        loss, loss_dict = distillation_loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            target_probs=target_probs,
            student_branches=student_branches,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        for key in running_component_losses:
            running_component_losses[key] += loss_dict.get(key, 0.0) * images.size(0)

        total += labels.size(0)
        correct += mixup_correct_predictions(
            student_logits,
            primary_labels,
            secondary_labels,
            lam,
        )

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    avg_component_losses = {
        key: value / total for key, value in running_component_losses.items()
    }

    return avg_loss, accuracy, avg_component_losses


@torch.no_grad()
def evaluate_student(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    """Evaluate the student model on the test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Test ]",
        leave=True,
        file=sys.stdout,
    )
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


def save_checkpoint(model: nn.Module, save_path: Path) -> None:
    """Save model weights."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def save_training_history(history: dict, save_path: Path) -> None:
    """Save training history as a JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


@torch.no_grad()
def initialize_branch_projections(
    student: nn.Module,
    distillation_loss_fn: nn.Module,
    device: torch.device,
) -> None:
    """Initialize reusable branch projection layers from a sample forward pass."""
    was_training = student.training
    student.eval()

    sample_images = torch.zeros(1, 3, 32, 32, device=device)
    _, sample_branches = student(sample_images, return_branches=True)
    distillation_loss_fn.initialize_branch_projections(
        sample_branches,
        num_classes=10,
    )

    if was_training:
        student.train()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # -------------------------
    # Fixed training setup
    # -------------------------
    student_model_name = args.student_model
    teacher_model_name = args.teacher_model
    teacher_checkpoint = resolve_teacher_checkpoint(
        teacher_model_name,
        args.teacher_checkpoint,
    )
    distillation_hparams = resolve_distillation_hyperparameters(args)
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    learning_rate = args.lr
    momentum = 0.9
    weight_decay = 5e-4

    # -------------------------
    # Paths (include config in filename to avoid overwriting)
    # -------------------------
    config_suffix = (
        f"{teacher_model_name}_{args.distillation_config}_"
        f"t{distillation_hparams['temperature']}_"
        f"ce{distillation_hparams['alpha_ce']}_"
        f"out{distillation_hparams['alpha_output_kl']}_"
        f"br{distillation_hparams['alpha_branch_kl']}"
    )
    config_suffix = config_suffix.replace(".", "_")  # Replace dots with underscores

    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    best_model_path = checkpoint_dir / f"repvgg_distilled_{config_suffix}_best.pth"
    last_model_path = checkpoint_dir / f"repvgg_distilled_{config_suffix}_last.pth"
    history_path = results_dir / f"repvgg_distilled_{config_suffix}_history.json"

    # -------------------------
    # Setup
    # -------------------------
    log("Starting distillation training script...")
    log(f"Student model: {student_model_name}")
    log(f"Teacher model: {teacher_model_name}")
    log(f"Teacher checkpoint: {teacher_checkpoint}")
    log(f"Distillation config: {args.distillation_config}")

    device = get_device(args.device)
    log(f"Requested device: {args.device}")
    log(f"Using device: {device}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    # Build models
    log("Building student model...")
    student = build_student_model(num_classes=10).to(device)

    log("Building teacher model...")
    teacher = build_teacher_model(
        teacher_model_name, num_classes=10
    ).to(device)

    if teacher_model_name.replace("_latest", "").lower() not in teacher_checkpoint.name.lower():
        log(
            "Warning: teacher model and checkpoint name do not obviously match. "
            "Double-check that the selected checkpoint belongs to the requested teacher architecture."
        )

    if not teacher_checkpoint.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_checkpoint}"
        )

    log("Loading teacher checkpoint...")
    load_teacher_checkpoint(teacher, teacher_checkpoint, device)

    # Setup distillation loss
    distillation_loss_fn = create_distillation_loss(
        temperature=distillation_hparams["temperature"],
        alpha_ce=distillation_hparams["alpha_ce"],
        alpha_output_kl=distillation_hparams["alpha_output_kl"],
        alpha_branch_kl=distillation_hparams["alpha_branch_kl"],
        label_smoothing=LABEL_SMOOTHING,
    ).to(device)
    initialize_branch_projections(student, distillation_loss_fn, device)

    trainable_params = list(
        chain(student.parameters(), distillation_loss_fn.parameters())
    )

    optimizer = optim.SGD(
        trainable_params,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
    )

    # Initialize history
    history = {
        "student_model": student_model_name,
        "teacher_model": teacher_model_name,
        "teacher_checkpoint": str(teacher_checkpoint),
        "distillation_config": args.distillation_config,
        "temperature": distillation_hparams["temperature"],
        "alpha_ce": distillation_hparams["alpha_ce"],
        "alpha_output_kl": distillation_hparams["alpha_output_kl"],
        "alpha_branch_kl": distillation_hparams["alpha_branch_kl"],
        "scheduler": "cosine_annealing",
        "mixup_alpha": MIXUP_ALPHA,
        "label_smoothing": LABEL_SMOOTHING,
        "cutout_length": CUTOUT_LENGTH,
        "train_loss": [],
        "train_ce_loss": [],
        "train_output_kl": [],
        "train_branch_kl": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": [],
        "epoch_time_sec": [],
    }

    best_test_acc = 0.0

    log("Distillation training starts now.\n")

    for epoch in range(1, num_epochs + 1):
        log(f"Starting epoch {epoch}/{num_epochs}...")
        start_time = time.time()

        train_loss, train_acc, train_loss_components = train_one_epoch_with_distillation(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            distillation_loss_fn=distillation_loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        log(f"Finished training epoch {epoch}, starting evaluation...")

        test_loss, test_acc = evaluate_student(
            model=student,
            dataloader=test_loader,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_ce_loss"].append(train_loss_components["ce_loss"])
        history["train_output_kl"].append(train_loss_components["output_kl"])
        history["train_branch_kl"].append(train_loss_components["branch_kl"])
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        log(
            f"\nEpoch [{epoch}/{num_epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"CE: {train_loss_components['ce_loss']:.4f} | "
            f"Output KL: {train_loss_components['output_kl']:.4f} | "
            f"Branch KL: {train_loss_components['branch_kl']:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(student, best_model_path)
            log(f"Best model saved with test accuracy: {best_test_acc:.2f}%")

        save_checkpoint(student, last_model_path)
        save_training_history(history, history_path)

        scheduler.step()
        log("")

    log(f"Distillation training completed. Best test accuracy: {best_test_acc:.2f}%")
    log(f"Best checkpoint: {best_model_path}")
    log(f"Last checkpoint: {last_model_path}")
    log(f"History file: {history_path}")


if __name__ == "__main__":
    main()
