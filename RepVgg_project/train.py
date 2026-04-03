"""
train.py

Purpose:
--------
This file trains image classification models for the RepVGG reproduction project.

Objective:
----------
Provide one unified training script that can train multiple models under
the same CIFAR-10 training pipeline for fair comparison.

Supported models:
-----------------
- repvgg
- plaincnn
- resnet18

Why this file matters:
----------------------
To evaluate RepVGG properly, we should not train it alone. We need to compare
it against baseline architectures using the same dataset, augmentation, optimizer,
scheduler, and number of epochs.

This file is used in Phase 2.1 of the project:
training baseline models on CIFAR-10 for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.repvgg_net import create_repvgg_small
from models.baselines import create_plain_cnn, create_resnet18
from evaluate import evaluate_model


def log(message: str) -> None:
    """
    Print a message immediately to the terminal.
    """
    print(message, flush=True)


def get_device() -> torch.device:
    """
    Return GPU if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train baseline models on CIFAR-10")

    parser.add_argument(
        "--model",
        type=str,
        default="repvgg",
        choices=["repvgg", "plaincnn", "resnet18"],
        help="Model architecture to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers (0 recommended on Windows)",
    )

    return parser.parse_args()


def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Create CIFAR-10 train and test dataloaders.
    """
    log("Preparing CIFAR-10 transforms...")

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    log("Dataloaders are ready.")
    return train_loader, test_loader


def build_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Build the selected model.
    """
    if model_name == "repvgg":
        return create_repvgg_small(num_classes=num_classes, deploy=False)

    if model_name == "plaincnn":
        return create_plain_cnn(num_classes=num_classes)

    if model_name == "resnet18":
        return create_resnet18(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    """
    Train the model for one epoch.
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=True,
        file=sys.stdout,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        current_loss = running_loss / total
        current_acc = 100.0 * correct / total

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.2f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.5f}",
        )

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_epochs: int,
) -> tuple[float, float]:
    """
    Evaluate the model on the test set.
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch}/{num_epochs} [Test ]",
        leave=True,
        file=sys.stdout,
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

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
    """
    Save model weights.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)


def save_training_history(history: dict, save_path: Path) -> None:
    """
    Save training history as a JSON file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def main() -> None:
    """
    Main training function.
    """
    args = parse_args()

    # -------------------------
    # Fixed training setup
    # -------------------------
    model_name = args.model
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epochs = args.epochs
    learning_rate = args.lr
    momentum = 0.9
    weight_decay = 5e-4
    step_size = max(1, num_epochs // 2)
    gamma = 0.1

    # -------------------------
    # Paths
    # -------------------------
    checkpoint_dir = Path("checkpoints")
    results_dir = Path("results")

    best_model_path = checkpoint_dir / f"{model_name}_best.pth"
    last_model_path = checkpoint_dir / f"{model_name}_last.pth"
    history_path = results_dir / f"{model_name}_history.json"

    # -------------------------
    # Setup
    # -------------------------
    log("Starting training script...")
    log(f"Selected model: {model_name}")

    device = get_device()
    log(f"Using device: {device}")
    log(f"CUDA available: {torch.cuda.is_available()}")

    train_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    log("Building model...")
    model = build_model(model_name=model_name, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
    )

    history = {
        "model_name": model_name,
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": [],
        "epoch_time_sec": [],
    }

    best_test_acc = 0.0

    log("Training starts now.\n")

    for epoch in range(1, num_epochs + 1):
        log(f"Starting epoch {epoch}/{num_epochs}...")
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        log(f"Finished training epoch {epoch}, starting evaluation...")

        test_loss, test_acc = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=device,
            desc=f"Epoch {epoch}/{num_epochs} [Test ]",
        )

        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(current_lr)
        history["epoch_time_sec"].append(epoch_time)

        log(
            f"\nEpoch [{epoch}/{num_epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.2f}% | "
            f"LR: {current_lr:.5f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, best_model_path)
            log(f"Best model saved with test accuracy: {best_test_acc:.2f}%")

        save_checkpoint(model, last_model_path)
        save_training_history(history, history_path)

        scheduler.step()
        log("")

    log(f"Training completed. Best test accuracy: {best_test_acc:.2f}%")
    log(f"Best checkpoint: {best_model_path}")
    log(f"Last checkpoint: {last_model_path}")
    log(f"History file: {history_path}")


if __name__ == "__main__":
    main()