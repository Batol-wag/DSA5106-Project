from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn


def build_history(config) -> dict:
    return {
        "data_dir": config.data_dir,
        "save_path": config.save_path,
        "data_augment": config.data_augment,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "label_smoothing": config.label_smoothing,
        "mixup_alpha": config.mixup_alpha,
        "train_loss": [],
        "train_top1_acc": [],
        "val_loss": [],
        "val_top1_acc": [],
        "val_top5_acc": [],
        "learning_rate": [],
        "epoch_time_sec": [],
    }


def append_epoch_metrics(
    history: dict,
    train_loss: float,
    train_top1_acc: float,
    val_loss: float,
    val_top1_acc: float,
    val_top5_acc: float,
    learning_rate: float,
    epoch_time_sec: float,
) -> None:
    history["train_loss"].append(train_loss)
    history["train_top1_acc"].append(train_top1_acc)
    history["val_loss"].append(val_loss)
    history["val_top1_acc"].append(val_top1_acc)
    history["val_top5_acc"].append(val_top5_acc)
    history["learning_rate"].append(learning_rate)
    history["epoch_time_sec"].append(epoch_time_sec)


def save_history(history: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)


def summarize_convergence(history: dict) -> dict:
    best_top1 = max(history["val_top1_acc"]) if history["val_top1_acc"] else 0.0
    best_epoch = history["val_top1_acc"].index(best_top1) + 1 if history["val_top1_acc"] else 0
    final_train_top1 = history["train_top1_acc"][-1] if history["train_top1_acc"] else 0.0
    final_val_top1 = history["val_top1_acc"][-1] if history["val_top1_acc"] else 0.0
    final_gap = final_train_top1 - final_val_top1

    return {
        "best_val_top1_acc": best_top1,
        "best_epoch": best_epoch,
        "final_train_top1_acc": final_train_top1,
        "final_val_top1_acc": final_val_top1,
        "final_generalization_gap": final_gap,
        "mean_epoch_time_sec": (
            sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
            if history["epoch_time_sec"]
            else 0.0
        ),
    }


def save_convergence_summary(history: dict, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_convergence(history)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def plot_convergence_curves(history: dict, save_dir: str | Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence Behavior: Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_convergence.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_top1_acc"], label="Train Top-1", linewidth=2)
    plt.plot(epochs, history["val_top1_acc"], label="Validation Top-1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "top1_accuracy.png", dpi=300)
    plt.close()


def get_evaluation_artifact_paths(checkpoint_path: str | Path) -> dict[str, Path]:
    checkpoint_path = Path(checkpoint_path)
    base_dir = checkpoint_path.parent
    stem = checkpoint_path.stem
    return {
        "history": base_dir / f"{stem}_history.json",
        "summary": base_dir / f"{stem}_convergence_summary.json",
        "figures": base_dir / f"{stem}_figures",
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, float, float]:
    model.eval()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, top1_predictions = outputs.max(dim=1)
        correct_top1 += top1_predictions.eq(labels).sum().item()

        top_k = min(5, outputs.size(1))
        _, topk_predictions = outputs.topk(top_k, dim=1, largest=True, sorted=True)
        correct_top5 += topk_predictions.eq(labels.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    top5_acc = 100.0 * correct_top5 / total
    return avg_loss, top1_acc, top5_acc
