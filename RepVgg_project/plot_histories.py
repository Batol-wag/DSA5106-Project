"""
plot_histories.py

Purpose:
--------
This file loads training history JSON files and generates comparison plots
for the CIFAR-10 experiments.

Objective:
----------
Visualize how RepVGG, PlainCNN, and ResNet-18 behave during training under
the same pipeline.

Generated plots:
----------------
1. Training loss vs epoch
2. Test accuracy vs epoch

Input files:
------------
- results/repvgg_history.json
- results/plaincnn_history.json
- results/resnet18_history.json

Output files:
-------------
- figures/train_loss_comparison.png
- figures/test_accuracy_comparison.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> dict:
    """
    Load a JSON history file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_train_loss(histories: dict[str, dict], save_path: Path) -> None:
    """
    Plot train loss vs epoch for multiple models.
    """
    plt.figure(figsize=(8, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        plt.plot(epochs, history["train_loss"], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss on CIFAR-10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_test_accuracy(histories: dict[str, dict], save_path: Path) -> None:
    """
    Plot test accuracy vs epoch for multiple models.
    """
    plt.figure(figsize=(8, 5))

    for model_name, history in histories.items():
        epochs = range(1, len(history["test_acc"]) + 1)
        plt.plot(epochs, history["test_acc"], label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy on CIFAR-10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    """
    Main plotting function.
    """
    project_root = Path(__file__).resolve().parent

    results_dir = project_root / "results"
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_files = {
        "RepVGG": results_dir / "repvgg_history.json",
        "PlainCNN": results_dir / "plaincnn_history.json",
        "ResNet-18": results_dir / "resnet18_history.json",
    }

    histories = {}
    for model_name, file_path in history_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing history file: {file_path}")
        histories[model_name] = load_history(file_path)

    plot_train_loss(
        histories=histories,
        save_path=figures_dir / "train_loss_comparison.png",
    )

    plot_test_accuracy(
        histories=histories,
        save_path=figures_dir / "test_accuracy_comparison.png",
    )

    print("Plots saved successfully.")
    print(f"Train loss plot: {figures_dir / 'train_loss_comparison.png'}")
    print(f"Test accuracy plot: {figures_dir / 'test_accuracy_comparison.png'}")


if __name__ == "__main__":
    main()