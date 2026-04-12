"""
plot_histories.py

Plot training histories for CIFAR-10 experiments.

This version is aligned with the cleaned CIFAR-10-only training pipeline
and the preset-based checkpoint/history naming scheme.

Expected history files
----------------------
Results directory should contain files like:

- cifar10_repvgg_A0_strong_subset1_history.json
- cifar10_repvgg_A1_strong_subset1_history.json
- cifar10_repvgg_A2_strong_subset1_history.json
- cifar10_repvgg_B0_strong_subset1_history.json
- cifar10_repvgg_B1_strong_subset1_history.json
- cifar10_resnet18_strong_subset1_history.json
- cifar10_resnet34_strong_subset1_history.json

Generated plots
---------------
- train_loss_comparison.png
- val_loss_comparison.png
- val_top1_comparison.png
- final_top1_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_history_path(
    results_dir: Path,
    model_name: str,
    augmentation_mode: str,
    subset_fraction: float,
) -> Path:
    subset_tag = f"subset{subset_fraction:g}"
    filename = f"cifar10_{model_name}_{augmentation_mode}_{subset_tag}_history.json"
    return results_dir / filename


def get_expected_runs(run_b_variants: bool) -> list[tuple[str, str]]:
    runs = [
        ("RepVGG-A0", "repvgg_A0"),
        ("RepVGG-A1", "repvgg_A1"),
        ("RepVGG-A2", "repvgg_A2"),
    ]

    if run_b_variants:
        runs.extend([
            ("RepVGG-B0", "repvgg_B0"),
            ("RepVGG-B1", "repvgg_B1"),
        ])

    runs.extend([
        ("ResNet-18", "resnet18"),
        ("ResNet-34", "resnet34"),
    ])

    return runs


def load_histories(
    results_dir: Path,
    augmentation_mode: str,
    subset_fraction: float,
    run_b_variants: bool,
) -> dict[str, dict]:
    histories: dict[str, dict] = {}
    expected_runs = get_expected_runs(run_b_variants)

    for display_name, model_name in expected_runs:
        history_path = build_history_path(
            results_dir=results_dir,
            model_name=model_name,
            augmentation_mode=augmentation_mode,
            subset_fraction=subset_fraction,
        )

        if not history_path.exists():
            print(f"[SKIP] Missing history file: {history_path}")
            continue

        histories[display_name] = load_history(history_path)

    if not histories:
        raise FileNotFoundError(
            "No history files were found. Check results_dir, augmentation_mode, and subset_fraction."
        )

    return histories


def plot_train_loss(histories: dict[str, dict], save_path: Path) -> None:
    plt.figure(figsize=(9, 5))

    for model_name, history in histories.items():
        values = history.get("train_loss", [])
        if not values:
            continue

        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss on CIFAR-10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_val_loss(histories: dict[str, dict], save_path: Path) -> None:
    plt.figure(figsize=(9, 5))

    for model_name, history in histories.items():
        values = history.get("val_loss", [])
        if not values:
            continue

        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss on CIFAR-10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_val_top1(histories: dict[str, dict], save_path: Path) -> None:
    plt.figure(figsize=(9, 5))

    for model_name, history in histories.items():
        if "val_acc" in history:
            values = history["val_acc"]
        else:
            values = history.get("test_acc", [])

        if not values:
            continue

        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=model_name)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Top-1 Accuracy (%)")
    plt.title("Validation Top-1 Accuracy on CIFAR-10")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_final_top1(histories: dict[str, dict], save_path: Path) -> None:
    model_names: list[str] = []
    final_accs: list[float] = []

    for model_name, history in histories.items():
        if "val_acc" in history and history["val_acc"]:
            final_value = history["val_acc"][-1]
        elif "test_acc" in history and history["test_acc"]:
            final_value = history["test_acc"][-1]
        else:
            continue

        model_names.append(model_name)
        final_accs.append(final_value)

    plt.figure(figsize=(10, 5))
    plt.bar(model_names, final_accs)
    plt.xlabel("Model")
    plt.ylabel("Final Top-1 Accuracy (%)")
    plt.title("Final Validation Top-1 Accuracy on CIFAR-10")
    plt.xticks(rotation=20, ha="right")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_summary_table(histories: dict[str, dict], save_path: Path) -> None:
    rows = []

    for model_name, history in histories.items():
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        val_acc = history.get("val_acc", history.get("test_acc", []))
        val_top5 = history.get("val_top5_acc", [])

        rows.append(
            {
                "model": model_name,
                "epochs": len(train_loss),
                "final_train_loss": train_loss[-1] if train_loss else None,
                "final_val_loss": val_loss[-1] if val_loss else None,
                "final_val_top1": val_acc[-1] if val_acc else None,
                "best_val_top1": max(val_acc) if val_acc else None,
                "final_val_top5": val_top5[-1] if val_top5 else None,
            }
        )

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CIFAR-10 training histories for RepVGG presets and baselines"
    )
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")
    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="strong",
        choices=["simple", "strong"],
    )
    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--run_b_variants", type=str, default="true")
    return parser.parse_args()


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def main() -> None:
    args = parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_b_variants = str2bool(args.run_b_variants)

    histories = load_histories(
        results_dir=results_dir,
        augmentation_mode=args.augmentation_mode,
        subset_fraction=args.subset_fraction,
        run_b_variants=run_b_variants,
    )

    plot_train_loss(histories, figures_dir / "train_loss_comparison.png")
    plot_val_loss(histories, figures_dir / "val_loss_comparison.png")
    plot_val_top1(histories, figures_dir / "val_top1_comparison.png")
    plot_final_top1(histories, figures_dir / "final_top1_comparison.png")
    save_summary_table(histories, figures_dir / "summary_table.json")

    print("Saved plots and summary to:", figures_dir)


if __name__ == "__main__":
    main()