"""
plot_results.py

Generate training result figures and summary statistics for all models.

This script reads saved training history JSON files and creates separate
plots for each trained model.

Main tasks performed:

1. Load history files saved after training.

2. Search for expected runs:
   - RepVGG-A0
   - RepVGG-A1
   - RepVGG-A2
   - RepVGG-B0
   - RepVGG-B1
   - ResNet-18
   - ResNet-34

3. Create three plots for each model:
   - Training vs validation loss
   - Training vs validation Top-1 accuracy
   - Validation Top-5 accuracy

4. Save plots into separate folders for each model.

5. Generate a summary table containing final and best metrics.

Used for report figures, model comparison, and experiment analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(path: Path) -> dict:
    """
    Load one training history JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_history_path(
    results_dir: Path,
    model_name: str,
    augmentation_mode: str,
    subset_fraction: float,
) -> Path:
    """
    Build expected history file path for one model run.
    """
    subset_tag = f"subset{subset_fraction:g}"
    filename = f"cifar10_{model_name}_{augmentation_mode}_{subset_tag}_history.json"

    return results_dir / filename


def get_expected_runs() -> list[tuple[str, str]]:
    """
    Return display names and internal file names for expected models.
    """
    return [
        ("RepVGG-A0", "repvgg_A0"),
        ("RepVGG-A1", "repvgg_A1"),
        ("RepVGG-A2", "repvgg_A2"),
        ("RepVGG-B0", "repvgg_B0"),
        ("RepVGG-B1", "repvgg_B1"),
        ("ResNet-18", "resnet18"),
        ("ResNet-34", "resnet34"),
    ]


def load_histories(
    results_dir: Path,
    augmentation_mode: str,
    subset_fraction: float,
) -> dict[str, dict]:
    """
    Load all available history files.
    """
    histories: dict[str, dict] = {}

    for display_name, model_name in get_expected_runs():
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
        raise FileNotFoundError("No history files found.")

    return histories


def safe_name(name: str) -> str:
    """
    Convert model name into a filesystem-safe format.
    """
    return name.lower().replace(" ", "_").replace("-", "_")


def plot_loss(
    model_name: str,
    history: dict,
    save_path: Path,
) -> None:
    """
    Plot training and validation loss.
    """
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])

    if not train_loss and not val_loss:
        print(f"[SKIP] {model_name} missing loss history")
        return

    plt.figure(figsize=(8, 5))

    if train_loss:
        plt.plot(
            range(1, len(train_loss) + 1),
            train_loss,
            label="Train Loss",
        )

    if val_loss:
        plt.plot(
            range(1, len(val_loss) + 1),
            val_loss,
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_top1(
    model_name: str,
    history: dict,
    save_path: Path,
) -> None:
    """
    Plot training and validation Top-1 accuracy.
    """
    train_top1 = history.get("train_top1", [])
    val_top1 = history.get("val_top1", [])

    if not train_top1 and not val_top1:
        print(f"[SKIP] {model_name} missing Top-1 history")
        return

    plt.figure(figsize=(8, 5))

    if train_top1:
        plt.plot(
            range(1, len(train_top1) + 1),
            train_top1,
            label="Train Top-1",
        )

    if val_top1:
        plt.plot(
            range(1, len(val_top1) + 1),
            val_top1,
            label="Validation Top-1",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title(f"{model_name} - Top-1 Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_top5(
    model_name: str,
    history: dict,
    save_path: Path,
) -> None:
    """
    Plot validation Top-5 accuracy.
    """
    val_top5 = history.get("val_top5", [])

    if not val_top5:
        print(f"[SKIP] {model_name} missing Top-5 history")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(val_top5) + 1),
        val_top5,
        label="Validation Top-5",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Top-5 Accuracy (%)")
    plt.title(f"{model_name} - Validation Top-5 Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_summary_table(
    histories: dict[str, dict],
    save_path: Path,
) -> None:
    """
    Save summary metrics for all models into JSON format.
    """
    rows = []

    for model_name, history in histories.items():
        train_loss = history.get("train_loss", [])
        val_loss = history.get("val_loss", [])
        train_top1 = history.get("train_top1", [])
        val_top1 = history.get("val_top1", [])
        val_top5 = history.get("val_top5", [])

        rows.append(
            {
                "model": model_name,
                "epochs": len(train_loss),
                "final_train_loss": train_loss[-1] if train_loss else None,
                "final_val_loss": val_loss[-1] if val_loss else None,
                "final_train_top1": train_top1[-1] if train_top1 else None,
                "final_val_top1": val_top1[-1] if val_top1 else None,
                "best_val_top1": max(val_top1) if val_top1 else None,
                "final_val_top5": val_top5[-1] if val_top5 else None,
            }
        )

    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(rows, file, indent=4)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create 3 plots for each model"
    )

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--figures_dir", type=str, default="figures")

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="recommended",
        choices=["simple", "recommended"],
    )

    parser.add_argument("--subset_fraction", type=float, default=1.0)

    return parser.parse_args()


def main() -> None:
    """
    Load histories, generate plots, and save summary table.
    """
    args = parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    histories = load_histories(
        results_dir=results_dir,
        augmentation_mode=args.augmentation_mode,
        subset_fraction=args.subset_fraction,
    )

    for model_name, history in histories.items():
        model_dir = figures_dir / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_tag = safe_name(model_name)

        plot_loss(
            model_name=model_name,
            history=history,
            save_path=model_dir / f"{model_tag}_loss.png",
        )

        plot_top1(
            model_name=model_name,
            history=history,
            save_path=model_dir / f"{model_tag}_top1.png",
        )

        plot_top5(
            model_name=model_name,
            history=history,
            save_path=model_dir / f"{model_tag}_top5.png",
        )

    save_summary_table(
        histories=histories,
        save_path=figures_dir / "summary_table.json",
    )



if __name__ == "__main__":
    main()