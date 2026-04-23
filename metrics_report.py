"""
metrics_report.py

Purpose:
--------
Collect and summarize experimental metrics for the RepVGG reproduction project.

Objective:
----------
- Load saved training histories for all models.
- Compute model size and parameter counts.
- Measure RepVGG deploy conversion accuracy gap when a checkpoint exists.
- Export a presentation-ready JSON and Markdown summary.

Usage:
------
python metrics_report.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from evaluate import evaluate_model, validate_repvgg_deploy
from models.baselines import create_plain_cnn, create_resnet18
from models.repvgg_net import create_repvgg_small


MODEL_FACTORIES = {
    "repvgg": create_repvgg_small,
    "plaincnn": create_plain_cnn,
    "resnet18": create_resnet18,
}

HUMAN_MODEL_NAMES = {
    "repvgg": "RepVGG",
    "plaincnn": "PlainCNN",
    "resnet18": "ResNet-18",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate metrics summary for RepVGG experiments."
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results"),
        help="Directory containing history JSON files.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory containing saved model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("reports"),
        help="Directory where summary outputs are written.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size used for any optional evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of data loader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device to use for evaluation.",
    )
    return parser.parse_args()


def get_test_loader(batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_bytes(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024 or unit == "GB":
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} GB"


def load_history(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_history(history: dict[str, Any]) -> dict[str, Any]:
    test_acc = history.get("test_acc", [])
    train_acc = history.get("train_acc", [])
    epoch_time = history.get("epoch_time_sec", [])

    summary: dict[str, Any] = {
        "final_test_acc": test_acc[-1] if test_acc else None,
        "best_test_acc": max(test_acc) if test_acc else None,
        "best_epoch": int(test_acc.index(max(test_acc)) + 1) if test_acc else None,
        "final_train_acc": train_acc[-1] if train_acc else None,
        "avg_epoch_time_sec": float(sum(epoch_time) / len(epoch_time)) if epoch_time else None,
        "total_training_time_sec": float(sum(epoch_time)) if epoch_time else None,
        "epochs": len(test_acc),
    }

    return summary


def build_model_metrics() -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}

    for model_name, factory in MODEL_FACTORIES.items():
        model = factory(num_classes=10)
        params = count_parameters(model)
        bytes_size = params["total"] * 4

        metrics[model_name] = {
            "display_name": HUMAN_MODEL_NAMES[model_name],
            "total_parameters": params["total"],
            "trainable_parameters": params["trainable"],
            "parameter_size_bytes": bytes_size,
            "parameter_size": format_bytes(bytes_size),
        }

    return metrics


def collect_metrics(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "models": {},
        "repvgg_deploy": None,
    }

    # Model architecture metrics
    summary["models"] = build_model_metrics()

    # Load histories
    for model_name in MODEL_FACTORIES:
        history_path = args.results_dir / f"{model_name}_history.json"
        history = load_history(history_path)
        if history is None:
            summary["models"][model_name]["history_available"] = False
            continue

        summary["models"][model_name]["history_available"] = True
        summary["models"][model_name].update(
            summarize_history(history)
        )

    # Evaluate RepVGG deploy conversion if checkpoint exists
    repvgg_checkpoint = args.checkpoints_dir / "repvgg_best.pth"
    if repvgg_checkpoint.exists():
        test_loader = get_test_loader(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        device = torch.device(args.device)

        before_acc, after_acc = validate_repvgg_deploy(
            test_loader=test_loader,
            device=device,
            checkpoint_path=repvgg_checkpoint,
        )

        summary["repvgg_deploy"] = {
            "checkpoint_path": str(repvgg_checkpoint.resolve()),
            "acc_before_deploy": before_acc,
            "acc_after_deploy": after_acc,
            "accuracy_gap": abs(before_acc - after_acc),
        }
    else:
        summary["repvgg_deploy"] = {
            "checkpoint_path": str(repvgg_checkpoint.resolve()),
            "message": "Checkpoint not found; deploy validation skipped.",
        }

    return summary


def save_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def save_markdown(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# RepVGG Experimental Metrics Summary",
        "",
        "This report summarizes the evaluation metrics collected from the CIFAR-10 experiments.",
        "",
    ]

    lines.append("## Model Architecture Metrics")
    lines.append("")
    lines.append(
        "| Model | Parameters | Trainable Parameters | Parameter Size | History Available | Best Test Accuracy | Final Test Accuracy | Avg Epoch Time (s) |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )

    for model_name, model_data in summary["models"].items():
        lines.append(
            f"| {model_data['display_name']} | {model_data['total_parameters']:,} | "
            f"{model_data['trainable_parameters']:,} | {model_data['parameter_size']} | "
            f"{model_data.get('history_available', False)} | "
            f"{model_data.get('best_test_acc', 'N/A')} | "
            f"{model_data.get('final_test_acc', 'N/A')} | "
            f"{model_data.get('avg_epoch_time_sec', 'N/A')} |"
        )

    lines.append("")
    lines.append("## RepVGG Deploy Conversion Metrics")
    lines.append("")
    if summary["repvgg_deploy"] and summary["repvgg_deploy"].get("acc_before_deploy") is not None:
        lines.extend(
            [
                f"- RepVGG checkpoint: `{summary['repvgg_deploy']['checkpoint_path']}`",
                f"- Accuracy before deploy conversion: {summary['repvgg_deploy']['acc_before_deploy']:.2f}%",
                f"- Accuracy after deploy conversion: {summary['repvgg_deploy']['acc_after_deploy']:.2f}%",
                f"- Accuracy gap: {summary['repvgg_deploy']['accuracy_gap']:.6f}%",
            ]
        )
    else:
        lines.append(
            "- RepVGG deploy validation was skipped because the checkpoint was not available."
        )

    lines.append("")
    lines.append("## Notes for Presentation")
    lines.append("")
    lines.append(
        "- Use the best test accuracy and training curve summaries to compare RepVGG, PlainCNN, and ResNet-18."
    )
    lines.append(
        "- Highlight the RepVGG deploy conversion gap as evidence that re-parameterization preserves accuracy."
    )
    lines.append(
        "- Report parameter counts and model size to show efficiency trade-offs."
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_metrics(args)

    json_path = output_dir / "metrics_summary.json"
    md_path = output_dir / "metrics_summary.md"

    save_json(summary, json_path)
    save_markdown(summary, md_path)

    print(f"Metrics summary written to: {json_path}")
    print(f"Markdown report written to: {md_path}")


if __name__ == "__main__":
    main()
