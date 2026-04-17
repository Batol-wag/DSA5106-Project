"""
run_all.py

Run the full CIFAR-10 experiment pipeline automatically.

This script executes all major project steps in sequence so the full
reproduction workflow can be completed with one command.

Main tasks performed:

1. Train RepVGG variants:
   - A0
   - A1
   - A2
   - B0 
   - B1

2. Train baseline models:
   - ResNet-18
   - ResNet-34

3. Run model comparison script after training.

4. Generate plots and visual summaries.

5. Print progress logs and stop if any stage fails.

Used for final experiments, large training runs,
and clean end-to-end project execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(
    command: list[str],
    workdir: Path,
) -> None:
    """
    Execute one command inside the project directory.

    Raise an error if the command fails.
    """
    process = subprocess.Popen(
        command,
        cwd=str(workdir),
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )

    process.wait()

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}"
        )


def str2bool(value: str) -> bool:
    """
    Convert common string inputs into boolean values.
    """
    return value.lower() in {
        "1",
        "true",
        "yes",
        "y",
    }


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the full pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run all CIFAR-10 training, comparison, and plotting jobs sequentially"
    )

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="./results")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_fraction", type=float, default=1.0)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine"],
    )

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--amp", type=str, default="true")

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="recommended",
        choices=["simple", "recommended"],
    )

    parser.add_argument(
        "--run_b_variants",
        type=str2bool,
        default=True,
        help="Whether to also run RepVGG B0 and B1.",
    )

    parser.add_argument(
        "--run_compare",
        type=str2bool,
        default=True,
        help="Whether to run compare_models.py after training.",
    )

    parser.add_argument(
        "--run_plot",
        type=str2bool,
        default=True,
        help="Whether to run plot_histories.py after training/comparison.",
    )

    return parser.parse_args()


def build_train_command(
    python_executable: str,
    args: argparse.Namespace,
    model: str,
    repvgg_preset: str | None = None,
) -> list[str]:
    """
    Build command for train.py.
    """
    command = [
        python_executable,
        "train.py",
        "--data_dir", args.data_dir,
        "--save_dir", args.results_dir,
        "--model", model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--subset_fraction", str(args.subset_fraction),
        "--scheduler", args.scheduler,
        "--warmup_epochs", str(args.warmup_epochs),
        "--amp", args.amp,
        "--augmentation_mode", args.augmentation_mode,
    ]

    if args.lr is not None:
        command += [
            "--lr",
            str(args.lr),
        ]

    if args.weight_decay is not None:
        command += [
            "--weight_decay",
            str(args.weight_decay),
        ]

    if repvgg_preset is not None:
        command += [
            "--repvgg_preset",
            repvgg_preset,
        ]

    return command


def build_compare_command(
    python_executable: str,
    args: argparse.Namespace,
) -> list[str]:
    """
    Build command for compare_models.py.
    """
    return [
        python_executable,
        "compare_models.py",
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--subset_fraction", str(args.subset_fraction),
        "--augmentation_mode", args.augmentation_mode,
    ]


def build_plot_command(
    python_executable: str,
    args: argparse.Namespace,
) -> list[str]:
    """
    Build command for plot_histories.py.
    """
    return [
        python_executable,
        "plot_histories.py",
        "--results_dir", args.results_dir,
        "--augmentation_mode", args.augmentation_mode,
        "--subset_fraction", str(args.subset_fraction),
        "--run_b_variants", str(args.run_b_variants),
    ]


def main() -> None:
    """
    Execute the full experiment pipeline.
    """
    args = parse_args()

    workdir = Path(__file__).resolve().parent
    python_executable = sys.executable

    repvgg_presets = ["A0", "A1", "A2"]

    if args.run_b_variants:
        repvgg_presets.extend(["B0", "B1"])

    print("Starting full reproduction pipeline...", flush=True)
    print(f"Data dir: {args.data_dir}", flush=True)
    print(f"Results dir: {args.results_dir}", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Augmentation mode: {args.augmentation_mode}", flush=True)
    print(f"Run B variants: {args.run_b_variants}", flush=True)
    print(f"Run compare: {args.run_compare}", flush=True)
    print(f"Run plot: {args.run_plot}", flush=True)

    for preset in repvgg_presets:
        print(f"\n[TRAIN] RepVGG-{preset}", flush=True)

        run_command(
            build_train_command(
                python_executable=python_executable,
                args=args,
                model="repvgg",
                repvgg_preset=preset,
            ),
            workdir,
        )

    for baseline in ["resnet18", "resnet34"]:
        print(f"\n[TRAIN] {baseline}", flush=True)

        run_command(
            build_train_command(
                python_executable=python_executable,
                args=args,
                model=baseline,
            ),
            workdir,
        )

    if args.run_compare:
        print("\n[COMPARE] Running compare_models.py", flush=True)

        run_command(
            build_compare_command(
                python_executable=python_executable,
                args=args,
            ),
            workdir,
        )

    if args.run_plot:
        print("\n[PLOT] Running plot_histories.py", flush=True)

        run_command(
            build_plot_command(
                python_executable=python_executable,
                args=args,
            ),
            workdir,
        )

    print(
        "\nFull reproduction pipeline completed successfully.",
        flush=True,
    )


if __name__ == "__main__":
    main()