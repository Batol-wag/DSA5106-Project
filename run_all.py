from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], workdir: Path) -> None:
    process = subprocess.Popen(
        command,
        cwd=str(workdir),
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all training and comparison jobs sequentially"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny_imagenet",
        choices=["cifar10", "tiny_imagenet"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--subset_fraction", type=float, default=1.0)

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine"],
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=0)

    parser.add_argument("--imagenet_image_size", type=int, default=64)
    parser.add_argument("--amp", type=str, default="true")
    parser.add_argument("--augmentation_mode", type=str, default="strong", choices=["simple", "strong"])

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

    return parser.parse_args()


def build_train_command(
    python_executable: str,
    args: argparse.Namespace,
    model: str,
    repvgg_preset: str | None = None,
) -> list[str]:
    command = [
        python_executable,
        "train.py",
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--model", model,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--subset_fraction", str(args.subset_fraction),
        "--scheduler", args.scheduler,
        "--warmup_epochs", str(args.warmup_epochs),
        "--imagenet_image_size", str(args.imagenet_image_size),
        "--amp", args.amp,
        "--augmentation_mode", args.augmentation_mode,
    ]

    if args.lr is not None:
        command += ["--lr", str(args.lr)]

    if args.weight_decay is not None:
        command += ["--weight_decay", str(args.weight_decay)]

    if repvgg_preset is not None:
        command += ["--repvgg_preset", repvgg_preset]

    return command


def build_compare_command(
    python_executable: str,
    args: argparse.Namespace,
    repvgg_preset: str,
) -> list[str]:
    command = [
        python_executable,
        "compare_models.py",
        "--dataset", args.dataset,
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--subset_fraction", str(args.subset_fraction),
        "--imagenet_image_size", str(args.imagenet_image_size),
        "--repvgg_preset", repvgg_preset,
        "--augmentation_mode", args.augmentation_mode,
        "--amp", args.amp,
    ]

    return command


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    python_executable = sys.executable

    repvgg_presets = ["A0", "A1", "A2"]
    if args.run_b_variants:
        repvgg_presets += ["B0", "B1"]

    baseline_models = ["resnet18", "resnet34"]

    print("\n" + "=" * 70)
    print("Starting FULL training pipeline")
    print("=" * 70)
    print(f"Dataset            : {args.dataset}")
    print(f"Data dir           : {args.data_dir}")
    print(f"Epochs             : {args.epochs}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Subset fraction    : {args.subset_fraction}")
    print(f"Augmentation mode  : {args.augmentation_mode}")
    print(f"Run B variants     : {args.run_b_variants}")
    print(f"Run compare        : {args.run_compare}")
    print("=" * 70 + "\n", flush=True)

    # --------------------------------------------------
    # 1) Train RepVGG presets
    # --------------------------------------------------
    for preset in repvgg_presets:
        print("\n" + "=" * 70)
        print(f"Starting training for RepVGG preset: {preset}")
        print("=" * 70 + "\n", flush=True)

        command = build_train_command(
            python_executable=python_executable,
            args=args,
            model="repvgg",
            repvgg_preset=preset,
        )

        run_command(command, workdir=project_root)

        print("\n" + "-" * 70)
        print(f"Finished training for RepVGG preset: {preset}")
        print("-" * 70 + "\n", flush=True)

    # --------------------------------------------------
    # 2) Train baselines
    # --------------------------------------------------
    for model in baseline_models:
        print("\n" + "=" * 70)
        print(f"Starting training for baseline: {model}")
        print("=" * 70 + "\n", flush=True)

        command = build_train_command(
            python_executable=python_executable,
            args=args,
            model=model,
            repvgg_preset=None,
        )

        run_command(command, workdir=project_root)

        print("\n" + "-" * 70)
        print(f"Finished training for baseline: {model}")
        print("-" * 70 + "\n", flush=True)

    # --------------------------------------------------
    # 3) Run comparisons
    # --------------------------------------------------
    if args.run_compare:
        for preset in repvgg_presets:
            print("\n" + "=" * 70)
            print(f"Running comparison for preset: {preset}")
            print("=" * 70 + "\n", flush=True)

            command = build_compare_command(
                python_executable=python_executable,
                args=args,
                repvgg_preset=preset,
            )

            run_command(command, workdir=project_root)

            print("\n" + "-" * 70)
            print(f"Finished comparison for preset: {preset}")
            print("-" * 70 + "\n", flush=True)

    print("\n" + "=" * 70)
    print("All runs completed successfully.")
    print("=" * 70 + "\n", flush=True)


if __name__ == "__main__":
    main()