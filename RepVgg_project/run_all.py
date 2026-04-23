"""
run_all.py

Purpose:
--------
Run training for all models (RepVGG, PlainCNN, ResNet-18) sequentially.

Objective:
----------
Automate Phase 2.1 experiments so all models are trained under
the same conditions without manual repetition.

This script makes sure train.py is executed from the correct project folder.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], workdir: Path) -> None:
    """
    Run a command and stream output live from the correct working directory.
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
        raise RuntimeError(f"Command failed: {' '.join(command)}")


def main() -> None:
    """
    Run all models sequentially.
    """
    project_root = Path(__file__).resolve().parent
    python_executable = sys.executable

    models = ["repvgg", "plaincnn", "resnet18"]

    for model in models:
        print("\n" + "=" * 60)
        print(f"Starting training for model: {model}")
        print("=" * 60 + "\n", flush=True)

        command = [
            python_executable,
            "train.py",
            "--model",
            model,
            "--epochs",
            "30",
            "--batch_size",
            "128",
            "--lr",
            "0.1",
        ]

        run_command(command, workdir=project_root)

        print("\n" + "-" * 60)
        print(f"Finished training for model: {model}")
        print("-" * 60 + "\n", flush=True)


if __name__ == "__main__":
    main()
