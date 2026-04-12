from __future__ import annotations

import argparse
from pathlib import Path

import torch

from evaluate import evaluate_model, validate_repvgg_deploy
from train import build_model, get_dataloaders, resolve_defaults, apply_repvgg_preset
from utils.metrics import compute_flops, count_parameters, measure_inference_latency


def str2bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained models on CIFAR-10")

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=str2bool, default=True)

    parser.add_argument("--subset_fraction", type=float, default=1.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--repvgg_variant",
        type=str,
        default="a",
        choices=["small", "a", "b"],
    )
    parser.add_argument(
        "--repvgg_preset",
        type=str,
        default=None,
        choices=["A0", "A1", "A2", "B0", "B1"],
    )
    parser.add_argument("--a_multiplier", type=float, default=1.0)
    parser.add_argument("--b_multiplier", type=float, default=2.5)

    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="strong",
        choices=["simple", "strong"],
    )
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--autoaugment", type=str2bool, default=False)
    parser.add_argument("--mixup_alpha", type=float, default=0.0)

    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["step", "cosine"],
    )
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--amp", type=str2bool, default=True)

    args = parser.parse_args()
    args = apply_repvgg_preset(args)
    args = resolve_defaults(args)
    return args


def get_input_shape() -> tuple[int, int, int, int]:
    return (1, 3, 32, 32)


def get_latency_shape(args: argparse.Namespace) -> tuple[int, int, int, int]:
    return (args.batch_size, 3, 32, 32)


def build_run_tag(
    args: argparse.Namespace,
    model_name: str,
) -> str:
    model_tag = model_name

    if model_name == "repvgg":
        if args.repvgg_preset is not None:
            model_tag += f"_{args.repvgg_preset}"
        else:
            model_tag += f"_{args.repvgg_variant}_a{args.a_multiplier:g}_b{args.b_multiplier:g}"

    return (
        f"cifar10_"
        f"{model_tag}_"
        f"{args.augmentation_mode}_"
        f"subset{args.subset_fraction:g}"
    )


def load_model(
    args: argparse.Namespace,
    model_name: str,
    ckpt_path: Path,
) -> torch.nn.Module:
    local_args = argparse.Namespace(**vars(args))
    local_args.model = model_name

    model = build_model(local_args)

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def format_flops(flops: float | None) -> str:
    return "N/A" if flops is None else f"{flops:.4f}"


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = get_dataloaders(args)
    checkpoint_dir = Path("checkpoints")

    repvgg_run_tag = build_run_tag(args, "repvgg")
    resnet18_run_tag = build_run_tag(args, "resnet18")
    resnet34_run_tag = build_run_tag(args, "resnet34")

    model_configs = [
        ("RepVGG (train)", "repvgg", f"{repvgg_run_tag}_best.pth", False),
        ("RepVGG (deploy)", "repvgg", f"{repvgg_run_tag}_best.pth", True),
        ("ResNet-18", "resnet18", f"{resnet18_run_tag}_best.pth", False),
        ("ResNet-34", "resnet34", f"{resnet34_run_tag}_best.pth", False),
    ]

    rows = []

    for display_name, model_name, ckpt_file, is_deploy in model_configs:
        ckpt_path = checkpoint_dir / ckpt_file
        if not ckpt_path.exists():
            print(f"[SKIP] Missing checkpoint: {ckpt_path}", flush=True)
            continue

        print(f"Evaluating {display_name}...", flush=True)

        model = load_model(args, model_name, ckpt_path).to(device)

        if is_deploy and hasattr(model, "switch_to_deploy"):
            model.switch_to_deploy()
            model = model.to(device)

        _, top1, top5 = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            desc=display_name,
        )

        params = count_parameters(model)
        flops = compute_flops(model, input_size=get_input_shape(), device=device)
        latency_ms = measure_inference_latency(
            model=model,
            input_size=get_latency_shape(args),
            device=device,
        )

        rows.append(
            {
                "model": display_name,
                "top1": top1,
                "top5": top5,
                "params_m": params / 1e6,
                "gflops": flops,
                "latency_ms": latency_ms,
            }
        )

    repvgg_ckpt = checkpoint_dir / f"{repvgg_run_tag}_best.pth"
    if repvgg_ckpt.exists():
        print("\nRunning RepVGG deploy consistency check...", flush=True)
        validate_repvgg_deploy(
            test_loader=val_loader,
            device=device,
            model_variant=args.repvgg_variant,
            num_classes=args.num_classes,
            checkpoint_path=repvgg_ckpt,
            a_multiplier=args.a_multiplier,
            b_multiplier=args.b_multiplier,
        )

    print("\n" + "=" * 78)
    print("Model Comparison Table")
    print("=" * 78)
    print(
        f"| {'Model':<20} | {'Top-1 (%)':>9} | {'Top-5 (%)':>9} "
        f"| {'Params (M)':>10} | {'GFLOPs':>8} | {'Latency (ms)':>13} |"
    )
    print(
        f"| {'-'*20} | {'-'*9} | {'-'*9} "
        f"| {'-'*10} | {'-'*8} | {'-'*13} |"
    )

    for row in rows:
        print(
            f"| {row['model']:<20} | {row['top1']:>9.2f} | {row['top5']:>9.2f} "
            f"| {row['params_m']:>10.2f} | {format_flops(row['gflops']):>8} | "
            f"{row['latency_ms']:>13.2f} |"
        )


if __name__ == "__main__":
    main()