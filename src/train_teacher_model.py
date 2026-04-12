import os
import random
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.evaluate import (
    append_epoch_metrics,
    build_history,
    evaluate_model,
    get_evaluation_artifact_paths,
    plot_convergence_curves,
    save_convergence_summary,
    save_history,
)
from src.models import build_resnet18


@dataclass
class Config:
    data_dir: str = "./data"
    save_path: str = "./outputs/best_resnet18_cifar10.pth"
    data_augment: str = "Cutout"
    batch_size: int = 128
    epochs: int = 120
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    num_classes: int = 10
    num_workers: int = 4
    seed: int = 42
    scheduler: str = "cosine_annealing"
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutout_holes: int = 1
    cutout_length: int = 16
    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: Config,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    total = 0
    use_amp = config.use_amp and device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, targets_a, targets_b, lam = mixup_data(
            images,
            labels,
            alpha=config.mixup_alpha,
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct_top1 += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    top1_acc = 100.0 * correct_top1 / total
    print(f"Epoch [{epoch}] Train Loss: {avg_loss:.4f} | Train Top-1 Acc: {top1_acc:.2f}%")
    return avg_loss, top1_acc


def train_teacher_model(config: Config) -> float:
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_loader, test_loader = get_dataloaders(config, device)
    model = build_resnet18(num_classes=config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.use_amp and device.type == "cuda"))

    save_dir = os.path.dirname(config.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    artifact_paths = get_evaluation_artifact_paths(config.save_path)
    history = build_history(config)
    best_acc = 0.0

    for epoch in range(1, config.epochs + 1):
        epoch_start_time = time.time()
        train_loss, train_top1_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            config=config,
        )

        val_loss, val_top1_acc, val_top5_acc = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            use_amp=(config.use_amp and device.type == "cuda"),
        )

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        append_epoch_metrics(
            history=history,
            train_loss=train_loss,
            train_top1_acc=train_top1_acc,
            val_loss=val_loss,
            val_top1_acc=val_top1_acc,
            val_top5_acc=val_top5_acc,
            learning_rate=current_lr,
            epoch_time_sec=epoch_time,
        )
        save_history(history, artifact_paths["history"])
        save_convergence_summary(history, artifact_paths["summary"])
        plot_convergence_curves(history, artifact_paths["figures"])

        print(
            f"Epoch [{epoch}] Val Loss: {val_loss:.4f} | "
            f"Val Top-1 Acc: {val_top1_acc:.2f}% | "
            f"Val Top-5 Acc: {val_top5_acc:.2f}% | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        )

        scheduler.step()

        if val_top1_acc > best_acc:
            best_acc = val_top1_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                    "config": asdict(config),
                    "history_path": str(artifact_paths["history"]),
                    "convergence_summary_path": str(artifact_paths["summary"]),
                },
                config.save_path,
            )
            print(f"Saved best model to {config.save_path}")

    print(f"Training finished. Best Top-1 Acc: {best_acc:.2f}%")
    print(f"History saved to {artifact_paths['history']}")
    print(f"Convergence summary saved to {artifact_paths['summary']}")
    print(f"Convergence figures saved to {artifact_paths['figures']}")
    return best_acc
