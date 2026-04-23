from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision import transforms


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 0.1
MIXUP_ALPHA = 0.2
LABEL_SMOOTHING = 0.1
CUTOUT_LENGTH = 16


def resolve_device(device_preference: str = "auto") -> torch.device:
    """Resolve the requested execution device with safe fallback behavior."""
    normalized = device_preference.lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if normalized == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if normalized == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("MPS was requested, but MPS is not available.")

    if normalized == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device preference: {device_preference}. "
        "Expected one of: auto, cuda, cpu, mps."
    )


class Cutout:
    """Randomly mask out a square patch from a tensor image."""

    def __init__(self, length: int = CUTOUT_LENGTH) -> None:
        self.length = length

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if self.length <= 0:
            return image

        _, height, width = image.shape
        center_y = torch.randint(0, height, (1,)).item()
        center_x = torch.randint(0, width, (1,)).item()
        half_length = self.length // 2

        top = max(center_y - half_length, 0)
        bottom = min(center_y + half_length, height)
        left = max(center_x - half_length, 0)
        right = min(center_x + half_length, width)

        masked_image = image.clone()
        masked_image[:, top:bottom, left:right] = 0.0
        return masked_image


def get_cifar10_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            Cutout(length=CUTOUT_LENGTH),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )

    return train_transform, test_transform


def build_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0.0, 1.0).")

    off_value = label_smoothing / num_classes
    on_value = 1.0 - label_smoothing + off_value

    targets = torch.full(
        (labels.size(0), num_classes),
        fill_value=off_value,
        device=labels.device,
        dtype=torch.float32,
    )
    targets.scatter_(1, labels.unsqueeze(1), on_value)
    return targets


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0.0:
        target_probs = build_soft_targets(
            labels,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
        )
        return images, target_probs, labels, labels, 1.0

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    permutation = torch.randperm(images.size(0), device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[permutation]
    primary_labels = labels
    secondary_labels = labels[permutation]

    primary_targets = build_soft_targets(
        primary_labels,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )
    secondary_targets = build_soft_targets(
        secondary_labels,
        num_classes=num_classes,
        label_smoothing=label_smoothing,
    )
    mixed_targets = lam * primary_targets + (1.0 - lam) * secondary_targets

    return mixed_images, mixed_targets, primary_labels, secondary_labels, lam


def soft_target_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def mixup_correct_predictions(
    logits: torch.Tensor,
    primary_labels: torch.Tensor,
    secondary_labels: torch.Tensor,
    lam: float,
) -> float:
    predictions = logits.argmax(dim=1)
    primary_correct = predictions.eq(primary_labels).sum().item()
    secondary_correct = predictions.eq(secondary_labels).sum().item()
    return lam * primary_correct + (1.0 - lam) * secondary_correct
