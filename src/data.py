import torch
import torchvision
import torchvision.transforms as transforms


class Cutout:
    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        height = img.size(1)
        width = img.size(2)
        mask = torch.ones((height, width), dtype=img.dtype, device=img.device)

        for _ in range(self.n_holes):
            center_y = torch.randint(0, height, (1,)).item()
            center_x = torch.randint(0, width, (1,)).item()

            y1 = max(0, center_y - self.length // 2)
            y2 = min(height, center_y + self.length // 2)
            x1 = max(0, center_x - self.length // 2)
            x2 = min(width, center_x + self.length // 2)
            mask[y1:y2, x1:x2] = 0

        return img * mask.expand_as(img)


def get_dataloaders(config, device: torch.device):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if config.data_augment == "Cutout":
        train_transforms.append(
            Cutout(n_holes=config.cutout_holes, length=config.cutout_length)
        )

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=config.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
    }

    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader
