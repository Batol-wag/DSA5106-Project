import torch.nn as nn
import torchvision


def build_resnet18(num_classes: int = 10) -> nn.Module:
    """Build a ResNet18 variant adapted for CIFAR-10."""

    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
