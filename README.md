# DSA5106-Project Teacher model part

`main.py` is the CLI entrypoint. It parses training arguments and starts CIFAR-10 teacher-model training with a CIFAR-adapted ResNet-18.

## Structure

- `src/data.py`: data augmentation and CIFAR-10 dataloaders
- `src/models.py`: ResNet18 model builder for CIFAR-10
- `src/train_teacher_model.py`: config, training loop, evaluation, checkpoint saving
- `main.py`: `argparse` interface and training startup

## Standardized Setup

- Dataset: CIFAR-10
- Epochs: 120
- Batch size: 128
- Scheduler: Cosine Annealing
- Augmentation: RandomCrop + HorizontalFlip + Cutout + Mixup (`alpha=0.2`) + Label Smoothing (`0.1`)

The teacher model is a CIFAR-10-adapted ResNet-18 with:

- `conv1 = 3x3, stride=1, padding=1`
- `maxpool = Identity()`

## Usage

```bash
python main.py --epochs 120 --batch-size 128 --save-path ./outputs/best_resnet18_cifar10.pth
```
