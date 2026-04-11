import argparse

from src.train_teacher_model import Config, train_teacher_model


def build_parser() -> argparse.ArgumentParser:
    defaults = Config()
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 teacher model on CIFAR-10."
    )
    parser.add_argument("--data-dir", default=defaults.data_dir, help="CIFAR-10 dataset directory.")
    parser.add_argument("--save-path", default=defaults.save_path, help="Path to save the best checkpoint.")
    parser.add_argument(
        "--data_augment",
        choices=["None", "Cutout"],
        default=defaults.data_augment,
        help="Select the training data augmentation method.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=defaults.epochs, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=defaults.lr, help="Initial learning rate.")
    parser.add_argument("--momentum", type=float, default=defaults.momentum, help="SGD momentum.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=defaults.weight_decay,
        help="Weight decay for SGD.",
    )
    parser.add_argument("--num-classes", type=int, default=defaults.num_classes, help="Number of output classes.")
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed.")
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=defaults.label_smoothing,
        help="Label smoothing factor.",
    )
    parser.add_argument("--mixup-alpha", type=float, default=defaults.mixup_alpha, help="Mixup alpha.")
    parser.add_argument("--cutout-holes", type=int, default=defaults.cutout_holes, help="Number of Cutout holes.")
    parser.add_argument("--cutout-length", type=int, default=defaults.cutout_length, help="Cutout mask size.")

    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--use-amp", dest="use_amp", action="store_true", help="Enable AMP training.")
    amp_group.add_argument("--no-amp", dest="use_amp", action="store_false", help="Disable AMP training.")
    parser.set_defaults(use_amp=defaults.use_amp)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config(**vars(args))
    train_teacher_model(config)


if __name__ == "__main__":
    main()
