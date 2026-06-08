from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split
from torchvision import datasets


CLASS_A = 7
CLASS_B = 8
CLASS_NAMES = ["Sneaker", "Bag"]


def default_data_root():
    return Path(__file__).resolve().parents[1] / "data"


def build_binary_dataset(dataset, class_a=CLASS_A, class_b=CLASS_B):
    data = dataset.data.float() / 255.0
    targets = dataset.targets

    mask = (targets == class_a) | (targets == class_b)
    data = data[mask]
    targets = targets[mask]

    data = (data - 0.5) / 0.5
    data = data.unsqueeze(1)

    binary_targets = torch.where(targets == class_a, 0, 1)
    return TensorDataset(data, binary_targets)


def load_fashion_binary(data_root=None, val_ratio=0.2, seed=42, download=False):
    root = Path(data_root) if data_root else default_data_root()

    train_raw = datasets.FashionMNIST(root=str(root), train=True, download=download)
    test_raw = datasets.FashionMNIST(root=str(root), train=False, download=download)

    train_full = build_binary_dataset(train_raw)
    test_dataset = build_binary_dataset(test_raw)

    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size

    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    return train_dataset, val_dataset, test_dataset
