"""
Dataset Loaders and Data Pipelines for Diffusion Training.
"""

from typing import Dict, Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Fashion-MNIST Class mapping dictionary
FASHION_CLASSES: Dict[int, str] = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def get_fashion_mnist_dataloader(
    batch_size: int = 128,
    root: str = "./data",
    download: bool = True,
    image_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[datasets.FashionMNIST, datasets.FashionMNIST, DataLoader, DataLoader]:
    """
    Load Fashion-MNIST dataset with transforms tailored for diffusion models:
    resizes images from 28x28 to 32x32 and normalizes pixel values to [-1, 1].

    Args:
        batch_size: Batch size for training / evaluation (default: 128).
        root: Storage directory for dataset (default: './data').
        download: Automatically download dataset if not locally present (default: True).
        image_size: Target spatial resolution (default: 32).
        num_workers: DataLoader subprocess workers (default: 2).
        pin_memory: Pin memory for GPU transfer acceleration (default: True).

    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
    ])

    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    return train_dataset, test_dataset, train_loader, test_loader


def get_dataset(
    dataset_name: str = "fashion_mnist",
    batch_size: int = 128,
    root: str = "./data",
    download: bool = True,
    image_size: int = 32,
    num_workers: int = 2,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, DataLoader, DataLoader]:
    """
    Generic dataset dispatcher supporting multiple benchmarks.

    Supported datasets: 'fashion_mnist', 'mnist', 'cifar10'.
    """
    name_lower = dataset_name.lower().replace("-", "_")

    if name_lower in ["fashion_mnist", "fashionmnist"]:
        return get_fashion_mnist_dataloader(
            batch_size=batch_size,
            root=root,
            download=download,
            image_size=image_size,
            num_workers=num_workers,
        )
    elif name_lower == "mnist":
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        train_ds = datasets.MNIST(root=root, train=True, download=download, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=download, transform=transform)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_ds, test_ds, train_ld, test_ld
    elif name_lower in ["cifar10", "cifar_10"]:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        train_ds = datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        test_ds = datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_ds, test_ds, train_ld, test_ld
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: 'fashion_mnist', 'mnist', 'cifar10'.")
