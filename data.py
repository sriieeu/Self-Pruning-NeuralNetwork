"""
data.py — CIFAR-10 Data Loaders
================================
Standard train / test splits with augmentation for CIFAR-10.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_loaders(
    batch_size: int = 256,
    data_dir: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for CIFAR-10.

    Training augmentations
    ----------------------
    - Random horizontal flip
    - Random crop with 4-pixel padding
    - Normalise with CIFAR-10 channel statistics

    Parameters
    ----------
    batch_size  : int  — mini-batch size (default 256)
    data_dir    : str  — directory to cache the dataset
    num_workers : int  — DataLoader worker processes

    Returns
    -------
    (train_loader, test_loader)
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(
        data_dir, train=True,  download=True, transform=transform_train
    )
    test_ds  = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader
