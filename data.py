"""
data.py
-------
WHY THIS FILE EXISTS:
    The instruction doc says we MUST use Dirichlet distribution
    alpha in {0.01, 0.1, 0.5, 1.0, IID} for data partitioning.
    This is the non-IID simulation — lower alpha = more heterogeneous data.
    This file handles EVERYTHING related to data: loading, splitting, distributing.

WHAT THIS FILE DOES:
    1. load_dataset()        — downloads MNIST/FMNIST/CIFAR-10 using torchvision
    2. dirichlet_split()     — splits dataset across N clients using Dirichlet distribution
    3. get_client_dataloader()— returns DataLoader for a specific client's data
    4. get_test_dataloader() — returns global test set loader for evaluation

HOW IT CONNECTS:
    client.py calls get_client_dataloader(client_id, dataset, alpha, num_clients)
    server.py calls get_test_dataloader(dataset) for global evaluation
    main.py uses this to set up the experiment

DIRICHLET EXPLAINED SIMPLY:
    For each class (e.g. digit 0-9), sample a distribution across N clients
    using Dir(alpha). Low alpha (0.01) means one client gets almost all
    samples of a class. High alpha (IID) means uniform distribution.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Optional
import os


# Fixed seed as required by instruction doc
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_dataset(dataset_name: str, data_dir: str = "./data"):
    """
    Download and return full train + test datasets.

    Args:
        dataset_name: "mnist", "fmnist", "cifar10", "cifar100"
        data_dir: where to store downloaded data

    Returns:
        (train_dataset, test_dataset)
    """
    os.makedirs(data_dir, exist_ok=True)
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
        test  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    elif dataset_name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test  = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train, test


def dirichlet_split(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = SEED
) -> List[List[int]]:
    """
    Split dataset indices across clients using Dirichlet distribution.

    This is the standard non-IID simulation used in both papers and the instruction doc.

    Args:
        dataset: PyTorch dataset with .targets attribute
        num_clients: number of FL clients (10, 50, or 100 per instructions)
        alpha: Dirichlet concentration parameter
               0.01  = extremely non-IID (one client per class almost)
               0.1   = highly non-IID
               0.5   = moderately non-IID
               1.0   = mildly non-IID
               1000  = approximately IID (use this for "IID" setting)
        seed: random seed for reproducibility

    Returns:
        List of length num_clients, each element is a list of sample indices
        for that client.

    Example:
        client_indices = dirichlet_split(train_dataset, num_clients=10, alpha=0.5)
        # client_indices[0] = [234, 1002, 5643, ...]  # indices for client 0
    """
    np.random.seed(seed)

    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels manually
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute its samples across clients using Dirichlet
    for class_id in range(num_classes):
        # Get all indices of this class
        class_indices = np.where(labels == class_id)[0]
        np.random.shuffle(class_indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Assign samples to clients proportionally
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, proportions)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # Shuffle each client's indices
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    return client_indices


def iid_split(dataset, num_clients: int, seed: int = SEED) -> List[List[int]]:
    """
    Split dataset equally and randomly across clients (IID setting).
    Used when alpha = "IID" in the config.
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(dataset))
    splits = np.array_split(indices, num_clients)
    return [split.tolist() for split in splits]


def get_client_dataloader(
    client_id: int,
    dataset_name: str,
    num_clients: int,
    alpha,  # float or "IID"
    batch_size: int = 32,
    data_dir: str = "./data"
) -> DataLoader:
    """
    Get DataLoader for a specific client.

    Args:
        client_id: 0-indexed client ID
        dataset_name: "mnist", "fmnist", "cifar10"
        num_clients: total number of clients
        alpha: Dirichlet alpha or "IID"
        batch_size: 32 as per instruction doc baseline config
        data_dir: data storage directory

    Returns:
        DataLoader for this client's local dataset

    Usage in client.py:
        loader = get_client_dataloader(self.client_id, "mnist", 10, 0.5)
        for batch_x, batch_y in loader:
            ...
    """
    train_dataset, _ = load_dataset(dataset_name, data_dir)

    # Partition data
    if alpha == "IID" or (isinstance(alpha, str) and alpha.upper() == "IID"):
        client_indices = iid_split(train_dataset, num_clients)
    else:
        client_indices = dirichlet_split(train_dataset, num_clients, float(alpha))

    # Get this client's subset
    this_client_indices = client_indices[client_id]
    client_subset = Subset(train_dataset, this_client_indices)

    return DataLoader(
        client_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Keep 0 for compatibility
        drop_last=False
    )


def get_test_dataloader(
    dataset_name: str,
    batch_size: int = 256,
    data_dir: str = "./data"
) -> DataLoader:
    """
    Get the global test set DataLoader.
    Used by server.py for evaluation every round.
    The instruction doc requires: 'Global Test Accuracy — evaluated every round
    on a held-out global test set'
    """
    _, test_dataset = load_dataset(dataset_name, data_dir)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def get_dataset_stats(
    dataset_name: str,
    num_clients: int,
    alpha,
    data_dir: str = "./data"
) -> dict:
    """
    Print statistics about the data distribution.
    Useful for verifying non-IID split worked correctly.

    Returns dict with per-client class distributions.
    """
    train_dataset, _ = load_dataset(dataset_name, data_dir)

    if alpha == "IID":
        client_indices = iid_split(train_dataset, num_clients)
    else:
        client_indices = dirichlet_split(train_dataset, num_clients, float(alpha))

    if hasattr(train_dataset, 'targets'):
        labels = np.array(train_dataset.targets)
    else:
        labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])

    num_classes = len(np.unique(labels))
    stats = {}

    for cid in range(num_clients):
        client_labels = labels[client_indices[cid]]
        class_counts = {c: int(np.sum(client_labels == c)) for c in range(num_classes)}
        stats[cid] = {
            "total_samples": len(client_indices[cid]),
            "class_distribution": class_counts
        }

    print(f"\n=== Data Distribution: {dataset_name}, {num_clients} clients, alpha={alpha} ===")
    for cid in range(min(5, num_clients)):  # print first 5 clients
        print(f"Client {cid}: {stats[cid]['total_samples']} samples, "
              f"classes: {stats[cid]['class_distribution']}")
    if num_clients > 5:
        print(f"... (showing first 5 of {num_clients} clients)")

    return stats
