import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticRegression(nn.Module):
    """
    Simple logistic regression for convex experiments.
    Paper 1 uses this on synthetic data and FEMNIST for convex analysis.
    We use it on MNIST (flattened 28x28 = 784 input, 10 classes).
    """
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Flatten image to vector
        x = x.view(x.size(0), -1)
        return self.linear(x)


class SimpleDNN(nn.Module):
    """
    One hidden layer feedforward network.
    Paper 1 uses this exact architecture on MNIST for non-convex experiments.
    Uses PCA projection layer conceptually — we skip the PCA and use a
    linear bottleneck layer instead for simplicity in Flower.
    Architecture: 784 -> 200 -> 10
    """
    def __init__(self, input_dim=784, hidden_dim=200, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SmallCNN(nn.Module):
    """
    Small CNN for FMNIST and CIFAR-10.
    Good balance between parameter count and accuracy.
    FMNIST: 1 channel, 28x28
    CIFAR-10: 3 channels, 32x32
    """
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the flattened size
        # FMNIST: 28x28 -> pool -> 14x14 -> pool -> 7x7, 64 channels = 64*7*7=3136
        # CIFAR10: 32x32 -> pool -> 16x16 -> pool -> 8x8, 64 channels = 64*8*8=4096
        self.fc_input_dim = None  # will be set in first forward pass
        self.fc1 = None
        self.fc2 = None
        self.in_channels = in_channels

        # Pre-compute fc input dim
        if in_channels == 1:
            self.fc_input_dim = 64 * 7 * 7   # FMNIST
        else:
            self.fc_input_dim = 64 * 8 * 8   # CIFAR-10

        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(dataset: str, model_type: str = "default") -> nn.Module:
    """
    Factory function — call this from client.py.

    Args:
        dataset: one of "mnist", "fmnist", "cifar10", "cifar100", "synthetic"
        model_type: "logreg", "dnn", "cnn", or "default" (auto-selects)

    Returns:
        Initialized model (not trained)

    Usage:
        model = get_model("mnist")
        model = get_model("mnist", model_type="logreg")  # force logistic regression
    """
    dataset = dataset.lower()

    if model_type == "logreg":
        if dataset in ("mnist", "fmnist", "synthetic"):
            return LogisticRegression(input_dim=784, num_classes=10)
        elif dataset == "cifar10":
            return LogisticRegression(input_dim=3072, num_classes=10)

    if model_type == "dnn":
        if dataset in ("mnist", "fmnist", "synthetic"):
            return SimpleDNN(input_dim=784, hidden_dim=200, num_classes=10)

    # default: auto-select best model for each dataset
    if dataset == "mnist":
        return SimpleDNN(input_dim=784, hidden_dim=200, num_classes=10)
    elif dataset == "fmnist":
        return SmallCNN(in_channels=1, num_classes=10)
    elif dataset == "cifar10":
        return SmallCNN(in_channels=3, num_classes=10)
    elif dataset == "cifar100":
        return SmallCNN(in_channels=3, num_classes=100)
    elif dataset == "synthetic":
        return LogisticRegression(input_dim=40, num_classes=10)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from: mnist, fmnist, cifar10, cifar100, synthetic")


def get_model_size_mb(model: nn.Module) -> float:
    """
    Returns size of model parameters in MB.
    Used to calculate communication cost metric required by the instruction doc.
    Communication cost = model_size_mb * num_clients * num_rounds
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 ** 2)


def get_parameters(model: nn.Module):
    """Extract model parameters as list of numpy arrays. Used by Flower."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters):
    """Load parameters from list of numpy arrays into model. Used by Flower."""
    import numpy as np
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)
