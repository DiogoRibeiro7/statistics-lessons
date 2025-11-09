"""Simple deep learning examples using PyTorch."""

from __future__ import annotations

from typing import Sequence, Tuple

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled in tests
    torch = None
    nn = None


def train_linear_regression(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[float, float]:
    """Train a linear regression model with PyTorch.

    Args:
        x: Feature values.
        y: Target values.

    Returns:
        Tuple containing learned slope and intercept.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if torch is None or nn is None:  # pragma: no cover - dependency missing
        raise ImportError("PyTorch is required for this function")

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()

    for _ in range(200):
        optimizer.zero_grad()
        pred = model(x_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()

    weight = float(model.weight.detach().numpy().squeeze())
    bias = float(model.bias.detach().numpy().squeeze())
    return weight, bias


def train_mlp_classifier(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
) -> Tuple[nn.Module, float]:
    """Train a simple multilayer perceptron classifier with PyTorch.

    Args:
        features: Two-dimensional feature matrix.
        labels: Target class labels starting from 0.

    Returns:
        Tuple of the trained model and the final training loss.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    if torch is None or nn is None:  # pragma: no cover - dependency missing
        raise ImportError("PyTorch is required for this function")

    torch.manual_seed(0)
    x_tensor = torch.tensor(features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    n_features = x_tensor.shape[1]
    n_classes = int(y_tensor.max().item() + 1)

    model = nn.Sequential(
        nn.Linear(n_features, 8),
        nn.ReLU(),
        nn.Linear(8, n_classes),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(200):
        optimizer.zero_grad()
        logits = model(x_tensor)
        loss = loss_fn(logits, y_tensor)
        loss.backward()
        optimizer.step()

    return model, float(loss.item())
