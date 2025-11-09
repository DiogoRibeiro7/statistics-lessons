"""Synthetic data generators for educational examples."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_linear_regression(
    n_samples: int = 100,
    *,
    slope: float = 1.0,
    intercept: float = 0.0,
    noise: float = 1.0,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a simple linear regression dataset.

    Args:
        n_samples: Number of observations to generate.
        slope: True slope of the underlying linear relationship.
        intercept: Intercept term of the line.
        noise: Standard deviation of the additive Gaussian noise.
        random_state: Seed controlling the randomness.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a one-column ``DataFrame`` of
        predictors and ``y`` is a ``Series`` of responses.
    """

    rng = np.random.default_rng(random_state)
    x = rng.uniform(-1.0, 1.0, size=n_samples)
    y = slope * x + intercept + rng.normal(0.0, noise, size=n_samples)
    X = pd.DataFrame({"x": x})
    y_series = pd.Series(y, name="y")
    return X, y_series


def make_classification(
    n_samples: int = 100,
    *,
    weights: Tuple[float, float] = (1.0, -1.0),
    bias: float = 0.0,
    random_state: int | None = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate a two-feature binary classification dataset.

    The target labels are drawn from a logistic model using the specified
    linear weights.

    Args:
        n_samples: Number of observations to generate.
        weights: Coefficients for the two input features.
        bias: Intercept term of the logistic model.
        random_state: Seed controlling the randomness.

    Returns:
        A tuple ``(X, y)`` where ``X`` is a two-column ``DataFrame`` of
        features and ``y`` is a binary ``Series``.
    """

    rng = np.random.default_rng(random_state)
    X_array = rng.normal(size=(n_samples, 2))
    linear = X_array @ np.asarray(weights) + bias
    probs = 1.0 / (1.0 + np.exp(-linear))
    y = rng.binomial(1, probs)
    X = pd.DataFrame(X_array, columns=["x1", "x2"])
    y_series = pd.Series(y, name="y")
    return X, y_series
