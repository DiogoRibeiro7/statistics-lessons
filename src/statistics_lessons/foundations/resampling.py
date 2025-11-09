"""Resampling utilities for bootstrapping analyses."""

from typing import Optional, Tuple

import numpy as np


def bootstrap_resample(
    data: np.ndarray, size: Optional[int] = None
) -> np.ndarray:
    """Create a bootstrap resample of the data.

    Args:
        data (numpy.ndarray): Original observations.
        size (Optional[int], optional): Number of samples to draw. Defaults to
            ``len(data)``.

    Returns:
        numpy.ndarray: Array containing the resampled data.

    Raises:
        ValueError: If ``data`` is empty or ``size`` is not positive.
    """
    if len(data) == 0:
        raise ValueError("Data must contain at least one observation.")
    if size is None:
        size = len(data)
    if size <= 0:
        raise ValueError("size must be positive.")
    # Randomly draw indices with replacement
    indices = np.random.randint(0, len(data), size=size)
    return data[indices]


def bootstrap_mean_ci(
    data: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95
) -> Tuple[float, float]:
    """Estimate a confidence interval for the mean via bootstrapping.

    Args:
        data (numpy.ndarray): Observed data sample.
        n_bootstrap (int, optional): Number of bootstrap resamples. Defaults to
            ``1000``.
        confidence (float, optional): Confidence level between 0 and 1.
            Defaults to ``0.95``.

    Returns:
        Tuple[float, float]: Lower and upper confidence interval bounds.

    Raises:
        ValueError: If ``data`` is empty, ``n_bootstrap`` is not positive, or
            ``confidence`` is not in (0, 1).
    """
    if len(data) == 0:
        raise ValueError("Data must contain at least one observation.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    means = np.empty(n_bootstrap)
    # Generate bootstrap resamples and compute their means
    for i in range(n_bootstrap):
        resample = bootstrap_resample(data)
        means[i] = np.mean(resample)

    alpha = 1.0 - confidence
    lower = np.percentile(means, 100 * (alpha / 2))
    upper = np.percentile(means, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=123)
    sample = rng.normal(loc=0.0, scale=1.0, size=50)
    ci_low, ci_high = bootstrap_mean_ci(sample)
    print("Bootstrap mean 95% CI:", ci_low, ci_high)
