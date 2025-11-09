"""Hierarchical shrinkage intervals for grouped data."""

from typing import Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t


def _validate_alpha(alpha: float) -> None:
    """Validate the significance or credible level.

    Args:
        alpha: Level between 0 and 1.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _validate_groups(data: Sequence[Sequence[float]]) -> list[np.ndarray]:
    """Validate and convert nested sequences to arrays.

    Args:
        data: Iterable of groups, each containing numeric observations.

    Returns:
        list[numpy.ndarray]: List of numeric arrays for each group.

    Raises:
        ValueError: If ``data`` is empty or groups are invalid.
    """

    if not data:
        raise ValueError("data must contain at least one group.")
    arrays = []
    for group in data:
        arr = np.asarray(group, dtype=float)
        if arr.size == 0:
            raise ValueError("each group must be non-empty.")
        if not np.isfinite(arr).all():
            raise ValueError("data must be numeric.")
        arrays.append(arr)
    return arrays


def frequentist_intervals(
    data: Sequence[Sequence[float]],
    alpha: float = 0.05,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute standalone confidence intervals for each group.

    Args:
        data: List of arrays, each representing scores for one group.
        alpha: Significance level. Defaults to ``0.05``.

    Returns:
        Dict[int, Tuple[float, float, float]]:
            Mapping of group index to mean and interval bounds.

    Raises:
        ValueError: If any group has fewer than two observations.
    """

    _validate_alpha(alpha)
    groups = _validate_groups(data)
    intervals = {}
    for i, arr in enumerate(groups):
        n = arr.size
        if n < 2:
            raise ValueError("each group needs at least two observations")
        mean = arr.mean()
        df = n - 1
        se = arr.std(ddof=1) / np.sqrt(n)
        tcrit = t.ppf(1 - alpha / 2, df)
        half_width = tcrit * se
        intervals[i] = (mean, mean - half_width, mean + half_width)
    return intervals


def hierarchical_intervals(
    data: Sequence[Sequence[float]],
    alpha: float = 0.05,
    sigma2: float = 1.0,
) -> Dict[int, Tuple[float, float, float]]:
    """Compute empirical Bayes credible intervals for each group.

    Model: ``y_ij ~ N(θ_i, σ²)`` with ``θ_i ~ N(μ0, τ²)`` and hyperparameters
    estimated via method-of-moments.

    Args:
        data: List of arrays for each group.
        alpha: Credible level. Defaults to ``0.05``.
        sigma2: Known within-group variance. Defaults to ``1.0``.

    Returns:
        Dict[int, Tuple[float, float, float]]:
            Mapping of group index to posterior mean and interval bounds.

    Raises:
        ValueError: If ``sigma2`` is not positive.
    """

    _validate_alpha(alpha)
    if sigma2 <= 0:
        raise ValueError("sigma2 must be positive.")
    groups = _validate_groups(data)
    m = len(groups)
    bar = np.array([np.mean(group) for group in groups])
    n = np.array([len(group) for group in groups], dtype=float)
    # Estimate hyperparameters
    mu0 = np.mean(bar)
    # variance of bar means
    s2_bar = np.var(bar, ddof=1)
    # average sampling variance
    vs = np.mean(sigma2 / n)
    tau2 = max(0.0, s2_bar - vs)
    # Posterior for each theta_i
    results = {}
    for i in range(m):
        vi = sigma2 / n[i]
        post_var = 1 / (1 / vi + 1 / tau2) if tau2 > 0 else vi
        if tau2 > 0:
            post_mean = (bar[i] / vi + mu0 / tau2) * post_var
        else:
            post_mean = bar[i]
        zcrit = norm.ppf(1 - alpha / 2)
        half = zcrit * np.sqrt(post_var)
        results[i] = (post_mean, post_mean - half, post_mean + half)
    return results


def plot_hierarchical(
    data: Sequence[Sequence[float]],
    freq: Dict[int, Tuple[float, float, float]],
    hier: Dict[int, Tuple[float, float, float]],
) -> None:
    """Plot standalone versus hierarchical intervals for comparison.

    Args:
        data: List of arrays per group.
        freq: Output from :func:`frequentist_intervals`.
        hier: Output from :func:`hierarchical_intervals`.
    """

    groups = _validate_groups(data)
    m = len(groups)
    schools = np.arange(m)
    # extract endpoints
    f_means = [freq[i][0] for i in schools]
    f_lows = [freq[i][1] for i in schools]
    f_highs = [freq[i][2] for i in schools]
    h_means = [hier[i][0] for i in schools]
    h_lows = [hier[i][1] for i in schools]
    h_highs = [hier[i][2] for i in schools]

    plt.figure(figsize=(8, 6))
    # Frequentist CIs
    for i in schools:
        plt.plot(
            [i - 0.1, i - 0.1],
            [f_lows[i], f_highs[i]],
            color="gray",
            lw=2,
        )
        plt.plot(i - 0.1, f_means[i], "o", color="gray")
    # Hierarchical CIs
    for i in schools:
        plt.plot(
            [i + 0.1, i + 0.1],
            [h_lows[i], h_highs[i]],
            color="blue",
            lw=2,
        )
        plt.plot(i + 0.1, h_means[i], "o", color="blue")
    plt.xticks(schools)
    plt.xlabel("School index")
    plt.ylabel("Estimated mean")
    plt.legend(["Freq CI", "Hier CI"], loc="upper right")
    plt.title("Standalone vs Hierarchical Shrinkage Intervals")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Simulate data: true thetas from N(50, 10^2), within-school sigma=1
    rng = np.random.default_rng(1)
    m = 10
    n_per = 20
    true_thetas = rng.normal(50, 10, size=m)
    data = [rng.normal(theta, 1, size=n_per) for theta in true_thetas]
    freq = frequentist_intervals(data)
    hier = hierarchical_intervals(data, sigma2=1.0)
    plot_hierarchical(data, freq, hier)
