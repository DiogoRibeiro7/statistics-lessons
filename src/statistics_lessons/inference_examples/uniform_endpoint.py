"""Endpoint estimation for a Uniform(0, θ) distribution."""

from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _validate_alpha(alpha: float) -> None:
    """Validate the significance level.

    Args:
        alpha: Significance level between 0 and 1.

    Raises:
        ValueError: If ``alpha`` is not strictly between 0 and 1.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _validate_data(data: Sequence[float]) -> np.ndarray:
    """Ensure the data are valid non-negative numbers.

    Args:
        data: Sequence of observed samples.

    Returns:
        numpy.ndarray: Array of validated samples.

    Raises:
        ValueError: If ``data`` is empty or contains negatives.
        ValueError: If ``data`` contains non-numeric values.
    """

    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError("data must be numeric.")
    if np.any(arr < 0):
        raise ValueError("data must be non-negative for Uniform(0, θ) model.")
    return arr


def uniform_endpoint_estimation(
    data: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Estimate the endpoint ``θ`` of a Uniform(0, ``θ``) distribution.

    Args:
        data: Observations from ``U(0, θ)``.
        alpha: Significance or credible level. Defaults to ``0.05``.

    Returns:
        Dict[str, float]:
            Dictionary with MLE and frequentist/Bayesian interval bounds.
    """

    _validate_alpha(alpha)
    arr = _validate_data(data)
    n = arr.size
    # MLE is the sample maximum
    mle = float(arr.max())
    # Frequentist 95% CI (as given): [mle, mle/alpha]
    ci_freq_low = mle
    ci_freq_high = mle / alpha
    # Bayesian 95% credible interval under flat prior => Pareto posterior
    # Posterior CDF: 1 - (mle/theta)^n => solve (mle/theta)^n = alpha
    ci_bayes_low = mle
    ci_bayes_high = mle / (alpha ** (1 / n))
    return {
        "mle": mle,
        "ci_freq_low": ci_freq_low,
        "ci_freq_high": ci_freq_high,
        "ci_bayes_low": ci_bayes_low,
        "ci_bayes_high": ci_bayes_high,
    }


def plot_uniform_endpoint(
    data: Sequence[float],
    intervals: Dict[str, float],
) -> None:
    """Visualize the MLE and confidence/credible intervals.

    Args:
        data: Observed samples.
        intervals: Output from :func:`uniform_endpoint_estimation`.
    """

    _validate_data(data)
    mle = intervals["mle"]
    f_low = intervals["ci_freq_low"]
    f_high = intervals["ci_freq_high"]
    b_low = intervals["ci_bayes_low"]
    b_high = intervals["ci_bayes_high"]

    # Plot histogram with interval markers
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=10, density=True, alpha=0.6, edgecolor="black")
    plt.axvline(mle, color="black", linestyle="--", label="MLE = max(data)")
    # Frequentist CI shading
    plt.axvspan(f_low, f_high, color="red", alpha=0.2, label="Freq. CI")
    # Bayesian CI shading
    plt.axvspan(b_low, b_high, color="blue", alpha=0.2, label="Bayes CI")
    plt.legend()
    plt.title("Endpoint Estimation: Uniform(0, θ)")
    plt.xlabel("Data values")
    plt.ylabel("Density")
    plt.show()


# Example usage
if __name__ == "__main__":
    sample = np.random.uniform(0, 10, size=30)
    ints = uniform_endpoint_estimation(sample)
    print(f"MLE: {ints['mle']}")
    print(f"Frequentist CI: [{ints['ci_freq_low']}, {ints['ci_freq_high']}]")
    print(f"Bayesian CI:   [{ints['ci_bayes_low']}, {ints['ci_bayes_high']}]")
    plot_uniform_endpoint(sample, ints)
