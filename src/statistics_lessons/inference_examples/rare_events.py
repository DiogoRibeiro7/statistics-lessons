"""Bounds for rare event probabilities in Poisson and Binomial models."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, gamma


def _validate_alpha(alpha: float) -> None:
    """Validate the significance level.

    Args:
        alpha: Significance level between 0 and 1.

    Raises:
        ValueError: If ``alpha`` is not within ``(0, 1)``.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def poisson_zero_upper_bound(T: float, alpha: float = 0.05) -> float:
    """Frequentist upper bound for a Poisson rate with zero observed events.

    Args:
        T: Observation time span.
        alpha: Significance level. Defaults to ``0.05``.

    Returns:
        float: Upper bound on the rate ``λ``.

    Raises:
        ValueError: If ``T`` is not positive.
    """

    if T <= 0:
        raise ValueError("Observation time T must be positive.")
    _validate_alpha(alpha)
    return -np.log(alpha) / T


def binomial_zero_upper_bound(n: int, alpha: float = 0.05) -> float:
    """Frequentist upper bound for binomial failure probability.

    Assumes zero failures in ``n`` trials.

    Args:
        n: Number of trials.
        alpha: Significance level. Defaults to ``0.05``.

    Returns:
        float: Upper bound on the probability ``p``.

    Raises:
        ValueError: If ``n`` is not positive.
    """

    if n <= 0:
        raise ValueError("Number of trials n must be positive.")
    _validate_alpha(alpha)
    return 1 - alpha ** (1 / n)


def bayesian_poisson_zero_bound(
    T: float,
    alpha: float = 0.05,
    a0: float = 1.0,
    b0: float = 1.0,
) -> float:
    """Bayesian upper bound for a Poisson rate with zero events.

    Args:
        T: Observation time span.
        alpha: Significance level. Defaults to ``0.05``.
        a0: Shape parameter of the Gamma prior. Defaults to ``1.0``.
        b0: Rate parameter of the Gamma prior. Defaults to ``1.0``.

    Returns:
        float: Upper bound on the rate ``λ``.

    Raises:
        ValueError: If ``T`` is negative.
        ValueError: If ``a0`` or ``b0`` are non-positive.
    """

    if T < 0:
        raise ValueError("Observation time T must be non-negative.")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("Gamma prior parameters a0 and b0 must be positive.")
    _validate_alpha(alpha)
    a_post = a0 + 0
    scale = 1 / (b0 + T)
    return gamma.ppf(1 - alpha, a=a_post, scale=scale)


def bayesian_binomial_zero_bound(
    n: int,
    alpha: float = 0.05,
    a0: float = 1.0,
    b0: float = 1.0,
) -> float:
    """Bayesian upper bound for binomial failure probability.

    Assumes zero failures and a ``Beta(a0, b0)`` prior.

    Args:
        n: Number of trials.
        alpha: Significance level. Defaults to ``0.05``.
        a0: Alpha parameter of the Beta prior. Defaults to ``1.0``.
        b0: Beta parameter of the Beta prior. Defaults to ``1.0``.

    Returns:
        float: Upper bound on the probability ``p``.

    Raises:
        ValueError: If ``n`` is negative.
        ValueError: If ``a0`` or ``b0`` are non-positive.
    """

    if n < 0:
        raise ValueError("Number of trials n must be non-negative.")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("Beta prior parameters a0 and b0 must be positive.")
    _validate_alpha(alpha)
    a_post = a0 + 0
    b_post = b0 + n
    return beta.ppf(1 - alpha, a_post, b_post)


def plot_rare_event_bounds(
    T: float,
    n: int,
    alpha: float = 0.05,
    a0: float = 1.0,
    b0: float = 1.0,
) -> None:
    """Plot frequentist and Bayesian bounds for zero-event scenarios.

    Args:
        T: Observation time span.
        n: Number of trials.
        alpha: Significance level. Defaults to ``0.05``.
        a0: Prior shape parameter for Gamma/Beta priors. Defaults to ``1.0``.
        b0: Prior rate parameter for Gamma/Beta priors. Defaults to ``1.0``.

    Raises:
        ValueError: If ``T`` or ``n`` are non-positive.
        ValueError: If ``a0`` or ``b0`` are non-positive.
    """

    if T <= 0:
        raise ValueError("Observation time T must be positive.")
    if n <= 0:
        raise ValueError("Number of trials n must be positive.")
    if a0 <= 0 or b0 <= 0:
        raise ValueError("Prior parameters a0 and b0 must be positive.")
    _validate_alpha(alpha)
    po_f = poisson_zero_upper_bound(T, alpha)
    bi_f = binomial_zero_upper_bound(n, alpha)
    po_b = bayesian_poisson_zero_bound(T, alpha, a0, b0)
    bi_b = bayesian_binomial_zero_bound(n, alpha, a0, b0)

    labels = [
        "Poisson (freq)",
        "Binomial (freq)",
        "Poisson (Bayes)",
        "Binomial (Bayes)",
    ]
    values = [po_f, bi_f, po_b, bi_b]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Upper bound")
    plt.title("Zero-event Upper Bounds (α={})".format(alpha))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    T = 10.0  # observation time
    n = 20  # trials
    bounds = {
        "poisson_freq": poisson_zero_upper_bound(T),
        "binomial_freq": binomial_zero_upper_bound(n),
        "poisson_bayes": bayesian_poisson_zero_bound(T),
        "binomial_bayes": bayesian_binomial_zero_bound(n),
    }
    for k, v in bounds.items():
        print(f"{k}: {v:.3f}")
    plot_rare_event_bounds(T, n)
