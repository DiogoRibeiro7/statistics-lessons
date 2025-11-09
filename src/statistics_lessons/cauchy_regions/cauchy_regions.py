"""Utilities for estimating the Cauchy location parameter.

The functions in this module illustrate likelihood-based and Bayesian
approaches for interval estimation. They are written with teaching and
demonstration purposes in mind and favor clarity over efficiency.
"""

from typing import Sequence, Tuple, Dict, Optional, Callable

import numpy as np
from scipy.stats import chi2, cauchy
import matplotlib.pyplot as plt


def cauchy_likelihood(theta: float, data: Sequence[float]) -> float:
    """Compute the joint likelihood for a Cauchy sample.

    Args:
        theta: Location parameter of the Cauchy distribution.
        data: Observed samples.

    Returns:
        Joint likelihood of ``data`` given ``theta``.
    """
    arr = np.asarray(data, dtype=float)
    dens = 1 / (np.pi * (1 + (arr - theta) ** 2))
    return float(np.prod(dens))


def compute_likelihood_grid(
    data: Sequence[float],
    window: float = 10,
    n_points: int = 10001,
    grid: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the likelihood across a grid of location values.

    Args:
        data: Observed samples.
        window: Range on either side of the sample median for the grid.
        n_points: Number of points in the grid.
        grid: Optional array of ``theta`` values. If ``None`` a symmetric grid
            is generated around the sample median.

    Returns:
        Tuple containing the ``theta`` grid and the corresponding likelihood
        evaluations.
    """
    if grid is None:
        # A symmetric grid around the sample median centers the search
        # on a robust location estimate.
        median = np.median(data)
        grid = np.linspace(median - window, median + window, n_points)
    # Evaluate the likelihood at each grid point.
    lik = np.array([cauchy_likelihood(t, data) for t in grid])
    return grid, lik


def mle_theta(grid: np.ndarray, lik_vals: np.ndarray) -> float:
    """Find the maximum-likelihood estimate from a grid.

    Args:
        grid: Candidate ``theta`` values.
        lik_vals: Likelihood evaluated at each point in ``grid``.

    Returns:
        Value of ``theta`` that maximizes the likelihood.
    """
    return float(grid[np.argmax(lik_vals)])


def likelihood_ratio_intervals(
    grid: np.ndarray, lik_vals: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """Construct likelihood-ratio confidence intervals.

    Args:
        grid: Candidate ``theta`` values.
        lik_vals: Likelihood evaluated at each point in ``grid``.
        alpha: Significance level for the interval.

    Returns:
        Array of interval endpoints ``[a1, b1, a2, b2, ...]``.
    """
    Lmax = np.max(lik_vals)
    # For a single parameter, -2 log(L(theta)/Lmax) asymptotically
    # follows chi^2_1.  Rearranging yields the likelihood threshold
    # L(theta) >= Lmax * exp(-chi2_{1,1-alpha}/2).
    thresh = np.exp(-chi2.ppf(1 - alpha, df=1) / 2)
    mask = lik_vals / Lmax >= thresh
    intervals = []
    in_seg = False
    for i, ok in enumerate(mask):
        if ok and not in_seg:
            start = grid[i]
            in_seg = True
        if not ok and in_seg:
            end = grid[i - 1]
            intervals.append((start, end))
            in_seg = False
    if in_seg:
        intervals.append((start, grid[-1]))
    return np.array([pt for seg in intervals for pt in seg])


def credible_region(
    grid: np.ndarray,
    lik_vals: np.ndarray,
    alpha: float = 0.05,
    prior: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Compute a highest posterior density region.

    Args:
        grid: Candidate ``theta`` values.
        lik_vals: Likelihood evaluated at each point in ``grid``.
        alpha: Credible level; ``1 - alpha`` of the posterior mass is included.
        prior: Function mapping ``grid`` to prior density values. If ``None``
            a flat prior is assumed.

    Returns:
        Array of interval endpoints forming the HPD region.
    """
    if prior is None:
        prior_vals = np.ones_like(grid)
    else:
        prior_vals = prior(grid)
    # Posterior is proportional to likelihood * prior.
    unnorm = lik_vals * prior_vals
    # Normalize to integrate to one using the trapezoid rule.
    post = unnorm / np.trapz(unnorm, grid)
    # Highest posterior density (HPD) region: sort grid points by
    # posterior density and include them until cumulative mass
    # reaches 1 - alpha.  The posterior value at the boundary gives
    # the density threshold for the HPD set.
    idx = np.argsort(post)[::-1]
    cum = 0.0
    thresh = 0.0
    dx = grid[1] - grid[0]
    for i in idx:
        cum += post[i] * dx
        if cum >= 1 - alpha:
            thresh = post[i]
            break
    mask = post >= thresh
    intervals = []
    in_seg = False
    for i, ok in enumerate(mask):
        if ok and not in_seg:
            start = grid[i]
            in_seg = True
        if not ok and in_seg:
            end = grid[i - 1]
            intervals.append((start, end))
            in_seg = False
    if in_seg:
        intervals.append((start, grid[-1]))
    return np.array([pt for seg in intervals for pt in seg])


def compute_regions(
    data: Sequence[float],
    alpha: float = 0.05,
    window: float = 10,
    n_points: int = 10001,
    prior: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Compute point and interval estimates for Cauchy data.

    Args:
        data: Observed samples.
        alpha: Significance level for intervals.
        window: Range on either side of the sample median for the grid.
        n_points: Number of points in the grid.
        prior: Prior density function used for the credible interval.

    Returns:
        Dictionary containing the grid, likelihood values, maximum-likelihood
        estimate, likelihood-ratio intervals, and credible intervals.
    """
    grid, lik = compute_likelihood_grid(data, window, n_points)
    mle = mle_theta(grid, lik)
    # Construct both frequentist (LR) and Bayesian (HPD) intervals
    # from the common likelihood grid.
    lr = likelihood_ratio_intervals(grid, lik, alpha)
    cred = credible_region(grid, lik, alpha, prior)
    return {
        "grid": grid,
        "likelihood": lik,
        "mle": mle,
        "lr_intervals": lr,
        "credible_intervals": cred,
    }


def plot_regions(
    data: Sequence[float],
    results: Dict[str, np.ndarray],
    figsize: Tuple[int, int] = (8, 5),
) -> None:
    """Plot likelihood and posterior intervals.

    Args:
        data: Sample used to compute the regions.
        results: Output dictionary from :func:`compute_regions`.
        figsize: Size of the figure in inches.
    """
    grid = results["grid"]
    lik = results["likelihood"]
    mle = results["mle"]
    lr = results["lr_intervals"]
    cred = results["credible_intervals"]
    # Under a flat prior the posterior is proportional to the likelihood.
    post = lik / np.trapz(lik, grid)

    plt.figure(figsize=figsize)
    # Likelihood
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(grid, lik)
    ax1.axvline(mle, linestyle="--")
    for i in range(0, len(lr), 2):
        ax1.axvspan(lr[i], lr[i + 1], color="gray", alpha=0.3)
    ax1.set_title("Likelihood and LR Confidence Intervals")
    # Posterior
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(grid, post)
    for i in range(0, len(cred), 2):
        ax2.axvspan(cred[i], cred[i + 1], color="gray", alpha=0.3)
    ax2.set_title("Posterior Density and Credible Regions")
    plt.tight_layout()
    plt.show()


def monte_carlo_coverage(
    true_theta: float,
    n: int,
    alpha: float = 0.05,
    trials: int = 10000,
    window: float = 10,
    n_points: int = 5001,
    method: str = "lr",
    prior: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Estimate coverage probability of interval procedures.

    Args:
        true_theta: True location parameter used for simulation.
        n: Sample size per simulation.
        alpha: Significance or credible level for intervals.
        trials: Number of Monte Carlo repetitions.
        window: Range on either side of the median for the likelihood grid.
        n_points: Number of grid points.
        method: ``"lr"`` for likelihood-ratio intervals or ``"credible"`` for
            Bayesian credible intervals.
        prior: Prior density function when ``method`` is ``"credible"``.

    Returns:
        Estimated coverage probability across the simulations.
    """
    count = 0
    for _ in range(trials):
        # Generate a fresh sample and compute the desired interval.
        sample = cauchy.rvs(loc=true_theta, size=n)
        res = compute_regions(sample, alpha, window, n_points, prior)
        if method == "lr":
            ints = res["lr_intervals"]
        else:
            ints = res["credible_intervals"]
        # Coverage: true parameter lies inside at least one interval.
        for i in range(0, len(ints), 2):
            a, b = ints[i], ints[i + 1]
            if a <= true_theta <= b:
                count += 1
                break
    return count / trials


# Example usage
if __name__ == "__main__":
    data = [0.5, -1.2, 2.0]
    res = compute_regions(data)
    plot_regions(data, res)
    # Coverage check for n=3, true_theta=0
    cov = monte_carlo_coverage(0.0, 3, trials=2000)
    print(f"Estimated LR coverage: {cov:.3f}")
