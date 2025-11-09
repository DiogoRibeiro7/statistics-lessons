from typing import Sequence, Tuple, Dict, Optional
import numpy as np
from scipy.stats import trim_mean, t
import matplotlib.pyplot as plt
from statsmodels.robust.scale import huber


def _validate_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")


def _validate_data(data: Sequence[float]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError("data must be numeric.")
    return arr


def trimmed_mean_ci(
    data: Sequence[float],
    proportiontocut: float = 0.1,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute trimmed mean and bootstrap CI.

    :param data: Observations
    :param proportiontocut: Fraction to trim on each side
    :param alpha: Significance level
    :param n_bootstrap: Number of bootstrap samples
    :param random_state: RNG seed
    :return: (trimmed_mean, ci_low, ci_high)
    """
    if not 0 <= proportiontocut < 0.5:
        raise ValueError("proportiontocut must be in [0, 0.5).")
    _validate_alpha(alpha)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    arr = _validate_data(data)
    tm = trim_mean(arr, proportiontocut)
    rng = np.random.default_rng(random_state)
    boots = []
    n = arr.size
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boots.append(trim_mean(sample, proportiontocut))
    low = np.percentile(boots, 100 * (alpha / 2))
    high = np.percentile(boots, 100 * (1 - alpha / 2))
    return tm, low, high


def huber_ci(
    data: Sequence[float],
    c: float = 1.345,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute Huber M-estimator location and bootstrap CI.

    :param data: Observations
    :param c: Tuning constant for Huber
    :param alpha: Significance level
    :param n_bootstrap: Number of bootstrap samples
    :param random_state: RNG seed
    :return: (huber_loc, ci_low, ci_high)
    """
    if c <= 0:
        raise ValueError("c must be positive.")
    _validate_alpha(alpha)
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")
    arr = _validate_data(data)
    loc, _ = huber(arr, c=c)
    rng = np.random.default_rng(random_state)
    boots = []
    n = arr.size
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        loc_bs, _ = huber(sample, c=c)
        boots.append(loc_bs)
    low = np.percentile(boots, 100 * (alpha / 2))
    high = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(loc), low, high


def bayesian_student_t_posterior(
    data: Sequence[float],
    df: int = 3,
    window: float = 5,
    n_points: int = 2001,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Compute posterior density on theta under Student-t likelihood and flat prior.

    :param data: Observations
    :param df: Degrees of freedom for t-likelihood
    :param window: Range around data median for grid
    :param n_points: Number of grid points
    :param alpha: Significance level
    :return: (theta_grid, posterior, hpd_low, hpd_intervals)
    """
    if df <= 0:
        raise ValueError("df must be positive.")
    if window <= 0:
        raise ValueError("window must be positive.")
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1.")
    _validate_alpha(alpha)
    arr = _validate_data(data)
    median = np.median(arr)
    grid = np.linspace(median - window, median + window, n_points)
    # Likelihood under t
    like = np.prod([t.pdf(arr - theta, df=df) for theta in grid], axis=1)
    # Posterior under flat prior
    post = like / np.trapz(like, grid)
    # HPD credible interval
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
    # find contiguous intervals
    intervals = []
    in_seg = False
    for i, ok in enumerate(mask):
        if ok and not in_seg:
            start = grid[i]; in_seg = True
        elif not ok and in_seg:
            end = grid[i-1]; intervals.append((start, end)); in_seg = False
    if in_seg:
        intervals.append((start, grid[-1]))
    # Flatten
    hpd = np.array([pt for seg in intervals for pt in seg])
    return grid, post, float(grid[np.argmax(post)]), hpd


def plot_outlier_robustness(
    data: Sequence[float],
    trim_res: Tuple[float, float, float],
    huber_res: Tuple[float, float, float],
    post_res: Tuple[np.ndarray, np.ndarray, float, np.ndarray]
) -> None:
    """
    Plot data, robust estimates, and Bayesian posterior.
    """
    _validate_data(data)
    grid, post, post_mode, hpd = post_res
    tm, tm_low, tm_high = trim_res
    hl, hl_low, hl_high = huber_res

    plt.figure(figsize=(10, 6))
    # Histogram
    plt.hist(data, bins=15, density=True, alpha=0.4, edgecolor='black')
    # Posterior
    plt.twinx()
    plt.plot(grid, post, label='Posterior (Student-t)', lw=2)
    for i in range(0, len(hpd), 2):
        plt.axvspan(hpd[i], hpd[i+1], color='gray', alpha=0.3)
    # Trimmed mean
    plt.axvline(tm, color='red', linestyle='--', label='Trimmed mean')
    plt.axvline(tm_low, color='red', alpha=0.5)
    plt.axvline(tm_high, color='red', alpha=0.5)
    # Huber
    plt.axvline(hl, color='blue', linestyle='--', label='Huber')
    plt.axvline(hl_low, color='blue', alpha=0.5)
    plt.axvline(hl_high, color='blue', alpha=0.5)
    plt.title('Outlier Robustness: Gaussian vs Student-t')
    plt.xlabel('Value')
    plt.ylabel('Density / Posterior')
    plt.legend(loc='upper left')
    plt.show()

# Example usage
if __name__ == '__main__':
    # Simulate normal data with outliers
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(0,1, size=50), rng.normal(0,10, size=5)])
    trim_res = trimmed_mean_ci(data)
    huber_res = huber_ci(data)
    post_res = bayesian_student_t_posterior(data)
    plot_outlier_robustness(data, trim_res, huber_res, post_res)
