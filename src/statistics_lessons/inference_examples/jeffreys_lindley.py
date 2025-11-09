from typing import Sequence, Dict
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def _validate_data(data: Sequence[float]) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        raise ValueError("data must be non-empty.")
    if not np.isfinite(arr).all():
        raise ValueError("data must be numeric.")
    return arr


def _validate_tau2(tau2: float) -> None:
    if tau2 <= 0:
        raise ValueError("tau2 must be positive.")


def _validate_n_values(n_values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(n_values, dtype=int)
    if arr.size == 0 or np.any(arr <= 0):
        raise ValueError("n_values must contain positive integers.")
    return arr


def z_test_pvalue(data: Sequence[float]) -> float:
    """
    Two-sided z-test p-value for H0: theta=0 assuming known sigma=1.

    :param data: Observed samples
    :return: two-sided p-value
    """
    arr = _validate_data(data)
    n = arr.size
    xbar = arr.mean()
    # standard error for sigma=1
    se = 1 / np.sqrt(n)
    z = xbar / se
    return 2 * (1 - norm.cdf(abs(z)))


def bayes_factor(data: Sequence[float], tau2: float) -> float:
    """
    Compute Bayes factor BF01 for H0: theta=0 vs H1: theta~N(0, tau2), sigma=1.

    :param data: Observed samples
    :param tau2: Prior variance on theta under H1
    :return: BF01 (evidence in favor of H0)
    """
    arr = _validate_data(data)
    _validate_tau2(tau2)
    n = arr.size
    xbar = arr.mean()
    # Log marginal likelihood under H0
    log_m0 = np.sum(norm.logpdf(arr, loc=0, scale=1))
    # Sum of squared deviations around xbar
    ss = np.sum((arr - xbar)**2)
    # Constant term
    c = -n/2 * np.log(2 * np.pi)
    # Log marginal under H1
    log_m1 = c - 0.5 * ss
    var_post = 1/n + tau2
    log_m1 += -0.5 * (np.log(2 * np.pi * var_post) + xbar**2 / var_post)
    # Return BF01
    return float(np.exp(log_m0 - log_m1))


def simulate_jl(
    delta: float,
    n_values: Sequence[int],
    tau2: float,
    trials: int = 5000
) -> Dict[str, np.ndarray]:
    """
    Simulate average p-values and BF01 across sample sizes.

    :param delta: true theta value used in simulation
    :param n_values: Sequence of sample sizes
    :param tau2: Prior variance for H1
    :param trials: Number of replicates per n
    :return: dict with keys 'n', 'p_means', 'bf_means'
    """
    _validate_tau2(tau2)
    if trials <= 0:
        raise ValueError("trials must be positive.")
    n_arr = _validate_n_values(n_values)
    p_means = []
    bf_means = []
    for n in n_arr:
        ps = []
        bfs = []
        for _ in range(trials):
            data = norm.rvs(loc=delta, scale=1, size=n)
            ps.append(z_test_pvalue(data))
            bfs.append(bayes_factor(data, tau2))
        p_means.append(np.mean(ps))
        bf_means.append(np.mean(bfs))
    return {'n': n_arr, 'p_means': np.array(p_means), 'bf_means': np.array(bf_means)}


def plot_jl(
    result: Dict[str, np.ndarray],
    delta: float,
    tau2: float
) -> None:
    """
    Plot simulation results for Jeffreys–Lindley paradox.

    :param result: Output of simulate_jl
    :param delta: true theta used
    :param tau2: Prior variance
    """
    n = result['n']
    p = result['p_means']
    bf = result['bf_means']

    fig, ax1 = plt.subplots()
    ax1.plot(n, p, label='Mean p-value')
    ax1.axhline(0.05, linestyle='--', label='α=0.05')
    ax1.set_xlabel('Sample size n')
    ax1.set_ylabel('p-value')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(n, bf, label='Mean BF01', linestyle='-', marker='o')
    ax2.axhline(1, linestyle='--', label='BF=1')
    ax2.set_ylabel('Bayes factor BF01')
    ax2.legend(loc='upper right')

    plt.title(f"Jeffreys–Lindley: δ={delta}, τ²={tau2}")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == '__main__':
    delta = 0.1
    tau2 = 1.0
    n_vals = np.arange(10, 201, 10)
    res = simulate_jl(delta, n_vals, tau2, trials=2000)
    plot_jl(res, delta, tau2)
