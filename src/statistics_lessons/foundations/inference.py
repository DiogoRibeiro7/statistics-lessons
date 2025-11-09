"""Statistical inference utilities."""

from typing import Tuple

import numpy as np
from scipy import stats


def confidence_interval_mean(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """Calculate a confidence interval for the population mean.

    Args:
        data (numpy.ndarray): Sample observations.
        confidence (float, optional): Confidence level between 0 and 1.
            Defaults to 0.95.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.size == 0:
        raise ValueError("data must not be empty")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must be numeric")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    n = len(data)
    mean_val = float(np.mean(data))
    # Standard error of the mean accounts for sample size
    sem = stats.sem(data)
    # ``stats.t`` provides the appropriate t multiplier
    margin = sem * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
    return mean_val - margin, mean_val + margin


def two_sample_ttest(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Perform a two-sample t-test for independent samples.

    Args:
        a (numpy.ndarray): First sample observations.
        b (numpy.ndarray): Second sample observations.

    Returns:
        Tuple[float, float]: t statistic and two-sided p-value.
    """
    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError("a and b must be numpy.ndarray")
    if a.size == 0 or b.size == 0:
        raise ValueError("a and b must not be empty")
    if not np.issubdtype(a.dtype, np.number) or not np.issubdtype(b.dtype, np.number):
        raise TypeError("a and b must be numeric")

    # ``stats.ttest_ind`` handles unequal variances by default
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
    return float(t_stat), float(p_val)


if __name__ == "__main__":
    # Simple usage example
    rng = np.random.default_rng(seed=123)
    sample1 = rng.normal(loc=5.0, scale=2.0, size=50)
    sample2 = rng.normal(loc=6.0, scale=2.0, size=60)

    ci_low, ci_high = confidence_interval_mean(sample1)
    print("95% CI for mean:", ci_low, ci_high)

    t_stat, p_val = two_sample_ttest(sample1, sample2)
    print("t statistic:", t_stat)
    print("p-value:", p_val)
