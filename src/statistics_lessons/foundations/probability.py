"""Utilities for working with probability distributions."""

from typing import Sequence

import numpy as np
import pandas as pd


def normal_sample(mean: float, std_dev: float, size: int) -> np.ndarray:
    """Generate a sample from a normal distribution.

    Args:
        mean (float): Mean of the distribution.
        std_dev (float): Standard deviation of the distribution.
        size (int): Number of points to sample.

    Returns:
        numpy.ndarray: Array of sampled values.
    """
    if std_dev <= 0:
        raise ValueError("std_dev must be positive")
    if not isinstance(size, int) or size <= 0:
        raise ValueError("size must be a positive integer")

    # NumPy handles random sampling efficiently
    return np.random.normal(loc=mean, scale=std_dev, size=size)


def empirical_cdf(series: pd.Series) -> pd.DataFrame:
    """Compute the empirical cumulative distribution function (ECDF)."""

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series must not be empty")
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("series must be numeric")

    # Sort the values to compute cumulative probabilities
    sorted_vals = np.sort(series.values)
    # Normalized ranks give us the ECDF
    ecdf = np.arange(1, len(sorted_vals) + 1) / float(len(sorted_vals))

    return pd.DataFrame({"value": sorted_vals, "ecdf": ecdf})


if __name__ == "__main__":
    # Example manual test
    sample = normal_sample(0.0, 1.0, 1000)
    df_ecdf = empirical_cdf(pd.Series(sample))
    print(df_ecdf.head())
