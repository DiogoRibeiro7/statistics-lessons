"""Descriptive statistics and visualization utilities."""

from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype


def summary_statistics(series: pd.Series) -> Dict[str, float]:
    """Compute basic summary statistics for a series.

    Args:
        series (pd.Series): Numeric data to describe.

    Returns:
        Dict[str, float]: Dictionary with mean, median, and standard deviation.

    Raises:
        ValueError: If ``series`` is empty.
        TypeError: If ``series`` is not numeric.
    """
    if series.empty:
        raise ValueError("Series must not be empty.")
    if not is_numeric_dtype(series):
        raise TypeError("Series must contain numeric values.")

    # Mean provides the central tendency
    mean_val = float(series.mean())
    # Median is robust to outliers
    median_val = float(series.median())
    # Standard deviation measures spread
    std_val = float(series.std())

    return {"mean": mean_val, "median": median_val, "std": std_val}


def median_absolute_deviation(series: pd.Series) -> float:
    """Compute the median absolute deviation (MAD).

    The MAD is a robust measure of statistical dispersion. It is the
    median of the absolute deviations from the series median.

    Args:
        series (pd.Series): Numeric data to describe.

    Returns:
        float: The median absolute deviation of the series.

    Raises:
        ValueError: If ``series`` is empty.
        TypeError: If ``series`` is not numeric.
    """
    if series.empty:
        raise ValueError("Series must not be empty.")
    if not is_numeric_dtype(series):
        raise TypeError("Series must contain numeric values.")

    median_val = series.median()
    deviations = (series - median_val).abs()
    return float(deviations.median())


def plot_histogram(series: pd.Series, bins: int = 20) -> plt.Axes:
    """Plot a histogram of the provided series.

    Args:
        series (pd.Series): Values to plot.
        bins (int, optional): Number of histogram bins. Defaults to 20.

    Returns:
        matplotlib.axes.Axes: Axis object containing the plot.

    Raises:
        ValueError: If ``series`` is empty or ``bins`` is not positive.
        TypeError: If ``series`` is not numeric.
    """
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if series.empty:
        raise ValueError("Series must not be empty.")
    if not is_numeric_dtype(series):
        raise TypeError("Series must contain numeric values.")

    fig, ax = plt.subplots()
    # Create histogram with chosen bin count
    series.plot(kind="hist", bins=bins, ax=ax, alpha=0.7)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution")
    return ax


if __name__ == "__main__":
    # Example usage for manual testing
    sample = pd.Series(range(100))
    stats = summary_statistics(sample)
    print(stats)
    plot_histogram(sample)
    plt.show()
