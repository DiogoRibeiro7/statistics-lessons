"""Utilities to compute robust trend features on time series."""

import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from scipy.stats import kendalltau

from typing import Tuple


def compute_theil_sen_slope(window_values: np.ndarray) -> float:
    """Estimate slope using Theil–Sen.

    Args:
        window_values (np.ndarray): 1D array of values in the rolling window.

    Returns:
        float: Estimated slope representing trend per unit time.
    """
    n = len(window_values)
    # Use integer positions as the time index for regression
    X = np.arange(n).reshape(-1, 1)
    model = TheilSenRegressor()
    # Fit model to obtain robust slope
    model.fit(X, window_values)
    return float(model.coef_[0])


def compute_mann_kendall_pvalue(window_values: np.ndarray) -> float:
    """Calculate p-value for a monotonic trend.

    Uses the Mann–Kendall approach via ``scipy.stats.kendalltau``.

    Args:
        window_values (np.ndarray): 1D array of values in the rolling window.

    Returns:
        float: Two-sided p-value for the presence of a monotonic trend.
    """
    # Use kendalltau as a proxy for Mann-Kendall
    # Time indices for the Kendall correlation
    times = np.arange(len(window_values))
    # ``kendalltau`` returns the correlation coefficient and p-value
    tau, p_value = kendalltau(times, window_values, nan_policy='omit')
    return float(p_value)


def trend_features(series: pd.Series, window: int) -> pd.DataFrame:
    """Compute rolling trend statistics.

    For each rolling window the function calculates a robust trend slope using
    the Theil–Sen estimator and assesses its significance with a
    Mann–Kendall-based p-value.

    Args:
        series (pd.Series): Time-indexed data to analyze.
        window (int): Size of the rolling window.

    Returns:
        pd.DataFrame: DataFrame containing ``theil_sen_slope`` and ``mk_pvalue``
        columns aligned with ``series``.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series cannot be empty")
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("series must contain numeric data")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    if window > len(series):
        raise ValueError("window cannot exceed series length")
    rolling = series.rolling(window=window, min_periods=window)

    slopes = rolling.apply(compute_theil_sen_slope, raw=True)
    pvalues = rolling.apply(compute_mann_kendall_pvalue, raw=True)

    return pd.DataFrame(
        {
            "theil_sen_slope": slopes,
            "mk_pvalue": pvalues,
        }
    )


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a simple rolling mean.

    Args:
        series (pd.Series): Series of numeric values.
        window (int): Length of the moving window.

    Returns:
        pd.Series: Rolling mean aligned with the original index.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series cannot be empty")
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("series must contain numeric data")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    # ``rolling`` automatically handles window boundaries and NaNs
    return series.rolling(window=window).mean()


def rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling standard deviation.

    Args:
        series (pd.Series): Series of numeric values.
        window (int): Length of the moving window.

    Returns:
        pd.Series: Rolling standard deviation aligned with ``series``.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series cannot be empty")
    if not np.issubdtype(series.dtype, np.number):
        raise TypeError("series must contain numeric data")
    if window <= 0:
        raise ValueError("window must be a positive integer")
    # ``rolling`` with ``std`` gives a measure of variability over time
    return series.rolling(window=window).std()


def detect_trend(series: pd.Series, window: int, alpha: float = 0.05) -> pd.Series:
    """Flag significant monotonic trends in a rolling window.

    This uses :func:`trend_features` internally and returns ``True`` for
    windows where the Mann--Kendall p-value is below ``alpha``.

    Args:
        series (pd.Series): Time-indexed data to analyze.
        window (int): Size of the rolling window.
        alpha (float, optional): Significance threshold. Defaults to ``0.05``.

    Returns:
        pd.Series: Boolean series indicating significant trends.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    # Compute slope and p-value for each window
    features = trend_features(series, window)
    # Determine if the p-value indicates significance
    return features["mk_pvalue"] < alpha


if __name__ == "__main__":
    # Example usage for manual testing
    # Generate sample data: cumulative sum of random noise
    dates = pd.date_range(start="2025-01-01", periods=100, freq="D")
    data = pd.Series(np.random.randn(100).cumsum(), index=dates)

    # Compute 21-day rolling trend features
    df_features = trend_features(data, window=21)
    print(df_features.head(25))
