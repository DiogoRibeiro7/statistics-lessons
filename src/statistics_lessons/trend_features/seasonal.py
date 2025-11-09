"""Seasonal decomposition utilities for time series."""

from typing import Dict

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def decompose_series(series: pd.Series, period: int) -> Dict[str, pd.Series]:
    """Decompose a time series into trend, seasonal, and residual components.

    This function wraps :func:`statsmodels.tsa.seasonal.seasonal_decompose` to
    split a series into its constituent parts. It's a thin convenience layer to
    keep the public API consistent with the rest of the repository.

    Args:
        series (pd.Series): Time-indexed data to decompose.
        period (int): Number of observations per cycle (e.g., 12 for monthly
            data with yearly seasonality).

    Returns:
        Dict[str, pd.Series]: Dictionary containing ``trend``, ``seasonal``, and
        ``resid`` series aligned with the input index.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series must not be empty")
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("series must be numeric")
    if period <= 0:
        raise ValueError("period must be positive")

    # statsmodels requires a `Series` with a date-like index
    decomposition = seasonal_decompose(series, period=period, model="additive")

    return {
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "resid": decomposition.resid,
    }

