"""Simple forecasting utilities for time series data."""

from typing import Tuple, Optional

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def arima_forecast(series: pd.Series, order: Tuple[int, int, int], steps: int) -> pd.Series:
    """Fit an ARIMA model and produce a forecast.

    Args:
        series (pd.Series): Time-indexed observations to model.
        order (Tuple[int, int, int]): ``(p, d, q)`` ARIMA parameters.
        steps (int): Number of future periods to forecast.

    Returns:
        pd.Series: Forecasted values indexed after the input series.
    """
    # Fit the ARIMA model on the provided series
    model = ARIMA(series, order=order)
    fitted = model.fit()

    # Generate out-of-sample forecast for the requested number of steps
    forecast = fitted.forecast(steps=steps)
    return forecast


def exponential_smoothing_forecast(
    series: pd.Series,
    trend: Optional[str] = None,
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
    steps: int = 1,
) -> pd.Series:
    """Apply Holt-Winters exponential smoothing to produce a forecast.

    Args:
        series (pd.Series): Time series data to model.
        trend (Optional[str], optional): Type of trend component, such as ``'add'`` or ``'mul'``. Defaults to ``None``.
        seasonal (Optional[str], optional): Type of seasonal component. Defaults to ``None``.
        seasonal_periods (Optional[int], optional): Number of periods in a full seasonal cycle. Required if ``seasonal`` is specified.
        steps (int, optional): Number of periods to forecast. Defaults to ``1``.

    Returns:
        pd.Series: Forecasted values for the specified horizon.
    """
    # Initialize the model with optional trend and seasonal components
    model = ExponentialSmoothing(
        series,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
    )
    fitted = model.fit()

    # Forecast the specified number of future steps
    forecast = fitted.forecast(steps)
    return forecast


if __name__ == "__main__":
    # Example usage when running this module directly
    rng = pd.date_range("2025-01-01", periods=50, freq="D")
    toy_series = pd.Series(range(50), index=rng)

    print(arima_forecast(toy_series, order=(1, 1, 0), steps=5))
    print(exponential_smoothing_forecast(toy_series, trend="add", steps=5))
