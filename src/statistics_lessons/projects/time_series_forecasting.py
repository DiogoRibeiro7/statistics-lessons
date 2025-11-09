"""Simple time series forecasting example using the sunspots dataset."""
from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

from statistics_lessons.trend_features.forecasting import (
    arima_forecast,
    exponential_smoothing_forecast,
)
from .data_loaders import load_sunspots


def sunspots_forecasting_example(
    order: tuple[int, int, int] = (2, 0, 2),
    steps: int = 5,
    plot: bool = False,
) -> float:
    """Forecast sunspots with ARIMA and return RMSE on a holdout set.

    Parameters
    ----------
    order : tuple[int, int, int], default (2, 0, 2)
        ARIMA ``(p, d, q)`` order.
    steps : int, default 5
        Number of future periods to forecast.
    plot : bool, default False
        If ``True``, display a plot comparing the forecast to the actual
        observations.
    """
    series = load_sunspots()
    train = series[:-steps]
    test = series[-steps:]
    forecast = arima_forecast(train, order, steps)
    rmse = sqrt(mean_squared_error(test, forecast))

    if plot:
        import matplotlib.pyplot as plt

        ax = series.plot(label="actual")
        pd.Series(forecast, index=test.index).plot(ax=ax, label="forecast")
        ax.legend()
        ax.set_title("Sunspots Forecast (ARIMA)")
        ax.set_ylabel("Sunspots")
        plt.tight_layout()
        plt.show()

    return rmse


def sunspots_exp_smoothing_example(
    trend: Optional[str] = "add",
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
    steps: int = 5,
    plot: bool = False,
) -> float:
    """Forecast sunspots with exponential smoothing and return RMSE.

    Parameters
    ----------
    trend, seasonal, seasonal_periods
        Model specification passed to
        :func:`exponential_smoothing_forecast`.
    steps : int, default 5
        Number of periods to forecast.
    plot : bool, default False
        If ``True``, display a plot comparing the forecast to the actual
        observations.
    """
    series = load_sunspots()
    train = series[:-steps]
    test = series[-steps:]
    forecast = exponential_smoothing_forecast(
        train,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        steps=steps,
    )
    rmse = sqrt(mean_squared_error(test, forecast))

    if plot:
        import matplotlib.pyplot as plt

        ax = series.plot(label="actual")
        pd.Series(forecast, index=test.index).plot(ax=ax, label="forecast")
        ax.legend()
        ax.set_title("Sunspots Forecast (Exp. Smoothing)")
        ax.set_ylabel("Sunspots")
        plt.tight_layout()
        plt.show()

    return rmse
