"""Convenience exports for time series utilities."""

from .trend_features import (
    compute_mann_kendall_pvalue,
    compute_theil_sen_slope,
    detect_trend,
    rolling_mean,
    rolling_std,
    trend_features,
)
from .seasonal import decompose_series
from .forecasting import (
    arima_forecast,
    exponential_smoothing_forecast,
)

__all__ = [
    "compute_mann_kendall_pvalue",
    "compute_theil_sen_slope",
    "detect_trend",
    "rolling_mean",
    "rolling_std",
    "trend_features",
    "decompose_series",
    "arima_forecast",
    "exponential_smoothing_forecast",
]

