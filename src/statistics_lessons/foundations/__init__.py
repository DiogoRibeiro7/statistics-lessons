"""Convenience imports for foundational statistics utilities."""

from .descriptive_stats import (
    plot_histogram,
    summary_statistics,
    median_absolute_deviation,
)
from .probability import empirical_cdf, normal_sample
from .preprocessing import drop_missing, standardize
from .inference import confidence_interval_mean, two_sample_ttest
from .resampling import bootstrap_resample, bootstrap_mean_ci

__all__ = [
    "plot_histogram",
    "summary_statistics",
    "median_absolute_deviation",
    "empirical_cdf",
    "normal_sample",
    "drop_missing",
    "standardize",
    "confidence_interval_mean",
    "two_sample_ttest",
    "bootstrap_resample",
    "bootstrap_mean_ci",
]
