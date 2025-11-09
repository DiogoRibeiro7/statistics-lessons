"""Basic survival analysis utilities for educational purposes."""

from __future__ import annotations

from math import erfc, sqrt
from typing import Tuple

import numpy as np


def kaplan_meier_estimator(
    durations: np.ndarray, event_observed: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate the survival function using the Kaplan-Meier estimator.

    Args:
        durations (numpy.ndarray): Times of events or censoring.
        event_observed (numpy.ndarray): Array indicating if the event was observed
            (``1``) or censored (``0``).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: Event times and corresponding survival
        probabilities.

    Raises:
        ValueError: If inputs are empty, mismatched, or not one-dimensional.
    """
    if durations.ndim != 1 or event_observed.ndim != 1:
        raise ValueError("durations and event_observed must be 1D arrays")
    if durations.size == 0 or event_observed.size == 0:
        raise ValueError("durations and event_observed cannot be empty")
    if durations.shape[0] != event_observed.shape[0]:
        raise ValueError("durations and event_observed must have the same length")

    order = np.argsort(durations)
    times = durations[order]
    events = event_observed[order]

    unique_times = np.unique(times[events == 1])
    if unique_times.size == 0:
        raise ValueError("at least one observed event is required")

    n = len(times)
    survival: list[float] = []
    s = 1.0
    for t in unique_times:
        d_i = np.sum((times == t) & (events == 1))
        n_i = np.sum(times >= t)
        s *= 1.0 - d_i / n_i
        survival.append(s)
    return unique_times, np.asarray(survival)


def log_rank_test(
    time_a: np.ndarray,
    event_a: np.ndarray,
    time_b: np.ndarray,
    event_b: np.ndarray,
) -> Tuple[float, float]:
    """Perform the log-rank test comparing two survival distributions.

    Args:
        time_a (numpy.ndarray): Event or censoring times for group A.
        event_a (numpy.ndarray): Event indicators for group A.
        time_b (numpy.ndarray): Event or censoring times for group B.
        event_b (numpy.ndarray): Event indicators for group B.

    Returns:
        Tuple[float, float]: Test statistic and p-value based on a chi-squared
        distribution with one degree of freedom.

    Raises:
        ValueError: If input arrays are empty or mismatched.
    """
    for arr in (time_a, event_a, time_b, event_b):
        if arr.ndim != 1:
            raise ValueError("all inputs must be 1D arrays")
        if arr.size == 0:
            raise ValueError("input arrays cannot be empty")
    if time_a.shape[0] != event_a.shape[0] or time_b.shape[0] != event_b.shape[0]:
        raise ValueError("time and event arrays must match in length for each group")

    all_times = np.unique(np.concatenate([time_a[event_a == 1], time_b[event_b == 1]]))
    if all_times.size == 0:
        raise ValueError("at least one observed event is required")

    oe_sum = 0.0
    var_sum = 0.0
    for t in all_times:
        n_a = np.sum(time_a >= t)
        n_b = np.sum(time_b >= t)
        d_a = np.sum((time_a == t) & (event_a == 1))
        d_b = np.sum((time_b == t) & (event_b == 1))
        d = d_a + d_b
        n = n_a + n_b
        if n == 0:
            continue
        expected_a = n_a / n * d
        oe = d_a - expected_a
        if n > 1:
            var = (n_a * n_b * d * (n - d)) / (n**2 * (n - 1))
        else:
            var = 0.0
        oe_sum += oe
        var_sum += var

    if var_sum == 0:
        raise ValueError("variance is zero; log-rank test cannot be computed")

    statistic = (oe_sum**2) / var_sum
    p_value = erfc(sqrt(statistic / 2.0))
    return statistic, p_value


def fit_cox_proportional_hazards(
    x: np.ndarray,
    durations: np.ndarray,
    event_observed: np.ndarray,
    learning_rate: float = 0.01,
    n_iter: int = 1000,
) -> np.ndarray:
    """Fit a basic Cox proportional hazards model using gradient ascent.

    Args:
        x (numpy.ndarray): Feature matrix of shape ``(n_samples, n_features)``.
        durations (numpy.ndarray): Event or censoring times.
        event_observed (numpy.ndarray): Event indicators (``1`` for event,
            ``0`` for censored).
        learning_rate (float, optional): Step size for gradient updates.
        n_iter (int, optional): Number of iterations. Defaults to ``1000``.

    Returns:
        numpy.ndarray: Estimated coefficient vector.

    Raises:
        ValueError: If inputs are invalid or mismatched.
    """
    if durations.ndim != 1 or event_observed.ndim != 1:
        raise ValueError("durations and event_observed must be 1D arrays")
    if x.ndim == 1:
        x_mat = x.reshape(-1, 1)
    elif x.ndim == 2:
        x_mat = x
    else:
        raise ValueError("x must be a 1D or 2D array")
    if x_mat.shape[0] != durations.shape[0] or durations.shape[0] != event_observed.shape[0]:
        raise ValueError("x, durations, and event_observed must have the same number of rows")
    if durations.size == 0:
        raise ValueError("input arrays cannot be empty")

    n_samples, n_features = x_mat.shape
    beta = np.zeros(n_features)

    for _ in range(n_iter):
        gradient = np.zeros_like(beta)
        for i in range(n_samples):
            if event_observed[i] == 1:
                risk = durations >= durations[i]
                x_risk = x_mat[risk]
                exp_term = np.exp(x_risk @ beta)
                gradient += x_mat[i] - (exp_term[:, None] * x_risk).sum(axis=0) / exp_term.sum()
        beta += learning_rate * gradient / n_samples
    return beta


__all__ = [
    "kaplan_meier_estimator",
    "log_rank_test",
    "fit_cox_proportional_hazards",
]
