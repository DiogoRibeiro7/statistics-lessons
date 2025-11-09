"""Probabilistic programming examples with PyMC."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

try:
    import pymc as pm
except Exception:  # pragma: no cover - handled in tests
    pm = None


def coin_flip_posterior(flips: Sequence[int]) -> float:
    """Compute posterior mean of a coin bias.

    Args:
        flips: Sequence of 0s and 1s representing tails and heads.

    Returns:
        Posterior mean of the probability of heads.

    Raises:
        ImportError: If PyMC is not installed.
    """
    if pm is None:  # pragma: no cover - dependency missing
        raise ImportError("PyMC is required for this function")

    flips = np.asarray(flips)
    with pm.Model():
        p = pm.Beta("p", 1, 1)
        pm.Bernoulli("obs", p=p, observed=flips)
        trace = pm.sample(500, chains=1, progressbar=False, random_seed=0)
    return float(trace.posterior["p"].mean().values)


def bayesian_linear_regression(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[float, float]:
    """Estimate slope and intercept using a Bayesian linear model.

    Args:
        x: Predictor values.
        y: Response values.

    Returns:
        Tuple of posterior means for the slope and intercept.

    Raises:
        ImportError: If PyMC is not installed.
    """
    if pm is None:  # pragma: no cover - dependency missing
        raise ImportError("PyMC is required for this function")

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    with pm.Model():
        slope = pm.Normal("slope", 0, 10)
        intercept = pm.Normal("intercept", 0, 10)
        sigma = pm.HalfNormal("sigma", 1)
        pm.Normal(
            "obs",
            mu=slope * x_arr + intercept,
            sigma=sigma,
            observed=y_arr,
        )
        trace = pm.sample(500, chains=1, progressbar=False, random_seed=0)

    slope_mean = float(trace.posterior["slope"].mean().values)
    intercept_mean = float(trace.posterior["intercept"].mean().values)
    return slope_mean, intercept_mean
