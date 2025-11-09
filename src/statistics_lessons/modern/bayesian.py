"""Bayesian modeling utilities using PyMC and PyStan."""

from __future__ import annotations

from typing import Sequence

import numpy as np


try:
    import pymc as pm
except Exception:  # pragma: no cover - handled in tests
    pm = None

try:
    import stan
except Exception:  # pragma: no cover - handled in tests
    stan = None


def estimate_normal_mean(data: Sequence[float]) -> float:
    """Estimate the mean of a normal distribution via Bayesian inference.

    Args:
        data: Observed samples assumed to come from a normal distribution
            with unknown mean and unit variance.

    Returns:
        Posterior mean estimate of the normal mean.

    Raises:
        ImportError: If PyMC is not installed.
    """
    if pm is None:  # pragma: no cover - dependency missing
        raise ImportError("PyMC is required for this function")

    data = np.asarray(data)
    with pm.Model():
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("obs", mu=mu, sigma=1, observed=data)
        trace = pm.sample(500, chains=1, progressbar=False, random_seed=0)
    return float(trace.posterior["mu"].mean().values)


def estimate_normal_mean_stan(data: Sequence[float]) -> float:
    """Estimate the mean of a normal distribution via PyStan.

    Args:
        data: Observed samples assumed to come from a normal distribution
            with unknown mean and unit variance.

    Returns:
        Posterior mean estimate of the normal mean.

    Raises:
        ImportError: If PyStan is not installed.
    """
    if stan is None:  # pragma: no cover - dependency missing
        raise ImportError("stan is required for this function")

    data = np.asarray(data)
    model_code = """
    data { int<lower=0> N; vector[N] y; }
    parameters { real mu; }
    model { y ~ normal(mu, 1); }
    """
    posterior = stan.build(
        model_code,
        data={"N": len(data), "y": data},
        random_seed=0,
    )
    fit = posterior.sample(num_chains=1, num_samples=500)
    return float(fit["mu"].mean())
