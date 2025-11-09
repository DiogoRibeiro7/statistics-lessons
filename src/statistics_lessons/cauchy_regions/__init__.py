"""Utilities for analyzing likelihood intervals of the Cauchy distribution."""

from .cauchy_regions import (
    cauchy_likelihood,
    compute_likelihood_grid,
    mle_theta,
    likelihood_ratio_intervals,
    credible_region,
    compute_regions,
    plot_regions,
    monte_carlo_coverage,
)

__all__ = [
    "cauchy_likelihood",
    "compute_likelihood_grid",
    "mle_theta",
    "likelihood_ratio_intervals",
    "credible_region",
    "compute_regions",
    "plot_regions",
    "monte_carlo_coverage",
]
