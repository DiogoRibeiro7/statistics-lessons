"""Example scripts demonstrating inferential statistics concepts."""

from .uniform_endpoint import (
    uniform_endpoint_estimation,
    plot_uniform_endpoint,
)
from .rare_events import (
    poisson_zero_upper_bound,
    binomial_zero_upper_bound,
    bayesian_poisson_zero_bound,
    bayesian_binomial_zero_bound,
)

__all__ = [
    "uniform_endpoint_estimation",
    "plot_uniform_endpoint",
    "poisson_zero_upper_bound",
    "binomial_zero_upper_bound",
    "bayesian_poisson_zero_bound",
    "bayesian_binomial_zero_bound",
]
