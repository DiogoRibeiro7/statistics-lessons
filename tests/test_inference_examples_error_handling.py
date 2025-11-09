import numpy as np
import pytest

from statistics_lessons.inference_examples.rare_events import (
    poisson_zero_upper_bound,
    binomial_zero_upper_bound,
    bayesian_poisson_zero_bound,
    bayesian_binomial_zero_bound,
)
from statistics_lessons.inference_examples.hierarchical_shrinkage import (
    frequentist_intervals,
)
from statistics_lessons.inference_examples.model_selection import (
    fit_models_and_criteria,
    simulate_model_selection,
)
from statistics_lessons.inference_examples.outlier_robustness import (
    trimmed_mean_ci,
)
from statistics_lessons.inference_examples.uniform_endpoint import (
    uniform_endpoint_estimation,
)
from statistics_lessons.inference_examples.jeffreys_lindley import (
    z_test_pvalue,
    bayes_factor,
    simulate_jl,
)


def test_poisson_zero_upper_bound_alpha():
    with pytest.raises(ValueError):
        poisson_zero_upper_bound(5.0, alpha=1.2)


def test_binomial_zero_upper_bound_alpha():
    with pytest.raises(ValueError):
        binomial_zero_upper_bound(5, alpha=-0.1)


def test_hierarchical_intervals_empty_group():
    with pytest.raises(ValueError):
        frequentist_intervals([[]])


def test_fit_models_and_criteria_shape_mismatch():
    X = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError):
        fit_models_and_criteria(X, y)


def test_simulate_model_selection_invalid():
    with pytest.raises(ValueError):
        simulate_model_selection(n=0, p=3, true_coefs=[1.0, 0.0], sigma=1.0)


def test_trimmed_mean_ci_proportion():
    with pytest.raises(ValueError):
        trimmed_mean_ci([1, 2, 3], proportiontocut=0.6)


def test_uniform_endpoint_negative_data():
    with pytest.raises(ValueError):
        uniform_endpoint_estimation([-1, 2, 3])


def test_uniform_endpoint_empty_data():
    with pytest.raises(ValueError):
        uniform_endpoint_estimation([])


def test_uniform_endpoint_alpha_invalid():
    with pytest.raises(ValueError):
        uniform_endpoint_estimation([1, 2, 3], alpha=0)


def test_z_test_pvalue_empty():
    with pytest.raises(ValueError):
        z_test_pvalue([])


def test_bayes_factor_tau2():
    with pytest.raises(ValueError):
        bayes_factor([1, 2, 3], tau2=0)


def test_simulate_jl_invalid_n():
    with pytest.raises(ValueError):
        simulate_jl(0.1, [10, 0], tau2=1.0, trials=10)


def test_bayesian_poisson_zero_bound_invalid_params():
    with pytest.raises(ValueError):
        bayesian_poisson_zero_bound(-1.0)
    with pytest.raises(ValueError):
        bayesian_poisson_zero_bound(1.0, a0=0)


def test_bayesian_binomial_zero_bound_invalid_params():
    with pytest.raises(ValueError):
        bayesian_binomial_zero_bound(-1)
    with pytest.raises(ValueError):
        bayesian_binomial_zero_bound(1, a0=0)
