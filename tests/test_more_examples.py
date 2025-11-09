import numpy as np
import pytest

from statistics_lessons.ml_models.regression_examples import (
    linear_regression_example,
    logistic_regression_example,
)
from statistics_lessons.inference_examples.uniform_endpoint import (
    uniform_endpoint_estimation,
)
from statistics_lessons.inference_examples.rare_events import (
    poisson_zero_upper_bound,
    binomial_zero_upper_bound,
)


def test_linear_regression_example_estimates_coefficients():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.0 * x + 1.0
    slope, intercept = linear_regression_example(x, y)
    assert np.isclose(slope, 2.0, atol=1e-2)
    assert np.isclose(intercept, 1.0, atol=1e-2)


def test_logistic_regression_example_coefficient_sign():
    x = np.array([-2, -1, -0.5, 0.5, 1, 2])
    y = np.array([0, 0, 0, 1, 1, 1])
    coef, intercept = logistic_regression_example(x, y)
    assert coef > 0
    assert abs(intercept) < 5


def test_uniform_endpoint_estimation_outputs():
    data = [1.0, 2.0, 3.0]
    res = uniform_endpoint_estimation(data, alpha=0.1)
    assert res['mle'] == 3.0
    assert np.isclose(res['ci_freq_high'], 30.0)
    expected_bayes = 3.0 / (0.1 ** (1/3))
    assert np.isclose(res['ci_bayes_high'], expected_bayes)


def test_rare_event_bounds_basic_values():
    po = poisson_zero_upper_bound(10.0, alpha=0.05)
    bi = binomial_zero_upper_bound(20, alpha=0.05)
    assert np.isclose(po, -np.log(0.05) / 10.0)
    assert np.isclose(bi, 1 - 0.05 ** (1/20))


@pytest.mark.parametrize("T", [-1.0, 0.0])
def test_poisson_zero_upper_bound_invalid(T):
    with pytest.raises(ValueError):
        poisson_zero_upper_bound(T)


@pytest.mark.parametrize("n", [0, -1])
def test_binomial_zero_upper_bound_invalid(n):
    with pytest.raises(ValueError):
        binomial_zero_upper_bound(n)
