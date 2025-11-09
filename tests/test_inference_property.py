import numpy as np
import pytest
from hypothesis import given, strategies as st

from statistics_lessons.inference_examples.uniform_endpoint import (
    uniform_endpoint_estimation,
)
from statistics_lessons.inference_examples.rare_events import (
    poisson_zero_upper_bound,
    binomial_zero_upper_bound,
)


@given(
    st.lists(
        st.floats(min_value=0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=50,
    ),
    st.floats(min_value=1e-6, max_value=0.99),
)
def test_uniform_endpoint_properties(data, alpha):
    res = uniform_endpoint_estimation(data, alpha)
    mle = max(data)
    assert res["mle"] == pytest.approx(mle)
    assert res["ci_freq_low"] == pytest.approx(mle)
    assert res["ci_bayes_low"] == pytest.approx(mle)
    assert res["ci_freq_high"] >= res["ci_bayes_high"] >= mle


@given(
    st.floats(min_value=1e-6, max_value=1e6),
    st.floats(min_value=1e-6, max_value=0.99),
)
def test_poisson_zero_upper_bound_formula(T, alpha):
    result = poisson_zero_upper_bound(T, alpha)
    expected = -np.log(alpha) / T
    assert result == pytest.approx(expected)


@given(
    st.integers(min_value=1, max_value=1000),
    st.floats(min_value=1e-6, max_value=0.99),
)
def test_binomial_zero_upper_bound_formula(n, alpha):
    result = binomial_zero_upper_bound(n, alpha)
    expected = 1 - alpha ** (1 / n)
    assert result == pytest.approx(expected)
