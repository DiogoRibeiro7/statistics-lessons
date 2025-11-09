import numpy as np
import pandas as pd
import pytest

from statistics_lessons.foundations.inference import confidence_interval_mean, two_sample_ttest
from statistics_lessons.foundations.probability import normal_sample, empirical_cdf


def test_confidence_interval_mean_invalid_type():
    with pytest.raises(TypeError):
        confidence_interval_mean([1, 2, 3])


def test_confidence_interval_mean_empty():
    with pytest.raises(ValueError):
        confidence_interval_mean(np.array([]))


def test_confidence_interval_mean_bad_confidence():
    with pytest.raises(ValueError):
        confidence_interval_mean(np.array([1, 2, 3]), confidence=1.5)


def test_two_sample_ttest_non_numeric():
    a = np.array(["a", "b"], dtype=object)
    b = np.array([1.0, 2.0])
    with pytest.raises(TypeError):
        two_sample_ttest(a, b)


def test_two_sample_ttest_empty():
    with pytest.raises(ValueError):
        two_sample_ttest(np.array([]), np.array([1.0]))


def test_normal_sample_invalid_std():
    with pytest.raises(ValueError):
        normal_sample(0.0, -1.0, 10)


def test_normal_sample_invalid_size():
    with pytest.raises(ValueError):
        normal_sample(0.0, 1.0, 0)


def test_empirical_cdf_non_series():
    with pytest.raises(TypeError):
        empirical_cdf([1, 2, 3])


def test_empirical_cdf_empty():
    with pytest.raises(ValueError):
        empirical_cdf(pd.Series([], dtype=float))


def test_empirical_cdf_non_numeric():
    with pytest.raises(TypeError):
        empirical_cdf(pd.Series(["a", "b"]))

