import numpy as np
import pytest

from statistics_lessons.foundations.resampling import (
    bootstrap_mean_ci,
    bootstrap_resample,
)


def test_bootstrap_mean_ci_invalid_confidence():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        bootstrap_mean_ci(data, confidence=1.5)


def test_bootstrap_mean_ci_invalid_n_bootstrap():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        bootstrap_mean_ci(data, n_bootstrap=0)


def test_bootstrap_resample_invalid_size():
    data = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        bootstrap_resample(data, size=0)
