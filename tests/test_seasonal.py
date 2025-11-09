import pandas as pd
import pytest

from statistics_lessons.trend_features.seasonal import decompose_series


def test_decompose_series_non_series():
    with pytest.raises(TypeError):
        decompose_series([1, 2, 3], period=2)  # type: ignore[arg-type]


def test_decompose_series_empty():
    with pytest.raises(ValueError):
        decompose_series(pd.Series([], dtype=float), period=2)


def test_decompose_series_non_numeric():
    with pytest.raises(TypeError):
        decompose_series(pd.Series(["a", "b"]), period=2)


def test_decompose_series_bad_period():
    series = pd.Series(range(10))
    with pytest.raises(ValueError):
        decompose_series(series, period=0)

