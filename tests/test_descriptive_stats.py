import pandas as pd
import pytest

from statistics_lessons.foundations.descriptive_stats import (
    median_absolute_deviation,
    summary_statistics,
)


def test_median_absolute_deviation_simple_case():
    series = pd.Series([1, 2, 3, 4, 5])
    # median is 3, deviations are [2,1,0,1,2], median is 1
    assert median_absolute_deviation(series) == 1.0


def test_median_absolute_deviation_requires_numeric():
    series = pd.Series(["a", "b"])
    with pytest.raises(TypeError):
        median_absolute_deviation(series)


def test_summary_statistics_empty_series():
    series = pd.Series([], dtype=float)
    with pytest.raises(ValueError):
        summary_statistics(series)
