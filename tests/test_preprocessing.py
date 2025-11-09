import pandas as pd
import pytest

from statistics_lessons.foundations.preprocessing import drop_missing, standardize


def test_drop_missing_invalid_columns():
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    with pytest.raises(ValueError):
        drop_missing(df, columns=["c"])


def test_standardize_non_numeric():
    series = pd.Series(["x", "y"])
    with pytest.raises(TypeError):
        standardize(series)
