import pandas as pd
import numpy as np
import pytest

from statistics_lessons.trend_features.trend_features import (
    detect_trend,
    trend_features,
)


def test_trend_features_linear_slope():
    series = pd.Series(np.arange(10))
    features = trend_features(series, window=5)

    # first window-1 entries should be NaN
    assert features["theil_sen_slope"].isna().sum() == 4
    assert features["mk_pvalue"].isna().sum() == 4

    # slope should capture increasing trend
    assert features["theil_sen_slope"].iloc[-1] == pytest.approx(1.0, rel=1e-6)


def test_trend_features_constant_series_zero_slope():
    series = pd.Series(np.ones(10))
    features = trend_features(series, window=3)

    slopes = features["theil_sen_slope"].dropna()
    assert np.allclose(slopes.values, 0.0)


def test_trend_features_invalid_inputs():
    series = pd.Series(np.arange(5))
    with pytest.raises(TypeError):
        trend_features(np.arange(5), window=2)
    with pytest.raises(ValueError):
        trend_features(series, window=0)
    with pytest.raises(ValueError):
        trend_features(series, window=10)


def test_detect_trend_invalid_alpha():
    series = pd.Series(np.arange(10))
    with pytest.raises(ValueError):
        detect_trend(series, window=3, alpha=1.2)
