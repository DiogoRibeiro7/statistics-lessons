from statistics_lessons.datasets import (
    make_linear_regression,
    make_classification,
)
import numpy as np


def test_make_linear_regression_shape_and_reproducible():
    X1, y1 = make_linear_regression(
        n_samples=10, slope=2.0, intercept=0.5, random_state=42
    )
    X2, y2 = make_linear_regression(
        n_samples=10, slope=2.0, intercept=0.5, random_state=42
    )
    assert X1.shape == (10, 1)
    assert y1.shape == (10,)
    assert np.allclose(X1.values, X2.values)
    assert np.allclose(y1.values, y2.values)


def test_make_classification_outputs_binary_labels():
    X, y = make_classification(n_samples=20, random_state=0)
    assert X.shape == (20, 2)
    assert set(y.unique()) <= {0, 1}
