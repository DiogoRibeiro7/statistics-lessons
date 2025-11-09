"""Regression models for educational purposes."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression


def linear_regression_example(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit a simple linear regression model.

    Args:
        x (numpy.ndarray): 1D array of predictor values.
        y (numpy.ndarray): 1D array of response values.

    Returns:
        Tuple[float, float]: Fitted slope and intercept.
    """
    # scikit-learn expects 2D input for features
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return slope, intercept


def logistic_regression_example(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Fit a simple logistic regression model.

    Args:
        x (numpy.ndarray): 1D array of predictor values.
        y (numpy.ndarray): Binary response values (0 or 1).

    Returns:
        Tuple[float, float]: Fitted coefficient and intercept.
    """
    # Reshape ``x`` to 2D for scikit-learn
    model = LogisticRegression()
    model.fit(x.reshape(-1, 1), y)
    coef = float(model.coef_[0][0])
    intercept = float(model.intercept_[0])
    return coef, intercept


if __name__ == "__main__":
    # Manual test with synthetic data
    rng = np.random.default_rng(seed=42)
    x_vals = rng.normal(size=100)
    y_vals = 2.0 * x_vals + 1.0 + rng.normal(scale=0.5, size=100)
    slope, intercept = linear_regression_example(x_vals, y_vals)
    print("Linear regression slope:", slope)
    print("Linear regression intercept:", intercept)

    # Generate logistic targets
    y_binary = (x_vals > 0).astype(int)
    coef, intercept_l = logistic_regression_example(x_vals, y_binary)
    print("Logistic regression coef:", coef)
    print("Logistic regression intercept:", intercept_l)
