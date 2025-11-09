"""Gaussian Process regression utilities for educational purposes."""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, WhiteKernel


def fit_gaussian_process(
    x: np.ndarray,
    y: np.ndarray,
    kernel: Optional[Kernel] = None,
    alpha: float = 1e-10,
) -> GaussianProcessRegressor:
    """Fit a Gaussian Process regression model.

    Args:
        x (numpy.ndarray): 1D or 2D array of predictor values.
        y (numpy.ndarray): 1D array of response values.
        kernel (Optional[Kernel], optional): Kernel to use. Defaults to
            ``RBF(1.0) + WhiteKernel(1.0)``.
        alpha (float, optional): Value added to the diagonal of the kernel
            matrix during fitting. Defaults to ``1e-10``.

    Returns:
        GaussianProcessRegressor: Trained Gaussian Process model.

    Raises:
        ValueError: If ``x`` and ``y`` have incompatible shapes or are empty.
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x
    if x_reshaped.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")

    if kernel is None:
        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)
    model.fit(x_reshaped, y)
    return model


def optimize_kernel_hyperparameters(
    x: np.ndarray,
    y: np.ndarray,
    kernel: Kernel,
    n_restarts: int = 5,
    alpha: float = 1e-10,
) -> GaussianProcessRegressor:
    """Optimize kernel hyperparameters for a Gaussian Process model.

    Args:
        x (numpy.ndarray): 1D or 2D array of predictor values.
        y (numpy.ndarray): 1D array of response values.
        kernel (Kernel): Initial kernel with hyperparameters to optimize.
        n_restarts (int, optional): Number of optimizer restarts. Defaults to ``5``.
        alpha (float, optional): Value added to the diagonal of the kernel
            matrix during fitting. Defaults to ``1e-10``.

    Returns:
        GaussianProcessRegressor: Model fitted with optimized kernel
        hyperparameters.

    Raises:
        ValueError: If ``n_restarts`` is negative or input arrays are invalid.
    """
    if n_restarts < 0:
        raise ValueError("n_restarts must be non-negative")
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    x_reshaped = x.reshape(-1, 1) if x.ndim == 1 else x
    if x_reshaped.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")

    model = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=n_restarts,
        alpha=alpha,
        normalize_y=True,
    )
    model.fit(x_reshaped, y)
    return model


def plot_gaussian_process(
    model: GaussianProcessRegressor,
    x: np.ndarray,
    y: np.ndarray,
    x_pred: np.ndarray,
) -> Figure:
    """Plot Gaussian Process predictions with uncertainty bands.

    This function assumes a one-dimensional input space for visualization.

    Args:
        model (GaussianProcessRegressor): Trained Gaussian Process model.
        x (numpy.ndarray): 1D array of training predictor values.
        y (numpy.ndarray): 1D array of training response values.
        x_pred (numpy.ndarray): 1D array of predictor values for prediction.

    Returns:
        matplotlib.figure.Figure: Figure containing the plot.

    Raises:
        ValueError: If the model is unfitted or input arrays have incorrect
            dimensions.
    """
    if not hasattr(model, "X_train_"):
        raise ValueError("model must be fitted before plotting")
    if x.ndim != 1 or y.ndim != 1 or x_pred.ndim != 1:
        raise ValueError("x, y, and x_pred must be 1D arrays")
    if x.size == 0 or y.size == 0 or x_pred.size == 0:
        raise ValueError("input arrays cannot be empty")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")

    x_pred_reshaped = x_pred.reshape(-1, 1)
    y_mean, y_std = model.predict(x_pred_reshaped, return_std=True)

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black", label="Observations")
    ax.plot(x_pred, y_mean, color="blue", label="Prediction")
    ax.fill_between(
        x_pred,
        y_mean - 2 * y_std,
        y_mean + 2 * y_std,
        color="blue",
        alpha=0.2,
        label="Uncertainty",
    )
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig


__all__ = [
    "fit_gaussian_process",
    "optimize_kernel_hyperparameters",
    "plot_gaussian_process",
]
