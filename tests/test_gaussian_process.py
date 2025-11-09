import numpy as np
import matplotlib.pyplot as plt
import pytest
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from statistics_lessons.ml_models.gaussian_process import (
    fit_gaussian_process,
    optimize_kernel_hyperparameters,
    plot_gaussian_process,
)


def test_fit_gaussian_process_returns_model() -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20)
    y = np.sin(2 * np.pi * x)
    model = fit_gaussian_process(x, y)
    preds = model.predict(x.reshape(-1, 1))
    assert preds.shape == y.shape


def test_optimize_kernel_changes_parameters() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 15)
    y = np.cos(2 * np.pi * x)
    kernel = RBF(length_scale=0.5) + WhiteKernel(noise_level=0.5)
    model = optimize_kernel_hyperparameters(x, y, kernel, n_restarts=1)
    assert not np.allclose(model.kernel_.theta, kernel.theta)


def test_plot_gaussian_process_returns_figure() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 10)
    y = np.sin(2 * np.pi * x)
    model = fit_gaussian_process(x, y)
    x_pred = np.linspace(0, 1, 50)
    fig = plot_gaussian_process(model, x, y, x_pred)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_fit_gaussian_process_shape_mismatch() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([0.0])
    with pytest.raises(ValueError):
        fit_gaussian_process(x, y)
