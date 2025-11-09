"""Gaussian Process regression example on a scikit-learn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..ml_models.gaussian_process import (
    optimize_kernel_hyperparameters,
    plot_gaussian_process,
)


def gp_regression_example(
    test_size: float = 0.2,
    random_state: int | None = 0,
    plot: bool = False,
) -> pd.DataFrame:
    """Compare Gaussian Process and linear regression on the diabetes dataset.

    This example uses a single feature of the diabetes dataset for clarity
    and visualisation purposes. It fits both a Gaussian Process regressor
    with hyperparameter optimisation and a standard linear regression model,
    reporting their mean squared errors on a held-out test set.

    Args:
        test_size: Proportion of the dataset to include in the test split.
        random_state: Seed controlling the train/test split.
        plot: If ``True``, display a plot of the Gaussian Process predictions
            with uncertainty bands and overlay test observations.

    Returns:
        pandas.DataFrame: Mean squared error for each model.

    Raises:
        ValueError: If ``test_size`` is not between 0 and 1.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    data = load_diabetes()
    x = data.data[:, 2]  # use BMI feature for 1D input
    y = data.target
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    gp_model = optimize_kernel_hyperparameters(
        x_train, y_train, kernel=kernel, n_restarts=3
    )
    y_pred_gp = gp_model.predict(x_test.reshape(-1, 1))
    mse_gp = mean_squared_error(y_test, y_pred_gp)

    lin_model = LinearRegression().fit(x_train.reshape(-1, 1), y_train)
    y_pred_lin = lin_model.predict(x_test.reshape(-1, 1))
    mse_lin = mean_squared_error(y_test, y_pred_lin)

    results = pd.DataFrame(
        {
            "model": ["gaussian_process", "linear_regression"],
            "mse": [float(mse_gp), float(mse_lin)],
        }
    )

    if plot:
        x_plot = np.linspace(x.min(), x.max(), 100)
        fig = plot_gaussian_process(gp_model, x_train, y_train, x_plot)
        ax = fig.axes[0]
        ax.scatter(x_test, y_test, color="red", label="Test data")
        ax.legend()
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.show()

    return results


if __name__ == "__main__":
    print(gp_regression_example())
