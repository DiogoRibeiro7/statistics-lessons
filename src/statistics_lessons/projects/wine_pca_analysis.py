"""Wine dataset PCA analysis using scikit-learn."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .data_loaders import load_wine_data


def wine_pca_example(n_components: float = 0.95, plot: bool = False) -> pd.DataFrame:
    """Run PCA on the wine dataset and return explained variance.

    Parameters
    ----------
    n_components : float, default 0.95
        Fraction of variance to retain when selecting principal components.
    plot : bool, default False
        If ``True``, display a bar chart of explained variance ratios.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing component numbers and explained variance ratios.
    """
    X, _ = load_wine_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    result = pd.DataFrame(
        {
            "component": range(1, pca.n_components_ + 1),
            "explained_variance": pca.explained_variance_ratio_,
        }
    )

    if plot:
        import matplotlib.pyplot as plt

        ax = result.plot(x="component", y="explained_variance", kind="bar", legend=False)
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("Wine PCA Explained Variance")
        plt.tight_layout()
        plt.show()

    return result


if __name__ == "__main__":
    print(wine_pca_example())
