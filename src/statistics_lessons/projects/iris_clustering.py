"""K-means clustering example on the Iris dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .iris_classification import load_iris_data


def iris_clustering_example(
    n_clusters: int = 3,
    random_state: int = 0,
    plot: bool = False,
) -> pd.DataFrame:
    """Cluster the Iris dataset using k-means and compute silhouette score.

    Parameters
    ----------
    n_clusters : int, default 3
        Number of clusters to form.
    random_state : int, default 0
        Seed for the KMeans algorithm.
    plot : bool, default False
        If ``True``, display a scatter plot of the first two features coloured
        by cluster assignment.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the silhouette score for the clustering.
    """
    X, _ = load_iris_data()
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)
    result = pd.DataFrame({"silhouette_score": [float(score)]})

    if plot:
        import matplotlib.pyplot as plt

        ax = plt.gca()
        ax.scatter(
            X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap="viridis", s=40
        )
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
        ax.set_title("Iris Clusters")
        plt.tight_layout()
        plt.show()

    return result


if __name__ == "__main__":
    print(iris_clustering_example())
