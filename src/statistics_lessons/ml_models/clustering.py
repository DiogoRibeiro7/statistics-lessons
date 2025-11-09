"""Clustering algorithms and evaluation helpers."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score


def kmeans_clustering(
    X: np.ndarray, n_clusters: int, random_state: Optional[int] = 0
) -> Tuple[KMeans, np.ndarray]:
    """Cluster data with the k-means algorithm.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        n_clusters: Desired number of clusters.
        random_state: Seed controlling initialization. Defaults to ``0``.

    Returns:
        Tuple[KMeans, np.ndarray]: Fitted model and integer labels per sample.
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def hierarchical_clustering(
    X: np.ndarray, n_clusters: int, linkage: str = "ward"
) -> Tuple[AgglomerativeClustering, np.ndarray]:
    """Apply agglomerative hierarchical clustering.

    Args:
        X: Feature matrix with shape ``(n_samples, n_features)``.
        n_clusters: Number of clusters to form.
        linkage: Linkage criterion, e.g. ``"ward"`` or ``"average"``.

    Returns:
        Tuple[AgglomerativeClustering, np.ndarray]: Trained estimator and labels.
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if n_clusters < 1:
        raise ValueError("n_clusters must be at least 1")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    return model, labels


def dbscan_clustering(
    X: np.ndarray, eps: float = 0.5, min_samples: int = 5
) -> Tuple[DBSCAN, np.ndarray]:
    """Perform density-based spatial clustering of applications with noise.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        eps: Maximum distance between samples for them to be considered neighbors. Defaults to ``0.5``.
        min_samples: Minimum number of points to form a dense region. Defaults to ``5``.

    Returns:
        Tuple[DBSCAN, np.ndarray]: Fitted DBSCAN instance and cluster labels.
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if min_samples < 1:
        raise ValueError("min_samples must be at least 1")
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return model, labels


def elbow_inertia(X: np.ndarray, k_range: Sequence[int]) -> Dict[int, float]:
    """Compute k-means inertia values for the elbow method.

    Args:
        X: Feature matrix with shape ``(n_samples, n_features)``.
        k_range: Sequence of cluster counts to evaluate.

    Returns:
        Dict[int, float]: Mapping of ``k`` to resulting inertia.
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if not k_range:
        raise ValueError("k_range cannot be empty")
    inertias: Dict[int, float] = {}
    for k in k_range:
        if k < 1:
            raise ValueError("k values must be at least 1")
        # Fit k-means and record the within-cluster sum-of-squares
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(X)
        inertias[k] = float(model.inertia_)
    return inertias


def silhouette_analysis(X: np.ndarray, k_range: Sequence[int]) -> Dict[int, float]:
    """Compute silhouette scores over a range of cluster counts.

    Args:
        X: Feature matrix ``(n_samples, n_features)``.
        k_range: Candidate numbers of clusters.

    Returns:
        Dict[int, float]: Mapping of ``k`` to average silhouette score.
    """
    if X.size == 0:
        raise ValueError("X cannot be empty")
    if not k_range:
        raise ValueError("k_range cannot be empty")
    scores: Dict[int, float] = {}
    for k in k_range:
        if k < 2:
            raise ValueError("k must be at least 2 for silhouette analysis")
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        scores[k] = float(score)
    return scores


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    X_demo = rng.normal(size=(200, 2))

    _, _ = kmeans_clustering(X_demo, n_clusters=3)
    _, _ = hierarchical_clustering(X_demo, n_clusters=3)
    _, _ = dbscan_clustering(X_demo)

    print("Elbow inertias:", elbow_inertia(X_demo, range(2, 5)))
    print("Silhouette scores:", silhouette_analysis(X_demo, range(2, 5)))
