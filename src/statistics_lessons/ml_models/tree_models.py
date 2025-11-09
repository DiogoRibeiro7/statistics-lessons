"""Decision tree and random forest examples for lessons."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def decision_tree_classifier_example(
    x: np.ndarray, y: np.ndarray, max_depth: Optional[int] = None
) -> DecisionTreeClassifier:
    """Fit a simple decision tree classifier.

    Args:
        x (numpy.ndarray): 2D array of features with shape ``(n_samples, n_features)``.
        y (numpy.ndarray): 1D array of class labels.
        max_depth (Optional[int], optional): Maximum depth of the tree. Defaults to ``None``.

    Returns:
        DecisionTreeClassifier: Trained decision tree model.
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    # Initialize classifier with optional depth limit
    clf = DecisionTreeClassifier(max_depth=max_depth)
    # Fit model to data
    clf.fit(x, y)
    return clf


def random_forest_classifier_example(
    x: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
) -> RandomForestClassifier:
    """Fit a random forest classifier.

    Args:
        x (numpy.ndarray): 2D array of features with shape ``(n_samples, n_features)``.
        y (numpy.ndarray): 1D array of class labels.
        n_estimators (int, optional): Number of trees in the forest. Defaults to ``100``.
        max_depth (Optional[int], optional): Maximum depth of each tree. Defaults to ``None``.

    Returns:
        RandomForestClassifier: Trained random forest model.
    """
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y cannot be empty")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must contain the same number of samples")
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    # Initialize the ensemble model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    # Fit model to data
    model.fit(x, y)
    return model


if __name__ == "__main__":
    # Simple usage example with synthetic data
    rng = np.random.default_rng(seed=42)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    tree_model = decision_tree_classifier_example(X, y, max_depth=3)
    rf_model = random_forest_classifier_example(X, y, n_estimators=10, max_depth=3)

    print("Tree depth:", tree_model.get_depth())
    print("Forest trees:", len(rf_model.estimators_))
