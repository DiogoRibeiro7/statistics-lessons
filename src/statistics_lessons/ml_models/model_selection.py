"""Utilities for cross-validation and hyperparameter tuning."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score


def cross_validated_score(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, scoring: str = "accuracy"
) -> float:
    """Compute a cross-validated score for the model.

    Args:
        model: Scikit-learn compatible estimator.
        X: Feature matrix with shape ``(n_samples, n_features)``.
        y: Target array with shape ``(n_samples,)``.
        cv: Number of folds for cross-validation. Defaults to ``5``.
        scoring: Metric name to optimize. Defaults to ``"accuracy"``.

    Returns:
        float: Mean cross-validated score across folds.
    """
    if X.size == 0 or y.size == 0:
        raise ValueError("X and y cannot be empty")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if cv < 2:
        raise ValueError("cv must be at least 2")
    # ``cross_val_score`` returns an array of scores for each fold
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    # Return the average performance across all folds
    return float(scores.mean())


def grid_search_best(
    model: Any,
    param_grid: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
) -> Tuple[Any, Dict[str, Any], float]:
    """Perform grid search to find the best hyperparameters.

    Args:
        model: Scikit-learn estimator with settable parameters.
        param_grid: Dictionary mapping parameter names to candidate values.
        X: Feature matrix with shape ``(n_samples, n_features)``.
        y: Target array with shape ``(n_samples,)``.
        cv: Number of cross-validation folds. Defaults to ``5``.
        scoring: Metric name for evaluation. Defaults to ``"accuracy"``.

    Returns:
        Tuple[Any, Dict[str, Any], float]: The best estimator, parameter
        settings, and cross-validated score.
    """
    if not param_grid:
        raise ValueError("param_grid cannot be empty")
    if X.size == 0 or y.size == 0:
        raise ValueError("X and y cannot be empty")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must contain the same number of samples")
    if cv < 2:
        raise ValueError("cv must be at least 2")
    # ``GridSearchCV`` exhaustively searches the parameter grid
    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)
    search.fit(X, y)
    best_estimator = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_
    return best_estimator, best_params, float(best_score)


if __name__ == "__main__":
    # Simple example using logistic regression
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    data = load_breast_cancer()
    X, y = data.data, data.target

    model = LogisticRegression(max_iter=1000)

    score = cross_validated_score(model, X, y, cv=3, scoring="accuracy")
    print("CV accuracy:", score)

    params = {"C": [0.1, 1.0, 10.0]}
    best_model, best_params, best_score = grid_search_best(
        model, params, X, y, cv=3, scoring="accuracy"
    )
    print("Best params:", best_params)
    print("Best score:", best_score)
