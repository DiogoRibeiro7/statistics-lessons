import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from statistics_lessons.ml_models.evaluation import classification_metrics
from statistics_lessons.ml_models.model_selection import (
    cross_validated_score,
    grid_search_best,
)
from statistics_lessons.ml_models.clustering import kmeans_clustering
from statistics_lessons.ml_models.tree_models import random_forest_classifier_example


def test_classification_metrics_mismatched_length():
    y_true = np.array([0, 1])
    y_pred = np.array([0])
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred)


def test_classification_metrics_invalid_proba():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_proba = np.array([1.2, -0.1])
    with pytest.raises(ValueError):
        classification_metrics(y_true, y_pred, y_proba)


def test_cross_validated_score_shape_mismatch():
    model = LogisticRegression()
    X = np.ones((5, 2))
    y = np.ones(4)
    with pytest.raises(ValueError):
        cross_validated_score(model, X, y)


def test_grid_search_best_empty_param_grid():
    model = LogisticRegression()
    X = np.ones((5, 2))
    y = np.ones(5)
    with pytest.raises(ValueError):
        grid_search_best(model, {}, X, y)


def test_kmeans_clustering_invalid_k():
    X = np.ones((3, 2))
    with pytest.raises(ValueError):
        kmeans_clustering(X, n_clusters=0)


def test_random_forest_invalid_estimators():
    X = np.ones((5, 2))
    y = np.zeros(5)
    with pytest.raises(ValueError):
        random_forest_classifier_example(X, y, n_estimators=0)

