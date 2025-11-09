"""Machine learning models used in lessons."""

from .regression_examples import linear_regression_example, logistic_regression_example
from .tree_models import (
    decision_tree_classifier_example,
    random_forest_classifier_example,
)
from .evaluation import classification_metrics
from .model_selection import cross_validated_score, grid_search_best
from .clustering import (
    kmeans_clustering,
    hierarchical_clustering,
    dbscan_clustering,
    elbow_inertia,
    silhouette_analysis,
)
from .gaussian_process import (
    fit_gaussian_process,
    optimize_kernel_hyperparameters,
    plot_gaussian_process,
)

__all__ = [
    "linear_regression_example",
    "logistic_regression_example",
    "decision_tree_classifier_example",
    "random_forest_classifier_example",
    "classification_metrics",
    "cross_validated_score",
    "grid_search_best",
    "kmeans_clustering",
    "hierarchical_clustering",
    "dbscan_clustering",
    "elbow_inertia",
    "silhouette_analysis",
    "fit_gaussian_process",
    "optimize_kernel_hyperparameters",
    "plot_gaussian_process",
]
