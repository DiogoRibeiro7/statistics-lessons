"""Case study modules for real-world datasets."""

from .data_loaders import (
    load_iris_data,
    load_breast_cancer_data,
    load_newsgroup_sentiment,
    load_wine_data,
    load_sunspots,
)
from .iris_classification import iris_classification_example
from .breast_cancer_classification import breast_cancer_classification
from .sentiment_analysis import sentiment_classification
from .prescriptive_optimization import production_planning_example
from .wine_pca_analysis import wine_pca_example
from .iris_clustering import iris_clustering_example
from .time_series_forecasting import (
    sunspots_forecasting_example,
    sunspots_exp_smoothing_example,
)
from .gp_regression_example import gp_regression_example

__all__ = [
    "load_iris_data",
    "iris_classification_example",
    "load_breast_cancer_data",
    "breast_cancer_classification",
    "load_newsgroup_sentiment",
    "sentiment_classification",
    "production_planning_example",
    "load_wine_data",
    "wine_pca_example",
    "iris_clustering_example",
    "gp_regression_example",
    "load_sunspots",
    "sunspots_forecasting_example",
    "sunspots_exp_smoothing_example",
]
