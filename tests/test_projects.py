import matplotlib
import pandas as pd
from statistics_lessons.projects import (
    wine_pca_example,
    production_planning_example,
    breast_cancer_classification,
    iris_classification_example,
    iris_clustering_example,
    sunspots_forecasting_example,
    sunspots_exp_smoothing_example,
    gp_regression_example,
)

matplotlib.use("Agg")


def test_wine_pca_example_variance_sum():
    df = wine_pca_example(n_components=0.95, plot=True)
    assert isinstance(df, pd.DataFrame)
    assert df.explained_variance.sum() >= 0.95


def test_production_planning_example_returns_positive_values():
    result = production_planning_example()
    assert all(result > 0)


def test_breast_cancer_classification_indices():
    df = breast_cancer_classification(test_size=0.2, random_state=0, plot=True)
    assert set(df.index) == {"logistic_regression", "random_forest"}


def test_iris_classification_accuracy_range():
    acc = iris_classification_example(test_size=0.3, random_state=0)
    assert 0.0 <= acc <= 1.0


def test_iris_clustering_example_score_range():
    df = iris_clustering_example(n_clusters=3, random_state=0, plot=True)
    score = df.silhouette_score.iloc[0]
    assert 0.0 <= score <= 1.0


def test_sunspots_forecasting_example_rmse_positive():
    rmse = sunspots_forecasting_example(order=(2, 0, 2), steps=3, plot=True)
    assert rmse >= 0.0


def test_sunspots_exp_smoothing_example_rmse_positive():
    rmse = sunspots_exp_smoothing_example(steps=3, plot=True)
    assert rmse >= 0.0


def test_gp_regression_example_mse_positive():
    df = gp_regression_example(test_size=0.2, random_state=0, plot=True)
    assert set(df.model) == {"gaussian_process", "linear_regression"}
    assert (df.mse >= 0).all()
