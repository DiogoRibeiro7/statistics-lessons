from statistics_lessons.projects import (
    load_iris_data,
    load_wine_data,
    load_breast_cancer_data,
)


def test_load_iris_data_shapes():
    X, y = load_iris_data()
    assert len(X) == len(y)
    assert not X.empty
    assert y.notnull().all()


def test_load_wine_data_shapes():
    X, y = load_wine_data()
    assert len(X) == len(y)
    assert X.shape[1] > 0


def test_load_breast_cancer_data_shapes():
    X, y = load_breast_cancer_data()
    assert len(X) == len(y)
    assert "mean radius" in X.columns
