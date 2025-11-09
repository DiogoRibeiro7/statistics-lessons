from statistics_lessons.ml_models.model_selection import cross_validated_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def test_cross_validated_score_returns_accuracy_between_zero_and_one():
    X, y = load_iris(return_X_y=True)
    model = LogisticRegression(max_iter=200)
    score = cross_validated_score(model, X, y, cv=3)
    assert 0.0 <= score <= 1.0
