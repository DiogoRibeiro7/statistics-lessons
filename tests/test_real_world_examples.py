from statistics_lessons.datasets import load_standard_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def test_breast_cancer_logistic_classification():
    """Breast Cancer logistic regression achieves high accuracy."""
    X_train, X_test, y_train, y_test = load_standard_dataset("breast_cancer")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert accuracy_score(y_test, preds) > 0.9
