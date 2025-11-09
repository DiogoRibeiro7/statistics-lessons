import numpy as np
from statistics_lessons.ml_models.evaluation import classification_metrics


def test_classification_metrics_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.1, 0.7, 0.2, 0.8])
    metrics = classification_metrics(y_true, y_pred, y_proba)
    assert set(metrics.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    }
    # expected values
    assert np.isclose(metrics["accuracy"], 0.5)
    assert np.isclose(metrics["precision"], 0.5)
    assert np.isclose(metrics["recall"], 0.5)
    assert np.isclose(metrics["f1"], 0.5)
    assert np.isclose(metrics["auc"], 0.75)
