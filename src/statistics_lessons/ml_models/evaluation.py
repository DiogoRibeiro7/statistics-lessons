"""Model evaluation utilities for classification tasks."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true (numpy.ndarray): Ground truth binary labels.
        y_pred (numpy.ndarray): Predicted binary labels.
        y_proba (Optional[numpy.ndarray], optional): Predicted probabilities for the
            positive class. Required for AUC. Defaults to ``None``.

    Returns:
        Dict[str, float]: Dictionary containing accuracy, precision, recall, F1,
        and AUC (if ``y_proba`` is provided).
    """
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional arrays")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred cannot be empty")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if y_proba is not None:
        if y_proba.shape != y_true.shape:
            raise ValueError("y_proba must have the same shape as y_true")
        if np.any((y_proba < 0) | (y_proba > 1)):
            raise ValueError("y_proba values must be between 0 and 1")

    # Basic metrics from labels alone
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }

    # Compute AUC if probability estimates are provided
    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        metrics["auc"] = float(auc)

    return metrics


if __name__ == "__main__":
    # Quick demonstration with synthetic predictions
    rng = np.random.default_rng(seed=0)
    true_labels = rng.integers(0, 2, size=100)
    pred_labels = rng.integers(0, 2, size=100)
    pred_probs = rng.random(size=100)

    results = classification_metrics(true_labels, pred_labels, pred_probs)
    print(results)
