"""Breast cancer classification case study using scikit-learn data."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..ml_models.evaluation import classification_metrics
from ..ml_models.model_selection import cross_validated_score
from .data_loaders import load_breast_cancer_data


def breast_cancer_classification(
    test_size: float = 0.2,
    random_state: int = 42,
    plot: bool = False,
) -> pd.DataFrame:
    """Train logistic regression and random forest models on the dataset.

    Parameters
    ----------
    test_size : float, default 0.2
        Fraction of data to use for testing.
    random_state : int, default 42
        Seed for reproducible splits.
    plot : bool, default False
        If ``True``, display confusion matrix plots for both models.

    Returns
    -------
    pandas.DataFrame
        Table of evaluation metrics for each model.
    """
    X, y = load_breast_cancer_data()

    # Standardize numeric features for logistic regression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    results = []

    # Logistic Regression model
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    y_proba_lr = logreg.predict_proba(X_test)[:, 1]
    metrics_lr = classification_metrics(
        y_test.to_numpy(), y_pred_lr, y_proba_lr
    )
    cv_lr = cross_validated_score(logreg, X_scaled, y.to_numpy(), cv=5)
    metrics_lr["cv_accuracy"] = cv_lr
    metrics_lr["model"] = "logistic_regression"
    results.append(metrics_lr)

    # Random Forest model
    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]
    metrics_rf = classification_metrics(
        y_test.to_numpy(), y_pred_rf, y_proba_rf
    )
    cv_rf = cross_validated_score(rf, X_scaled, y.to_numpy(), cv=5)
    metrics_rf["cv_accuracy"] = cv_rf
    metrics_rf["model"] = "random_forest"
    results.append(metrics_rf)

    results_df = pd.DataFrame(results).set_index("model")

    if plot:
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_lr, ax=axes[0], colorbar=False
        )
        axes[0].set_title("Logistic Regression")
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred_rf, ax=axes[1], colorbar=False
        )
        axes[1].set_title("Random Forest")
        fig.tight_layout()
        plt.show()

    return results_df


if __name__ == "__main__":
    print(breast_cancer_classification())
