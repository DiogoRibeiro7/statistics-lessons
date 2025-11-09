"""Case study using the Iris dataset for classification."""

from typing import Tuple

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .data_loaders import load_iris_data


def iris_classification_example(
    test_size: float = 0.2, random_state: int = 0
) -> float:
    """Train a logistic regression classifier on the Iris dataset.

    The function splits the data into train and test subsets, fits a logistic
    regression model, and returns the accuracy on the held-out test set.

    Args:
        test_size (float, optional): Fraction of data to use for testing.
            Defaults to ``0.2``.
        random_state (int, optional): Seed for reproducible splits. Defaults
            to ``0``.

    Returns:
        float: Classification accuracy on the test portion of the data.
    """
    # Load full dataset
    X, y = load_iris_data()

    # Split into training and testing partitions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Fit a simple logistic regression classifier
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Predict and evaluate accuracy
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return float(accuracy)


if __name__ == "__main__":
    # Example manual execution
    acc = iris_classification_example()
    print("Test accuracy:", acc)
