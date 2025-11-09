"""Sentiment analysis example using 20 Newsgroups data."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from ..ml_models.evaluation import classification_metrics
from ..ml_models.model_selection import cross_validated_score
from .data_loaders import load_newsgroup_sentiment


def sentiment_classification(
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train a logistic regression model on text data and evaluate performance.

    Args:
        test_size: Fraction of samples to reserve for testing. Defaults to
            ``0.2``.
        random_state: Seed controlling the train/test split. Defaults to
            ``42``.

    Returns:
        pandas.DataFrame: Metrics summarizing accuracy and AUC on the test
        set and cross-validated accuracy.
    """
    texts, labels = load_newsgroup_sentiment()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    # Pipeline combines TF-IDF vectorization with logistic regression
    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        LogisticRegression(max_iter=1000),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(y_test, y_pred, y_proba)
    cv_acc = cross_validated_score(model, texts, labels, cv=5)
    metrics["cv_accuracy"] = cv_acc
    metrics["model"] = "logistic_regression"
    return pd.DataFrame([metrics]).set_index("model")


if __name__ == "__main__":
    print(sentiment_classification())
