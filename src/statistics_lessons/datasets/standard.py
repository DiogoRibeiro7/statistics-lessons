"""Helpers to fetch and split popular toy datasets."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.model_selection import train_test_split

_DATASET_LOADERS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
}


def load_standard_dataset(
    name: str,
    *,
    test_size: float = 0.2,
    random_state: int | None = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load a dataset and return train/test splits.

    Args:
        name: Dataset identifier. Supported options are ``"iris"``,
            ``"wine"``, and ``"breast_cancer"``.
        test_size: Fraction of samples to allocate to the test set.
        random_state: Seed controlling the train/test split.

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)`` containing the
        feature matrices and target vectors for the train and test partitions.

    Raises:
        ValueError: If ``name`` is not a recognized dataset.
    """

    if name not in _DATASET_LOADERS:
        available = ", ".join(_DATASET_LOADERS)
        message = f"Unknown dataset '{name}'. Available options: {available}"
        raise ValueError(message)

    dataset = _DATASET_LOADERS[name](as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )
    return X_train, X_test, y_train, y_test
