from statistics_lessons.datasets import load_standard_dataset
import pytest


def test_load_standard_dataset_returns_splits():
    X_train, X_test, y_train, y_test = load_standard_dataset(
        "iris", test_size=0.3, random_state=1
    )
    assert len(X_train) + len(X_test) == 150
    assert X_train.shape[1] == 4
    assert y_train.nunique() == 3


def test_load_standard_dataset_bad_name():
    with pytest.raises(ValueError):
        load_standard_dataset("unknown")
