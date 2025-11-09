"""Utility functions to load common datasets used across project examples."""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd
from sklearn.datasets import (
    fetch_20newsgroups,
    load_breast_cancer,
    load_iris,
    load_wine,
)
from statsmodels.datasets import sunspots


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return the classic Iris flower dataset."""
    dataset = load_iris(as_frame=True)
    return dataset.data, dataset.target


def load_wine_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return the UCI wine recognition dataset."""
    dataset = load_wine(as_frame=True)
    return dataset.data, dataset.target


def load_breast_cancer_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Return the Wisconsin breast cancer dataset."""
    dataset = load_breast_cancer(as_frame=True)
    return dataset.data, dataset.target


def load_newsgroup_sentiment() -> Tuple[List[str], List[int]]:
    """Fetch a two-category subset of the 20 Newsgroups text corpus."""
    categories = ["rec.sport.hockey", "sci.space"]
    data = fetch_20newsgroups(
        subset="all",
        categories=categories,
        remove=("headers", "footers", "quotes"),
    )
    labels = [
        1 if data.target_names[t] == "rec.sport.hockey" else 0
        for t in data.target
    ]
    return data.data, labels


def load_sunspots() -> pd.Series:
    """Load annual sunspot counts as a series indexed by year."""
    data = sunspots.load_pandas().data
    return pd.Series(
        data["SUNACTIVITY"].values,
        index=pd.Index(data["YEAR"].astype(int), name="Year"),
    )

