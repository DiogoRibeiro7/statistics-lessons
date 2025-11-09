"""Utilities for downloading and preparing sample datasets."""

from .standard import load_standard_dataset
from .synthetic import make_linear_regression, make_classification

__all__ = [
    "load_standard_dataset",
    "make_linear_regression",
    "make_classification",
]
