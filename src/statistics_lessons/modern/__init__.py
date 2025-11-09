"""Modern statistical technique examples."""

from .bayesian import estimate_normal_mean, estimate_normal_mean_stan
from .probabilistic_programming import (
    coin_flip_posterior,
    bayesian_linear_regression,
)
from .deep_learning import train_linear_regression, train_mlp_classifier
from .causality import backdoor_adjustment_sets, is_d_separated
from .nlp import summarize_text, extract_named_entities

__all__ = [
    "estimate_normal_mean",
    "estimate_normal_mean_stan",
    "coin_flip_posterior",
    "bayesian_linear_regression",
    "train_linear_regression",
    "train_mlp_classifier",
    "is_d_separated",
    "backdoor_adjustment_sets",
    "summarize_text",
    "extract_named_entities",
]
