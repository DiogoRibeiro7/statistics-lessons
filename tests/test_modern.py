"""Tests for modern statistical techniques modules."""

import pytest

from statistics_lessons import modern


def test_estimate_normal_mean():
    pytest.importorskip("pymc")
    est = modern.estimate_normal_mean([0, 1, 2])
    assert abs(est - 1) < 0.2


def test_estimate_normal_mean_stan():
    pytest.importorskip("stan")
    est = modern.estimate_normal_mean_stan([0, 1, 2])
    assert abs(est - 1) < 0.2


def test_coin_flip_posterior():
    pytest.importorskip("pymc")
    posterior = modern.coin_flip_posterior([1, 0, 1, 1])
    assert 0 <= posterior <= 1


def test_bayesian_linear_regression():
    pytest.importorskip("pymc")
    slope, intercept = modern.bayesian_linear_regression([0, 1, 2], [1, 3, 5])
    assert pytest.approx(slope, rel=0.2) == 2
    assert pytest.approx(intercept, rel=0.2) == 1


def test_train_linear_regression():
    pytest.importorskip("torch")
    weight, bias = modern.train_linear_regression([0, 1, 2], [1, 3, 5])
    assert pytest.approx(weight, rel=0.2) == 2
    assert pytest.approx(bias, rel=0.2) == 1


def test_train_mlp_classifier():
    torch = pytest.importorskip("torch")
    features = [[0.0], [0.1], [1.0], [1.1]]
    labels = [0, 0, 1, 1]
    model, loss = modern.train_mlp_classifier(features, labels)
    assert loss < 0.1
    tensor = torch.tensor(features, dtype=torch.float32)
    preds = model(tensor).argmax(dim=1).tolist()
    assert preds == labels


def test_is_d_separated():
    pytest.importorskip("causalgraphicalmodels")
    edges = [("X", "Z"), ("Z", "Y")]
    assert modern.is_d_separated(edges, "X", "Y", {"Z"})


def test_backdoor_adjustment_sets():
    pytest.importorskip("causalgraphicalmodels")
    edges = [("U", "X"), ("U", "Y"), ("X", "Y")]
    adj_sets = modern.backdoor_adjustment_sets(edges, "X", "Y")
    assert frozenset({"U"}) in adj_sets


def test_summarize_text():
    pytest.importorskip("transformers")
    summary = modern.summarize_text(
        "This library provides many practical examples.",
    )
    assert isinstance(summary, str) and summary


def test_extract_named_entities():
    pytest.importorskip("transformers")
    text = "Hugging Face is based in New York City."
    entities = modern.extract_named_entities(text)
    assert entities and isinstance(entities[0][0], str)
