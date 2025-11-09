import numpy as np
import pytest

from statistics_lessons.survival_analysis import (
    kaplan_meier_estimator,
    log_rank_test,
    fit_cox_proportional_hazards,
)


def test_kaplan_meier_estimator_basic() -> None:
    durations = np.array([5, 6, 6, 2, 4, 3])
    events = np.array([1, 1, 0, 1, 0, 1])
    times, survival = kaplan_meier_estimator(durations, events)
    assert times.shape == survival.shape
    assert np.all(np.diff(survival) <= 0)


def test_log_rank_test_detects_difference() -> None:
    time_a = np.array([5, 6, 7, 8, 9])
    event_a = np.ones_like(time_a)
    time_b = np.array([10, 11, 12, 13, 14])
    event_b = np.ones_like(time_b)
    stat, p = log_rank_test(time_a, event_a, time_b, event_b)
    assert stat > 0
    assert p < 0.05


def test_fit_cox_proportional_hazards_returns_coefficients() -> None:
    x = np.arange(1, 5).reshape(-1, 1)
    durations = np.array([10, 8, 6, 4])
    events = np.ones_like(durations)
    beta = fit_cox_proportional_hazards(x, durations, events, n_iter=2000, learning_rate=0.1)
    assert beta.shape == (1,)
    assert beta[0] > 0


def test_kaplan_meier_estimator_mismatched_input() -> None:
    durations = np.array([1, 2])
    events = np.array([1])
    with pytest.raises(ValueError):
        kaplan_meier_estimator(durations, events)
