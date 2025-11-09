import matplotlib
import numpy as np
from statistics_lessons.inference_examples import uniform_endpoint, rare_events
from statistics_lessons.inference_examples.hierarchical_shrinkage import (
    frequentist_intervals,
    hierarchical_intervals,
    plot_hierarchical,
)

matplotlib.use("Agg")


def test_uniform_endpoint_plot_runs():
    data = [1.0, 2.0, 3.0]
    ints = uniform_endpoint.uniform_endpoint_estimation(data)
    uniform_endpoint.plot_uniform_endpoint(data, ints)


def test_rare_event_plot_runs():
    rare_events.plot_rare_event_bounds(T=10.0, n=20)


def test_hierarchical_plot_runs():
    data = [np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 2.5, 3.0, 3.5])]
    freq = frequentist_intervals(data)
    hier = hierarchical_intervals(data, sigma2=1.0)
    plot_hierarchical(data, freq, hier)
