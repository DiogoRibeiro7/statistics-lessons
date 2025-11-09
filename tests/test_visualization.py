import numpy as np
import pytest
from plotly.graph_objs import Figure

from statistics_lessons.visualization import (
    interactive_histogram,
    interactive_scatter,
)


def test_interactive_histogram_returns_figure():
    data = np.random.normal(size=100)
    fig = interactive_histogram(data, bins=20, title="hist")
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "histogram"


def test_interactive_histogram_empty_data():
    with pytest.raises(ValueError):
        interactive_histogram([], bins=5)


def test_interactive_scatter_returns_figure():
    x = np.arange(10)
    y = x**2
    fig = interactive_scatter(x, y, title="scatter")
    assert isinstance(fig, Figure)
    assert fig.data[0].type == "scatter"


def test_interactive_scatter_length_mismatch():
    with pytest.raises(ValueError):
        interactive_scatter([1, 2], [1])
