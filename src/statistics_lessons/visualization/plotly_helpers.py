"""Utility functions for interactive visualizations using Plotly."""

from collections.abc import Sequence

import numpy as np
import plotly.express as px
from plotly.graph_objs import Figure


def interactive_histogram(
    data: Sequence[float],
    bins: int = 10,
    title: str = "Histogram",
) -> Figure:
    """Create an interactive histogram.

    Args:
        data: Iterable of numeric values to plot.
        bins: Number of histogram bins.
        title: Title for the plot.

    Returns:
        plotly.graph_objs.Figure: Histogram figure.

    Raises:
        ValueError: If ``data`` is empty.
    """
    arr = np.asarray(data)
    if arr.size == 0:
        raise ValueError("data must contain at least one value")
    fig = px.histogram(arr, nbins=bins, title=title)
    return fig


def interactive_scatter(
    x: Sequence[float],
    y: Sequence[float],
    title: str = "Scatter Plot",
) -> Figure:
    """Create an interactive scatter plot.

    Args:
        x: Sequence of x-axis values.
        y: Sequence of y-axis values.
        title: Title for the plot.

    Returns:
        plotly.graph_objs.Figure: Scatter figure.

    Raises:
        ValueError: If ``x`` and ``y`` have different lengths or are empty.
    """
    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.size == 0 or y_arr.size == 0:
        raise ValueError("x and y must contain at least one value")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length")
    fig = px.scatter(x=x_arr, y=y_arr, title=title)
    return fig
