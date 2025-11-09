# Statistics Lessons

This project contains a collection of small utilities and lesson notes for teaching basic statistics with an emphasis on machine learning applications. Each lesson follows the roadmap to gradually introduce new concepts.

## Documentation

The latest documentation is available at <https://diogoribeiro7.github.io/statistics_lessons/>.

## Getting Started

### Installation

Clone the repository and install the package:

```bash
git clone https://github.com/DiogoRibeiro7/statistics_lessons.git
cd statistics_lessons
pip install -e .  # or use poetry install
```

This installs required packages like `numpy`, `scipy`, `pandas`,
`matplotlib`, `scikit-learn`, and `statsmodels`. All Python modules live
under `src/statistics_lessons`.

### First steps

After installation, open a Python session and compute basic statistics:

```python
>>> import pandas as pd
>>> from statistics_lessons.foundations.descriptive_stats import summary_statistics
>>> data = pd.Series([1, 2, 3, 4, 5])
>>> summary_statistics(data)
{'mean': 3.0, 'median': 3.0, 'std': 1.5811...}
```

You can also detect trends in a time series:

```python
>>> from statistics_lessons.trend_features import detect_trend
>>> s = pd.Series([1, 2, 3, 4, 5, 6, 7])
>>> detect_trend(s, window=3)
0    False
1    False
2    False
3     True
4     True
5     True
6     True
dtype: bool
```

Cluster samples into groups:

```python
>>> import numpy as np
>>> from statistics_lessons.ml_models.clustering import kmeans_clustering
>>> X = np.array([[1, 2], [1, 0], [4, 5], [5, 4]])
>>> model, labels = kmeans_clustering(X, n_clusters=2)
>>> labels
array([0, 0, 1, 1], dtype=int32)
```

Load a prepared dataset for modeling:

```python
>>> from statistics_lessons.datasets import load_standard_dataset
>>> X_train, X_test, y_train, y_test = load_standard_dataset("iris")
>>> X_train.head()
    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                 ...               ...                 ...               ...
```

Generate synthetic data for experimentation:

```python
>>> from statistics_lessons.datasets import make_linear_regression
>>> X, y = make_linear_regression(n_samples=50, slope=2.0, intercept=-1.0, random_state=0)
>>> X.head()
     x
0 ...
```

Create an interactive histogram:

```python
>>> import numpy as np
>>> from statistics_lessons.visualization import interactive_histogram
>>> data = np.random.randn(500)
>>> fig = interactive_histogram(data, bins=30, title="Normal Distribution")
>>> fig.show()  # Opens an interactive plot in supported environments
```

Explore modern techniques:

```python
>>> from statistics_lessons.modern import (
...     estimate_normal_mean,
...     estimate_normal_mean_stan,
...     coin_flip_posterior,
...     bayesian_linear_regression,
...     train_linear_regression,
...     train_mlp_classifier,
...     summarize_text,
...     extract_named_entities,
...     is_d_separated,
...     backdoor_adjustment_sets,
... )
>>> estimate_normal_mean([1, 2, 3])  # PyMC implementation
1.5...
>>> estimate_normal_mean_stan([1, 2, 3])  # PyStan implementation
1.5...
>>> coin_flip_posterior([1, 0, 1, 1])  # PyMC coin flip example
0.7...
>>> bayesian_linear_regression([0, 1, 2], [1, 3, 5])  # PyMC regression
(2.0..., 1.0...)
>>> train_linear_regression([0, 1, 2], [1, 3, 5])  # PyTorch regression
(2.0..., 1.0...)
>>> model, loss = train_mlp_classifier([[0.0], [0.1], [1.0], [1.1]], [0, 0, 1, 1])
>>> loss < 0.1
True
>>> summarize_text("This library contains many examples for demonstration purposes.")
"summary"
>>> extract_named_entities("Hugging Face is based in New York City.")
[('Hugging Face', 'ORG'), ('New York City', 'LOC')]
>>> is_d_separated([("X", "Z"), ("Z", "Y")], "X", "Y", {"Z"})
True
>>> backdoor_adjustment_sets([("U", "X"), ("U", "Y"), ("X", "Y")], "X", "Y")
frozenset({frozenset({'U'})})
```

### Real-world example: Breast Cancer classification

Use a real dataset to train and evaluate a classifier:

```python
>>> from statistics_lessons.datasets import load_standard_dataset
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import accuracy_score
>>> X_train, X_test, y_train, y_test = load_standard_dataset("breast_cancer")
>>> model = LogisticRegression(max_iter=1000)
>>> model.fit(X_train, y_train)
>>> accuracy_score(y_test, model.predict(X_test))
0.95...
```

### Minimum dependency versions

The library is tested with the following minimum versions of its core dependencies:

- Python 3.10
- NumPy 2.0
- SciPy 1.13
- pandas 2.3
- Matplotlib 3.9
- scikit-learn 1.6
- statsmodels 0.14

The `foundations` package offers descriptive statistics, probability, preprocessing,
inference, and resampling utilities. The `trend_features` module demonstrates how to extract
rolling trend metrics from time series and provides helpers for computing
rolling means, rolling standard deviations, decomposing a series into seasonal
components, detecting statistically significant trends, and generating simple
forecasts with ARIMA or exponential smoothing. The `ml_models`
package provides quick examples of linear
and logistic regression, decision tree and random forest utilities, clustering
algorithms like k-means, hierarchical clustering, and DBSCAN, functions for
evaluating classification models, and helpers for cross-validation and
hyperparameter tuning.

The new `projects` directory showcases case studies with public datasets. The
first example demonstrates logistic regression on the classic Iris data set,
a second example trains multiple models on the Breast Cancer dataset, and a
third script illustrates basic sentiment analysis using a subset of the
20 Newsgroups corpus. Another script runs PCA on the Wine dataset to
highlight dimensionality reduction techniques. A clustering example groups the
Iris samples with k-means and reports the silhouette score. Finally, a
prescriptive analytics example shows how to solve a simple production planning
problem with linear programming. Time series scripts demonstrate forecasting
annual sunspot counts using both ARIMA and exponential smoothing models.
Many of these case studies include optional plotting to visualize outputs such
as confusion matrices, cluster scatterplots, PCA variance bar charts, and
forecast comparisons.

## Documentation

API documentation can be generated with [Sphinx](https://www.sphinx-doc.org/).
After installing the optional ``docs`` dependencies, build the HTML site with:

```bash
cd docs
make html
```

## Example notebooks

Example Jupyter notebooks can be found in the `notebooks` directory. These
notebooks reproduce key lessons and let you experiment with the utilities in an
interactive environment.

For a brief summary of insights gained from these projects and suggestions for
next steps, see [REFLECTION.md](REFLECTION.md).

See [SUGGESTIONS.md](SUGGESTIONS.md) for a detailed list of advanced study topics.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute and refer to [CITATION.cff](CITATION.cff) if you use this material. All interactions should follow the [Code of Conduct](CODE_OF_CONDUCT.md).
