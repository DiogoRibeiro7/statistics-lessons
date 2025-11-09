from typing import Tuple, Dict, Sequence
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


def fit_models_and_criteria(
    X_full: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """
    Fit a full and nested linear model, return AIC, BIC, and approximate Bayes factor.

    The nested model drops the last predictor in X_full.
    Bayes factor BF10 approximated via BIC difference: BF10 ≈ exp((BIC_null - BIC_alt)/2).

    :param X_full: design matrix including intercept (n x p)
    :param y: response vector (n,)
    :return: dict with keys 'AIC_null','AIC_alt','BIC_null','BIC_alt','BF10'
    """
    X_full = np.asarray(X_full, dtype=float)
    y = np.asarray(y, dtype=float)
    if X_full.ndim != 2 or y.ndim != 1:
        raise ValueError("X_full must be 2D and y must be 1D.")
    if X_full.shape[0] != y.size:
        raise ValueError("X_full and y must have the same number of rows.")
    if X_full.shape[1] < 2:
        raise ValueError("X_full must have at least two columns including intercept.")
    if X_full.size == 0 or y.size == 0:
        raise ValueError("X_full and y must be non-empty.")
    X_null = X_full[:, :-1]
    model_null = sm.OLS(y, X_null).fit()
    model_alt = sm.OLS(y, X_full).fit()
    # Extract criteria
    aic_null = model_null.aic
    aic_alt = model_alt.aic
    bic_null = model_null.bic
    bic_alt = model_alt.bic
    # Approximate Bayes factor in favor of alternative
    bf10 = np.exp((bic_null - bic_alt) / 2)
    return {
        'AIC_null': aic_null,
        'AIC_alt': aic_alt,
        'BIC_null': bic_null,
        'BIC_alt': bic_alt,
        'BF10': bf10
    }


def simulate_model_selection(
    n: int,
    p: int,
    true_coefs: Sequence[float],
    sigma: float,
    trials: int = 1000,
    seed: int = 0
) -> pd.DataFrame:
    """
    Monte Carlo simulation comparing model selection by AIC, BIC, and Bayes factor.

    :param n: sample size
    :param p: number of predictors in full model (including intercept)
    :param true_coefs: list of true betas for p-1 predictors (excluding intercept)
    :param sigma: noise standard deviation
    :param trials: number of simulations
    :param seed: RNG seed
    :return: DataFrame with counts of times full model preferred
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if p < 2:
        raise ValueError("p must be at least 2 including intercept.")
    if len(true_coefs) != p - 1:
        raise ValueError("true_coefs must have length p-1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if trials <= 0:
        raise ValueError("trials must be positive.")
    rng = np.random.default_rng(seed)
    results = {'AIC': 0, 'BIC': 0, 'BF10': 0}
    for _ in range(trials):
        # simulate predictors
        X = rng.standard_normal((n, p-1))
        X_full = np.column_stack([np.ones(n), X])
        # true intercept = 0
        beta = np.array([0.0] + list(true_coefs))
        y = X_full.dot(beta) + rng.normal(0, sigma, size=n)
        crit = fit_models_and_criteria(X_full, y)
        # full preferred if criterion lower for alternative or BF10 > 1
        if crit['AIC_alt'] < crit['AIC_null']:
            results['AIC'] += 1
        if crit['BIC_alt'] < crit['BIC_null']:
            results['BIC'] += 1
        if crit['BF10'] > 1:
            results['BF10'] += 1
    # convert counts to proportions
    for key in results:
        results[key] /= trials
    return pd.DataFrame([results])


def plot_selection_frequencies(
    freq_df: pd.DataFrame
) -> None:
    """
    Bar plot of selection frequencies for AIC, BIC, and Bayes factor.

    :param freq_df: DataFrame from simulate_model_selection
    """
    if freq_df.empty:
        raise ValueError("freq_df must be non-empty.")
    freqs = freq_df.iloc[0]
    labels = ['AIC','BIC','BF10']
    values = [freqs['AIC'], freqs['BIC'], freqs['BF10']]
    plt.figure(figsize=(6,4))
    plt.bar(labels, values, edgecolor='black')
    plt.ylabel('Proportion full model selected')
    plt.title('Model selection frequencies')
    plt.ylim(0,1)
    plt.show()

# Example usage
if __name__ == '__main__':
    # e.g. two predictors, true second coef zero (so nested true)
    df = simulate_model_selection(
        n=100, p=3, true_coefs=[1.0, 0.0], sigma=1.0, trials=500
    )
    print(df)
    plot_selection_frequencies(df)
