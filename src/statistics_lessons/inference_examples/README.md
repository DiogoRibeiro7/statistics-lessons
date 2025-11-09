# Inference Examples

This repository, organized under the `inference_examples/` folder, contains a collection of Python scripts demonstrating and comparing frequentist and Bayesian inference methods across various classic statistical problems.

## Directory Structure

```
inference_examples/
├── cauchy_regions.py          # Cauchy likelihood vs. posterior regions example
├── jeffreys_lindley.py        # Jeffreys–Lindley paradox simulation
├── uniform_endpoint.py        # Estimating Uniform(0,θ) endpoint example
├── outlier_robustness.py      # Robust estimation: trimmed mean, Huber, Student-t
├── rare_events.py             # Zero-event bounds for Poisson/Binomial
├── model_selection.py         # AIC/BIC vs. Bayes factor for regression
└── hierarchical_shrinkage.py  # Hierarchical vs standalone CIs example
README.md
```

## Scripts Overview

### 1. Cauchy Regions (`cauchy_regions.py`)

* Computes the joint likelihood for three Cauchy observations.
* Finds maximum likelihood estimator (MLE) via grid search.
* Constructs frequentist likelihood-ratio confidence intervals (possibly disjoint).
* Builds objective Bayesian highest-posterior-density credible sets under a flat prior.
* Plots likelihood and posterior with shaded intervals.
* Monte Carlo routine to estimate coverage probabilities.

### 2. Jeffreys–Lindley Paradox (`jeffreys_lindley.py`)

* Performs two-sided $z$-tests for $H_0: 	heta=0$ (known variance).
* Calculates Bayes factor $BF_{01}$ under a normal prior on $	heta$.
* Simulates mean p-values and Bayes factors as sample size grows, illustrating the paradox.
* Plots p-value and $BF_{01}$ curves over varying $n$.

### 3. Uniform Endpoint (`uniform_endpoint.py`)

* Estimates the maximum $	heta$ of a Uniform(0,$	heta$) using MLE (sample maximum).
* Computes a 95% frequentist CI: $[\max Y_i,\; \max Y_i / \alpha]$.
* Derives Bayesian 95% credible interval under an improper flat prior (Pareto posterior).
* Visualizes data histogram with both intervals.

### 4. Outlier Robustness (`outlier_robustness.py`)

* **Frequentist**:

  * Trimmed mean with bootstrap confidence interval.
  * Huber $M$-estimator with bootstrap CI.
* **Bayesian**:

  * Student-$t$ likelihood (low degrees of freedom) posterior under a flat prior.
  * Highest-posterior-density credible interval.
* Plot overlays of histogram, robust estimates, and posterior.

### 5. Rare Events (`rare_events.py`)

* **Frequentist**:

  * One-sided 95% upper bound on Poisson rate given zero events: $-\ln(\alpha)/T$.
  * One-sided 95% upper bound on Binomial failure probability after zero failures.
* **Bayesian**:

  * Gamma posterior for Poisson rate, quantile bound.
  * Beta posterior for Binomial probability, quantile bound.
* Bar chart comparing all four bounds.

### 6. Model Selection (`model_selection.py`)

* Fits nested linear models (drop last predictor).
* Computes AIC, BIC, and approximates Bayes factor via BIC difference.
* Monte Carlo simulation of selection frequency by each criterion.
* Bar plot of how often the full model is chosen.

### 7. Hierarchical Shrinkage (`hierarchical_shrinkage.py`)

* **Frequentist**:

  * Computes separate 95% t-based CIs for each group’s mean.
* **Bayesian (Empirical Bayes)**:

  * Estimates hyperparameters by method-of-moments under a normal-normal model.
  * Calculates shrunk posterior means and credible intervals.
* Plot comparing standalone vs. hierarchical intervals across groups.

## Getting Started

1. Install dependencies:

   ```bash
   pip install numpy scipy statsmodels matplotlib
   ```
2. Run any example:

   ```bash
   python inference_examples/jeffreys_lindley.py
   ```
3. Explore and adapt parameters (e.g., sample size, priors, trimming proportions) to see how frequentist and Bayesian conclusions differ.

## License

This work is provided for educational purposes. No warranty is implied.
