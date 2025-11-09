# Rationale and Explanation

This module is designed to capture meaningful trend signals from time series data with robustness and statistical rigor. It also includes helpers to decompose a series into seasonal, trend, and residual components. Below are the main motivations and advantages behind using the Theil–Sen estimator and the Mann–Kendall test for rolling trend features.

## 1. Robust Slope Estimation with Theil–Sen

- **Resilience to Outliers:** Unlike ordinary least squares, which minimizes squared errors and can be skewed by extreme values, Theil–Sen computes the median of all pairwise slopes. This median-based approach shields the slope estimate from single anomalous points.
- **No Distributional Assumptions:** Theil–Sen does not assume normally distributed residuals or homoscedasticity. It simply relies on order statistics, making it suitable for real-world business metrics that often deviate from ideal statistical conditions.
- **Easy Interpretation:** The output is a “trend per unit time” (e.g., change per day), just like OLS. This consistency allows analysts and stakeholders to understand and compare slopes across different metrics.

## 2. Nonparametric Significance Testing with Mann–Kendall

- **Monotonic Trend Detection:** The Mann–Kendall test is designed to detect any consistent upward or downward movement, regardless of linearity. It is rank-based, so it flags monotonic shifts even if the data curve is nonlinear.
- **Handling Missing & Irregular Data:** Business data often has gaps or variable sampling intervals. Mann–Kendall naturally accommodates missing values (via `nan_policy='omit'`) and does not require equally spaced time stamps.
- **Extension for Seasonality:** When periodic patterns exist (e.g., weekly or monthly seasonality), the Seasonal Kendall variant can separate seasonal cycles from long-term trends, ensuring that significance relates to genuine drift rather than cyclical patterns.

## 3. Complementary Strengths

By combining Theil–Sen and Mann–Kendall, you get both: a robust magnitude of trend and a reliable assessment of its statistical significance. This duo outperforms traditional OLS+p-value methods when data are noisy, irregular, or non-normally distributed.

## 4. Practical Business Use Cases

- **Financial Metrics:** Track account balances or transaction volumes where spikes or drops may occur (e.g., market events). Theil–Sen avoids being misled by short-lived spikes, while Mann–Kendall confirms persistent direction.
- **Operational KPIs:** Monitor server response times or system errors over time, where sudden infractions should not distort the long-term slope.
- **Customer Behavior:** Analyze daily active users or churn rates, helping product teams understand genuine momentum rather than reacting to transient fluctuations.

## 5. Implementation Notes

- **Window Size Selection:** Choose a window length that balances responsiveness with stability. Short windows react quickly but can be noisy; longer windows smooth more but may lag in detecting change.
- **Thresholding:** Use the p-value from Mann–Kendall to filter for statistically significant trends (e.g., `p < 0.05`). For practical applications, adjust the threshold to control false alarms.

This explanation should help you understand the 'why' behind each component. Let me know if you’d like deeper notes on seasonal adjustments or performance considerations.

## 6. Forecasting Models

Beyond trend extraction, this package offers basic wrappers for ARIMA and
Holt-Winters exponential smoothing. They illustrate classic forecasting methods
using ``statsmodels`` with an interface consistent with the other helpers.
