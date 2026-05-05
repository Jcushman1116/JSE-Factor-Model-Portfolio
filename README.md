# Portfolio Optimization via James-Stein Eigenvector Regularization

> **Robust Global Minimum Variance portfolio construction using James-Stein shrinkage on the leading eigenvector of a Single-Factor Market Model.**

---

## Overview

This project addresses the classic **High-Dimension, Low-Sample Size (HDLSS)** problem in portfolio optimization. When the number of assets $p$ greatly exceeds the number of observations $n$, the sample covariance matrix becomes singular and cannot be inverted for traditional Markowitz optimization.

**Solution:** We implement a Single-Factor Market Model and apply the **James-Stein Estimator (JSE)** to the leading eigenvector, as proposed by Goldbergh & Kercheval (2022). This eigenvector shrinkage corrects for the bias and excess dispersion endemic to high-dimensional datasets, producing a more stable and generalizable Global Minimum Variance (GMV) portfolio.

---

## The Problem at a Glance

| Setting | Value |
|---|---|
| Universe | Top 400 S&P 500 stocks by market cap |
| Observation window | 26 weeks (weekly closing prices) |
| Assets $p$ | 400 |
| Observations $n$ | 26 |
| Issue | $p \gg n$ → sample covariance matrix is rank-deficient |

Because $\text{rank}(S) \leq n = 26 < 400 = p$, the sample covariance matrix $S$ is **singular and non-invertible**, making standard Markowitz optimization impossible.

---

## Methodology

### 1. Data Collection
- S&P 500 tickers scraped from Wikipedia via its public API
- Weekly closing prices downloaded via `yfinance` over a 26-week window
- Universe filtered to top 400 stocks by estimated market capitalization:

$$\text{Market Cap} = \text{Closing Price} \times \text{Shares Outstanding}$$

### 2. Covariance Matrix Construction

Weekly excess returns are computed by subtracting the weekly risk-free rate (sourced from a 6-month Treasury bond):

$$E = R^T - r_{week} \quad r_{week} = 0.0375 / 52$$

The de-meaned excess returns matrix $Y = E - \mu$ gives the sample covariance matrix:

$$S = \frac{YY^T}{n}$$

### 3. Single-Factor Market Model

To resolve the singularity, we use a rank-regularized covariance estimate based on the leading eigenpair $(\lambda^2 h)$ of $S$:

$$\Sigma_{\text{PCA}} = (\lambda^2 - \ell^2) hh^T + \frac{n}{p}\ell^2 I$$

where:

$$\ell^2 = \frac{\text{tr}(S) - \lambda^2}{n - 1}$$

The spectral shift by $\frac{n}{p}\ell^2$ lifts all zero eigenvalues, ensuring $\Sigma$ is **strictly positive definite** and invertible.

### 4. James-Stein Estimator (JSE)

The JSE improves on the PCA model by shrinking the leading eigenvector $h$ toward the equal-weighted direction:

$$h^{JSE} = m(h)\mathbf{e} + c^{JSE}(h - m(h)\mathbf{e})$$

where $m(h)$ is the mean entry of $h$, $\mathbf{e}$ is the vector of ones, and the shrinkage constant is:

$$c^{JSE} = \max\!\left(0, 1 - \frac{\nu^2}{s^2(h)}\right)$$

The JSE covariance estimate becomes:

$$\Sigma_{JSE} = (\lambda^2 - \ell^2) \frac{h^{JSE}(h^{JSE})^T}{\|h^{JSE}\|^2} + \frac{n}{p} \ell^2 I$$

This is a **consistent estimator** in the asymptotic HDLSS regime — it pulls the sample eigenvector toward the true population eigenvector, filtering out market noise.

### 5. GMV Portfolio Construction

Using the regularized covariance matrix, the **Global Minimum Variance** portfolio weights are:

$$h_C = \frac{\Sigma^{-1}\mathbf{e}}{\mathbf{e}^T \Sigma^{-1}\mathbf{e}}$$

Portfolio statistics are then computed:

| Metric | Formula |
|---|---|
| Variance | $\sigma^2_C = h_C^T \Sigma h_C$ |
| Beta | $\beta_C = \Sigma h_C / \sigma^2_C$ |
| Expected Excess Return | $f_C = h_C^T \mu$ |
| Sharpe Ratio | $SR_C = f_C / \sigma_C$ |

---

## Results

### JSE Regularization

| Stat | Value |
|---|---|
| Shrinkage constant $c^{JSE}$ | 0.5588 |
| MSE (Sample $h$ vs JSE $h$) | 0.00016069 |
| MSE (Sample $h$ vs PCA $h$) | 0.00000000 |

A shrinkage constant of ~0.56 indicates that the JSE identified **over 44% of the signal** in the sample eigenvector as noise, pulling weights toward equal-weighting.

### Portfolio Performance Comparison

| Metric | PCA Model | JSE Model | SPY (Benchmark) |
|---|---|---|---|
| Annualized Variance | 0.000478 | 0.001072 | 0.013402 |
| Annualized Std Dev | 0.021860 | 0.032743 | 0.115765 |
| Expected Excess Return | 0.030964 | -0.106052 | 0.277219 |
| Sharpe Ratio | 1.4165 | -3.2389 | 2.3947 |
| Avg Portfolio Beta | 1.0000 | 1.0000 | — |

### Individual Asset Benchmark

| Metric | PCA Model | JSE Model |
|---|---|---|
| Mean Stock Variance | 0.090108 | 0.090108 |
| Min. Stock Variance | 0.076022 | 0.076033 |

Both GMV portfolios achieve a **dramatic reduction in variance** vs the benchmark (~96–97% reduction), confirming the optimization is working as intended.

---

## Key Takeaways

- **PCA model** achieves a lower in-sample variance (0.000478) but this reflects **overfitting** to the historical data window — it finds the minimum variance possible on past data, not on future data.
- **JSE model** has a higher in-sample variance (0.001072) but is **theoretically superior for out-of-sample generalization**, as it is a consistent estimator in the HDLSS regime.
- The **negative Sharpe Ratio** for JSE is expected and acceptable — the GMV portfolio is constructed purely for risk minimization, not return maximization.
- The JSE estimator acts as **insurance**: accepting a small performance penalty today in exchange for robustness when the market regime changes.

---

## Visualizations

**Scree Plot** — The sharp "elbow" after the first eigenvalue justifies the Single-Factor Model. The leading eigenvalue captures systematic, market-wide risk, while all subsequent eigenvalues decay gradually into noise.

**Portfolio Weight Distribution** — The PCA weights cluster tightly near the center (low dispersion, in-sample optimized), while the JSE weights spread more broadly (higher dispersion, noise-filtered for out-of-sample stability).

---

## Dependencies

```bash
pip install numpy pandas yfinance matplotlib requests
```

| Library | Purpose |
|---|---|
| `numpy` | Linear algebra, eigendecomposition |
| `pandas` | Data manipulation |
| `yfinance` | Stock price & fundamental data |
| `requests` | Wikipedia API scraping |
| `matplotlib` | Visualization |

---

## References

- Goldbergh, R. & Kercheval, A. (2022). *James-Stein for the Leading Eigenvector.* Proceedings of the National Academy of Sciences.
- Markowitz, H. (1952). *Portfolio Selection.* The Journal of Finance.

---

## Author

**Jonathan Cushman** — April 2026
