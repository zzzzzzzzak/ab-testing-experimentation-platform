# Enterprise A/B Testing & Decision Platform

[![CI](https://github.com/zzzzzzzzak/ab-testing-experimentation-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/zzzzzzzzak/ab-testing-experimentation-platform/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Author**: Siddiqui Zakir Ali • Senior ML & Analytics Engineer • [zak.siddiqui16@gmail.com](mailto:zak.siddiqui16@gmail.com) • [Live Portfolio](https://zzzzzzzzak.github.io/portfolio/)

A production-grade, mathematically rigorous experimentation and decision platform designed for enterprise data science teams. Implements **CUPED (Controlled-experiment Using Pre-Experiment Data)** variance reduction, **Wald's Sequential Probability Ratio Test (SPRT)** continuous monitoring, and **Sample Ratio Mismatch (SRM)** diagnostic forensics.

---

## Executive Summary: The Business Value

In high-growth tech enterprises, standard A/B testing faces three critical bottlenecks:
1. **Sample Inefficiency**: Standard tests require millions of impressions and weeks of execution to detect subtle single-digit metric lifts ($2\%-5\%$).
2. **The "Peeking" Fallacy**: Stakeholders checking dashboards daily inflate the true false-positive rate from the nominal $\alpha = 5\%$ to **over $25\%-30\%$**, shipping neutral or harmful features.
3. **Silent Telemetry Corruption**: Undetected Sample Ratio Mismatches (SRMs) caused by bot filters or mobile browser redirect drops corrupt causal inference.

This platform solves all three challenges:
- **CUPED Variance Reduction**: Achieves **$35\%-50\%$ variance reduction**, slashing required experiment duration and sample size by **$\sim 40\%$** without inducing estimator bias.
- **Sequential SPRT**: Permits continuous real-time peeking while mathematically bounding Type I error strictly $\le \alpha$.
- **Automated SRM Diagnostics**: Detects assignment skews via Chi-square goodness-of-fit and isolates the root-cause dimension (browser, OS, country).

---

## Mathematical Architecture

```
Experimentation System Architecture:
├── Pre-Experiment Baseline Telemetry (Covariates X_1, X_2)
│     │
│     ├──> 1. PowerPlanner (Sample Size & MDE Calculation)
│     │      ├── Analytical: N = 2 * (Z_alpha + Z_beta)^2 * (sigma^2 / delta^2)
│     │      └── CUPED-Adjusted: N_cuped = N * (1 - R^2)
│     │
│     ├──> 2. SRM Diagnostic Engine (Chi-Square Goodness-of-Fit)
│     │      ├── Chi2 = sum (Obs - Exp)^2 / Exp  (df = 1)
│     │      └── Multi-Dimensional Slicing (Localizing Biased Redirects)
│     │
│     ├──> 3. Multivariate CUPED Engine (OLS Covariate Residualization)
│     │      ├── theta = (X^T X)^(-1) X^T Y
│     │      └── Y_cuped = Y - theta * (X - E[X])
│     │
│     └──> 4. Wald's Sequential SPRT (Continuous Early Stopping)
│            ├── LLR_n = (delta / sigma^2) * sum(d_i - delta / 2)
│            ├── Upper Bound A = ln((1 - beta) / alpha)  --> Early Winner
│            └── Lower Bound B = ln(beta / (1 - alpha))  --> Early Futility
```

### 1. CUPED Variance Reduction
Given post-experiment outcome $Y$ and pre-experiment covariate vector $X$:
$$\hat{Y}_{\text{CUPED}} = Y - \theta (X - E[X])$$

Where $\theta$ is selected via ordinary least squares to minimize the variance of $\hat{Y}_{\text{CUPED}}$:
$$\theta = \frac{\text{Cov}(Y, X)}{\text{Var}(X)}$$

The resulting variance reduction ratio directly mirrors the coefficient of determination $R^2$:
$$\text{Var}(\hat{Y}_{\text{CUPED}}) = \text{Var}(Y) \cdot (1 - \rho^2)$$

### 2. Wald's Sequential Probability Ratio Test (SPRT)
For continuous monitoring without p-value peeking penalties, the cumulative log-likelihood ratio under normal incremental lift $d_i \sim \mathcal{N}(\delta, 2\sigma^2)$ is evaluated against analytical thresholds:

$$\Lambda_n = \frac{\delta_1}{2\sigma^2} \sum_{i=1}^n \left(d_i - \frac{\delta_1}{2}\right)$$

- **Reject $H_0$ (Declare Winner)** if:
  $$\Lambda_n \ge \ln\left(\frac{1 - \beta}{\alpha}\right)$$
- **Accept $H_0$ (Declare Futility)** if:
  $$\Lambda_n \le \ln\left(\frac{\beta}{1 - \alpha}\right)$$

---

## Empirical Benchmark Results

Benchmarked across 300 Monte Carlo experiment simulations with $N = 3,000$ users per iteration:

### 1. Sample Size Savings via CUPED

| Relative MDE | Standard N / Variant | CUPED N / Variant | Samples Saved | Sample Savings % |
| :---: | :---: | :---: | :---: | :---: |
| **3.0%** | 10,885 | 6,531 | **4,354** | **40.0%** |
| **5.0%** | 3,919 | 2,351 | **1,568** | **40.0%** |
| **8.0%** | 1,531 | 919 | **612** | **40.0%** |
| **10.0%** | 980 | 588 | **392** | **40.0%** |

### 2. Peeking Error Control (H0 True Null Simulation)

| Method | Nominal $\alpha$ | Observed False-Positive Rate | Outcome |
| :--- | :---: | :---: | :--- |
| **Fixed Horizon Test** | 5.0% | 4.9% | Baseline standard |
| **Naive Daily Peeking (t-test)** | 5.0% | **24.6%** | **4.92× Alpha Inflation** (Severe Error) |
| **Sequential SPRT** | 5.0% | **4.1%** | **Strict Error Bounded** ($\le \alpha$) |

---

## Project Structure

```
ab-testing-experimentation-platform/
├── .github/
│   └── workflows/
│       └── ci.yml               # Multi-version CI matrix with pytest & coverage
├── benchmarks/
│   ├── run_benchmarks.py        # Monte Carlo CUPED & SPRT benchmark runner
│   └── experimentation_benchmark_results.json
├── src/
│   ├── config.py                # Dataclasses & experiment specifications
│   ├── power_analysis.py        # Sample size & MDE sensitivity calculations
│   ├── core/
│   │   ├── cuped.py             # Univariate & multivariate CUPED engine
│   │   ├── srm.py               # Chi-Square goodness-of-fit SRM diagnostics
│   │   ├── sequential_sprt.py   # Wald's SPRT sequential testing monitor
│   │   └── hypothesis.py        # Welch's t-test, Mann-Whitney U, Bootstrap BCa
│   └── data/
│       └── synthetic_generator.py # User telemetry & SRM corruptor generator
├── tests/
│   ├── test_cuped.py            # Unbiasedness, variance reduction & CI parity
│   ├── test_srm.py              # Balanced vs skewed traffic checks
│   ├── test_sequential_sprt.py  # Stopping boundary & peeking validation
│   ├── test_hypothesis.py       # Welch, Mann-Whitney & Bootstrap tests
│   ├── test_power_analysis.py   # Power & sample size formulas
│   └── test_synthetic_generator.py
├── app.py                       # Interactive Streamlit command center
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quickstart & Installation

```bash
# Clone the repository
git clone https://github.com/zzzzzzzzak/ab-testing-experimentation-platform.git
cd ab-testing-experimentation-platform

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run unit tests
pytest tests/ -v --cov=src

# Run Monte Carlo benchmarks
python benchmarks/run_benchmarks.py

# Launch interactive Streamlit decision dashboard
streamlit run app.py
```

---

## Programmatic API Example

```python
from src.core.cuped import CUPEDEngine
from src.core.srm import calculate_srm
from src.data.synthetic_generator import generate_experiment_data

# 1. Generate experiment telemetry
df = generate_experiment_data(n_users=5000, true_lift_pct=0.05, seed=42)

# 2. Check Sample Ratio Mismatch (SRM)
c_count = (df["variant"] == "control").sum()
t_count = (df["variant"] == "treatment").sum()
srm_result = calculate_srm(c_count, t_count)
print(f"SRM Status: {srm_result.severity} (p-value: {srm_result.p_value:.4f})")

# 3. Apply CUPED Variance Reduction
engine = CUPEDEngine(covariates=["pre_exp_revenue", "pre_exp_sessions"])
cuped_result = engine.analyze_experiment(df, variant_col="variant", target_metric="revenue")

print(f"Unadjusted Lift: ${cuped_result.unadjusted_diff:.2f} (p = {cuped_result.unadjusted_p_value:.4f})")
print(f"CUPED Adjusted Lift: ${cuped_result.adjusted_diff:.2f} (p = {cuped_result.adjusted_p_value:.4f})")
print(f"Variance Reduced: {cuped_result.variance_reduction_pct:.1f}%")
print(f"Required Sample Size Factor: {cuped_result.sample_size_reduction_factor:.2f}x")
```

---

## License

MIT License. Developed by Siddiqui Zakir Ali.
