"""Robust Hypothesis Testing, Bootstrap Estimation & Multiple Testing Corrections."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import HypothesisTestResult


def welch_ttest(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Welch's t-test allowing for unequal sample sizes and variances.
    
    Robust baseline test for continuous metric comparison.
    """
    c = np.asarray(control, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.float64)

    mean_c = float(np.mean(c))
    mean_t = float(np.mean(t))
    diff = mean_t - mean_c
    relative_lift = (diff / (mean_c + 1e-12)) * 100.0

    stat, p_val = stats.ttest_ind(t, c, equal_var=False)

    # Analytical Confidence Interval
    var_c = np.var(c, ddof=1) / len(c)
    var_t = np.var(t, ddof=1) / len(t)
    se = np.sqrt(var_c + var_t)

    # Welch-Satterthwaite degrees of freedom
    df = ((var_c + var_t) ** 2) / (
        (var_c ** 2) / (len(c) - 1) + (var_t ** 2) / (len(t) - 1) + 1e-12
    )
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    ci_lower = float(diff - t_crit * se)
    ci_upper = float(diff + t_crit * se)

    return HypothesisTestResult(
        test_type="Welch's t-test",
        control_mean=mean_c,
        treatment_mean=mean_t,
        diff=diff,
        relative_lift_pct=relative_lift,
        p_value=float(p_val),
        stat=float(stat),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        is_significant=bool(p_val < alpha),
    )


def mann_whitney_test(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
) -> HypothesisTestResult:
    """Mann-Whitney U rank-sum non-parametric test.
    
    Ideal for heavy-tailed, zero-inflated, or highly skewed distributions.
    """
    c = np.asarray(control, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.float64)

    stat, p_val = stats.mannwhitneyu(t, c, alternative="two-sided")
    mean_c = float(np.mean(c))
    mean_t = float(np.mean(t))
    diff = mean_t - mean_c
    relative_lift = (diff / (mean_c + 1e-12)) * 100.0

    return HypothesisTestResult(
        test_type="Mann-Whitney U Test",
        control_mean=mean_c,
        treatment_mean=mean_t,
        diff=diff,
        relative_lift_pct=relative_lift,
        p_value=float(p_val),
        stat=float(stat),
        ci_lower=0.0,
        ci_upper=0.0,
        is_significant=bool(p_val < alpha),
    )


def bootstrap_ci(
    control: np.ndarray,
    treatment: np.ndarray,
    n_bootstraps: int = 1500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute empirical percentile bootstrap confidence interval for mean difference.
    
    Returns:
        (diff, ci_lower, ci_upper)
    """
    c = np.asarray(control, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.float64)
    diff = float(np.mean(t) - np.mean(c))

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        boot_c = rng.choice(c, size=len(c), replace=True)
        boot_t = rng.choice(t, size=len(t), replace=True)
        boot_diffs[i] = np.mean(boot_t) - np.mean(boot_c)

    ci_lower = float(np.percentile(boot_diffs, (alpha / 2.0) * 100.0))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha / 2.0) * 100.0))
    return diff, ci_lower, ci_upper


def multiple_testing_correction(
    p_values: List[float],
    method: str = "fdr_bh",
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """Apply False Discovery Rate (Benjamini-Hochberg) or FWER (Bonferroni) corrections.
    
    Returns:
        (adjusted_p_values, is_significant_flags)
    """
    m = len(p_values)
    if m == 0:
        return [], []

    p_arr = np.asarray(p_values, dtype=np.float64)

    if method == "bonferroni":
        adj_p = np.minimum(1.0, p_arr * m)
        rejected = adj_p < alpha
    elif method == "fdr_bh":
        # Benjamini-Hochberg rank adjustment
        order = np.argsort(p_arr)
        sorted_p = p_arr[order]
        ranks = np.arange(1, m + 1)
        
        # p_adj = min(1, p * m / rank) with monotonicity constraint backwards
        raw_adj = sorted_p * (m / ranks)
        adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
        adj_sorted = np.minimum(1.0, adj_sorted)
        
        adj_p = np.empty_like(adj_sorted)
        adj_p[order] = adj_sorted
        rejected = adj_p < alpha
    else:
        raise ValueError(f"Unknown correction method: {method}")

    return [float(x) for x in adj_p], [bool(x) for x in rejected]
