"""Unit tests for hypothesis testing and multiple testing corrections."""

import pytest
import numpy as np
from src.core.hypothesis import (
    welch_ttest,
    mann_whitney_test,
    bootstrap_ci,
    multiple_testing_correction,
)


def test_welch_ttest():
    rng = np.random.default_rng(123)
    c = rng.normal(50.0, 5.0, 500)
    t = rng.normal(53.0, 6.0, 500)  # Significant lift

    res = welch_ttest(c, t, alpha=0.05)
    assert res.is_significant
    assert res.p_value < 1e-4
    assert res.diff > 2.0
    assert res.ci_lower < res.diff < res.ci_upper


def test_mann_whitney_u():
    rng = np.random.default_rng(456)
    c = rng.exponential(10.0, 400)
    t = rng.exponential(15.0, 400)

    res = mann_whitney_test(c, t, alpha=0.05)
    assert res.is_significant
    assert res.p_value < 0.05


def test_bootstrap_ci():
    rng = np.random.default_rng(789)
    c = rng.normal(20.0, 2.0, 200)
    t = rng.normal(22.0, 2.0, 200)

    diff, low, high = bootstrap_ci(c, t, n_bootstraps=500, alpha=0.05, seed=42)
    assert low < diff < high
    assert low > 0.5  # Lift is distinctly positive


def test_multiple_testing_benjamini_hochberg():
    raw_p = [0.001, 0.015, 0.035, 0.045, 0.20, 0.65]
    adj_p, rejections = multiple_testing_correction(raw_p, method="fdr_bh", alpha=0.05)

    assert len(adj_p) == len(raw_p)
    # p-values must be monotonically adjusted
    for r, a in zip(raw_p, adj_p):
        assert a >= r
    assert rejections[0] is True  # 0.001 should remain significant


def test_multiple_testing_bonferroni():
    raw_p = [0.01, 0.04, 0.10]
    adj_p, rejections = multiple_testing_correction(raw_p, method="bonferroni", alpha=0.05)
    # Bonferroni multiplies by m=3
    assert np.isclose(adj_p[0], 0.03)
    assert np.isclose(adj_p[1], 0.12)
    assert np.isclose(adj_p[2], 0.30)
