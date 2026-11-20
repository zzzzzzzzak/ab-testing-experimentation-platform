"""Unit tests for CUPED variance reduction."""

import pytest
import numpy as np
import pandas as pd
from src.core.cuped import CUPEDEngine
from src.data.synthetic_generator import generate_experiment_data


def test_cuped_unbiasedness_and_mean_preservation():
    # CUPED estimator must be unbiased: E[Y_cuped] = E[Y]
    df = generate_experiment_data(n_users=5000, true_lift_pct=0.05, seed=123)
    engine = CUPEDEngine(covariates=["pre_exp_revenue"])
    engine.fit(df, "revenue")
    y_cuped = engine.transform(df, "revenue")

    mean_raw = np.mean(df["revenue"])
    mean_cuped = np.mean(y_cuped)
    assert np.isclose(mean_raw, mean_cuped, rtol=1e-4)


def test_cuped_variance_reduction():
    # Positively correlated pre-experiment covariate must reduce variance
    df = generate_experiment_data(n_users=5000, pre_post_correlation=0.75, seed=42)
    engine = CUPEDEngine(covariates=["pre_exp_revenue"])
    res = engine.analyze_experiment(df, "variant", "revenue")

    assert res.variance_reduction_pct > 25.0
    assert res.sample_size_reduction_factor < 0.75
    assert res.adjusted_var < res.unadjusted_var
    # Theta must be positive
    assert res.theta["pre_exp_revenue"] > 0.0


def test_cuped_confidence_interval_tightening():
    df = generate_experiment_data(n_users=6000, pre_post_correlation=0.70, seed=99)
    engine = CUPEDEngine(covariates=["pre_exp_revenue", "pre_exp_sessions"])
    res = engine.analyze_experiment(df, "variant", "revenue")

    raw_se = np.sqrt(res.unadjusted_var)
    cuped_se = np.sqrt(res.adjusted_var)
    assert cuped_se < raw_se
    # Confidence interval width must be narrower
    raw_ci_width = 2 * 1.96 * raw_se
    cuped_ci_width = res.ci_upper - res.ci_lower
    assert cuped_ci_width < raw_ci_width


def test_cuped_multivariate_covariates():
    df = generate_experiment_data(n_users=4000, seed=77)
    engine = CUPEDEngine(covariates=["pre_exp_revenue", "pre_exp_sessions"])
    engine.fit(df, "revenue")
    assert "pre_exp_revenue" in engine.thetas
    assert "pre_exp_sessions" in engine.thetas
