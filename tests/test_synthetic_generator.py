"""Unit tests for synthetic experimentation telemetry generator."""

import pytest
import numpy as np
from src.data.synthetic_generator import generate_experiment_data
from src.core.srm import calculate_srm


def test_synthetic_data_schema():
    df = generate_experiment_data(n_users=1000, seed=42)
    expected_cols = [
        "user_id", "variant", "device", "browser", "country",
        "pre_exp_sessions", "pre_exp_revenue", "post_exp_sessions", "revenue"
    ]
    for col in expected_cols:
        assert col in df.columns

    assert len(df) == 1000
    assert set(df["variant"].unique()) == {"control", "treatment"}


def test_synthetic_data_correlation():
    df = generate_experiment_data(n_users=3000, pre_post_correlation=0.70, seed=123)
    # Check correlation between pre and post revenue
    corr = np.corrcoef(df["pre_exp_revenue"], df["revenue"])[0, 1]
    assert corr > 0.40  # Positively correlated


def test_synthetic_data_srm_injection():
    # When srm_bias_factor > 0, SRM check must flag divergence
    df_srm = generate_experiment_data(n_users=8000, srm_bias_factor=0.6, seed=77)
    c_cnt = int(np.sum(df_srm["variant"] == "control"))
    t_cnt = int(np.sum(df_srm["variant"] == "treatment"))
    res = calculate_srm(c_cnt, t_cnt)
    assert res.is_srm
