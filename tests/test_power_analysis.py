"""Unit tests for power analysis and sample size planning."""

import pytest
import numpy as np
from src.power_analysis import PowerPlanner


def test_power_sample_size_calculation():
    # Standard benchmark: baseline=100, std=20, MDE=5% (delta=5)
    res = PowerPlanner.calculate_sample_size(
        baseline_mean=100.0,
        baseline_std=20.0,
        mde_relative=0.05,
        alpha=0.05,
        power=0.80,
    )

    assert res.required_sample_size_per_variant > 200
    assert res.mde_absolute == 5.0
    assert np.isclose(res.sample_savings_pct, 0.0, atol=1e-5)


def test_power_cuped_savings():
    # With 40% variance reduction (factor=0.60), CUPED should require 40% fewer samples
    res = PowerPlanner.calculate_sample_size(
        baseline_mean=100.0,
        baseline_std=20.0,
        mde_relative=0.05,
        alpha=0.05,
        power=0.80,
        variance_reduction_factor=0.60,
    )

    assert res.required_sample_size_cuped < res.required_sample_size_per_variant
    assert 38.0 <= res.sample_savings_pct <= 42.0


def test_mde_sensitivity_curve():
    sizes = [500, 1000, 2500, 5000]
    curve = PowerPlanner.generate_mde_sensitivity_curve(
        sample_sizes=sizes,
        baseline_mean=50.0,
        baseline_std=15.0,
        cuped_r2=0.36,
    )

    assert len(curve["standard_mde_pct"]) == len(sizes)
    assert len(curve["cuped_mde_pct"]) == len(sizes)
    # Larger sample size must yield smaller MDE
    assert curve["standard_mde_pct"][0] > curve["standard_mde_pct"][-1]
    # CUPED MDE must be strictly smaller than standard MDE
    for std_mde, cup_mde in zip(curve["standard_mde_pct"], curve["cuped_mde_pct"]):
        assert cup_mde < std_mde
