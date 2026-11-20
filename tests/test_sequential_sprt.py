"""Unit tests for Wald's Sequential Probability Ratio Test (SPRT)."""

import pytest
import numpy as np
from src.core.sequential_sprt import SequentialSPRT, simulate_peeking_problem


def test_sprt_stopping_boundaries():
    sprt = SequentialSPRT(alpha=0.05, beta=0.20, mde_abs=2.0)
    # Upper bound = ln((1 - 0.20) / 0.05) = ln(16) ~ 2.772588
    # Lower bound = ln(0.20 / (1 - 0.05)) = ln(0.20 / 0.95) ~ -1.558145
    assert np.isclose(sprt.upper_bound, np.log(16.0), atol=1e-4)
    assert np.isclose(sprt.lower_bound, np.log(0.20 / 0.95), atol=1e-4)


def test_sprt_early_winner_stopping():
    # Strong treatment effect should stop early with STOP_REJECT_NULL
    rng = np.random.default_rng(42)
    c = rng.normal(100.0, 5.0, 2000)
    t = rng.normal(108.0, 5.0, 2000)  # Strong lift of +8.0

    sprt = SequentialSPRT(alpha=0.05, beta=0.20, mde_abs=2.0, assumed_sigma=5.0)
    res = sprt.evaluate_stream(c, t, step_batch_size=50)

    assert res.decision == "STOP_REJECT_NULL"
    assert res.stopped_at_step < 2000
    assert res.final_llr >= res.upper_bound


def test_sprt_futility_stopping():
    # Negative treatment effect should trigger futility early
    rng = np.random.default_rng(99)
    c = rng.normal(100.0, 5.0, 2000)
    t = rng.normal(95.0, 5.0, 2000)  # Degraded treatment

    sprt = SequentialSPRT(alpha=0.05, beta=0.20, mde_abs=2.0, assumed_sigma=5.0)
    res = sprt.evaluate_stream(c, t, step_batch_size=50)

    assert res.decision == "STOP_ACCEPT_NULL"
    assert res.stopped_at_step < 2000
    assert res.final_llr <= res.lower_bound


def test_peeking_simulation_error_control():
    # Quick test verifying SPRT maintains lower false positive rate than naive peeking
    naive_fpr, sprt_fpr = simulate_peeking_problem(n_sims=80, sample_size=400, peeks=10, alpha=0.05)
    # Naive peeking should show elevated FPR relative to SPRT
    assert naive_fpr >= sprt_fpr
