"""Monte Carlo Experimentation Benchmarks: CUPED Variance Reduction & SPRT Error Control."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats

from src.core.cuped import CUPEDEngine
from src.core.sequential_sprt import SequentialSPRT, simulate_peeking_problem
from src.core.srm import calculate_srm
from src.power_analysis import PowerPlanner
from src.data.synthetic_generator import generate_experiment_data


def run_benchmarks(n_iterations: int = 200, n_users: int = 5000):
    print("=" * 70)
    print(" ENTERPRISE EXPERIMENTATION BENCHMARK SUITE")
    print("=" * 70)

    # 1. Benchmark: Empirical CUPED Variance Reduction & CI Tightening
    print("\n[1/3] Benchmarking CUPED Variance Reduction across Monte Carlo runs...")
    var_reductions = []
    sample_factors = []
    raw_ci_widths = []
    cuped_ci_widths = []

    for i in range(n_iterations):
        df = generate_experiment_data(
            n_users=n_users,
            true_lift_pct=0.04,
            pre_post_correlation=0.70,
            seed=1000 + i,
        )
        engine = CUPEDEngine(covariates=["pre_exp_revenue", "pre_exp_sessions"])
        res = engine.analyze_experiment(df, "variant", "revenue")
        
        var_reductions.append(res.variance_reduction_pct)
        sample_factors.append(res.sample_size_reduction_factor)
        
        raw_se = np.sqrt(res.unadjusted_var)
        cuped_se = np.sqrt(res.adjusted_var)
        raw_ci_widths.append(2 * 1.96 * raw_se)
        cuped_ci_widths.append(res.ci_upper - res.ci_lower)

    avg_var_reduction = float(np.mean(var_reductions))
    avg_sample_factor = float(np.mean(sample_factors))
    avg_raw_ci_width = float(np.mean(raw_ci_widths))
    avg_cuped_ci_width = float(np.mean(cuped_ci_widths))
    ci_tightening_pct = float((1.0 - avg_cuped_ci_width / avg_raw_ci_width) * 100.0)

    print(f"  -> Mean Variance Reduction: {avg_var_reduction:.2f}%")
    print(f"  -> Required Sample Size Factor: {avg_sample_factor:.3f}x ({(1 - avg_sample_factor)*100:.1f}% sample savings)")
    print(f"  -> 95% Confidence Interval Width: {avg_raw_ci_width:.3f} (Raw) vs {avg_cuped_ci_width:.3f} (CUPED) [-{ci_tightening_pct:.1f}%]")

    # 2. Benchmark: Naive Continuous Peeking vs. Wald's SPRT Alpha-Inflation
    print("\n[2/3] Benchmarking Peeking Problem under Null Hypothesis (H0: lift=0)...")
    t0 = time.time()
    naive_fpr, sprt_fpr = simulate_peeking_problem(
        n_sims=300,
        sample_size=600,
        peeks=15,
        alpha=0.05,
        sigma=10.0,
    )
    sim_time = time.time() - t0
    print(f"  -> Naive Continuous Peeking False-Positive Rate: {naive_fpr * 100:.2f}% (Expected ~20-25%)")
    print(f"  -> Sequential SPRT False-Positive Rate: {sprt_fpr * 100:.2f}% (Guaranteed <= 5.0%)")
    print(f"  -> Alpha Inflation Factor of Naive Peeking: {naive_fpr / (sprt_fpr + 1e-6):.2f}x")

    # 3. Benchmark: Power Analysis & Time-to-Decision Comparison
    print("\n[3/3] Benchmarking Sample Size Savings across Relative MDE targets...")
    baseline_mean = 100.0
    baseline_std = 25.0
    mdes = [0.03, 0.05, 0.08, 0.10]
    power_summary = []

    for mde in mdes:
        p_res = PowerPlanner.calculate_sample_size(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde_relative=mde,
            variance_reduction_factor=avg_sample_factor,
        )
        power_summary.append({
            "relative_mde_pct": mde * 100,
            "required_n_standard": p_res.required_sample_size_per_variant,
            "required_n_cuped": p_res.required_sample_size_cuped,
            "samples_saved": p_res.required_sample_size_per_variant - p_res.required_sample_size_cuped,
            "savings_pct": round(p_res.sample_savings_pct, 1),
        })

    # Summary Output
    benchmark_payload = {
        "cuped_benchmarks": {
            "iterations": n_iterations,
            "mean_variance_reduction_pct": round(avg_var_reduction, 2),
            "sample_size_multiplier": round(avg_sample_factor, 3),
            "ci_width_reduction_pct": round(ci_tightening_pct, 2),
        },
        "sequential_peeking_benchmarks": {
            "simulations": 300,
            "nominal_alpha": 0.05,
            "naive_peeking_false_positive_rate": round(naive_fpr, 4),
            "sprt_false_positive_rate": round(sprt_fpr, 4),
            "alpha_inflation_ratio": round(naive_fpr / (sprt_fpr + 1e-6), 2),
        },
        "sample_size_savings_by_mde": power_summary,
    }

    out_dir = os.path.dirname(__file__)
    out_file = os.path.join(out_dir, "experimentation_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print("\n" + "=" * 80)
    print(" EXPERIMENTATION BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Relative MDE':<15} | {'Std N / Variant':<18} | {'CUPED N / Variant':<18} | {'Samples Saved':<15} | {'Savings %'}")
    print("-" * 80)
    for row in power_summary:
        print(f"{row['relative_mde_pct']:>13.1f}% | {row['required_n_standard']:>18,d} | {row['required_n_cuped']:>18,d} | {row['samples_saved']:>15,d} | {row['savings_pct']:>8.1f}%")
    print("=" * 80)
    print(f"Results successfully saved to: {out_file}\n")


if __name__ == "__main__":
    run_benchmarks(n_iterations=100, n_users=3000)
