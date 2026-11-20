"""Power Analysis, Minimum Detectable Effect (MDE), and Sample Size Planning."""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

from src.config import PowerAnalysisResult


class PowerPlanner:
    """Enterprise Experiment Power & Sample Size Planner."""

    @staticmethod
    def calculate_sample_size(
        baseline_mean: float,
        baseline_std: float,
        mde_relative: float = 0.05,
        alpha: float = 0.05,
        power: float = 0.80,
        variance_reduction_factor: float = 1.0,
    ) -> PowerAnalysisResult:
        """Calculate required sample size per variant for continuous metrics.
        
        Formula:
            N = 2 * (Z_{1-alpha/2} + Z_{1-beta})^2 * (sigma^2 / delta^2)
            
        Under CUPED with variance reduction ratio R:
            N_cuped = N * R
        """
        delta = abs(baseline_mean * mde_relative)
        if delta == 0:
            raise ValueError("MDE absolute difference cannot be zero.")

        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(power)
        multiplier = (z_alpha + z_beta) ** 2

        # Standard unadjusted sample size per variant
        n_raw = 2.0 * multiplier * (baseline_std ** 2) / (delta ** 2)
        n_per_variant = int(np.ceil(n_raw))

        # CUPED reduced sample size
        n_cuped_raw = n_raw * variance_reduction_factor
        n_cuped = int(np.ceil(n_cuped_raw))
        savings_pct = 0.0 if n_cuped >= n_per_variant else float((1.0 - n_cuped / n_per_variant) * 100.0)

        return PowerAnalysisResult(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            mde_relative=mde_relative,
            mde_absolute=delta,
            alpha=alpha,
            power=power,
            required_sample_size_per_variant=n_per_variant,
            required_sample_size_cuped=n_cuped,
            sample_savings_pct=savings_pct,
        )

    @staticmethod
    def calculate_mde(
        sample_size_per_variant: int,
        baseline_mean: float,
        baseline_std: float,
        alpha: float = 0.05,
        power: float = 0.80,
        variance_reduction_factor: float = 1.0,
    ) -> float:
        """Calculate the Minimum Detectable Effect achievable for a given sample size."""
        if sample_size_per_variant <= 0:
            return 0.0

        z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
        z_beta = stats.norm.ppf(power)
        multiplier = z_alpha + z_beta

        effective_std = baseline_std * np.sqrt(variance_reduction_factor)
        delta = multiplier * np.sqrt(2.0 * (effective_std ** 2) / sample_size_per_variant)
        return float(delta / (baseline_mean + 1e-12))

    @classmethod
    def generate_mde_sensitivity_curve(
        cls,
        sample_sizes: List[int],
        baseline_mean: float,
        baseline_std: float,
        cuped_r2: float = 0.40,
        alpha: float = 0.05,
        power: float = 0.80,
    ) -> Dict[str, List[float]]:
        """Generate comparative MDE curve data (Standard vs CUPED) across sample sizes."""
        standard_mdes = []
        cuped_mdes = []
        vr_factor = 1.0 - cuped_r2

        for n in sample_sizes:
            mde_std = cls.calculate_mde(n, baseline_mean, baseline_std, alpha, power, 1.0)
            mde_cup = cls.calculate_mde(n, baseline_mean, baseline_std, alpha, power, vr_factor)
            standard_mdes.append(round(mde_std * 100, 2))
            cuped_mdes.append(round(mde_cup * 100, 2))

        return {
            "sample_sizes": sample_sizes,
            "standard_mde_pct": standard_mdes,
            "cuped_mde_pct": cuped_mdes,
        }
