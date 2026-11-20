"""Synthetic User Telemetry Generator for Experimentation."""

from typing import Optional
import numpy as np
import pandas as pd


def generate_experiment_data(
    n_users: int = 10000,
    traffic_split: float = 0.5,
    true_lift_pct: float = 0.04,  # 4% true treatment lift
    pre_post_correlation: float = 0.65,
    srm_bias_factor: float = 0.0,  # > 0 injects intentional Sample Ratio Mismatch
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic user-level experimentation data with pre- and post-experiment metrics.
    
    Features:
        - Pre-experiment revenue and sessions with log-normal distribution.
        - Correlated post-experiment outcome matching configurable rho.
        - Multi-categorical dimensions (device, browser, geo) for SRM slice forensics.
        - True causal lift injected into treatment.
    """
    rng = np.random.default_rng(seed)

    # 1. Dimensions
    devices = rng.choice(["Mobile", "Desktop", "Tablet"], size=n_users, p=[0.60, 0.35, 0.05])
    browsers = rng.choice(["Chrome", "Safari", "Edge", "Firefox"], size=n_users, p=[0.55, 0.30, 0.10, 0.05])
    countries = rng.choice(["US", "UK", "DE", "IN", "CA"], size=n_users, p=[0.45, 0.20, 0.15, 0.12, 0.08])

    # 2. Pre-experiment baseline metrics (correlated bivariate log-normal)
    # Latent base spend factor
    z_base = rng.standard_normal(n_users)
    z_noise = rng.standard_normal(n_users)

    pre_sessions = np.clip(np.exp(1.8 + 0.5 * z_base), 1, 200).round()
    pre_revenue = np.clip(np.exp(3.5 + 0.8 * z_base + 0.3 * z_noise), 0, 5000).round(2)

    # 3. Variant Assignment
    # If srm_bias_factor > 0, divert treatment users on Safari (common mobile redirect bug)
    if srm_bias_factor > 0:
        p_treatment = np.where(browsers == "Safari", traffic_split * (1.0 - srm_bias_factor), traffic_split)
        variants = np.where(rng.uniform(size=n_users) < p_treatment, "treatment", "control")
    else:
        variants = np.where(rng.uniform(size=n_users) < traffic_split, "treatment", "control")

    # 4. Post-experiment metric generation with target correlation rho
    # Y = rho * X_std + sqrt(1 - rho^2) * noise
    rho = float(np.clip(pre_post_correlation, 0.0, 0.95))
    noise = rng.standard_normal(n_users)

    # Standardized pre-revenue
    pre_rev_log = np.log1p(pre_revenue)
    pre_std = (pre_rev_log - np.mean(pre_rev_log)) / (np.std(pre_rev_log) + 1e-12)

    post_latent = rho * pre_std + np.sqrt(1.0 - rho ** 2) * noise

    # Map back to post revenue scale
    post_revenue = np.exp(3.6 + 0.8 * post_latent)
    post_sessions = np.clip(pre_sessions + rng.integers(-2, 5, size=n_users), 1, 300)

    # 5. Inject true treatment effect
    treatment_mask = variants == "treatment"
    post_revenue[treatment_mask] *= (1.0 + true_lift_pct)

    df = pd.DataFrame({
        "user_id": [f"usr_{i:06d}" for i in range(1, n_users + 1)],
        "variant": variants,
        "device": devices,
        "browser": browsers,
        "country": countries,
        "pre_exp_sessions": pre_sessions.astype(int),
        "pre_exp_revenue": pre_revenue,
        "post_exp_sessions": post_sessions.astype(int),
        "revenue": np.round(post_revenue, 2),
    })

    return df
