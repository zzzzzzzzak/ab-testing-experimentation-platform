"""Configuration dataclasses for experiment design, CUPED, SPRT, and SRM."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """Experiment specification and metadata."""
    name: str = "pricing_checkout_v2"
    control_label: str = "control"
    treatment_label: str = "treatment"
    traffic_split: float = 0.5  # 50% treatment
    target_metric: str = "revenue"
    covariates: List[str] = field(default_factory=lambda: ["pre_exp_revenue", "pre_exp_sessions"])
    alpha: float = 0.05
    power: float = 0.80
    mde_relative: float = 0.05  # 5% relative minimum detectable effect


@dataclass
class SRMResult:
    """Sample Ratio Mismatch diagnostic outcome."""
    control_count: int
    treatment_count: int
    total_count: int
    expected_ratio: float
    observed_ratio: float
    chi2_stat: float
    p_value: float
    is_srm: bool
    severity: str  # "NONE", "WARNING", "CRITICAL"
    segment_breakdown: Optional[Dict[str, Dict[str, float]]] = None


@dataclass
class CUPEDResult:
    """CUPED variance reduction and adjusted treatment estimation."""
    unadjusted_control_mean: float
    unadjusted_treatment_mean: float
    unadjusted_diff: float
    unadjusted_var: float
    unadjusted_p_value: float
    
    adjusted_control_mean: float
    adjusted_treatment_mean: float
    adjusted_diff: float
    adjusted_var: float
    adjusted_p_value: float
    
    theta: Dict[str, float]
    variance_reduction_pct: float
    sample_size_reduction_factor: float
    ci_lower: float
    ci_upper: float


@dataclass
class SPRTStep:
    """A single sequential decision step."""
    step: int
    control_mean: float
    treatment_mean: float
    log_likelihood_ratio: float
    lower_bound: float
    upper_bound: float
    decision: str  # "CONTINUE", "STOP_REJECT_NULL", "STOP_ACCEPT_NULL"


@dataclass
class SPRTResult:
    """Outcome of Sequential Probability Ratio Test."""
    decision: str
    stopped_at_step: int
    total_steps: int
    final_llr: float
    lower_bound: float
    upper_bound: float
    alpha: float
    beta: float
    steps: List[SPRTStep] = field(default_factory=list)


@dataclass
class HypothesisTestResult:
    """Comprehensive test outcome with frequentist and bootstrap estimates."""
    test_type: str
    control_mean: float
    treatment_mean: float
    diff: float
    relative_lift_pct: float
    p_value: float
    stat: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


@dataclass
class PowerAnalysisResult:
    """Power and sample size recommendation."""
    baseline_mean: float
    baseline_std: float
    mde_relative: float
    mde_absolute: float
    alpha: float
    power: float
    required_sample_size_per_variant: int
    required_sample_size_cuped: int
    sample_savings_pct: float
