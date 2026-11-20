"""Sample Ratio Mismatch (SRM) Diagnostic Engine."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.config import SRMResult


def calculate_srm(
    control_count: int,
    treatment_count: int,
    expected_ratio: float = 0.5,
    alpha_critical: float = 0.001,
    alpha_warning: float = 0.01,
) -> SRMResult:
    """Evaluate Sample Ratio Mismatch via Chi-Square goodness-of-fit test.
    
    Args:
        control_count: Observed number of units in control.
        treatment_count: Observed number of units in treatment.
        expected_ratio: Expected proportion in treatment (default 0.5 for 50/50 split).
        alpha_critical: P-value threshold for critical failure.
        alpha_warning: P-value threshold for warning alert.
        
    Returns:
        SRMResult with chi2 statistic, p-value, and severity classification.
    """
    total = control_count + treatment_count
    if total == 0:
        return SRMResult(
            control_count=0,
            treatment_count=0,
            total_count=0,
            expected_ratio=expected_ratio,
            observed_ratio=0.0,
            chi2_stat=0.0,
            p_value=1.0,
            is_srm=False,
            severity="NONE",
        )

    expected_control = total * (1.0 - expected_ratio)
    expected_treatment = total * expected_ratio
    observed = np.array([control_count, treatment_count], dtype=np.float64)
    expected = np.array([expected_control, expected_treatment], dtype=np.float64)

    # Chi-square test with 1 degree of freedom
    chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(stats.chi2.sf(chi2_stat, df=1))

    observed_ratio = treatment_count / total

    if p_value < alpha_critical:
        severity = "CRITICAL"
        is_srm = True
    elif p_value < alpha_warning:
        severity = "WARNING"
        is_srm = True
    else:
        severity = "NONE"
        is_srm = False

    return SRMResult(
        control_count=control_count,
        treatment_count=treatment_count,
        total_count=total,
        expected_ratio=expected_ratio,
        observed_ratio=observed_ratio,
        chi2_stat=chi2_stat,
        p_value=p_value,
        is_srm=is_srm,
        severity=severity,
    )


def diagnose_srm_by_dimension(
    df: pd.DataFrame,
    variant_col: str,
    dimension_cols: List[str],
    control_val: str = "control",
    treatment_val: str = "treatment",
    expected_ratio: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    """Breakdown SRM checks across categorical slices to isolate root causes.
    
    Identifies if SRM is localized to specific operating systems, browsers,
    geos, or tracking event discrepancies.
    """
    diagnostics = {}
    
    for col in dimension_cols:
        if col not in df.columns:
            continue
            
        rows = []
        for val, group in df.groupby(col):
            c_cnt = int(np.sum(group[variant_col] == control_val))
            t_cnt = int(np.sum(group[variant_col] == treatment_val))
            res = calculate_srm(c_cnt, t_cnt, expected_ratio=expected_ratio)
            rows.append({
                "dimension_value": str(val),
                "control": c_cnt,
                "treatment": t_cnt,
                "total": res.total_count,
                "treatment_pct": round(res.observed_ratio * 100, 2),
                "chi2_stat": round(res.chi2_stat, 4),
                "p_value": res.p_value,
                "severity": res.severity,
            })
            
        diag_df = pd.DataFrame(rows).sort_values("p_value", ascending=True)
        diagnostics[col] = diag_df
        
    return diagnostics
