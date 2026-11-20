"""Unit tests for Sample Ratio Mismatch (SRM) diagnostic engine."""

import pytest
import numpy as np
import pandas as pd
from src.core.srm import calculate_srm, diagnose_srm_by_dimension


def test_srm_balanced_traffic():
    # 5,000 vs 5,000 with 50/50 expected split
    res = calculate_srm(5000, 5000, expected_ratio=0.5)
    assert not res.is_srm
    assert res.severity == "NONE"
    assert res.p_value > 0.90
    assert res.chi2_stat < 0.05


def test_srm_detected_divergence():
    # Severe mismatch: 4,700 vs 5,300
    res = calculate_srm(4700, 5300, expected_ratio=0.5)
    assert res.is_srm
    assert res.severity == "CRITICAL"
    assert res.p_value < 1e-4
    assert res.chi2_stat > 15.0


def test_srm_warning_threshold():
    # Moderate mismatch triggering warning
    # e.g., 4,870 vs 5,130 on N=10,000 -> chi2 ~ 6.76, p ~ 0.009
    res = calculate_srm(4870, 5130, expected_ratio=0.5)
    assert res.is_srm
    assert res.severity in ("WARNING", "CRITICAL")
    assert res.p_value < 0.01


def test_srm_segment_diagnostics():
    df = pd.DataFrame({
        "variant": ["control"] * 500 + ["treatment"] * 500 + ["control"] * 800 + ["treatment"] * 200,
        "browser": ["Chrome"] * 1000 + ["Safari"] * 1000,
        "device": ["Desktop"] * 1000 + ["Mobile"] * 1000,
    })
    
    diag = diagnose_srm_by_dimension(df, "variant", ["browser", "device"])
    assert "browser" in diag
    safari_row = diag["browser"][diag["browser"]["dimension_value"] == "Safari"].iloc[0]
    assert safari_row["severity"] == "CRITICAL"
    assert safari_row["p_value"] < 1e-10

    chrome_row = diag["browser"][diag["browser"]["dimension_value"] == "Chrome"].iloc[0]
    assert chrome_row["severity"] == "NONE"
