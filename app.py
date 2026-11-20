"""Interactive Streamlit Experimentation Command Center."""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from src.core.cuped import CUPEDEngine
from src.core.srm import calculate_srm, diagnose_srm_by_dimension
from src.core.sequential_sprt import SequentialSPRT
from src.core.hypothesis import welch_ttest, bootstrap_ci, multiple_testing_correction
from src.power_analysis import PowerPlanner
from src.data.synthetic_generator import generate_experiment_data

st.set_page_config(
    page_title="Enterprise Experimentation Platform",
    page_icon="🧪",
    layout="wide",
)

st.title("🧪 Enterprise Experimentation & Decision Platform")
st.markdown(
    "**Author:** Siddiqui Zakir Ali • Senior ML & Analytics Engineer  \n"
    "Production-grade experimentation suite featuring **CUPED variance reduction**, "
    "**Wald's Sequential Probability Ratio Test (SPRT)**, and **Sample Ratio Mismatch (SRM) diagnostics**."
)

tabs = st.tabs([
    "📊 Experiment Design & Power Planner",
    "🛡️ SRM Diagnostic Engine",
    "⚡ CUPED Variance Reduction & Lift",
    "📈 Sequential SPRT Peeking Monitor",
])

# ==============================================================================
# TAB 1: POWER PLANNER
# ==============================================================================
with tabs[0]:
    st.header("Experiment Sample Size & MDE Planner")
    st.markdown("Calculate required sample size and assess the impact of CUPED covariate variance reduction.")

    col1, col2 = st.columns(2)
    with col1:
        base_mean = st.number_input("Baseline Metric Mean ($)", value=100.0, min_value=1.0, step=5.0)
        base_std = st.number_input("Baseline Standard Deviation ($)", value=25.0, min_value=1.0, step=5.0)
        mde_pct = st.slider("Target Relative MDE (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100.0
    with col2:
        alpha = st.selectbox("Significance Level (Alpha)", [0.01, 0.05, 0.10], index=1)
        power = st.selectbox("Statistical Power (1 - Beta)", [0.80, 0.90, 0.95], index=0)
        cuped_r2 = st.slider("Anticipated Pre-Post Covariate R² (CUPED)", min_value=0.0, max_value=0.85, value=0.45, step=0.05)

    vr_factor = 1.0 - cuped_r2
    plan = PowerPlanner.calculate_sample_size(
        baseline_mean=base_mean,
        baseline_std=base_std,
        mde_relative=mde_pct,
        alpha=alpha,
        power=power,
        variance_reduction_factor=vr_factor,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Standard N / Variant", f"{plan.required_sample_size_per_variant:,}")
    m2.metric("CUPED N / Variant", f"{plan.required_sample_size_cuped:,}")
    m3.metric("Sample Reduction", f"{plan.sample_savings_pct:.1f}%", delta=f"-{plan.sample_savings_pct:.1f}%")
    m4.metric("Absolute MDE", f"${plan.mde_absolute:.2f}")

    # Sensitivity curve
    sizes = list(range(500, 15000, 500))
    curve = PowerPlanner.generate_mde_sensitivity_curve(sizes, base_mean, base_std, cuped_r2, alpha, power)
    df_curve = pd.DataFrame({
        "Sample Size per Variant": sizes * 2,
        "Standard MDE (%)": curve["standard_mde_pct"] + curve["cuped_mde_pct"],
        "Method": ["Standard A/B"] * len(sizes) + ["CUPED Adjusted"] * len(sizes),
    })

    fig = px.line(
        df_curve,
        x="Sample Size per Variant",
        y="Standard MDE (%)",
        color="Method",
        title="Minimum Detectable Effect (MDE) vs Sample Size: Standard vs CUPED",
        labels={"Standard MDE (%)": "Relative MDE (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# TAB 2: SRM DIAGNOSTICS
# ==============================================================================
with tabs[1]:
    st.header("Sample Ratio Mismatch (SRM) Diagnostic Engine")
    st.markdown("Validates assignment integrity using Chi-Square goodness-of-fit and isolates corrupted slices.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Data Generation / Injection")
        n_samples = st.number_input("Experiment Sample Size", value=10000, step=1000)
        srm_bias = st.slider("Inject Safari Traffic Divergence (SRM Bug)", 0.0, 0.8, 0.0, 0.1)
        sim_btn = st.button("Generate & Audit Experiment Logs", key="srm_gen")

    df_exp = generate_experiment_data(n_users=n_samples, srm_bias_factor=srm_bias, seed=42)
    c_cnt = int(np.sum(df_exp["variant"] == "control"))
    t_cnt = int(np.sum(df_exp["variant"] == "treatment"))
    srm_res = calculate_srm(c_cnt, t_cnt)

    with col2:
        st.subheader("Overall Assignment Audit")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Control Count", f"{c_cnt:,}")
        s2.metric("Treatment Count", f"{t_cnt:,}")
        s3.metric("Chi-Square Stat", f"{srm_res.chi2_stat:.3f}")
        s4.metric("P-Value", f"{srm_res.p_value:.4e}")

        if srm_res.severity == "CRITICAL":
            st.error(f"🚨 CRITICAL SRM DETECTED (p = {srm_res.p_value:.4e} < 0.001)! Experiment validity is compromised. Do not trust lift estimates.")
        elif srm_res.severity == "WARNING":
            st.warning(f"⚠️ SRM WARNING (p = {srm_res.p_value:.4e} < 0.01). Potential tracking anomaly detected.")
        else:
            st.success(f"✅ PASSED (p = {srm_res.p_value:.3f}). Traffic split adheres to 50/50 ratio without divergence.")

    st.subheader("Slice Forensics: Localizing Divergence across Dimensions")
    diags = diagnose_srm_by_dimension(df_exp, "variant", ["browser", "device", "country"])
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Browser Dimension Breakdown**")
        st.dataframe(diags["browser"], use_container_width=True)
    with col_b:
        st.write("**Device Dimension Breakdown**")
        st.dataframe(diags["device"], use_container_width=True)

# ==============================================================================
# TAB 3: CUPED LIFT ESTIMATOR
# ==============================================================================
with tabs[2]:
    st.header("CUPED Variance Reduction & Treatment Lift")
    st.markdown("Removes pre-experiment variance to tighten confidence intervals and increase statistical power.")

    df_cuped = generate_experiment_data(n_users=6000, true_lift_pct=0.04, pre_post_correlation=0.72, seed=88)
    engine = CUPEDEngine(covariates=["pre_exp_revenue", "pre_exp_sessions"])
    cuped_res = engine.analyze_experiment(df_cuped, "variant", "revenue")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unadjusted Lift", f"${cuped_res.unadjusted_diff:.2f}", f"p={cuped_res.unadjusted_p_value:.4f}")
    k2.metric("CUPED Adjusted Lift", f"${cuped_res.adjusted_diff:.2f}", f"p={cuped_res.adjusted_p_value:.4f}")
    k3.metric("Variance Reduction", f"{cuped_res.variance_reduction_pct:.1f}%")
    k4.metric("Sample Size Savings", f"{(1 - cuped_res.sample_size_reduction_factor)*100:.1f}%")

    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        # Confidence Interval Comparison
        raw_se = np.sqrt(cuped_res.unadjusted_var)
        cuped_se = np.sqrt(cuped_res.adjusted_var)
        
        ci_df = pd.DataFrame({
            "Method": ["Unadjusted Standard", "CUPED Adjusted"],
            "Lift": [cuped_res.unadjusted_diff, cuped_res.adjusted_diff],
            "CI_Lower": [cuped_res.unadjusted_diff - 1.96 * raw_se, cuped_res.ci_lower],
            "CI_Upper": [cuped_res.unadjusted_diff + 1.96 * raw_se, cuped_res.ci_upper],
        })
        ci_df["Error_Minus"] = ci_df["Lift"] - ci_df["CI_Lower"]
        ci_df["Error_Plus"] = ci_df["CI_Upper"] - ci_df["Lift"]

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=ci_df["Lift"],
            y=ci_df["Method"],
            mode="markers",
            marker=dict(size=12, color=["#ef4444", "#10b981"]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=ci_df["Error_Plus"],
                arrayminus=ci_df["Error_Minus"],
                thickness=3,
                width=10,
            ),
        ))
        fig_ci.update_layout(title="95% Confidence Interval Comparison: Unadjusted vs CUPED", xaxis_title="Treatment Lift ($)")
        st.plotly_chart(fig_ci, use_container_width=True)

    with col_plot2:
        # Distribution comparison
        y_cuped_series = engine.transform(df_cuped, "revenue")
        dist_df = pd.DataFrame({
            "Raw Revenue": df_cuped["revenue"],
            "CUPED Adjusted Revenue": y_cuped_series,
        })
        fig_dist = px.histogram(
            dist_df,
            x=["Raw Revenue", "CUPED Adjusted Revenue"],
            barmode="overlay",
            nbins=50,
            title="Metric Variance Compression: Raw vs CUPED Adjusted Distribution",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

# ==============================================================================
# TAB 4: SEQUENTIAL SPRT
# ==============================================================================
with tabs[3]:
    st.header("Sequential Probability Ratio Test (SPRT) Peeking Monitor")
    st.markdown("Safely evaluate experiments continuously without inflating Type I false-positive error.")

    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        sprt_alpha = st.selectbox("Nominal Alpha", [0.01, 0.05, 0.10], index=1, key="sprt_a")
        sprt_beta = st.selectbox("Nominal Beta", [0.10, 0.20], index=1, key="sprt_b")
        sprt_mde = st.number_input("Expected Effect Delta", value=3.0, step=0.5)
        true_effect = st.slider("Simulated True Lift", -4.0, 6.0, 3.5, 0.5)
        run_seq = st.button("Run Sequential Stream Simulation")

    rng = np.random.default_rng(42)
    c_stream = rng.normal(100.0, 10.0, 3000)
    t_stream = rng.normal(100.0 + true_effect, 10.0, 3000)

    sprt = SequentialSPRT(alpha=sprt_alpha, beta=sprt_beta, mde_abs=sprt_mde, assumed_sigma=10.0)
    res_sprt = sprt.evaluate_stream(c_stream, t_stream, step_batch_size=50)

    with col_s2:
        if res_sprt.decision == "STOP_REJECT_NULL":
            st.success(f"🏆 SUCCESS: Early Victory Declared at sample N = {res_sprt.stopped_at_step:,}! Treatment is statistically superior.")
        elif res_sprt.decision == "STOP_ACCEPT_NULL":
            st.info(f"🛑 FUTILITY: Early Stopping Declared at sample N = {res_sprt.stopped_at_step:,}. Null hypothesis accepted (no detectable lift).")
        else:
            st.warning("⏳ CONTINUE: Boundaries not crossed. Maintain experiment execution.")

        step_df = pd.DataFrame([{
            "Step": s.step,
            "Log-Likelihood Ratio": s.log_likelihood_ratio,
            "Upper Bound (Win)": s.upper_bound,
            "Lower Bound (Futility)": s.lower_bound,
        } for s in res_sprt.steps])

        fig_sprt = go.Figure()
        fig_sprt.add_trace(go.Scatter(x=step_df["Step"], y=step_df["Log-Likelihood Ratio"], mode="lines+markers", name="Observed LLR", line=dict(color="#3b82f6", width=2)))
        fig_sprt.add_trace(go.Scatter(x=step_df["Step"], y=step_df["Upper Bound (Win)"], mode="lines", name="Upper Threshold (A)", line=dict(color="#10b981", dash="dash")))
        fig_sprt.add_trace(go.Scatter(x=step_df["Step"], y=step_df["Lower Bound (Futility)"], mode="lines", name="Lower Threshold (B)", line=dict(color="#ef4444", dash="dash")))
        fig_sprt.update_layout(title="Wald's SPRT Trajectory vs Analytical Boundaries", xaxis_title="Cumulative Sample Size", yaxis_title="Log-Likelihood Ratio")
        st.plotly_chart(fig_sprt, use_container_width=True)
