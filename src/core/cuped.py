"""CUPED (Controlled-experiment Using Pre-Experiment Data) Engine."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from src.config import CUPEDResult


class CUPEDEngine:
    """Multivariate CUPED variance reduction engine.
    
    Transforms the observed post-experiment metric Y into an unbiased
    reduced-variance estimate:
        Y_cuped = Y - theta * (X - E[X])
        
    Where theta is chosen to minimize Var(Y_cuped):
        theta = Cov(Y, X) / Var(X)  (univariate)
        theta = (X^T X)^(-1) X^T Y  (multivariate OLS)
    """

    def __init__(self, covariates: Optional[List[str]] = None):
        self.covariates = covariates or ["pre_exp_revenue"]
        self.thetas: Dict[str, float] = {}
        self.covariate_means: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, target_metric: str) -> "CUPEDEngine":
        """Compute optimal theta coefficients using pooled or control data."""
        y = df[target_metric].values.astype(np.float64)
        
        if len(self.covariates) == 1:
            cov = self.covariates[0]
            x = df[cov].values.astype(np.float64)
            var_x = float(np.var(x, ddof=1))
            cov_xy = float(np.cov(x, y, ddof=1)[0, 1])
            theta = cov_xy / (var_x + 1e-12) if var_x > 0 else 0.0
            self.thetas = {cov: theta}
            self.covariate_means = {cov: float(np.mean(x))}
        else:
            # Multivariate OLS regression without intercept (centered)
            X = df[self.covariates].values.astype(np.float64)
            means_x = np.mean(X, axis=0)
            X_centered = X - means_x
            y_centered = y - np.mean(y)
            
            # OLS: (X^T X)^-1 X^T y
            try:
                theta_vec, _, _, _ = np.linalg.lstsq(X_centered, y_centered, rcond=None)
            except np.linalg.LinAlgError:
                theta_vec = np.zeros(len(self.covariates))
                
            self.thetas = {cov: float(theta_vec[i]) for i, cov in enumerate(self.covariates)}
            self.covariate_means = {cov: float(means_x[i]) for i, cov in enumerate(self.covariates)}
            
        return self

    def transform(self, df: pd.DataFrame, target_metric: str) -> pd.Series:
        """Apply the fitted CUPED transformation to produce Y_cuped."""
        y = df[target_metric].copy().astype(np.float64)
        adjustment = np.zeros(len(df), dtype=np.float64)
        
        for cov, theta in self.thetas.items():
            if cov in df.columns:
                x = df[cov].values.astype(np.float64)
                x_mean = self.covariate_means[cov]
                adjustment += theta * (x - x_mean)
                
        return pd.Series(y - adjustment, index=df.index, name=f"{target_metric}_cuped")

    def fit_transform(self, df: pd.DataFrame, target_metric: str) -> pd.Series:
        """Convenience method to fit and transform in one call."""
        return self.fit(df, target_metric).transform(df, target_metric)

    def analyze_experiment(
        self,
        df: pd.DataFrame,
        variant_col: str,
        target_metric: str,
        control_val: str = "control",
        treatment_val: str = "treatment",
        alpha: float = 0.05,
    ) -> CUPEDResult:
        """Execute full unadjusted vs. CUPED-adjusted comparative analysis."""
        self.fit(df, target_metric)
        df_eval = df.copy()
        df_eval[f"{target_metric}_cuped"] = self.transform(df, target_metric)

        ctrl_mask = df_eval[variant_col] == control_val
        trt_mask = df_eval[variant_col] == treatment_val

        # Unadjusted statistics
        y_c = df_eval.loc[ctrl_mask, target_metric].values
        y_t = df_eval.loc[trt_mask, target_metric].values

        mean_c_unadj = float(np.mean(y_c))
        mean_t_unadj = float(np.mean(y_t))
        diff_unadj = mean_t_unadj - mean_c_unadj
        var_unadj = float(np.var(y_c, ddof=1) / len(y_c) + np.var(y_t, ddof=1) / len(y_t))
        t_stat_unadj, p_val_unadj = stats.ttest_ind(y_t, y_c, equal_var=False)

        # CUPED-adjusted statistics
        y_c_cuped = df_eval.loc[ctrl_mask, f"{target_metric}_cuped"].values
        y_t_cuped = df_eval.loc[trt_mask, f"{target_metric}_cuped"].values

        mean_c_adj = float(np.mean(y_c_cuped))
        mean_t_adj = float(np.mean(y_t_cuped))
        diff_adj = mean_t_adj - mean_c_adj
        var_adj = float(np.var(y_c_cuped, ddof=1) / len(y_c_cuped) + np.var(y_t_cuped, ddof=1) / len(y_t_cuped))
        t_stat_adj, p_val_adj = stats.ttest_ind(y_t_cuped, y_c_cuped, equal_var=False)

        # Variance reduction ratio
        variance_reduction_pct = max(0.0, (1.0 - var_adj / (var_unadj + 1e-12)) * 100.0)
        # Required sample size is proportional to variance: N_cuped = N_raw * (var_adj / var_unadj)
        sample_size_factor = float(var_adj / (var_unadj + 1e-12))

        # 95% Confidence Interval for difference
        se_adj = np.sqrt(var_adj)
        z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_lower = float(diff_adj - z_crit * se_adj)
        ci_upper = float(diff_adj + z_crit * se_adj)

        return CUPEDResult(
            unadjusted_control_mean=mean_c_unadj,
            unadjusted_treatment_mean=mean_t_unadj,
            unadjusted_diff=diff_unadj,
            unadjusted_var=var_unadj,
            unadjusted_p_value=float(p_val_unadj),
            adjusted_control_mean=mean_c_adj,
            adjusted_treatment_mean=mean_t_adj,
            adjusted_diff=diff_adj,
            adjusted_var=var_adj,
            adjusted_p_value=float(p_val_adj),
            theta=self.thetas,
            variance_reduction_pct=variance_reduction_pct,
            sample_size_reduction_factor=sample_size_factor,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
