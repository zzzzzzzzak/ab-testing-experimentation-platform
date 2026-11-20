"""Wald's Sequential Probability Ratio Test (SPRT) Engine."""

from typing import List, Optional, Tuple
import numpy as np
from scipy import stats

from src.config import SPRTResult, SPRTStep


class SequentialSPRT:
    """Wald's Sequential Probability Ratio Test for continuous A/B monitoring.
    
    Prevents alpha-inflation from repeated continuous 'peeking' by evaluating
    the log-likelihood ratio against analytical stopping boundaries:
        Upper bound A = ln((1 - beta) / alpha)  --> Reject H0 (Declare winner)
        Lower bound B = ln(beta / (1 - alpha))  --> Accept H0 (Declare futility)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        beta: float = 0.20,
        mde_abs: float = 1.0,
        assumed_sigma: float = 10.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.mde = mde_abs
        self.sigma = assumed_sigma
        
        # Wald analytical thresholds
        self.upper_bound = float(np.log((1.0 - self.beta) / self.alpha))
        self.lower_bound = float(np.log(self.beta / (1.0 - self.alpha)))

    def evaluate_stream(
        self,
        control_stream: np.ndarray,
        treatment_stream: np.ndarray,
        step_batch_size: int = 50,
    ) -> SPRTResult:
        """Evaluate sequential experiment across streaming sample batches.
        
        Args:
            control_stream: Array of sequentially arrived control metric observations.
            treatment_stream: Array of sequentially arrived treatment metric observations.
            step_batch_size: Evaluation frequency (check every N samples).
            
        Returns:
            SPRTResult with step trajectories and final decision.
        """
        n_obs = min(len(control_stream), len(treatment_stream))
        steps: List[SPRTStep] = []
        
        decision = "CONTINUE"
        stopped_step = n_obs
        final_llr = 0.0

        # We evaluate paired difference d_i = treatment_i - control_i
        # Under H0: d ~ N(0, 2*sigma^2)
        # Under H1: d ~ N(mde, 2*sigma^2)
        diff_var = 2.0 * (self.sigma ** 2)

        for i in range(step_batch_size, n_obs + 1, step_batch_size):
            c_slice = control_stream[:i]
            t_slice = treatment_stream[:i]
            
            mean_c = float(np.mean(c_slice))
            mean_t = float(np.mean(t_slice))
            
            # Cumulative Log-Likelihood Ratio
            # sum ln(f1 / f0) = (mde / diff_var) * sum(d_i - mde / 2)
            #               = (mde / diff_var) * [ n * (mean_t - mean_c) - n * mde / 2 ]
            llr = float((self.mde / diff_var) * (i * (mean_t - mean_c) - i * (self.mde / 2.0)))
            
            current_decision = "CONTINUE"
            if llr >= self.upper_bound:
                current_decision = "STOP_REJECT_NULL"
            elif llr <= self.lower_bound:
                current_decision = "STOP_ACCEPT_NULL"

            step_entry = SPRTStep(
                step=i,
                control_mean=mean_c,
                treatment_mean=mean_t,
                log_likelihood_ratio=llr,
                lower_bound=self.lower_bound,
                upper_bound=self.upper_bound,
                decision=current_decision,
            )
            steps.append(step_entry)

            if current_decision != "CONTINUE":
                decision = current_decision
                stopped_step = i
                final_llr = llr
                break
        else:
            final_llr = steps[-1].log_likelihood_ratio if steps else 0.0

        return SPRTResult(
            decision=decision,
            stopped_at_step=stopped_step,
            total_steps=n_obs,
            final_llr=final_llr,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            alpha=self.alpha,
            beta=self.beta,
            steps=steps,
        )


def simulate_peeking_problem(
    n_sims: int = 1000,
    sample_size: int = 1000,
    peeks: int = 20,
    alpha: float = 0.05,
    sigma: float = 10.0,
) -> Tuple[float, float]:
    """Demonstrate the mathematical explosion of false-positive rate under naive peeking.
    
    Under the true null (H0: effect=0):
    - A single test has false-positive rate exactly equal to alpha (5%).
    - Continuous peeking with naive t-test inflates false-positive rate to 20-30%.
    - Sequential SPRT strictly maintains false-positive rate <= alpha.
    """
    naive_false_positives = 0
    sprt_false_positives = 0
    
    mde_hypothetical = sigma * 0.2
    sprt = SequentialSPRT(alpha=alpha, beta=0.20, mde_abs=mde_hypothetical, assumed_sigma=sigma)
    peek_indices = np.linspace(50, sample_size, peeks, dtype=int)

    for _ in range(n_sims):
        # Draw true null samples (mean=0)
        c = np.random.normal(100.0, sigma, sample_size)
        t = np.random.normal(100.0, sigma, sample_size)
        
        # 1. Naive Peeking test
        naive_rejected = False
        for idx in peek_indices:
            _, p = stats.ttest_ind(t[:idx], c[:idx], equal_var=True)
            if p < alpha:
                naive_rejected = True
                break
        if naive_rejected:
            naive_false_positives += 1

        # 2. SPRT test
        res = sprt.evaluate_stream(c, t, step_batch_size=sample_size // peeks)
        if res.decision == "STOP_REJECT_NULL":
            sprt_false_positives += 1

    naive_fpr = float(naive_false_positives / n_sims)
    sprt_fpr = float(sprt_false_positives / n_sims)
    return naive_fpr, sprt_fpr
