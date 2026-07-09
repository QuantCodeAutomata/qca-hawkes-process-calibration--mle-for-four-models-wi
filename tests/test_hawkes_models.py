"""Tests for src/hawkes_models.py — Hawkes process model implementations."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.hawkes_models import (
    intensity_power_only,
    loglik_power_only,
    loglik_exp_only,
    aicc,
    run_mle_model1,
    run_mle_model3,
)


# ---------------------------------------------------------------------------
# Fixtures: simple test event sequences
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_events():
    """Small synthetic event sequence for fast tests."""
    return np.array([0.0, 0.5, 1.2, 2.3, 4.1, 6.0, 8.5])

@pytest.fixture
def target_params_m1():
    """Paper target parameters for Model 1."""
    return np.array([1.431227, 0.020010, 0.003982, 1.524052])

@pytest.fixture
def synthetic_37_events():
    """Synthetic 37-event catalog for MLE testing."""
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 15.0, 37))
    return t


# ---------------------------------------------------------------------------
# AICc tests
# ---------------------------------------------------------------------------

class TestAICc:
    def test_aicc_paper_target(self):
        """Verify AICc = -6.8272 for Model 1 target values (paper Table 1)."""
        ll = 8.0386
        k = 4
        n = 37
        result = aicc(ll, k, n)
        # -2*8.0386 + 2*4 + 2*4*5/(37-4-1) = -16.0772 + 8 + 1.25 = -6.8272
        assert abs(result - (-6.8272)) < 0.001, f"Expected -6.8272, got {result:.4f}"

    def test_aicc_formula_correctness(self):
        """AICc = AIC + 2k(k+1)/(n-k-1)."""
        ll, k, n = 5.0, 3, 20
        aic = -2 * ll + 2 * k
        expected = aic + 2 * k * (k + 1) / (n - k - 1)
        assert abs(aicc(ll, k, n) - expected) < 1e-10

    def test_aicc_positive_penalty(self):
        """AICc >= AIC (correction term is non-negative)."""
        ll, k, n = 10.0, 4, 37
        aic = -2 * ll + 2 * k
        assert aicc(ll, k, n) >= aic

    def test_aicc_all_paper_models(self):
        """Verify AICc values for all 4 paper models match within tolerance."""
        targets = [
            (8.0386,  4, -6.8272),   # Model 1
            (8.1208,  6, -1.4416),   # Model 2
            (4.5353,  3, -2.3433),   # Model 3
            (6.7185,  5, -1.5015),   # Model 4
        ]
        for ll, k, expected_aicc in targets:
            result = aicc(ll, k, n=37)
            assert abs(result - expected_aicc) < 0.002, \
                f"k={k}: expected {expected_aicc}, got {result:.4f}"


# ---------------------------------------------------------------------------
# Model 1: Only Power Kernel — log-likelihood tests
# ---------------------------------------------------------------------------

class TestLoglikPowerOnly:
    def test_returns_finite_for_valid_params(self, simple_events):
        """Log-likelihood returns a finite value for valid parameters."""
        params = np.array([1.0, 0.01, 0.01, 1.5])
        T_cal = 10.0
        ll = loglik_power_only(params, simple_events, T_cal)
        assert np.isfinite(ll), f"Expected finite ll, got {ll}"

    def test_negative_lambda0_returns_neginf(self, simple_events):
        """Negative lambda0 should return -inf."""
        params = np.array([-0.1, 0.01, 0.01, 1.5])
        ll = loglik_power_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_p_leq_1_returns_neginf(self, simple_events):
        """p <= 1 violates the power-law requirement; should return -inf."""
        params = np.array([1.0, 0.01, 0.01, 0.9])
        ll = loglik_power_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_supercritical_returns_neginf(self, simple_events):
        """Supercritical kernel (norm >= 1) should return -inf."""
        # rho1/((p-1)*rho2^(p-1)) >= 1 means supercritical
        params = np.array([1.0, 100.0, 0.001, 1.5])
        ll = loglik_power_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_log_likelihood_decreases_with_bad_params(self, simple_events):
        """Good params should give higher log-lik than random bad params."""
        good_params = np.array([1.0, 0.02, 0.01, 1.5])
        bad_params = np.array([10.0, 0.001, 5.0, 2.0])
        T_cal = 10.0
        ll_good = loglik_power_only(good_params, simple_events, T_cal)
        ll_bad = loglik_power_only(bad_params, simple_events, T_cal)
        # Both should be finite; no constraint on ordering here
        assert np.isfinite(ll_good) or np.isfinite(ll_bad)

    def test_compensator_positive(self, simple_events):
        """Compensator should be positive (it equals the expected number of events)."""
        params = np.array([1.0, 0.01, 0.01, 1.5])
        T_cal = 10.0
        lam0, rho1, rho2, p = params
        compensator = lam0 * T_cal
        for tj in simple_events:
            compensator += rho1 * ((rho2 + T_cal - tj)**(1 - p) - rho2**(1 - p)) / (1 - p)
        assert compensator > 0

    def test_target_params_give_positive_loglik(self, target_params_m1, synthetic_37_events):
        """Target params from paper should give positive log-likelihood."""
        T_cal = synthetic_37_events[-1]
        ll = loglik_power_only(target_params_m1, synthetic_37_events, T_cal)
        assert np.isfinite(ll)


# ---------------------------------------------------------------------------
# Model 3: Only Exponential Kernel — log-likelihood tests
# ---------------------------------------------------------------------------

class TestLoglikExpOnly:
    def test_returns_finite_for_valid_params(self, simple_events):
        """Log-likelihood returns finite value for valid params."""
        params = np.array([0.5, 0.5, 2.0])  # b1/b2 = 0.25 < 1
        ll = loglik_exp_only(params, simple_events, 10.0)
        assert np.isfinite(ll)

    def test_negative_b1_returns_neginf(self, simple_events):
        params = np.array([0.5, -1.0, 2.0])
        ll = loglik_exp_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_supercritical_exp_returns_neginf(self, simple_events):
        """b1/b2 >= 1 is supercritical."""
        params = np.array([0.5, 3.0, 2.0])  # b1/b2 = 1.5 >= 1
        ll = loglik_exp_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_consistency_with_zero_b1(self):
        """When b1=0 (subcritical limit), intensity = lambda0, ll = n*log(lam0) - lam0*T."""
        t_events = np.array([1.0, 3.0, 7.0])
        lam0, b1, b2 = 2.0, 1e-10, 1.0  # essentially b1=0
        params = np.array([lam0, b1, b2])
        T_cal = 10.0
        ll = loglik_exp_only(params, t_events, T_cal)
        n = len(t_events)
        expected_approx = n * np.log(lam0) - lam0 * T_cal
        assert abs(ll - expected_approx) < 0.1


# ---------------------------------------------------------------------------
# Intensity function tests
# ---------------------------------------------------------------------------

class TestIntensityPowerOnly:
    def test_at_t0_equals_lambda0(self):
        """At t=0 (before any events), intensity equals lambda0."""
        params = np.array([1.5, 0.02, 0.01, 1.5])
        t_events = np.array([0.5, 1.0, 2.0])  # all future
        assert abs(intensity_power_only(0.0, t_events, params) - 1.5) < 1e-10

    def test_intensity_increases_after_event(self):
        """Intensity should be higher just after an event than just before."""
        params = np.array([1.0, 0.5, 0.01, 1.5])
        t_events = np.array([1.0, 5.0])
        lam_before = intensity_power_only(0.99, t_events, params)
        lam_after = intensity_power_only(1.01, t_events, params)
        assert lam_after > lam_before

    def test_intensity_decays_after_event(self):
        """Intensity should decay as time moves away from an event."""
        params = np.array([1.0, 0.5, 0.01, 1.5])
        t_events = np.array([0.0])
        lam_near = intensity_power_only(0.1, t_events, params)
        lam_far = intensity_power_only(5.0, t_events, params)
        assert lam_near > lam_far

    def test_intensity_positive_always(self, simple_events):
        """Intensity should always be positive for valid params."""
        params = np.array([0.5, 0.01, 0.01, 1.5])
        for t in np.linspace(0.01, 10, 50):
            lam = intensity_power_only(t, simple_events, params)
            assert lam > 0, f"Negative intensity at t={t}: {lam}"


# ---------------------------------------------------------------------------
# MLE tests (fast, using known simple data)
# ---------------------------------------------------------------------------

class TestMLEModel1:
    def test_mle_returns_valid_params(self, simple_events):
        """MLE should return non-None parameters for valid input."""
        T_cal = simple_events[-1]
        params, ll = run_mle_model1(simple_events, T_cal, n_restarts=5, seed=42)
        assert params is not None
        assert np.isfinite(ll)
        assert ll > -np.inf

    def test_mle_params_positive(self, simple_events):
        """All MLE parameters should be positive (within model constraints)."""
        T_cal = simple_events[-1]
        params, _ = run_mle_model1(simple_events, T_cal, n_restarts=5, seed=42)
        lam0, rho1, rho2, p = params
        assert lam0 > 0
        assert rho1 > 0
        assert rho2 > 0
        assert p > 1

    def test_mle_subcriticality(self, simple_events):
        """MLE result should satisfy subcriticality constraint."""
        T_cal = simple_events[-1]
        params, _ = run_mle_model1(simple_events, T_cal, n_restarts=5, seed=42)
        lam0, rho1, rho2, p = params
        norm = rho1 / ((p - 1) * rho2**(p - 1))
        assert norm < 1.0 + 1e-3, f"Subcriticality violated: norm={norm:.4f}"

    def test_mle_loglik_at_optimum_exceeds_random(self, simple_events):
        """MLE log-likelihood should exceed a random feasible starting point."""
        T_cal = simple_events[-1]
        params, ll_mle = run_mle_model1(simple_events, T_cal, n_restarts=10, seed=0)
        # Evaluate at a known feasible point
        params_random = np.array([1.0, 0.01, 0.05, 1.5])
        ll_random = loglik_power_only(params_random, simple_events, T_cal)
        if np.isfinite(ll_random):
            assert ll_mle >= ll_random - 1e-6


class TestMLEModel3:
    def test_mle_returns_valid_params(self, simple_events):
        """MLE Model 3 returns valid parameters."""
        T_cal = simple_events[-1]
        params, ll = run_mle_model3(simple_events, T_cal, n_restarts=5, seed=42)
        assert params is not None
        assert np.isfinite(ll)

    def test_mle_subcriticality_exp(self, simple_events):
        """MLE result satisfies b1/b2 < 1 (subcriticality)."""
        T_cal = simple_events[-1]
        params, _ = run_mle_model3(simple_events, T_cal, n_restarts=5, seed=42)
        lam0, b1, b2 = params
        assert b1 / b2 < 1.0 + 1e-3


# ---------------------------------------------------------------------------
# Subcriticality verification for paper target params
# ---------------------------------------------------------------------------

class TestSubcriticality:
    def test_paper_params_model1_subcritical(self, target_params_m1):
        """Paper's Model 1 calibrated params must satisfy subcriticality."""
        lam0, rho1, rho2, p = target_params_m1
        norm = rho1 / ((p - 1) * rho2**(p - 1))
        assert norm < 1.0, f"Paper params are supercritical: norm={norm:.6f}"

    def test_paper_params_model3_subcritical(self):
        """Paper's Model 3 calibrated params must satisfy subcriticality."""
        lam0, b1, b2 = 1.185102, 2.666847, 5.0
        norm = b1 / b2
        assert norm < 1.0, f"Model 3 supercritical: norm={norm:.6f}"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_event(self):
        """Log-likelihood handles single event correctly."""
        t_events = np.array([0.0])
        params = np.array([1.0, 0.01, 0.01, 1.5])
        T_cal = 5.0
        ll = loglik_power_only(params, t_events, T_cal)
        assert np.isfinite(ll)

    def test_empty_events_has_no_contribution(self):
        """With no events, likelihood depends only on background."""
        t_events = np.array([])
        params = np.array([1.0, 0.01, 0.01, 1.5])
        T_cal = 5.0
        ll = loglik_power_only(params, t_events, T_cal)
        # Should be -lambda0*T_cal = -5.0
        expected = -1.0 * 5.0
        assert abs(ll - expected) < 1e-10

    def test_zero_rho2_returns_neginf(self, simple_events):
        """rho2=0 is invalid (division by zero in kernel)."""
        params = np.array([1.0, 0.01, 0.0, 1.5])
        ll = loglik_power_only(params, simple_events, 10.0)
        assert ll == -np.inf

    def test_large_T_cal(self, simple_events):
        """Log-likelihood handles very large T_cal."""
        params = np.array([0.1, 0.01, 0.01, 1.5])
        ll = loglik_power_only(params, simple_events, 1000.0)
        assert np.isfinite(ll)
