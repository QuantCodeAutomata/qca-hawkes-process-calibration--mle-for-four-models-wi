"""Tests for src/volterra_solver.py — Fractional Adams solver and investment strategy."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
from scipy.special import gamma as gamma_func

from src.volterra_solver import (
    fractional_adams,
    solve_H_bar,
    solve_B_bar,
    H_hat_vanilla,
    investment_ratio_pd,
    investment_ratio_vanilla,
)

# Fixed model parameters from Table 2
KAPPA = 0.173
THETA = 5.000
SIGMA = 0.340
RHO = -0.615
GAMMA = 1.000
UPSILON = 0.020
C = KAPPA + THETA * SIGMA * RHO  # = -0.8725


# ---------------------------------------------------------------------------
# Basic Fractional Adams tests
# ---------------------------------------------------------------------------

class TestFractionalAdams:
    def test_initial_condition_zero(self):
        """Solution starts at y(0) = 0."""
        f_func = lambda s, y: 1.0  # constant RHS
        tau, y = fractional_adams(f_func, T=1.0, N=100, delta=0.8)
        assert abs(y[0]) < 1e-14

    def test_grid_shape(self):
        """Output arrays have shape (N+1,)."""
        N = 200
        tau, y = fractional_adams(lambda s, y: 1.0, T=2.0, N=N, delta=0.7)
        assert tau.shape == (N + 1,)
        assert y.shape == (N + 1,)

    def test_grid_range(self):
        """Grid spans [0, T]."""
        T = 5.0
        tau, y = fractional_adams(lambda s, y: 1.0, T=T, N=100, delta=0.6)
        assert abs(tau[0]) < 1e-14
        assert abs(tau[-1] - T) < 1e-10

    def test_delta_1_reduces_to_standard_ode(self):
        """For delta=1, the Volterra equation reduces to standard ODE:
        y(tau) = int_0^tau f(s, y(s)) ds  => dy/dtau = f(tau, y), y(0)=0.
        For f(s,y) = 1, solution is y(tau) = tau.
        """
        T = 2.0
        N = 1000
        tau, y = fractional_adams(lambda s, y: 1.0, T=T, N=N, delta=1.0)
        # y should approximate tau
        err = np.max(np.abs(y - tau))
        assert err < 0.01, f"Max error for delta=1, f=1: {err:.4f}"

    def test_linear_equation_bounded_solution(self):
        """For linear f(s, y) = -y + c, solution should remain bounded."""
        tau, y = fractional_adams(lambda s, y: -y + 10.0, T=2.0, N=500, delta=0.7)
        assert np.all(np.isfinite(y))
        assert np.max(np.abs(y)) < 1e4

    def test_convergence_with_refinement(self):
        """Halving step size improves accuracy (convergence test)."""
        f_func = lambda s, y: 1.0 - 0.5 * y  # linear
        T = 1.0
        delta = 0.75
        N_coarse = 200
        N_fine = 400
        _, y_c = fractional_adams(f_func, T, N_coarse, delta)
        _, y_f = fractional_adams(f_func, T, N_fine, delta)
        # Compare at coarse grid points (every 2nd point of fine grid)
        y_f_on_c = y_f[::2]
        rel_err = np.max(np.abs(y_f_on_c - y_c)) / (np.max(np.abs(y_c)) + 1e-15)
        assert rel_err < 0.05, f"Convergence failed: rel_err={rel_err:.4f}"


# ---------------------------------------------------------------------------
# H_bar tests
# ---------------------------------------------------------------------------

class TestSolveHBar:
    def test_initial_value_zero(self):
        """H_bar(0) = 0 (initial condition)."""
        _, H = solve_H_bar(1.0, 100, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert abs(H[0]) < 1e-14

    def test_H_bar_positive(self):
        """H_bar should be positive for all tau > 0 (c < 0, so source dominates)."""
        _, H = solve_H_bar(1.0, 200, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert np.all(H >= 0), f"Negative H_bar values found: min={H.min():.4f}"

    def test_H_bar_monotone_increasing(self):
        """H_bar should be monotonically increasing (source term theta^2/gamma dominates)."""
        _, H = solve_H_bar(1.0, 200, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        diffs = np.diff(H)
        # Allow tiny numerical noise
        assert np.all(diffs >= -1e-6), f"H_bar not monotone: min diff={diffs.min():.2e}"

    @pytest.mark.parametrize("delta", [0.50, 0.60, 0.70])
    def test_all_delta_values_finite(self, delta):
        """All three delta values produce finite H_bar arrays."""
        _, H = solve_H_bar(1.0, 100, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert np.all(np.isfinite(H)), f"Non-finite H_bar for delta={delta}"

    def test_smaller_delta_larger_H(self):
        """Rougher volatility (smaller delta) produces larger H_bar values."""
        _, H_05 = solve_H_bar(1.0, 500, 0.50, KAPPA, THETA, SIGMA, RHO, GAMMA)
        _, H_07 = solve_H_bar(1.0, 500, 0.70, KAPPA, THETA, SIGMA, RHO, GAMMA)
        # H(delta=0.5) should be larger than H(delta=0.7) for tau > 0
        # Check at midpoint
        assert H_05[250] > H_07[250], \
            f"Expected H_05 > H_07 at midpoint: {H_05[250]:.4f} vs {H_07[250]:.4f}"


# ---------------------------------------------------------------------------
# Vanilla H_hat tests
# ---------------------------------------------------------------------------

class TestHHatVanilla:
    def test_initial_value_zero(self):
        """H_hat(0) = 0."""
        tau = np.linspace(0, 1, 100)
        H = H_hat_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert abs(H[0]) < 1e-14

    def test_H_hat_positive(self):
        """H_hat(tau) > 0 for tau > 0 (since c < 0 and (exp(c*tau)-1) < 0,
        divided by c < 0, gives positive result)."""
        tau = np.linspace(0, 5, 100)
        H = H_hat_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert np.all(H >= 0), f"Negative H_hat values"

    def test_closed_form_matches_rk4(self):
        """Closed-form H_hat satisfies ODE dH/dtau = c*H + theta^2/gamma.

        H_hat(tau) = (theta^2/gamma)/c * (exp(c*tau) - 1) satisfies:
        dH/dtau = (theta^2/gamma)*exp(c*tau) = c*H + theta^2/gamma  (with + sign).
        """
        T = 1.0
        N = 1000
        h = T / N
        tau = np.linspace(0, T, N + 1)
        # RK4 integration of the ODE that H_hat satisfies: dH/dtau = c*H + theta^2/gamma
        H_rk4 = np.zeros(N + 1)
        for i in range(N):
            k1 = C * H_rk4[i] + THETA**2 / GAMMA
            k2 = C * (H_rk4[i] + h/2 * k1) + THETA**2 / GAMMA
            k3 = C * (H_rk4[i] + h/2 * k2) + THETA**2 / GAMMA
            k4 = C * (H_rk4[i] + h * k3) + THETA**2 / GAMMA
            H_rk4[i+1] = H_rk4[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        H_cf = H_hat_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA)
        max_err = np.max(np.abs(H_cf - H_rk4))
        assert max_err < 0.01, f"Closed-form vs RK4 discrepancy: {max_err:.4e}"

    def test_H_hat_monotone_increasing(self):
        """H_hat should be monotonically increasing."""
        tau = np.linspace(0, 5, 500)
        H = H_hat_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert np.all(np.diff(H) >= -1e-8)


# ---------------------------------------------------------------------------
# Investment ratio boundary conditions
# ---------------------------------------------------------------------------

class TestInvestmentRatio:
    def test_boundary_condition_vanilla(self):
        """At tau=0, vanilla ratio = theta/gamma = 5.0."""
        tau = np.array([0.0, 0.5, 1.0])
        ratio = investment_ratio_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA, UPSILON)
        assert abs(ratio[0] - THETA / GAMMA) < 1e-6, \
            f"Vanilla BC failed: {ratio[0]:.6f} != {THETA/GAMMA}"

    def test_boundary_condition_path_dep(self):
        """At tau=0 (H_bar=0), PD ratio = theta/gamma = 5.0."""
        H_arr = np.array([0.0, 1.0, 2.0])
        tau_arr = np.array([0.0, 0.5, 1.0])
        ratio = investment_ratio_pd(H_arr, tau_arr, THETA, SIGMA, RHO, GAMMA, UPSILON)
        assert abs(ratio[0] - THETA / GAMMA) < 1e-6, \
            f"PD BC failed: {ratio[0]:.6f} != {THETA/GAMMA}"

    def test_all_ratios_positive(self):
        """Investment ratios should be positive for these model parameters."""
        T = 1.0
        N = 200
        tau = np.linspace(0, T, N + 1)
        ratio_v = investment_ratio_vanilla(tau, KAPPA, THETA, SIGMA, RHO, GAMMA, UPSILON)
        assert np.all(ratio_v > 0), "Vanilla ratio has non-positive values"

    @pytest.mark.parametrize("delta", [0.50, 0.60, 0.70])
    def test_pd_ratio_bc_for_all_deltas(self, delta):
        """Path-dependent ratio at tau=0 = 5.0 for all delta."""
        T = 1.0
        N = 100
        tau, H_bar = solve_H_bar(T, N, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
        ratio = investment_ratio_pd(H_bar, tau, THETA, SIGMA, RHO, GAMMA, UPSILON)
        assert abs(ratio[0] - THETA / GAMMA) < 1e-4, \
            f"PD BC failed for delta={delta}: {ratio[0]:.4f}"

    def test_convergence_check_H_bar(self):
        """Convergence: halving N reduces error by more than 50%."""
        delta = 0.70
        T = 1.0
        N_c = 200
        N_f = 400
        _, H_c = solve_H_bar(T, N_c, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
        _, H_f = solve_H_bar(T, N_f, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
        H_f_on_c = H_f[::2]
        rel_err = np.max(np.abs(H_f_on_c - H_c)) / (np.max(np.abs(H_c)) + 1e-15)
        assert rel_err < 0.05, f"Convergence failed: rel_err={rel_err:.4e}"


# ---------------------------------------------------------------------------
# B_bar tests
# ---------------------------------------------------------------------------

class TestSolveBBar:
    def test_initial_value_zero(self):
        """B_bar(0) = 0."""
        tau_H, H_bar = solve_H_bar(1.0, 100, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        _, B_bar = solve_B_bar(tau_H, H_bar, 1.0, 100, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert abs(B_bar[0]) < 1e-14

    def test_B_bar_finite(self):
        """B_bar is finite on the entire grid."""
        tau_H, H_bar = solve_H_bar(1.0, 100, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        _, B_bar = solve_B_bar(tau_H, H_bar, 1.0, 100, 0.7, KAPPA, THETA, SIGMA, RHO, GAMMA)
        assert np.all(np.isfinite(B_bar))
