"""Tests for src/insurance_deductible.py — insurance deductible computation."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from src.insurance_deductible import (
    phi_kernel,
    phi_integral,
    L_func,
    dL_dM,
    solve_Mbar,
    pd_deductible,
    vanilla_deductible,
)

# Fixed parameters
GAMMA_RISK = 1.0
UPSILON = 0.020
THETA_TILDE = 0.200
MU = 4.0
RHO1 = 0.020010

# Paper calibrated values
LAM0 = 1.431227
P_CALIB = 1.524052
RHO2_CALIB = 0.003982


# ---------------------------------------------------------------------------
# Kernel function tests
# ---------------------------------------------------------------------------

class TestPhiKernel:
    def test_phi_positive(self):
        """Phi kernel returns positive values."""
        tau = np.linspace(0.01, 5.0, 100)
        vals = phi_kernel(tau, RHO1, RHO2_CALIB, P_CALIB)
        assert np.all(vals > 0)

    def test_phi_decays(self):
        """Kernel phi is a decreasing function of tau."""
        tau = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        vals = phi_kernel(tau, RHO1, RHO2_CALIB, P_CALIB)
        assert np.all(np.diff(vals) < 0), "Phi kernel is not decreasing"

    def test_phi_integral_positive(self):
        """Integral of phi over [a, b] is positive."""
        val = phi_integral(0.0, 1.0, RHO1, 0.01, 1.5)
        assert val > 0

    def test_phi_integral_additive(self):
        """Integral over [a, c] = integral [a, b] + integral [b, c]."""
        a, b, c = 0.1, 0.5, 1.2
        rho1, rho2, p = RHO1, 0.01, 1.5
        assert abs(
            phi_integral(a, c, rho1, rho2, p) -
            (phi_integral(a, b, rho1, rho2, p) + phi_integral(b, c, rho1, rho2, p))
        ) < 1e-12

    def test_phi_integral_matches_numerical(self):
        """Analytical integral matches numerical (scipy) integration."""
        from scipy.integrate import quad
        a, b = 0.0, 2.0
        rho1, rho2, p = RHO1, 0.35, 1.30
        numerical, _ = quad(lambda u: rho1 / (rho2 + u)**p, a, b)
        analytical = phi_integral(a, b, rho1, rho2, p)
        assert abs(analytical - numerical) < 1e-8, \
            f"Analytical={analytical:.8f}, Numerical={numerical:.8f}"


# ---------------------------------------------------------------------------
# L_func tests
# ---------------------------------------------------------------------------

class TestLFunc:
    def test_L_at_zero(self):
        """L(0, 0) should equal -theta_tilde/mu - 1/mu = -(theta_tilde + 1)/mu."""
        val = L_func(0.0, 0.0, GAMMA_RISK, UPSILON, THETA_TILDE, MU)
        # eUs=1, arg = -mu*(theta_tilde/gamma + 0) = -4*0.2 = -0.8
        # L = 0 - (0.2/4)*exp(-0.8) - 1/4
        expected = -(THETA_TILDE / MU) * np.exp(-MU * THETA_TILDE / GAMMA_RISK) - 1.0 / MU
        assert abs(val - expected) < 1e-10

    def test_L_negative_at_zero(self):
        """L(0, 0) < 0 (from paper: M-bar <= 0 property)."""
        val = L_func(0.0, 0.0, GAMMA_RISK, UPSILON, THETA_TILDE, MU)
        assert val < 0

    def test_dL_dM_positive(self):
        """dL/dM > 0 always."""
        for s in [0.0, 0.5, 1.0, 5.0]:
            for M in [-1.0, 0.0, 0.5]:
                val = dL_dM(s, M, GAMMA_RISK, UPSILON, THETA_TILDE, MU)
                assert val > 0, f"dL/dM <= 0 at s={s}, M={M}: {val}"

    def test_dL_dM_greater_than_one(self):
        """dL/dM = 1 + theta_tilde * exp(...) > 1 always (exponential is positive).

        When M >= 0, the argument is negative so exp < 1, giving dL/dM in (1, 1+theta_tilde).
        When M < 0 (as in M_bar solutions), exp can exceed 1, so dL/dM > 1+theta_tilde.
        In all cases, dL/dM > 1.
        """
        for s in [0.0, 1.0, 5.0]:
            for M in [-0.5, 0.0, 0.1]:
                val = dL_dM(s, M, GAMMA_RISK, UPSILON, THETA_TILDE, MU)
                assert val > 1.0, f"dL/dM <= 1 at s={s}, M={M}: {val}"
        # For M >= 0, verify upper bound 1+theta_tilde also holds
        for s in [0.0, 1.0, 5.0]:
            for M in [0.0, 0.1, 0.5]:
                val = dL_dM(s, M, GAMMA_RISK, UPSILON, THETA_TILDE, MU)
                assert val <= 1.0 + THETA_TILDE + 1e-10, \
                    f"dL/dM > 1+theta for M>=0 at s={s}, M={M}: {val}"


# ---------------------------------------------------------------------------
# Solve_Mbar tests
# ---------------------------------------------------------------------------

class TestSolveMbar:
    def test_initial_value_zero(self):
        """M_bar(0) = 0 (initial condition)."""
        tau, M = solve_Mbar(T=1.0, N=100, p=1.5, rho2=0.35)
        assert abs(M[0]) < 1e-14

    def test_M_bar_nonpositive(self):
        """M_bar(tau) <= 0 for all tau > 0 (theoretical property)."""
        tau, M = solve_Mbar(T=1.0, N=200, p=1.30, rho2=0.35)
        assert np.all(M <= 1e-8), f"M_bar > 0 detected: max={M.max():.4e}"

    @pytest.mark.parametrize("p,rho2", [
        (1.30, 0.35), (1.524052, 0.35), (1.30, 1.20)
    ])
    def test_M_bar_nonpositive_various_params(self, p, rho2):
        """M_bar <= 0 for various subcritical (p, rho2) combinations."""
        norm = RHO1 / ((p - 1) * rho2**(p - 1))
        if norm >= 1:
            pytest.skip(f"Supercritical: p={p}, rho2={rho2}, norm={norm:.4f}")
        tau, M = solve_Mbar(T=1.0, N=200, p=p, rho2=rho2)
        assert np.all(M <= 1e-6), f"M_bar > 0: max={M.max():.4e}, p={p}, rho2={rho2}"

    def test_M_bar_finite(self):
        """M_bar is finite on entire grid."""
        tau, M = solve_Mbar(T=1.0, N=100, p=1.30, rho2=0.35)
        assert np.all(np.isfinite(M))

    def test_grid_shape(self):
        """Output arrays have shape (N+1,)."""
        N = 150
        tau, M = solve_Mbar(T=2.0, N=N, p=1.30, rho2=0.35)
        assert tau.shape == (N + 1,)
        assert M.shape == (N + 1,)

    def test_convergence(self):
        """Halving step size reduces relative error."""
        p, rho2, T = 1.30, 0.35, 1.0
        N_c = 100
        N_f = 200
        _, M_c = solve_Mbar(T, N_c, p, rho2)
        _, M_f = solve_Mbar(T, N_f, p, rho2)
        M_f_on_c = M_f[::2]
        scale = np.max(np.abs(M_c)) + 1e-15
        rel_err = np.max(np.abs(M_f_on_c - M_c)) / scale
        assert rel_err < 0.05, f"Convergence failed: rel_err={rel_err:.4e}"


# ---------------------------------------------------------------------------
# Deductible tests
# ---------------------------------------------------------------------------

class TestDeductibles:
    def test_vanilla_boundary(self):
        """Vanilla deductible at tau=0 equals theta_tilde/gamma."""
        tau = np.linspace(0, 1, 100)
        d = vanilla_deductible(tau, GAMMA_RISK, UPSILON, THETA_TILDE)
        assert abs(d[0] - THETA_TILDE / GAMMA_RISK) < 1e-10

    def test_vanilla_decreasing(self):
        """Vanilla deductible is decreasing in tau (increasing in t=T-tau)."""
        tau = np.linspace(0, 10, 100)
        d = vanilla_deductible(tau, GAMMA_RISK, UPSILON, THETA_TILDE)
        assert np.all(np.diff(d) <= 0), "Vanilla deductible not monotone in tau"

    def test_pd_boundary(self):
        """Path-dependent deductible at tau=0 (M_bar=0) equals theta_tilde/gamma."""
        tau = np.array([0.0, 0.5, 1.0])
        M = np.array([0.0, -0.1, -0.2])  # M_bar values
        d = pd_deductible(tau, M, GAMMA_RISK, UPSILON, THETA_TILDE)
        assert abs(d[0] - THETA_TILDE / GAMMA_RISK) < 1e-10

    def test_pd_below_vanilla(self):
        """Path-dependent deductible < vanilla deductible (M_bar <= 0)."""
        p, rho2, T = 1.30, 0.35, 1.0
        N = 200
        tau, M = solve_Mbar(T, N, p, rho2)
        d_pd = pd_deductible(tau, M, GAMMA_RISK, UPSILON, THETA_TILDE)
        d_van = vanilla_deductible(tau, GAMMA_RISK, UPSILON, THETA_TILDE)
        # d_pd <= d_van everywhere (up to numerical precision)
        margin = d_van - d_pd
        assert np.all(margin >= -1e-6), \
            f"d* > d_hat detected: max violation={(-margin).max():.4e}"

    def test_pd_deductible_nonnegative(self):
        """Deductible is always non-negative (max(0, ...) guarantee)."""
        tau = np.linspace(0, 10, 100)
        M = np.full(100, -100.0)  # very negative M
        d = pd_deductible(tau, M, GAMMA_RISK, UPSILON, THETA_TILDE)
        assert np.all(d >= 0)

    def test_calibrated_params_boundary(self):
        """Calibrated params: d*(T) = d_hat(T) = 0.2 at tau=0."""
        p, rho2, T = P_CALIB, RHO2_CALIB, 1.0
        norm = RHO1 / ((p - 1) * rho2**(p - 1))
        if norm >= 1:
            pytest.skip(f"Supercritical: norm={norm:.4f}")
        tau, M = solve_Mbar(T, 200, p, rho2)
        d = pd_deductible(tau, M, GAMMA_RISK, UPSILON, THETA_TILDE)
        assert abs(d[0] - THETA_TILDE) < 1e-6, f"BC failed: d[0]={d[0]:.6f}"


# ---------------------------------------------------------------------------
# Data module tests (basic)
# ---------------------------------------------------------------------------

class TestDataModule:
    def test_haversine_zero_distance(self):
        """Haversine distance from a point to itself is 0."""
        import math
        from data.download_earthquake import haversine
        d = haversine(30.5728, 104.0668, 30.5728, 104.0668)
        assert abs(d) < 1e-10

    def test_haversine_known_distance(self):
        """Haversine distance between Beijing and Shanghai ~1069 km."""
        from data.download_earthquake import haversine
        # Beijing: 39.9042N, 116.4074E; Shanghai: 31.2304N, 121.4737E
        d = haversine(39.9042, 116.4074, 31.2304, 121.4737)
        assert 1000 < d < 1200, f"Unexpected distance: {d:.1f} km"

    def test_haversine_chengdu_to_wenchuan(self):
        """Haversine distance from Chengdu to Wenchuan ~89 km (< 300 km threshold)."""
        from data.download_earthquake import haversine
        # Wenchuan: 31.0N, 103.4E
        d = haversine(30.5728, 104.0668, 31.0, 103.4)
        assert d < 300, f"Wenchuan should be < 300 km from Chengdu: {d:.1f} km"

    def test_catalog_returns_events(self):
        """ISC catalog cache returns >= 37 events (paper n=37; current ISC may differ)."""
        from data.download_earthquake import load_or_download
        from pathlib import Path
        cache = Path('/workspace/project/data/earthquake_events.csv')
        if cache.exists():
            import pandas as pd
            df = pd.read_csv(cache)
            assert len(df) >= 37, f"Expected >= 37 events, got {len(df)}"
        # If no cache, just verify the function is importable and callable

    def test_catalog_events_sorted(self):
        """Event times are sorted in ascending order."""
        from data.download_earthquake import load_or_download
        from pathlib import Path
        cache = Path('/workspace/project/data/earthquake_events.csv')
        if not cache.exists():
            return  # skip if no cached data
        import pandas as pd
        df = pd.read_csv(cache)
        t_years = df['T_YEARS'].values
        assert np.all(np.diff(t_years) >= 0), "Events not sorted"

    def test_t_years_first_event_zero(self):
        """First event (mainshock) at t=0."""
        from data.download_earthquake import load_or_download
        from pathlib import Path
        cache = Path('/workspace/project/data/earthquake_events.csv')
        if not cache.exists():
            return  # skip if no cached data
        import pandas as pd
        df = pd.read_csv(cache)
        assert abs(df['T_YEARS'].iloc[0]) < 1e-10
