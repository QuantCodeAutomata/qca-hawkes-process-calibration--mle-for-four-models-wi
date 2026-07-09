# src/insurance_deductible.py
"""Insurance deductible computation via product-integration for Volterra equations.

Implements the nonlinear Volterra solver for M-bar(tau) and the equilibrium
optimal deductible d*(t) under the Hawkes power kernel model.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


def phi_kernel(tau: np.ndarray, rho1: float, rho2: float, p: float) -> np.ndarray:
    """Modified Omori kernel: phi(tau) = rho1 / (rho2 + tau)^p."""
    return rho1 / (rho2 + tau)**p


def phi_integral(a: float, b: float, rho1: float, rho2: float, p: float) -> float:
    """Analytical integral of phi from a to b (for p > 1).

    = rho1 * [(rho2+a)^{1-p} - (rho2+b)^{1-p}] / (p-1)
    """
    return rho1 * ((rho2 + a)**(1 - p) - (rho2 + b)**(1 - p)) / (p - 1)


def L_func(
    s: float,
    Mbar: float,
    gamma_risk: float = 1.0,
    Upsilon: float = 0.020,
    theta_tilde: float = 0.200,
    mu: float = 4.0,
) -> float:
    """Nonlinear integrand L(s, M-bar) for the M-bar Volterra equation.

    L(s, M) = M - (theta_tilde/mu)*e^{Ups*s}*exp(-mu*(theta_tilde/(gamma*e^{Ups*s}) + M/e^{Ups*s}))
              - e^{Ups*s}/mu

    arg is clipped to [-700, 700] to prevent float64 overflow in exp.
    """
    eUs = np.exp(Upsilon * s)
    arg = -mu * (theta_tilde / (gamma_risk * eUs) + Mbar / eUs)
    arg = np.clip(arg, -700.0, 700.0)
    return Mbar - (theta_tilde / mu) * eUs * np.exp(arg) - eUs / mu


def dL_dM(
    s: float,
    Mbar: float,
    gamma_risk: float = 1.0,
    Upsilon: float = 0.020,
    theta_tilde: float = 0.200,
    mu: float = 4.0,
) -> float:
    """Partial derivative dL/dM for Newton's method.

    dL/dM = 1 + theta_tilde * exp(-mu*(theta_tilde/(gamma*e^{Ups*s}) + M/e^{Ups*s}))

    arg clipped to avoid overflow.
    """
    eUs = np.exp(Upsilon * s)
    arg = -mu * (theta_tilde / (gamma_risk * eUs) + Mbar / eUs)
    arg = np.clip(arg, -700.0, 700.0)
    return 1.0 + theta_tilde * np.exp(arg)


def solve_Mbar(
    T: float,
    N: int,
    p: float,
    rho2: float,
    rho1: float = 0.020010,
    gamma_risk: float = 1.0,
    Upsilon: float = 0.020,
    theta_tilde: float = 0.200,
    mu: float = 4.0,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve nonlinear Volterra equation for M-bar(tau).

    M-bar(tau) = int_0^tau phi(tau-s) * L(s, M-bar(s)) ds
    where phi(u) = rho1/(rho2+u)^p  (Modified Omori kernel).
    Uses product-integration with implicit Newton correction.

    Parameters
    ----------
    T       : terminal time
    N       : number of steps
    p, rho2 : kernel parameters
    rho1    : kernel amplitude (default calibrated value)
    gamma_risk, Upsilon, theta_tilde, mu : financial parameters

    Returns
    -------
    tau : ndarray(N+1,)
    M   : ndarray(N+1,), M-bar values
    """
    h = T / N
    tau = np.linspace(0, T, N + 1)
    M = np.zeros(N + 1)
    L_vals = np.zeros(N + 1)
    L_vals[0] = L_func(tau[0], M[0], gamma_risk, Upsilon, theta_tilde, mu)

    # Precompute self-weight: integral of phi over [0, h]
    w_self = phi_integral(0.0, h, rho1, rho2, p)

    for n in range(N):
        # Vectorized predictor weights
        j_arr = np.arange(n + 1, dtype=float)
        s_lo = (n - j_arr) * h          # tau_{n+1} - tau_{j+1}
        s_hi = (n + 1 - j_arr) * h      # tau_{n+1} - tau_j
        # Analytical integral of phi over each sub-interval
        w_arr = rho1 * ((rho2 + s_lo)**(1 - p) - (rho2 + s_hi)**(1 - p)) / (p - 1)
        M_pred = np.dot(w_arr, L_vals[:n + 1])

        # Newton's method: solve G(M) = M - M_pred - w_self * L(tau_{n+1}, M) = 0
        # Initial guess: predicted value, clamped to non-positive (theoretical property)
        M_cur = min(M_pred, 0.0)
        for _ in range(max_iter):
            L_cur = L_func(tau[n + 1], M_cur, gamma_risk, Upsilon, theta_tilde, mu)
            G = M_cur - M_pred - w_self * L_cur
            if abs(G) < tol:
                break
            dG = 1.0 - w_self * dL_dM(tau[n + 1], M_cur, gamma_risk, Upsilon, theta_tilde, mu)
            if abs(dG) < 1e-15:
                break
            step = G / dG
            # Damped Newton: limit step to avoid large oscillations
            step = np.clip(step, -5.0, 5.0)
            M_cur -= step

        # M-bar is theoretically non-positive; clamp to remove small numerical positives
        M[n + 1] = min(M_cur, 0.0)
        L_vals[n + 1] = L_func(tau[n + 1], M[n + 1], gamma_risk, Upsilon, theta_tilde, mu)

    return tau, M


def pd_deductible(
    tau_arr: np.ndarray,
    Mbar_arr: np.ndarray,
    gamma_risk: float = 1.0,
    Upsilon: float = 0.020,
    theta_tilde: float = 0.200,
) -> np.ndarray:
    """Path-dependent equilibrium deductible.

    d*(t) = max(0, e^{-Upsilon*tau} * (theta_tilde/gamma + M-bar(tau)))
    where tau = T - t.
    """
    return np.maximum(
        0.0,
        np.exp(-Upsilon * tau_arr) * (theta_tilde / gamma_risk + Mbar_arr)
    )


def vanilla_deductible(
    tau_arr: np.ndarray,
    gamma_risk: float = 1.0,
    Upsilon: float = 0.020,
    theta_tilde: float = 0.200,
) -> np.ndarray:
    """Vanilla (Poisson) equilibrium deductible.

    d-hat(t) = theta_tilde / (gamma * e^{Upsilon*tau})
    """
    return theta_tilde / (gamma_risk * np.exp(Upsilon * tau_arr))
