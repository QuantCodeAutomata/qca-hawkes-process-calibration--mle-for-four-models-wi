# src/volterra_solver.py
"""Fractional Adams predictor-corrector solver for Volterra integral equations.

Implements Diethelm et al. (2004) Adams PC method for equations of the form:
  y(tau) = int_0^tau (tau-s)^(delta-1)/Gamma(delta) * f(s, y(s)) ds
"""
from __future__ import annotations
import numpy as np
from scipy.special import gamma as gamma_func
from scipy.interpolate import interp1d
from typing import Callable, Tuple


def fractional_adams(
    f_func: Callable[[float, float], float],
    T: float,
    N: int,
    delta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve y(tau) = int_0^tau (tau-s)^(delta-1)/Gamma(delta)*f(s,y(s))ds.

    Uses the fractional Adams predictor-corrector scheme (Diethelm et al., 2004).
    Initial condition: y(0) = 0.

    Parameters
    ----------
    f_func : callable(s, y) -> float
        Right-hand side function.
    T : float
        Terminal time.
    N : int
        Number of time steps.
    delta : float
        Fractional order (0 < delta <= 1).

    Returns
    -------
    tau : ndarray of shape (N+1,)
    y   : ndarray of shape (N+1,)
    """
    h = T / N
    tau = np.linspace(0, T, N + 1)
    y = np.zeros(N + 1)
    y[0] = 0.0
    f_vals = np.zeros(N + 1)
    f_vals[0] = f_func(tau[0], y[0])

    Gd2 = gamma_func(delta + 2)  # Gamma(delta+2)
    coeff = h**delta / Gd2

    for n in range(N):
        # --- PREDICTOR ---
        # y^P_{n+1} = coeff * sum_{j=0}^{n} b_{j,n+1} * f_vals[j]
        # b_{j,n+1} = (n+1-j)^delta - (n-j)^delta
        j_arr = np.arange(n + 1, dtype=float)
        b_pred = (n + 1 - j_arr)**delta - (n - j_arr)**delta
        y_pred = coeff * np.dot(b_pred, f_vals[:n + 1])
        f_pred = f_func(tau[n + 1], y_pred)

        # --- CORRECTOR ---
        # Weights a^c_{j, n+1}
        a_corr = np.zeros(n + 2)
        if n == 0:
            # j=0: n^{d+1} - (n-delta)*(n+1)^delta with n=0 => 0 - (0-delta)*1 = delta
            a_corr[0] = delta
            a_corr[1] = 1.0  # j=n+1=1
        else:
            # j=0:
            a_corr[0] = n**(delta + 1) - (n - delta) * (n + 1)**delta
            # j=1..n:
            jj = np.arange(1, n + 1, dtype=float)
            nj = n - jj
            a_corr[1:n + 1] = (nj + 2)**(delta + 1) + nj**(delta + 1) - 2 * (nj + 1)**(delta + 1)
            # j=n+1:
            a_corr[n + 1] = 1.0

        y[n + 1] = coeff * (
            a_corr[n + 1] * f_pred +
            np.dot(a_corr[:n + 1], f_vals[:n + 1])
        )
        f_vals[n + 1] = f_func(tau[n + 1], y[n + 1])

    return tau, y


def solve_H_bar(
    T: float,
    N: int,
    delta: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Volterra equation for H-bar(tau).

    Equation: H-bar(tau) = int_0^tau (tau-s)^(delta-1)/Gamma(delta) * f_H(s, H-bar(s)) ds
    f_H(s, H) = -c*H + theta^2/gamma  where  c = kappa + theta*sigma*rho

    Parameters
    ----------
    T, N, delta : time horizon, steps, fractional order
    kappa, theta, sigma, rho, gamma : model parameters

    Returns
    -------
    tau, H_bar arrays
    """
    c = kappa + theta * sigma * rho
    theta2_gamma = theta**2 / gamma

    def f_H(s: float, H: float) -> float:
        return -c * H + theta2_gamma

    return fractional_adams(f_H, T, N, delta)


def solve_B_bar(
    tau_H: np.ndarray,
    H_bar: np.ndarray,
    T: float,
    N: int,
    delta: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the Volterra equation for B-bar(tau).

    Equation: B-bar(tau) = int_0^tau (tau-s)^(delta-1)/Gamma(delta) * g(s, B-bar(s)) ds
    g(s, B) = -kappa*B - theta*sigma*rho*H(s) - gamma*sigma^2*(1-rho^2)/2 * H(s)^2 + theta^2/(2*gamma)

    Parameters
    ----------
    tau_H, H_bar : precomputed H-bar on grid
    T, N, delta, kappa, theta, sigma, rho, gamma : model parameters

    Returns
    -------
    tau, B_bar arrays
    """
    H_interp = interp1d(tau_H, H_bar, kind='linear', fill_value='extrapolate')
    sig2 = sigma**2
    rho2 = rho**2
    coeff_H2 = gamma * sig2 * (1 - rho2) / 2
    theta2_2g = theta**2 / (2 * gamma)

    def g_B(s: float, B: float) -> float:
        Hs = H_interp(s)
        return (
            -kappa * B
            - theta * sigma * rho * Hs
            - coeff_H2 * Hs**2
            + theta2_2g
        )

    return fractional_adams(g_B, T, N, delta)


def H_hat_vanilla(
    tau_arr: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
) -> np.ndarray:
    """Closed-form vanilla Heston H-hat(tau) for comparison.

    ODE: dH/dtau = c*H - theta^2/gamma, H(0)=0
    Solution: H-hat(tau) = (theta^2/gamma)/c * (exp(c*tau) - 1)
    where c = kappa + theta*sigma*rho.
    """
    c = kappa + theta * sigma * rho
    return (theta**2 / gamma) / c * (np.exp(c * tau_arr) - 1)


def investment_ratio_pd(
    H_arr: np.ndarray,
    tau_arr: np.ndarray,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
    Upsilon: float,
) -> np.ndarray:
    """Path-dependent equilibrium investment ratio alpha*_t * X_t / sqrt(v_t).

    = [theta - gamma*sigma*rho*H_bar(tau)] / [gamma * exp(Upsilon*tau)]
    where tau = T - t.
    """
    return (theta - gamma * sigma * rho * H_arr) / (gamma * np.exp(Upsilon * tau_arr))


def investment_ratio_vanilla(
    tau_arr: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    gamma: float,
    Upsilon: float,
) -> np.ndarray:
    """Vanilla Heston equilibrium investment ratio.

    = [theta - sigma*rho*gamma*H_hat(tau)] / [gamma * exp(Upsilon*tau)]
    """
    H_hat = H_hat_vanilla(tau_arr, kappa, theta, sigma, rho, gamma)
    return (theta - sigma * rho * gamma * H_hat) / (gamma * np.exp(Upsilon * tau_arr))
