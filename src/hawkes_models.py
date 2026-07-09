# src/hawkes_models.py
"""
Hawkes process model implementations: 4 variants with MLE and AICc.

Models:
  1. Only Power Kernel      – lambda0, rho1, rho2, p          (k=4)
  2. Mean-Rev & Power Ker.  – lambda0, a0, a1, rho1, rho2, p  (k=6)
  3. Only Exponential Ker.  – lambda0, b1, b2                 (k=3)
  4. Mean-Rev & Exp. Ker.   – lambda0, a0, a1, b1, b2         (k=5)

Model 4 uses an exact Markovian ODE formulation between events (O(n) cost).
Model 2 uses a vectorised Volterra grid with a coarser mesh (h=0.05 yr).
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Model 1: Only Power Kernel  (closed-form compensator)
# lambda_t = lambda0 + sum_{t_j < t} rho1/(rho2 + t - t_j)^p
# ---------------------------------------------------------------------------

def intensity_power_only(
    t: float,
    t_events: np.ndarray,
    params: np.ndarray,
) -> float:
    """Conditional intensity for Model 1 at time t."""
    lam0, rho1, rho2, p = params
    past = t_events[t_events < t]
    return float(lam0 + np.sum(rho1 / (rho2 + t - past) ** p))


def loglik_power_only(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
) -> float:
    """Log-likelihood for Model 1: Only Power Kernel (closed-form compensator)."""
    lam0, rho1, rho2, p = params
    if lam0 <= 0 or rho1 <= 0 or rho2 <= 0 or p <= 1:
        return -np.inf
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        norm = rho1 / ((p - 1) * rho2 ** (p - 1))
    if not np.isfinite(norm) or norm >= 1:
        return -np.inf

    n = len(t_events)
    log_lam_sum = 0.0
    for i in range(n):
        ti = t_events[i]
        prev = t_events[:i]
        lam_i = lam0 + (np.sum(rho1 / (rho2 + ti - prev) ** p) if i > 0 else 0.0)
        if lam_i <= 0:
            return -np.inf
        log_lam_sum += np.log(lam_i)

    # Compensator (closed form, p > 1)
    gaps = T_cal - t_events                  # shape (n,)
    compensator = lam0 * T_cal + np.sum(
        rho1 * ((rho2 + gaps) ** (1 - p) - rho2 ** (1 - p)) / (1 - p)
    )
    return log_lam_sum - compensator


def run_mle_model1(
    t_events: np.ndarray,
    T_cal: float,
    n_restarts: int = 25,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """MLE for Model 1 with multiple random restarts (SLSQP)."""
    rng = np.random.default_rng(seed)
    best_ll, best_params = -np.inf, None

    def neg_ll(x: np.ndarray) -> float:
        ll = loglik_power_only(x, t_events, T_cal)
        return -ll if np.isfinite(ll) else 1e12

    bounds = [(1e-8, None), (1e-8, None), (1e-8, None), (1.001, None)]
    constraints = [{'type': 'ineq',
                    'fun': lambda x: 1 - x[1] / ((x[3] - 1) * x[2] ** (x[3] - 1)) - 1e-4}]

    for _ in range(n_restarts):
        lam0 = np.exp(rng.uniform(-2, 2))
        rho1 = np.exp(rng.uniform(-6, 0))
        rho2 = np.exp(rng.uniform(-7, -1))
        p = 1.0 + np.exp(rng.uniform(-2, 1))
        x0 = np.array([lam0, rho1, rho2, p])
        try:
            res = minimize(neg_ll, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'ftol': 1e-12, 'maxiter': 2000})
            if res.fun < 1e11:
                ll = -res.fun
                if ll > best_ll:
                    best_ll = ll
                    best_params = res.x.copy()
        except Exception:
            pass

    return best_params, best_ll


# ---------------------------------------------------------------------------
# Model 3: Only Exponential Kernel  (recursive + closed-form compensator)
# lambda_t = lambda0 + sum_{t_j < t} b1*exp(-b2*(t-t_j))
# ---------------------------------------------------------------------------

def loglik_exp_only(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
) -> float:
    """Log-likelihood for Model 3: Only Exponential Kernel."""
    lam0, b1, b2 = params
    if lam0 <= 0 or b1 <= 0 or b2 <= 0 or b1 / b2 >= 1:
        return -np.inf

    n = len(t_events)
    log_lam_sum = 0.0
    A = 0.0  # recursive auxiliary: A_i = sum_{j<i} exp(-b2*(t_i - t_j))
    for i in range(n):
        ti = t_events[i]
        if i > 0:
            A = np.exp(-b2 * (ti - t_events[i - 1])) * (1.0 + A)
        lam_i = lam0 + b1 * A
        if lam_i <= 0:
            return -np.inf
        log_lam_sum += np.log(lam_i)

    compensator = lam0 * T_cal + (b1 / b2) * np.sum(1 - np.exp(-b2 * (T_cal - t_events)))
    return log_lam_sum - compensator


def run_mle_model3(
    t_events: np.ndarray,
    T_cal: float,
    n_restarts: int = 25,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """MLE for Model 3 with multiple random restarts (SLSQP)."""
    rng = np.random.default_rng(seed)
    best_ll, best_params = -np.inf, None

    def neg_ll(x: np.ndarray) -> float:
        ll = loglik_exp_only(x, t_events, T_cal)
        return -ll if np.isfinite(ll) else 1e12

    bounds = [(1e-8, None), (1e-8, None), (1e-8, None)]
    constraints = [{'type': 'ineq', 'fun': lambda x: 1 - x[1] / x[2] - 1e-4}]

    for _ in range(n_restarts):
        lam0 = np.exp(rng.uniform(-2, 2))
        b1 = np.exp(rng.uniform(-2, 2))
        b2 = np.exp(rng.uniform(-1, 3))
        x0 = np.array([lam0, b1, b2])
        try:
            res = minimize(neg_ll, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'ftol': 1e-12, 'maxiter': 2000})
            if res.fun < 1e11:
                ll = -res.fun
                if ll > best_ll:
                    best_ll = ll
                    best_params = res.x.copy()
        except Exception:
            pass

    return best_params, best_ll


# ---------------------------------------------------------------------------
# Model 2: Mean-Reverting & Power Kernel  (vectorised Volterra grid, h=0.05)
# lambda_t = lambda0 + int_0^t [rho1/(rho2+t-r)^p](a0+a1*lam_r) dr
#          + sum_{t_j<t} rho1/(rho2+t-t_j)^p
# params = [lambda0, a0, a1, rho1, rho2, p]
# ---------------------------------------------------------------------------

def solve_lambda_grid_power(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
    h: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve conditional intensity on a uniform grid for Model 2.

    Uses the Volterra-trapezoid scheme.  Grid step h=0.05 yr gives
    ~300 points for T=15 yr, keeping cost manageable.
    """
    lam0, a0, a1, rho1, rho2, p = params
    N = max(int(T_cal / h) + 1, 2)
    t_grid = np.linspace(0, T_cal, N)
    lam_grid = np.empty(N)
    lam_grid[0] = lam0

    for i in range(1, N):
        t = t_grid[i]
        # Jump sum: events strictly before t
        mask_ev = t_events < t
        jump_sum = (np.sum(rho1 / (rho2 + t - t_events[mask_ev]) ** p)
                    if mask_ev.any() else 0.0)
        # Volterra integral: trapezoid over past grid slice
        dt_arr = t - t_grid[:i]          # (i,) — all positive
        phi = rho1 / (rho2 + dt_arr) ** p
        integrand = phi * (a0 + a1 * lam_grid[:i])
        # Trapezoid weights: h * [0.5, 1, 1, …, 1, 0.5] — simplified
        integral = np.trapz(integrand, t_grid[:i])
        lam_grid[i] = lam0 + integral + jump_sum

    return t_grid, lam_grid


def loglik_power_meanrev(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
    h: float = 0.05,
) -> float:
    """Log-likelihood for Model 2: Mean-Reverting & Power Kernel."""
    lam0, a0, a1, rho1, rho2, p = params
    if lam0 <= 0 or rho1 <= 0 or rho2 <= 0 or p <= 1:
        return -np.inf
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        norm = rho1 * max(a1 + 1.0, 0.0) / ((p - 1) * rho2 ** (p - 1))
    if not np.isfinite(norm) or norm >= 1:
        return -np.inf
    try:
        t_grid, lam_grid = solve_lambda_grid_power(params, t_events, T_cal, h)
    except Exception:
        return -np.inf
    if np.any(lam_grid <= 0):
        return -np.inf
    lam_at_ev = np.interp(t_events, t_grid, lam_grid)
    if np.any(lam_at_ev <= 0):
        return -np.inf
    compensator = np.trapz(lam_grid, t_grid)
    return float(np.sum(np.log(lam_at_ev)) - compensator)


def run_mle_model2(
    t_events: np.ndarray,
    T_cal: float,
    n_restarts: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """MLE for Model 2 with multiple random restarts (SLSQP)."""
    rng = np.random.default_rng(seed)
    best_ll, best_params = -np.inf, None

    def neg_ll(x: np.ndarray) -> float:
        ll = loglik_power_meanrev(x, t_events, T_cal)
        return -ll if np.isfinite(ll) else 1e12

    bounds = [(1e-8, None), (1e-8, None), (None, None),
              (1e-8, None), (1e-8, None), (1.001, None)]
    constraints = [{'type': 'ineq',
                    'fun': lambda x: (1 - x[3] * max(x[2] + 1, 0)
                                      / ((x[5] - 1) * x[4] ** (x[5] - 1)) - 1e-4)}]

    for _ in range(n_restarts):
        lam0 = np.exp(rng.uniform(-1, 2))
        a0 = np.exp(rng.uniform(-5, 0))
        a1 = rng.uniform(-3, 0)
        rho1 = np.exp(rng.uniform(-3, 0))
        rho2 = np.exp(rng.uniform(-7, -1))
        p = 1.0 + np.exp(rng.uniform(-3, 0))
        x0 = np.array([lam0, a0, a1, rho1, rho2, p])
        try:
            res = minimize(neg_ll, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'ftol': 1e-10, 'maxiter': 2000})
            if res.fun < 1e11:
                ll = -res.fun
                if ll > best_ll:
                    best_ll = ll
                    best_params = res.x.copy()
        except Exception:
            pass

    return best_params, best_ll


# ---------------------------------------------------------------------------
# Model 4: Mean-Reverting & Exponential Kernel  (Markovian ODE, O(n) cost)
# lambda_t = lambda0 + int_0^t b1*exp(-b2*(t-r))*(a0+a1*lam_r) dr
#          + sum_{t_j<t} b1*exp(-b2*(t-t_j))
# params = [lambda0, a0, a1, b1, b2]
#
# ODE state y=(X,J) between events:
#   dX/dt = b1*(a0 + a1*(lambda0+X+J)) - b2*X
#   dJ/dt = -b2*J
# At each event t_j+:  J -> J + b1
# lambda_t = lambda0 + X_t + J_t
# ---------------------------------------------------------------------------

def _advance_exp_meanrev(
    X0: float, J0: float, Z0: float,
    dt: float,
    lam0: float, a0: float, a1: float, b1: float, b2: float,
) -> Tuple[float, float, float]:
    """
    Exact analytical advance of Model 4 ODE state (X, J, Z) by time step dt.

    The ODE between events is linear:
        dX/dt = b1*(a0 + a1*lambda0) + (b1*a1 - b2)*X + b1*a1*J
        dJ/dt = -b2*J
        dZ/dt = lambda0 + X + J   (compensator accumulator)

    Eigenvalues: mu1 = b1*a1 - b2,  mu2 = -b2.
    Exact solution derived via integrating-factor method.
    """
    mu1 = b1 * a1 - b2   # X eigenvalue
    mu2 = -b2             # J eigenvalue (= -b2)
    c0 = b1 * (a0 + a1 * lam0)  # constant forcing term for X

    e1 = np.exp(mu1 * dt)
    e2 = np.exp(mu2 * dt)   # = exp(-b2*dt)

    # --- X(dt) -----------------------------------------------------------
    if abs(mu1) < 1e-10:
        # mu1 ≈ 0: Taylor-expand exp(mu1*dt) ≈ 1 + mu1*dt
        X_new = X0 + c0 * dt + b1 * a1 * J0 * dt
    elif abs(mu1 - mu2) < 1e-10:
        # mu1 ≈ mu2: repeated eigenvalue limit
        X_new = (X0 + b1 * a1 * J0 * dt) * e1 + c0 / mu1 * (e1 - 1.0)
    else:
        X_new = (X0 * e1
                 + c0 / mu1 * (e1 - 1.0)
                 + b1 * a1 * J0 / (mu1 - mu2) * (e1 - e2))

    # --- J(dt) -----------------------------------------------------------
    J_new = J0 * e2

    # --- Z(dt): int_0^dt (lam0 + X_s + J_s) ds --------------------------
    dZ = lam0 * dt

    # int_0^dt J_s ds = J0 * (1 - exp(-b2*dt)) / b2
    if abs(b2) < 1e-10:
        dZ += J0 * dt
    else:
        dZ += J0 * (1.0 - e2) / b2

    # int_0^dt X_s ds (analytical)
    if abs(mu1) < 1e-10:
        # mu1 ≈ 0
        dZ += X0 * dt + c0 * dt ** 2 / 2.0
    elif abs(mu1 - mu2) < 1e-10:
        # repeated eigenvalue
        dZ += (X0 * (e1 - 1.0) / mu1
               + c0 / mu1 * ((e1 - 1.0) / mu1 - dt)
               + b1 * a1 * J0 / mu1 * ((e1 - 1.0) / mu1 - dt * e1))
    else:
        dZ += (X0 * (e1 - 1.0) / mu1
               + c0 / mu1 * ((e1 - 1.0) / mu1 - dt)
               + b1 * a1 * J0 / (mu1 - mu2) * ((e1 - 1.0) / mu1 - (e2 - 1.0) / mu2))

    return X_new, J_new, Z0 + dZ


def loglik_exp_meanrev(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
) -> float:
    """
    Log-likelihood for Model 4: Mean-Reverting & Exponential Kernel.

    Exact O(n) computation via analytical ODE solution between events.
    No numerical ODE solver required — eliminates the main performance bottleneck.
    """
    lam0, a0, a1, b1, b2 = params
    if lam0 <= 0 or b1 <= 0 or b2 <= 0:
        return -np.inf
    with np.errstate(over='ignore'):
        norm = b1 * max(a1 + 1.0, 0.0) / b2
    if not np.isfinite(norm) or norm >= 1:
        return -np.inf

    n = len(t_events)
    log_lam_sum = 0.0
    X, J, Z = 0.0, 0.0, 0.0  # initial state
    t_cur = 0.0

    for i in range(n):
        t_ev = t_events[i]
        dt = t_ev - t_cur
        if dt > 1e-15:
            X, J, Z = _advance_exp_meanrev(X, J, Z, dt, lam0, a0, a1, b1, b2)
            if not np.isfinite(X):
                return -np.inf
        t_cur = t_ev
        # Intensity at event (before jump)
        lam_ev = lam0 + X + J
        if lam_ev <= 0:
            return -np.inf
        log_lam_sum += np.log(lam_ev)
        # Apply jump
        J += b1

    # Advance to T_cal for compensator
    dt_final = T_cal - t_cur
    if dt_final > 1e-15:
        _, _, Z = _advance_exp_meanrev(X, J, Z, dt_final, lam0, a0, a1, b1, b2)

    return log_lam_sum - Z


def solve_lambda_grid_exp(
    params: np.ndarray,
    t_events: np.ndarray,
    T_cal: float,
    n_grid: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate Model 4 conditional intensity on a fine uniform grid for plotting.

    Uses analytical ODE advancement between grid points (with event jumps applied).
    """
    lam0, a0, a1, b1, b2 = params
    t_grid = np.linspace(0, T_cal, n_grid)
    lam_grid = np.empty(n_grid)

    X, J, Z = 0.0, 0.0, 0.0
    t_cur = 0.0
    ev_idx = 0
    n_ev = len(t_events)

    for gi, t_q in enumerate(t_grid):
        # Apply all events between t_cur and t_q
        while ev_idx < n_ev and t_events[ev_idx] <= t_q:
            t_ev = t_events[ev_idx]
            dt_ev = t_ev - t_cur
            if dt_ev > 1e-15:
                X, J, Z = _advance_exp_meanrev(X, J, Z, dt_ev, lam0, a0, a1, b1, b2)
            J += b1
            t_cur = t_ev
            ev_idx += 1
        # Advance to t_q
        dt_q = t_q - t_cur
        if dt_q > 1e-15:
            X_q, J_q, _ = _advance_exp_meanrev(X, J, Z, dt_q, lam0, a0, a1, b1, b2)
        else:
            X_q, J_q = X, J
        lam_grid[gi] = lam0 + X_q + J_q

    return t_grid, lam_grid


def run_mle_model4(
    t_events: np.ndarray,
    T_cal: float,
    n_restarts: int = 20,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """MLE for Model 4 with multiple random restarts (SLSQP)."""
    rng = np.random.default_rng(seed)
    best_ll, best_params = -np.inf, None

    def neg_ll(x: np.ndarray) -> float:
        ll = loglik_exp_meanrev(x, t_events, T_cal)
        return -ll if np.isfinite(ll) else 1e12

    bounds = [(1e-8, None), (1e-8, None), (None, None), (1e-8, None), (1e-8, None)]
    constraints = [{'type': 'ineq',
                    'fun': lambda x: 1 - x[3] * max(x[2] + 1, 0) / x[4] - 1e-4}]

    for _ in range(n_restarts):
        lam0 = np.exp(rng.uniform(-3, 2))
        a0 = np.exp(rng.uniform(-1, 2))
        a1 = rng.uniform(-4, 0)
        b1 = np.exp(rng.uniform(-1, 2))
        b2 = np.exp(rng.uniform(-3, 2))
        x0 = np.array([lam0, a0, a1, b1, b2])
        try:
            res = minimize(neg_ll, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints,
                           options={'ftol': 1e-10, 'maxiter': 2000})
            if res.fun < 1e11:
                ll = -res.fun
                if ll > best_ll:
                    best_ll = ll
                    best_params = res.x.copy()
        except Exception:
            pass

    return best_params, best_ll


# ---------------------------------------------------------------------------
# AICc
# ---------------------------------------------------------------------------

def aicc(log_lik: float, k: int, n: int) -> float:
    """
    Corrected AIC (AICc) for small samples.

    AIC  = -2*l + 2k
    AICc = AIC + 2k(k+1)/(n-k-1)
    """
    aic = -2.0 * log_lik + 2.0 * k
    return aic + 2.0 * k * (k + 1) / (n - k - 1)
