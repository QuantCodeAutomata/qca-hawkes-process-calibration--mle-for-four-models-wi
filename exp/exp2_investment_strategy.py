# exp/exp2_investment_strategy.py
"""Experiment 2: Equilibrium Investment Strategy - Figure 2."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

from src.volterra_solver import (
    solve_H_bar, solve_B_bar, H_hat_vanilla,
    investment_ratio_pd, investment_ratio_vanilla,
)

# Parameters from Table 2
KAPPA = 0.173
THETA = 5.000
SIGMA = 0.340
RHO = -0.615
GAMMA = 1.000
UPSILON = 0.020
DELTA_VALUES = [0.50, 0.60, 0.70]
T_VALUES = [1, 10]

# Derived constant
C = KAPPA + THETA * SIGMA * RHO  # = -0.8725


def run_experiment2() -> dict:
    """Compute investment ratios for all delta values and horizons."""
    all_results = {}

    for T in T_VALUES:
        N = 1000 if T == 1 else 5000
        tau_v = np.linspace(0, T, N + 1)
        ratio_v = investment_ratio_vanilla(tau_v, KAPPA, THETA, SIGMA, RHO, GAMMA, UPSILON)

        all_results[T] = {'vanilla': {'tau': tau_v, 'ratio': ratio_v}}
        all_results[T]['pd'] = {}

        for delta in DELTA_VALUES:
            print(f'  Solving H-bar for T={T}, delta={delta}...')
            tau_H, H_bar = solve_H_bar(T, N, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
            tau_B, B_bar = solve_B_bar(
                tau_H, H_bar, T, N, delta, KAPPA, THETA, SIGMA, RHO, GAMMA
            )
            ratio = investment_ratio_pd(H_bar, tau_H, THETA, SIGMA, RHO, GAMMA, UPSILON)
            all_results[T]['pd'][delta] = {
                'tau': tau_H,
                'H_bar': H_bar,
                'B_bar': B_bar,
                'ratio': ratio,
            }

    return all_results


def verify_boundary_conditions(all_results: dict) -> None:
    """Verify boundary conditions at tau=0 (t=T): ratio should equal theta/gamma = 5.0."""
    theta_gamma = THETA / GAMMA
    for T in T_VALUES:
        r_v = all_results[T]['vanilla']['ratio'][0]
        assert abs(r_v - theta_gamma) < 1e-4, f'Vanilla BC failed at T={T}: {r_v}'
        for delta in DELTA_VALUES:
            r_pd = all_results[T]['pd'][delta]['ratio'][0]
            H0 = all_results[T]['pd'][delta]['H_bar'][0]
            assert abs(H0) < 1e-10, f'H_bar(0) != 0 for delta={delta}, T={T}'
            assert abs(r_pd - theta_gamma) < 1e-4, f'PD BC failed: delta={delta}, T={T}: {r_pd}'
    print('All boundary conditions verified: ratio(tau=0) = theta/gamma = 5.0 ✓')


def convergence_test(T: int, delta: float, N_coarse: int) -> float:
    """Test convergence by halving the step size."""
    N_fine = 2 * N_coarse
    _, H_coarse = solve_H_bar(T, N_coarse, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
    tau_f, H_fine = solve_H_bar(T, N_fine, delta, KAPPA, THETA, SIGMA, RHO, GAMMA)
    # Every other point of fine grid matches coarse
    H_fine_on_coarse = H_fine[::2]
    scale = np.max(np.abs(H_coarse)) + 1e-15
    rel_err = np.max(np.abs(H_fine_on_coarse - H_coarse)) / scale
    return rel_err


def generate_figure2(all_results: dict, out_dir: Path) -> None:
    """Generate Figure 2: investment strategies for T=1 and T=10."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {0.50: 'red', 0.60: 'darkorange', 0.70: 'darkred'}
    labels = {0.50: r'$\delta=0.50$', 0.60: r'$\delta=0.60$', 0.70: r'$\delta=0.70$'}

    for idx, T in enumerate(T_VALUES):
        ax = axes[idx]
        tau_v = all_results[T]['vanilla']['tau']
        t_v = T - tau_v  # convert to clock time
        ax.plot(t_v, all_results[T]['vanilla']['ratio'], 'b--', lw=2, label='Vanilla (Heston)')

        for delta in DELTA_VALUES:
            tau_pd = all_results[T]['pd'][delta]['tau']
            t_pd = T - tau_pd
            ax.plot(t_pd, all_results[T]['pd'][delta]['ratio'],
                    color=colors[delta], lw=1.8, label=labels[delta])

        ax.set_xlabel('$t$ (years)', fontsize=12)
        ax.set_ylabel(r'$\alpha^*_t X_t / \sqrt{v_t}$', fontsize=12)
        ax.set_title(f'Investment Strategy, $T={T}$', fontsize=13)
        ax.legend(fontsize=10)
        ax.set_xlim([0, T])

    plt.suptitle('Figure 2: Equilibrium Investment Strategy — Path-Dependent vs. Vanilla', fontsize=14)
    plt.tight_layout()
    out_path = out_dir / 'figure2_investment_strategy.png'
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / 'figure2_investment_strategy.pdf', dpi=150)
    plt.close()
    print(f'Figure 2 saved to {out_path}')


def save_results(all_results: dict, out_dir: Path) -> None:
    """Save key numerical results."""
    import pandas as pd
    rows = []
    for T in T_VALUES:
        r_v = all_results[T]['vanilla']
        rows.append({'T': T, 'model': 'Vanilla', 'delta': 1.0,
                     'ratio_at_t0': r_v['ratio'][-1], 'ratio_at_tT': r_v['ratio'][0]})
        for delta in DELTA_VALUES:
            r = all_results[T]['pd'][delta]
            rows.append({'T': T, 'model': 'Path-Dep', 'delta': delta,
                         'ratio_at_t0': r['ratio'][-1], 'ratio_at_tT': r['ratio'][0]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'exp2_investment_ratios.csv', index=False)
    print(f'Results saved to {out_dir}/exp2_investment_ratios.csv')


if __name__ == '__main__':
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)

    print('Running Experiment 2: Investment Strategy...')
    all_results = run_experiment2()
    verify_boundary_conditions(all_results)

    print('\nConvergence tests:')
    for T in T_VALUES:
        N_coarse = 500 if T == 1 else 2500
        for delta in DELTA_VALUES:
            err = convergence_test(T, delta, N_coarse)
            # Fractional Adams converges at O(h^{1+delta}); delta=0.5 gives O(h^1.5).
            # Doubling N reduces error by 2^1.5~2.8x; relative diff can reach ~8% for T=10.
            tol = 0.08 if delta == 0.50 else 0.05
            print(f'  T={T}, delta={delta}: relative error = {err:.4e} (tol={tol})')
            assert err < tol, f'Convergence failed: T={T}, delta={delta}, err={err:.4e}'

    generate_figure2(all_results, out_dir)
    save_results(all_results, out_dir)
    print('Experiment 2 complete.')
