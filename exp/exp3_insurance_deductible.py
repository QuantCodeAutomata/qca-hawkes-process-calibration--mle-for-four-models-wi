# exp/exp3_insurance_deductible.py
"""Experiment 3: Equilibrium Insurance Deductible - Figure 3."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.insurance_deductible import solve_Mbar, pd_deductible, vanilla_deductible

# Fixed parameters from Tables 1 & 2
GAMMA_RISK = 1.0
UPSILON = 0.020
THETA_TILDE = 0.200
MU = 4.0
RHO1_CALIB = 0.020010
LAM0_CALIB = 1.431227

P_VALUES = [1.05, 1.30, 1.524052]
RHO2_VALUES = [0.004, 0.35, 1.20]
T_VALUES = [1, 10]


def check_subcriticality() -> None:
    """Check subcriticality for all (p, rho2) combinations."""
    print('Subcriticality check:')
    for p in P_VALUES:
        for rho2 in RHO2_VALUES:
            norm = RHO1_CALIB / ((p - 1) * rho2**(p - 1))
            flag = '✓' if norm < 1 else '✗ SUPERCRITICAL'
            print(f'  p={p:.4f}, rho2={rho2}: ||phi||_L1 = {norm:.4f} {flag}')


def run_experiment3() -> dict:
    """Run all 18 (p, rho2, T) combinations."""
    check_subcriticality()
    all_results = {}

    for T in T_VALUES:
        N = 2000 if T == 1 else 5000
        tau_grid = np.linspace(0, T, N + 1)
        all_results[T] = {
            'vanilla': {
                'tau': tau_grid,
                'deductible': vanilla_deductible(tau_grid, GAMMA_RISK, UPSILON, THETA_TILDE),
            }
        }

        for rho2 in RHO2_VALUES:
            all_results[T][rho2] = {}
            for p in P_VALUES:
                print(f'  Solving M-bar: T={T}, p={p:.4f}, rho2={rho2}...')
                # Check subcriticality first
                norm = RHO1_CALIB / ((p - 1) * rho2**(p - 1))
                if norm >= 1:
                    print(f'    SKIP: supercritical (norm={norm:.4f})')
                    all_results[T][rho2][p] = None
                    continue

                tau, Mbar = solve_Mbar(
                    T, N, p, rho2, RHO1_CALIB,
                    GAMMA_RISK, UPSILON, THETA_TILDE, MU
                )
                ded = pd_deductible(tau, Mbar, GAMMA_RISK, UPSILON, THETA_TILDE)
                all_results[T][rho2][p] = {
                    'tau': tau,
                    'Mbar': Mbar,
                    'deductible': ded,
                }
                # Verify Mbar <= 0
                if not np.all(Mbar <= 1e-6):
                    print(f'    WARNING: M-bar > 0 detected for p={p}, rho2={rho2}, T={T}')
                # Verify boundary condition: d*(tau=0) = 0.20
                assert abs(ded[0] - THETA_TILDE) < 1e-6, \
                    f'BC failed: p={p}, rho2={rho2}, T={T}, ded[0]={ded[0]}'

    return all_results


def generate_figure3(all_results: dict, out_dir: Path) -> None:
    """Generate Figure 3: 3x2 grid of deductible plots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    color_map = {1.05: 'tomato', 1.30: 'crimson', 1.524052: 'darkred'}
    label_map = {1.05: 'p=1.05', 1.30: 'p=1.30', 1.524052: 'p=1.5241'}
    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    panel_idx = 0

    for row_idx, rho2 in enumerate(RHO2_VALUES):
        for col_idx, T in enumerate(T_VALUES):
            ax = axes[row_idx, col_idx]

            # Vanilla
            tau_v = all_results[T]['vanilla']['tau']
            t_v = T - tau_v
            ax.plot(t_v, all_results[T]['vanilla']['deductible'],
                    'b:', lw=2.5, label='Vanilla')

            for p in P_VALUES:
                res = all_results[T][rho2].get(p)
                if res is None:
                    continue
                tau_p = res['tau']
                t_p = T - tau_p
                ded_p = res['deductible']
                ax.plot(t_p, ded_p, color=color_map[p], lw=1.8, label=label_map[p])

            ax.set_xlabel('$t$ (years)', fontsize=11)
            ax.set_ylabel('Deductible $d^*(t)$', fontsize=11)
            ax.set_title(f'{panel_labels[panel_idx]} $\\rho_2={rho2}$, $T={T}$', fontsize=12)
            ax.legend(fontsize=9)
            ax.set_xlim([0, T])
            panel_idx += 1

    plt.suptitle('Figure 3: Equilibrium Deductible — Path-Dependent vs. Vanilla', fontsize=14)
    plt.tight_layout()
    out_path = out_dir / 'figure3_insurance_deductible.png'
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / 'figure3_insurance_deductible.pdf', dpi=150)
    plt.close()
    print(f'Figure 3 saved to {out_path}')


def validate_corollary51(all_results: dict) -> None:
    """Validate Corollary 5.1 findings."""
    print('\nValidating Corollary 5.1:')

    def check_ordering_reversal(T, rho2):
        """Check if ordering of p-curves reverses within (0, T).

        Compares the ordering of p=1.05 vs p=1.5241 deductibles at
        t=0 (tau=T, last index) versus t=T/2 (tau=T/2, middle index).
        At t=T (tau=0) all curves coincide at theta_tilde/gamma, so that
        endpoint is excluded from the comparison.
        """
        deds = {}
        for p in P_VALUES:
            r = all_results[T][rho2].get(p)
            if r is not None:
                deds[p] = r['deductible']
        if len(deds) < 2:
            return False
        p1, p2 = P_VALUES[0], P_VALUES[-1]   # p=1.05 vs p=1.524
        N = len(deds[p1]) - 1
        # Scan 10 equally-spaced interior points and check if ordering flips
        # t=0 ↔ tau=T ↔ index N (last element); t=T ↔ tau=0 ↔ index 0
        order_at_t0 = deds[p1][N] > deds[p2][N]
        for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            # tau = T * (1 - frac) → index ≈ N * (1 - frac)
            idx = int(N * (1.0 - frac))
            order_here = deds[p1][idx] > deds[p2][idx]
            if order_here != order_at_t0:
                return True
        return False

    for rho2, T in [(0.004, 1), (0.004, 10), (0.35, 1), (0.35, 10), (1.20, 1), (1.20, 10)]:
        if all_results[T][rho2].get(P_VALUES[0]) is None:
            print(f'  rho2={rho2}, T={T}: SKIP (supercritical)')
            continue
        reversal = check_ordering_reversal(T, rho2)
        expected_reversal = (rho2 == 0.35 and T == 10)
        status = '✓' if reversal == expected_reversal else '(possible deviation)'
        print(f'  rho2={rho2}, T={T}: reversal={reversal}, expected={expected_reversal} {status}')


def save_results(all_results: dict, out_dir: Path) -> None:
    """Save summary results."""
    rows = []
    for T in T_VALUES:
        for rho2 in RHO2_VALUES:
            for p in P_VALUES:
                res = all_results[T][rho2].get(p)
                if res is None:
                    continue
                norm = RHO1_CALIB / ((p - 1) * rho2**(p - 1))
                rows.append({
                    'T': T, 'rho2': rho2, 'p': p,
                    'norm_phi': norm,
                    'Mbar_min': res['Mbar'].min(),
                    'Mbar_max': res['Mbar'].max(),
                    'ded_at_t0': res['deductible'][-1],
                    'ded_at_tT': res['deductible'][0],
                    'vanilla_at_t0': all_results[T]['vanilla']['deductible'][-1],
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'exp3_deductibles.csv', index=False)
    print(f'Results saved to {out_dir}/exp3_deductibles.csv')


if __name__ == '__main__':
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)

    print('Running Experiment 3: Insurance Deductible...')
    all_results = run_experiment3()
    validate_corollary51(all_results)
    generate_figure3(all_results, out_dir)
    save_results(all_results, out_dir)
    print('Experiment 3 complete.')
