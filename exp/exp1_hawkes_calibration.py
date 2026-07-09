# exp/exp1_hawkes_calibration.py
"""Experiment 1: Hawkes Process Calibration - Table 1 and Figure 1."""
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
from datetime import datetime

from data.download_earthquake import load_or_download
from src.hawkes_models import (
    run_mle_model1, run_mle_model3, run_mle_model2, run_mle_model4,
    loglik_power_only, aicc, intensity_power_only
)


def run_experiment1(n_restarts: int = 25) -> dict:
    """
    Run full Experiment 1: calibrate 4 Hawkes models and compute AICc.

    Data source: ISC FDSN catalog (n=44 events, MW>=5.0, dist<=300 km).
    Paper reports n=37 events (ISC catalog version at time of writing).
    All four models are fitted and compared via AICc; the power kernel
    is expected to win (lowest AICc) consistent with the paper's finding.
    T_cal = end of observation window (2023-12-31 = 15.64 yr from mainshock)
    following the paper's instruction to use T_cal = (2023-12-31 - 2008-05-12)/365.25.
    """
    df = load_or_download()
    t_events = df['T_YEARS'].values
    n = len(t_events)
    # T_cal: paper instructs using end of window 2023-12-31
    mainshock_dt = datetime(2008, 5, 12, 6, 28, 0)  # actual mainshock UTC
    T_cal = (datetime(2023, 12, 31, 23, 59, 59) - mainshock_dt).total_seconds() / (365.25 * 86400)
    T_cal_last = t_events[-1]
    print(f'Data source: ISC FDSN | n = {n} events')
    print(f'T_cal_last = {T_cal_last:.4f} yr  T_cal_end = {T_cal:.4f} yr')
    print(f'Using T_cal = T_cal_end = {T_cal:.4f} yr (full observation window)')
    print('Note: paper uses n=37 (ISC catalog version at publication; current ISC = 44 events)')

    results = {}

    print('\nFitting Model 1: Only Power Kernel (k=4)...')
    p1, ll1 = run_mle_model1(t_events, T_cal, n_restarts=n_restarts)
    results['model1'] = {'params': p1, 'loglik': ll1, 'k': 4, 'T_cal': T_cal}
    print(f'  log-lik = {ll1:.4f}  params = {np.round(p1,6)}')

    print('\nFitting Model 3: Only Exponential Kernel (k=3)...')
    p3, ll3 = run_mle_model3(t_events, T_cal, n_restarts=n_restarts)
    results['model3'] = {'params': p3, 'loglik': ll3, 'k': 3, 'T_cal': T_cal}
    print(f'  log-lik = {ll3:.4f}  params = {np.round(p3,6)}')

    print('\nFitting Model 2: Mean-Reverting & Power Kernel (k=6)...')
    p2, ll2 = run_mle_model2(t_events, T_cal, n_restarts=n_restarts)
    results['model2'] = {'params': p2, 'loglik': ll2, 'k': 6, 'T_cal': T_cal}
    print(f'  log-lik = {ll2:.4f}  params = {np.round(p2,6) if p2 is not None else None}')

    print('\nFitting Model 4: Mean-Reverting & Exponential Kernel (k=5)...')
    p4, ll4 = run_mle_model4(t_events, T_cal, n_restarts=n_restarts)
    results['model4'] = {'params': p4, 'loglik': ll4, 'k': 5, 'T_cal': T_cal}
    print(f'  log-lik = {ll4:.4f}  params = {np.round(p4,6) if p4 is not None else None}')

    for key in ['model1', 'model2', 'model3', 'model4']:
        ll = results[key]['loglik']
        k = results[key]['k']
        results[key]['aicc'] = aicc(ll, k, n)

    return results, t_events, T_cal


def print_table1(results: dict) -> None:
    """Print Table 1: model comparison."""
    print('\n' + '='*80)
    print('TABLE 1: Hawkes Model Comparison')
    print('='*80)
    model_names = {
        'model1': 'Only Power Kernel      ',
        'model2': 'Mean-Rev. & Power      ',
        'model3': 'Only Exponential       ',
        'model4': 'Mean-Rev. & Exponential',
    }
    print(f'{"Model":<30} {"k":>4} {"log-lik":>10} {"AICc":>10}')
    print('-'*60)
    for key in ['model1', 'model2', 'model3', 'model4']:
        r = results[key]
        print(f'{model_names[key]} {r["k"]:>4} {r["loglik"]:>10.4f} {r["aicc"]:>10.4f}')
    print('='*80)
    print('Target: Model 1 AICc should be approximately -6.8272 (lowest)')


def generate_figure1(
    results: dict,
    t_events: np.ndarray,
    T_cal: float,
    out_dir: Path,
) -> None:
    """Generate Figure 1: calibrated intensity vs. calendar year."""
    # Use best-model params (Model 1)
    params = results['model1']['params']
    lam0, rho1, rho2, p = params
    print(f'\nFigure 1 params: lam0={lam0:.6f}, rho1={rho1:.6f}, rho2={rho2:.6f}, p={p:.6f}')

    # Fine grid
    n_grid = 3000
    t_grid = np.linspace(0, T_cal, n_grid)
    lam_grid = np.array([intensity_power_only(t, t_events, params) for t in t_grid])

    # Convert decimal years to calendar year
    t_ref_year = 2008.0 + 132 / 365.25  # 2008-05-12 = day 132
    cal_grid = t_ref_year + t_grid
    cal_events = t_ref_year + t_events
    lam_at_events = np.array([intensity_power_only(tj, t_events, params) for tj in t_events])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(cal_grid, lam_grid, 'b-', lw=1.5, label=r'Fitted intensity $\lambda_t$')
    ax.scatter(cal_events, lam_at_events, marker='x', color='red', s=80, zorder=5,
               label='Observed events')
    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel(r'Conditional intensity $\lambda_t$ (events/year)', fontsize=13)
    ax.set_title('Figure 1: Calibrated Hawkes Intensity (Only Power Kernel)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([2008, 2024])
    plt.tight_layout()
    out_path = out_dir / 'figure1_hawkes_intensity.png'
    plt.savefig(out_path, dpi=150)
    plt.savefig(out_dir / 'figure1_hawkes_intensity.pdf', dpi=150)
    plt.close()
    print(f'Figure 1 saved to {out_path}')


def save_results(results: dict, out_dir: Path) -> None:
    """Save experiment results to CSV."""
    rows = []
    model_names = {
        'model1': 'Only Power Kernel',
        'model2': 'Mean-Rev. & Power',
        'model3': 'Only Exponential',
        'model4': 'Mean-Rev. & Exponential',
    }
    for key, name in model_names.items():
        r = results[key]
        rows.append({
            'model': name,
            'k': r['k'],
            'log_lik': r['loglik'],
            'aicc': r['aicc'],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'exp1_table1.csv', index=False)
    print(f'Table 1 saved to {out_dir}/exp1_table1.csv')


if __name__ == '__main__':
    out_dir = Path(__file__).parent.parent / 'results'
    out_dir.mkdir(exist_ok=True)

    results, t_events, T_cal = run_experiment1(n_restarts=20)
    print_table1(results)
    generate_figure1(results, t_events, T_cal, out_dir)
    save_results(results, out_dir)

    best_key = min(results, key=lambda k: results[k]['aicc'])
    print(f'\nBest model by AICc: {best_key} (AICc = {results[best_key]["aicc"]:.4f})')
    if best_key == 'model1':
        print('CONFIRMED: Only Power Kernel is the best model (consistent with paper).')
    else:
        print(f'NOTE: Best model is {best_key}; paper finds model1 best on ISC n=37 dataset.')
