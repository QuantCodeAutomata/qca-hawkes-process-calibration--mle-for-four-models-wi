# Hawkes Process Calibration: MLE for Four Models with AICc Comparison

This repository implements three interrelated experiments on Hawkes processes,
Volterra integral equations, and equilibrium stochastic control in quantitative finance.

## Experiments

### Exp 1 — Hawkes Process Calibration (Table 1 & Figure 1)
Calibrate four Hawkes process models to the Wenchuan earthquake aftershock sequence
(2008–2023, n=37 events, Mw≥5.0) via MLE and compare with AICc.

**Models:**
| Model | Kernel | Params (k) |
|-------|--------|-----------|
| Only Power Kernel | ρ₁/(ρ₂+Δt)^p | 4 |
| Mean-Reverting & Power | Volterra integral | 6 |
| Only Exponential | b₁·exp(−b₂Δt) | 3 |
| Mean-Reverting & Exponential | Markovian ODE | 5 |

**Target (Table 1):**
- Model 1 (Only Power): log-lik=8.0386, AICc=**−6.8272** (best)
- Model 2 (MR & Power): log-lik=8.1208, AICc=−1.4416
- Model 3 (Only Exp): log-lik=4.5353, AICc=−2.3433
- Model 4 (MR & Exp): log-lik=6.7185, AICc=−1.5015

### Exp 2 — Equilibrium Investment Strategy (Figure 2)
Compute equilibrium portfolio allocation under the Volterra-Heston rough volatility
model (δ∈{0.50,0.60,0.70}) versus classical Heston (δ=1), for horizons T∈{1,10}.

Uses the fractional Adams predictor-corrector method (Diethelm et al. 2004)
to solve weakly-singular Volterra integral equations for H̄(τ) and B̄(τ).

### Exp 3 — Equilibrium Insurance Deductible (Figure 3)
Compute optimal insurance deductible d*(t) under Hawkes power-kernel intensity
versus vanilla Poisson, varying p∈{1.05,1.30,1.524} and ρ₂∈{0.004,0.35,1.20}
for horizons T∈{1,10}. Verifies Corollary 5.1 (horizon effect).

## Repository Structure
```
repo/
├── data/
│   ├── download_earthquake.py   # ISC catalog download & preprocessing
│   └── earthquake_events.csv    # Cached event catalog (generated on run)
├── src/
│   ├── hawkes_models.py          # 4 Hawkes model variants, MLE, AICc
│   ├── volterra_solver.py        # Fractional Adams solver, investment ratios
│   └── insurance_deductible.py  # Product-integration Volterra, deductible
├── exp/
│   ├── exp1_hawkes_calibration.py   # Exp 1: Table 1 + Figure 1
│   ├── exp2_investment_strategy.py  # Exp 2: Figure 2
│   └── exp3_insurance_deductible.py # Exp 3: Figure 3
├── tests/
│   ├── test_hawkes_models.py        # Unit tests for src/hawkes_models.py
│   ├── test_volterra_solver.py      # Unit tests for src/volterra_solver.py
│   └── test_insurance_deductible.py # Unit tests for src/insurance_deductible.py
├── results/
│   ├── RESULTS.md                   # Experiment summaries and findings
│   ├── exp1_table1.csv              # Table 1 reproduction
│   ├── figure1_hawkes_intensity.png # Figure 1
│   ├── figure2_investment_strategy.png # Figure 2
│   └── figure3_insurance_deductible.png # Figure 3
├── README.md
└── .gitignore
```

## Installation

```bash
pip install numpy scipy matplotlib pandas requests
```

## Usage

```bash
# Run all experiments
python exp/exp1_hawkes_calibration.py
python exp/exp2_investment_strategy.py
python exp/exp3_insurance_deductible.py

# Run tests
pytest tests/ -v
```

## Data

Earthquake catalog is downloaded from the International Seismological Centre (ISC)
bulletin. If the download fails (network unavailable), a synthetic fallback catalog
is generated via thinning simulation using calibrated power-kernel parameters.

**Parameters:**
- Center: Chengdu (30.5728°N, 104.0668°E), radius ≤ 300 km
- Magnitude: Mw ≥ 5.0
- Period: 2008-05-12 to 2023-12-31
- Events: n = 37

## Key Results

See [results/RESULTS.md](results/RESULTS.md) for detailed experiment outcomes.

## Mathematical Background

The **Only Power Kernel** Hawkes model achieves lowest AICc = −6.8272, confirming
that the modified Omori power-law decay (φ(t) = ρ₁/(ρ₂+t)^p) describes Wenchuan
aftershock clustering better than exponential decay, consistent with seismological
literature on the Omori-Utsu law.

For the **investment strategy** under rough volatility (δ < 1), smaller δ (rougher)
means more path memory, which amplifies the short-horizon investment relative to
classical Heston, but leads to more cautious long-horizon investment.

For the **insurance deductible**, path-dependent (Hawkes) models prescribe lower
deductibles (more insurance) than vanilla Poisson at every time point, reflecting
the clustering risk of earthquake losses.
