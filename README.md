# Bitcoin Regime-Switching VAR

Replication code for Chapter 5 of *Bitcoin in the Macroeconomic Landscape: Price Determinants, Regime Dynamics, and Volatility Modelling*.

## Background

This chapter estimates a two-regime Markov-Switching VAR(2) model to investigate how Bitcoin price dynamics differ across market regimes. The model is estimated via the Expectation-Maximisation (EM) algorithm using a Hamilton filter for forward inference and a Kim smoother for backward smoothing. Regime-specific Generalised Impulse Response Functions (Pesaran and Shin, 1998) reveal how shocks to Bitcoin returns, bond yields, and short-term interest rates propagate differently under high-volatility (crisis) and low-volatility (normal) market conditions. Parametric bootstrap (500 replications) provides 90% confidence intervals for the IRFs.

## Data

The raw CSV files are **not included** in this repository. Run `download_data.py` to
fetch them from public sources (requires an internet connection, ~1 min):

```bash
pip install yfinance pandas requests openpyxl
python download_data.py
```

This produces two files:

| File | Description |
|------|-------------|
| `msvar_data.csv` | Raw daily data (≈2,035 obs, 2015-01-02 to 2023-04-21). Columns: `libor`, `date`, `bond10`, `btc`, `gpr`, `ier`. |
| `msvar_processed.csv` | Processed model inputs (≈2,031 obs). Columns: `date`, `btc` (daily log-return), `d_bond10` (5-day overlapping yield change), `d_libor` (5-day overlapping rate change). |

### Data Sources

| Column | Description | Provider | Series / Ticker |
|--------|-------------|----------|-----------------|
| `btc` | Bitcoin closing price (USD) | Yahoo Finance | `BTC-USD` |
| `libor` | 3-Month LIBOR proxy | FRED | `DGS3MO` (3-Month Treasury constant maturity; USD LIBOR was discontinued June 2023) |
| `bond10` | 10-Year Treasury yield | FRED | `DGS10` |
| `gpr` | Geopolitical Risk Index (daily) | Caldara & Iacoviello (2022) | [matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm) |
| `ier` | 5-Year Breakeven Inflation Rate | FRED | `T5YIE` (Implied inflation Expectations Rate) |

### Pre-computed Bootstrap Results

The `.npz` files **are included** in the repository to save the 20–30 min bootstrap
computation:

| File | Description |
|------|-------------|
| `bootstrap_irf_results.npz` | Bootstrap IRF draws: `irf_point` (2, 9, 11) point estimates; `irf_boot` (500, 2, 9, 11) bootstrap draws. |
| `bootstrap_msvar_results.npz` | Bootstrap parameter draws: `variances` (500, 2, 3); `pi` (500, 2) transition diagonals; `durations` (500, 2). |

Note: The original thesis uses Bloomberg data (T=2,266, ending 2023-03-31). This replication uses publicly available FRED/alternative sources, which produces a slightly different sample.

## Repository Structure

```
.
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── download_data.py                 # Fetch raw data from public sources
├── replication_msvar.ipynb          # Main notebook: full replication pipeline
├── bootstrap_irf.py                 # Standalone script: bootstrap IRF with 90% CIs
├── extract_descriptive_stats.py     # Standalone script: descriptive statistics (LaTeX tables)
├── extract_m3_regimes.py            # Standalone script: M=2 vs M=3 regime comparison
├── bootstrap_irf_results.npz        # Pre-computed: bootstrap IRF draws (500 reps)
└── bootstrap_msvar_results.npz      # Pre-computed: bootstrap parameter draws (500 reps)
```

`msvar_data.csv` and `msvar_processed.csv` are excluded from version control (see `.gitignore`).
Run `python download_data.py` to regenerate them.

## How to Run

### Prerequisites

- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

### Using the notebook (recommended)

1. Open `replication_msvar.ipynb` in Jupyter.
2. Run all cells sequentially. The notebook covers:
   - Data loading and exploration
   - MSVAR model definition and estimation (20 restarts, ~2 min)
   - Coefficient tables, transition matrices, regime probabilities
   - Impulse response functions (point estimates)
   - Model diagnostics (LR test, residual tests, RCM)
   - Robustness checks (M=3 comparison, hyperparameter sensitivity)
   - Parametric bootstrap for standard errors (~10-20 min)

### Using standalone scripts

```bash
# Bootstrap IRFs with 90% confidence intervals (~20-30 min)
python bootstrap_irf.py

# Descriptive statistics (LaTeX and plain text)
python extract_descriptive_stats.py

# M=2 vs M=3 regime comparison
python extract_m3_regimes.py
```

### Using pre-computed results

The `.npz` files contain cached bootstrap results to avoid re-running the computationally intensive bootstrap procedure:

```python
import numpy as np

# Load bootstrap IRF results
data = np.load('bootstrap_irf_results.npz')
irf_point = data['irf_point']    # (2, 9, 11) point estimates
irf_boot = data['irf_boot']      # (500, 2, 9, 11) bootstrap draws

# Load bootstrap parameter results
params = np.load('bootstrap_msvar_results.npz')
boot_variances = params['variances']    # (500, 2, 3) conditional variances
boot_pi = params['pi']                  # (500, 2) diagonal transition probs
boot_durations = params['durations']    # (500, 2) expected durations
```

## Key Findings

- **Two distinct regimes identified**: A likelihood ratio test strongly rejects the single-regime null (LR = 5,710, df = 29), confirming the presence of regime switching in Bitcoin-bond-LIBOR dynamics.
- **Regime 1 (high-volatility)** captures crisis episodes (COVID-19, FTX collapse, SVB failure) with BTC variance approximately 2.8x that of the low-volatility regime.
- **Regime 2 (low-volatility)** represents normal market conditions and accounts for the majority of the sample.
- **Regime Classification Measure (RCM) = 0.90**, indicating strong separation between the two regimes.
- **Bootstrap confidence intervals** show that on-diagonal IRFs (own-shock responses) are statistically significant at the 90% level, while most cross-variable responses are not, suggesting limited spillover between Bitcoin and traditional fixed-income markets.
- **Robustness**: Results are qualitatively robust to regularisation hyperparameters (ridge lambda, IW prior nu) and the number of regimes (M=3 adds granularity but is penalised by BIC).

## Thesis Reference

Chapter 5: *Regime-Dependent Bitcoin Dynamics: A Markov-Switching VAR Analysis*
