#!/usr/bin/env python3
"""Extract descriptive statistics for MS-VAR variables.

Computes mean, std, skew, kurtosis, min, max, and correlation matrix
for the three MS-VAR variables: r_BTC, delta_y10, delta_i3.
Outputs both LaTeX and plain-text table formats.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


def main():
    # Load processed MS-VAR data
    df = pd.read_csv('msvar_processed.csv', parse_dates=['date'])

    print(f'Data range: {df.date.min()} to {df.date.max()}')
    print(f'Total observations: {len(df)}\n')

    var_names = ['btc', 'd_bond10', 'd_libor']
    var_labels = [r'$r_{\text{BTC}}$', r'$\Delta y_{10}$', r'$\Delta i_3$']
    data = df[var_names].values

    # Compute descriptive statistics
    stats = {
        'Obs': len(df),
        'Mean': np.mean(data, axis=0),
        'Std Dev': np.std(data, axis=0, ddof=1),
        'Skewness': skew(data, axis=0),
        'Kurtosis': kurtosis(data, axis=0, fisher=False),  # Pearson (normal=3)
        'Min': np.min(data, axis=0),
        'Max': np.max(data, axis=0),
    }

    corr_matrix = np.corrcoef(data, rowvar=False)

    # --- LaTeX output ---
    print('=' * 80)
    print('Table: Descriptive Statistics of MS-VAR Variables')
    print('=' * 80)
    print()

    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Descriptive Statistics of MS-VAR Variables}')
    print(r'\label{tab:msvar_descriptive}')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r'Statistic & $r_{\text{BTC}}$ & $\Delta y_{10}$ & $\Delta i_3$ \\')
    print(r'\midrule')

    for stat_name, values in stats.items():
        if stat_name == 'Obs':
            print(f'{stat_name:<15s} & {values:>12d} & {values:>12d} & {values:>12d} \\\\')
        else:
            print(f'{stat_name:<15s} & {values[0]:>12.4f} & {values[1]:>12.4f} & {values[2]:>12.4f} \\\\')

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print()

    # --- LaTeX correlation matrix ---
    print('=' * 80)
    print('Table: Unconditional Correlation Matrix')
    print('=' * 80)
    print()

    print(r'\begin{table}[htbp]')
    print(r'\centering')
    print(r'\caption{Unconditional Correlation Matrix}')
    print(r'\label{tab:msvar_correlation}')
    print(r'\begin{tabular}{lccc}')
    print(r'\toprule')
    print(r' & $r_{\text{BTC}}$ & $\Delta y_{10}$ & $\Delta i_3$ \\')
    print(r'\midrule')

    for i, label in enumerate(var_labels):
        row = f'{label:<20s}'
        for j in range(3):
            row += f' & {corr_matrix[i, j]:>8.4f}'
        row += r' \\'
        print(row)

    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print()

    # --- Plain-text output ---
    print('=' * 80)
    print('PLAIN TEXT VERSION')
    print('=' * 80)
    print()

    print('Descriptive Statistics:')
    print('-' * 80)
    print(f'{"Statistic":<15s} {"r_BTC":>12s} {"Delta_y10":>12s} {"Delta_i3":>12s}')
    print('-' * 80)
    for stat_name, values in stats.items():
        if stat_name == 'Obs':
            print(f'{stat_name:<15s} {values:>12d} {values:>12d} {values:>12d}')
        else:
            print(f'{stat_name:<15s} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f}')
    print()

    print('Correlation Matrix:')
    print('-' * 60)
    print(f'{"":>15s} {"r_BTC":>12s} {"Delta_y10":>12s} {"Delta_i3":>12s}')
    print('-' * 60)
    for i, name in enumerate(['r_BTC', 'Delta_y10', 'Delta_i3']):
        row = f'{name:<15s}'
        for j in range(3):
            row += f' {corr_matrix[i, j]:>12.4f}'
        print(row)
    print()

    # --- Notes ---
    print('=' * 80)
    print('NOTES')
    print('=' * 80)
    print(f'Sample period: {df.date.min().strftime("%Y-%m-%d")} to {df.date.max().strftime("%Y-%m-%d")}')
    print(f'Total observations: {len(df)}')
    print('r_BTC: Daily log returns of Bitcoin')
    print('Delta_y10: 5-day change in 10-year Treasury yield (overlapping)')
    print('Delta_i3: 5-day change in 3-month LIBOR (overlapping)')
    print()
    print('Kurtosis reported is Pearson kurtosis (normal = 3.0)')
    print('Excess kurtosis (normal = 0.0) would subtract 3 from these values')


if __name__ == '__main__':
    main()
