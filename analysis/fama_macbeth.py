"""
Fama-MacBeth Regressions with Newey-West standard errors.

Implements the two-step FM procedure:
1. Cross-sectional regressions each period
2. Time-series average of coefficients with NW-corrected t-stats
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config.settings import NW_LAGS, MIN_OBS_FM, LABELS


def fama_macbeth_nw(df, y_col, x_cols, date_col='week',
                     min_obs=None, nw_lags=None):
    """
    Fama-MacBeth regression with Newey-West corrected t-statistics.

    Parameters
    ----------
    df : pd.DataFrame — panel data
    y_col : str — dependent variable (e.g., 'ret_lead1')
    x_cols : list[str] — independent variables
    date_col : str — time period column
    min_obs : int — minimum obs per cross-section
    nw_lags : int — Newey-West lag truncation

    Returns
    -------
    pd.DataFrame with columns: variable, coef, nw_tstat, n_periods, avg_obs, coef_ts
    """
    if min_obs is None:
        min_obs = MIN_OBS_FM
    if nw_lags is None:
        nw_lags = NW_LAGS

    dates = df[date_col].unique()
    all_vars = [y_col] + x_cols
    coeffs = {v: [] for v in ['const'] + x_cols}
    n_obs_per_period = []

    for d in dates:
        sub = df.loc[df[date_col] == d, all_vars].dropna()
        if len(sub) < min_obs:
            continue
        y = sub[y_col].values
        X = sm.add_constant(sub[x_cols].values)
        try:
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            coeffs['const'].append(b[0])
            for i, v in enumerate(x_cols):
                coeffs[v].append(b[i + 1])
            n_obs_per_period.append(len(sub))
        except Exception:
            continue

    rows = []
    for v in ['const'] + x_cols:
        arr = np.array(coeffs[v])
        if len(arr) < 10:
            continue
        mean_c = arr.mean()
        T = len(arr)
        gamma0 = np.var(arr, ddof=1)
        nw_se_sq = gamma0
        for lag in range(1, nw_lags + 1):
            w = 1 - lag / (nw_lags + 1)
            gamma_l = np.sum((arr[lag:] - mean_c) * (arr[:-lag] - mean_c)) / (T - 1)
            nw_se_sq += 2 * w * gamma_l
        nw_se = np.sqrt(max(nw_se_sq, 1e-20) / T)
        t = mean_c / nw_se if nw_se > 0 else 0
        rows.append({
            'variable': v,
            'coef': mean_c * 100,  # ×100 for readability (% units)
            'nw_tstat': t,
            'n_periods': T,
            'avg_obs': np.mean(n_obs_per_period),
            'coef_ts': arr,
        })

    return pd.DataFrame(rows)


def fm_to_latex(fm_results, specs_to_show, vars_to_show, filename,
                table_dir=None):
    """Generate a LaTeX table from FM regression results."""
    from config.settings import TABLE_DIR
    import os
    if table_dir is None:
        table_dir = TABLE_DIR

    lines = [
        r'\begin{table}[htbp]',
        r'\centering',
        r'\caption{Fama-MacBeth Regressions: Weekly Returns}',
        r'\label{tab:fm}',
        r'\small',
    ]
    n_specs = len(specs_to_show)
    lines.append(r'\begin{tabular}{l' + 'c' * n_specs + '}')
    lines.append(r'\hline\hline')
    header = ' & '.join([''] + [f'({i+1})' for i in range(n_specs)])
    lines.append(header + r' \\')
    lines.append(r'\hline')

    for var in vars_to_show:
        coef_row = [LABELS.get(var, var)]
        tstat_row = ['']
        for spec in specs_to_show:
            row = fm_results[(fm_results['spec'] == spec) &
                              (fm_results['variable'] == var)]
            if row.empty:
                coef_row.append('')
                tstat_row.append('')
            else:
                c = row['coef'].iloc[0]
                t = row['nw_tstat'].iloc[0]
                sig = ('^{***}' if abs(t) > 2.576 else
                       '^{**}' if abs(t) > 1.96 else
                       '^{*}' if abs(t) > 1.645 else '')
                coef_row.append(f'${c:.4f}{sig}$')
                tstat_row.append(f'$({t:.2f})$')
        lines.append(' & '.join(coef_row) + r' \\')
        lines.append(' & '.join(tstat_row) + r' \\[3pt]')

    lines.append(r'\hline')
    nper_row = ['$T$ (weeks)']
    for spec in specs_to_show:
        row = fm_results[fm_results['spec'] == spec]
        if not row.empty:
            nper_row.append(f'{row["n_periods"].iloc[0]:,}')
        else:
            nper_row.append('')
    lines.append(' & '.join(nper_row) + r' \\')
    lines += [r'\hline\hline', r'\end{tabular}',
              r'\begin{tablenotes}\small',
              r'\item Newey-West corrected $t$-statistics (6 lags) in parentheses.',
              r'$^{***}$, $^{**}$, $^{*}$ denote significance at 1\%, 5\%, 10\%.',
              r'\end{tablenotes}', r'\end{table}']

    path = os.path.join(table_dir, f'{filename}.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  ✓ Saved {filename}.tex")
