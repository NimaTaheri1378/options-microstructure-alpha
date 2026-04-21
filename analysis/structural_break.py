"""
Structural Break Analysis — Chow tests and sub-period comparisons.

Tests whether signal coefficients are stable across sub-periods,
particularly around the 2013 structural break identified in
the O/S ratio signal.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from config.settings import NW_LAGS, BASE_CONTROLS


SUB_PERIODS = [
    ('1996–2005', '1996-01-01', '2005-12-31'),
    ('2006–2012', '2006-01-01', '2012-12-31'),
    ('2013–2019', '2013-01-01', '2019-12-31'),
    ('2020–2024', '2020-01-01', '2024-12-31'),
]


def chow_test(coef_ts, dates_used, break_year):
    """
    Test for structural break in FM coefficient time series.

    Parameters
    ----------
    coef_ts : np.array — per-period FM coefficients
    dates_used : list — period labels (matching coef_ts length)
    break_year : int — year of hypothesized break

    Returns
    -------
    dict with pre/post means, Welch t-stat, p-value, NW dummy coef
    """
    T_total = len(coef_ts)
    break_date = pd.Period(f'{break_year}-01-01', freq='W')

    break_idx = None
    for i, d in enumerate(dates_used):
        if d >= break_date:
            break_idx = i
            break

    if break_idx is None or break_idx < 10 or break_idx > T_total - 10:
        return None

    pre = coef_ts[:break_idx]
    post = coef_ts[break_idx:]

    # Welch's t-test
    t_chow, p_chow = stats.ttest_ind(pre, post, equal_var=False)

    # NW regression with dummy
    dummy = np.zeros(T_total)
    dummy[break_idx:] = 1
    X_chow = sm.add_constant(dummy)
    chow_ols = sm.OLS(coef_ts, X_chow).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})

    return {
        'break_year': break_year,
        'pre_mean': np.mean(pre) * 100,
        'post_mean': np.mean(post) * 100,
        'delta': (np.mean(post) - np.mean(pre)) * 100,
        'welch_t': t_chow,
        'welch_p': p_chow,
        'nw_dummy_coef': chow_ols.params[1] * 100,
        'nw_dummy_t': chow_ols.tvalues[1],
        'n_pre': len(pre),
        'n_post': len(post),
    }


def subperiod_analysis(panel, fm_func, portfolio_func, signal='os_ratio'):
    """
    Run FM regressions and portfolio sorts for each sub-period.

    Returns
    -------
    tuple(sub_fm_df, sub_hedge_ew_df, sub_hedge_vw_df)
    """
    sub_fm = []
    sub_hedge_ew = []
    sub_hedge_vw = []

    for name, s, e in SUB_PERIODS:
        sub = panel[(panel['date'] >= s) & (panel['date'] <= e)]
        if len(sub) < 1000:
            print(f"  {name}: skip ({len(sub)} obs)")
            continue
        print(f"\n  {name} ({len(sub):,} obs, {sub['permno'].nunique():,} stocks)")

        # Adaptive controls
        sub_controls = []
        for c in BASE_CONTROLS:
            cov = sub[c].notna().mean()
            if cov >= 0.50:
                sub_controls.append(c)
            else:
                print(f"    ⚠ Dropping {c} ({cov*100:.0f}% coverage)")

        # FM
        x_vars = [signal] + sub_controls
        sub_clean = sub.dropna(subset=['ret_lead1'] + x_vars)
        if len(sub_clean) < 500:
            continue

        res = fm_func(sub_clean, 'ret_lead1', x_vars)
        if res.empty:
            continue

        os_row = res[res['variable'] == signal]
        if not os_row.empty:
            c, t = os_row['coef'].iloc[0], os_row['nw_tstat'].iloc[0]
            print(f"    {signal} FM coef: {c:.4f} (t={t:.2f})")
            sub_fm.append({
                'period': name, 'coef': c, 'tstat': t,
                'n_obs': len(sub_clean),
                'coef_ts': os_row['coef_ts'].iloc[0]
            })

        # EW Hedge
        port_ew, _ = portfolio_func(sub, signal, label=f'{signal} [{name}] EW',
                                     weighting='ew')
        h = port_ew[port_ew['Q'] == 'Q1-Q5']
        if not h.empty:
            sub_hedge_ew.append({
                'period': name,
                'hedge_ret': h['mean_ret'].iloc[0],
                'hedge_t': h['t_stat'].iloc[0],
                'c4_alpha': h['c4_alpha'].iloc[0],
                'c4_t': h['c4_tstat'].iloc[0]
            })

        # VW Hedge
        port_vw, _ = portfolio_func(sub, signal, label=f'{signal} [{name}] VW',
                                     weighting='vw')
        h_vw = port_vw[port_vw['Q'] == 'Q1-Q5']
        if not h_vw.empty:
            sub_hedge_vw.append({
                'period': name,
                'hedge_ret': h_vw['mean_ret'].iloc[0],
                'hedge_t': h_vw['t_stat'].iloc[0],
                'c4_alpha': h_vw['c4_alpha'].iloc[0],
                'c4_t': h_vw['c4_tstat'].iloc[0]
            })

    return (pd.DataFrame(sub_fm),
            pd.DataFrame(sub_hedge_ew),
            pd.DataFrame(sub_hedge_vw))
