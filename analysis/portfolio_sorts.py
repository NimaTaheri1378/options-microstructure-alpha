"""
Portfolio Sorts — Quintile portfolios with risk-adjusted alphas.

Implements EW and VW quintile sorts with CAPM, FF3, and Carhart alpha
computation. Exact logic from the original analysis.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config.settings import NW_LAGS, N_QUINTILES, LABELS


def portfolio_sort(df, signal, ret_col='ret_lead1', n_q=None,
                   label='Signal', weighting='ew', me_col='me',
                   ff_weekly=None):
    """
    Quintile portfolio sort with risk-adjusted alphas.

    Parameters
    ----------
    df : pd.DataFrame — panel with signal, returns, week, etc.
    signal : str — signal column name
    ret_col : str — return column
    n_q : int — number of quantiles
    weighting : str — 'ew' or 'vw'
    ff_weekly : pd.DataFrame — FF factors at weekly freq

    Returns
    -------
    tuple(pd.DataFrame results, pd.DataFrame portfolio time-series)
    """
    if n_q is None:
        n_q = N_QUINTILES

    sub = df.dropna(subset=[signal, ret_col]).copy()

    # Extract FF factors if not provided
    if ff_weekly is None:
        ff_weekly = sub[['week', 'mktrf', 'smb', 'hml', 'umd', 'rf']].drop_duplicates('week')

    # Assign quintiles
    def assign_quintile(x):
        n = len(x)
        if n < n_q:
            return pd.Series(np.nan, index=x.index)
        ranks = x.rank(method='first')
        return pd.cut(ranks, bins=n_q, labels=range(1, n_q + 1)).astype(float)

    sub['quintile'] = sub.groupby('week')[signal].transform(assign_quintile)
    sub = sub.dropna(subset=['quintile'])
    sub['quintile'] = sub['quintile'].astype(int)

    # Portfolio returns
    if weighting == 'vw':
        sub['me_pos'] = sub[me_col].clip(lower=0.01)
        sub['wt'] = sub.groupby(['week', 'quintile'])['me_pos'].transform(
            lambda x: x / x.sum())
        sub['wret'] = sub[ret_col] * sub['wt']
        port_ret = sub.groupby(['week', 'quintile'])['wret'].sum().reset_index()
        port_ret = port_ret.rename(columns={'wret': ret_col})
    else:
        port_ret = sub.groupby(['week', 'quintile'])[ret_col].mean().reset_index()

    port_ret = port_ret.pivot(index='week', columns='quintile', values=ret_col)
    port_ret = port_ret.reset_index().merge(
        ff_weekly[['week', 'mktrf', 'smb', 'hml', 'umd', 'rf']],
        on='week', how='left')
    port_ret = port_ret.dropna(subset=['mktrf'])

    # Compute stats for each quintile
    results = []
    for q in range(1, n_q + 1):
        r = port_ret[q].values
        mu = r.mean() * 100
        nw = sm.OLS(r, np.ones(len(r))).fit(
            cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        t_raw = nw.tvalues[0]

        y_ex = r - port_ret['rf'].values
        X_capm = sm.add_constant(port_ret[['mktrf']].values)
        a_capm = sm.OLS(y_ex, X_capm).fit(
            cov_type='HAC', cov_kwds={'maxlags': NW_LAGS}).params[0] * 100

        X_ff3 = sm.add_constant(port_ret[['mktrf', 'smb', 'hml']].values)
        ff3_fit = sm.OLS(y_ex, X_ff3).fit(
            cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        a_ff3 = ff3_fit.params[0] * 100
        t_ff3 = ff3_fit.tvalues[0]

        X_c4 = sm.add_constant(
            port_ret[['mktrf', 'smb', 'hml', 'umd']].fillna(0).values)
        c4_fit = sm.OLS(y_ex, X_c4).fit(
            cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
        a_c4 = c4_fit.params[0] * 100
        t_c4 = c4_fit.tvalues[0]

        results.append({
            'Q': q, 'mean_ret': mu, 't_stat': t_raw,
            'capm_alpha': a_capm, 'ff3_alpha': a_ff3, 'ff3_tstat': t_ff3,
            'c4_alpha': a_c4, 'c4_tstat': t_c4
        })

    # Hedge Q1-Q5
    hedge = port_ret[1].values - port_ret[n_q].values
    nw_h = sm.OLS(hedge, np.ones(len(hedge))).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    y_ex_h = hedge - port_ret['rf'].values
    X_ff3 = sm.add_constant(port_ret[['mktrf', 'smb', 'hml']].values)
    ff3_h = sm.OLS(y_ex_h, X_ff3).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    X_c4 = sm.add_constant(
        port_ret[['mktrf', 'smb', 'hml', 'umd']].fillna(0).values)
    c4_h = sm.OLS(y_ex_h, X_c4).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    X_capm_h = sm.add_constant(port_ret[['mktrf']].values)
    capm_h = sm.OLS(y_ex_h, X_capm_h).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})

    results.append({
        'Q': 'Q1-Q5', 'mean_ret': hedge.mean() * 100,
        't_stat': nw_h.tvalues[0],
        'capm_alpha': capm_h.params[0] * 100,
        'ff3_alpha': ff3_h.params[0] * 100, 'ff3_tstat': ff3_h.tvalues[0],
        'c4_alpha': c4_h.params[0] * 100, 'c4_tstat': c4_h.tvalues[0]
    })

    res_df = pd.DataFrame(results)
    wt_label = 'VW' if weighting == 'vw' else 'EW'
    print(f"\n  ── {label} ({wt_label}) ──")
    print(f"  {'Q':<6s} {'Mean%':>8s} {'t':>7s} {'CAPM α':>8s} "
          f"{'FF3 α':>8s} {'FF3 t':>7s} {'C4 α':>8s} {'C4 t':>7s}")
    for _, r in res_df.iterrows():
        q = str(r['Q'])
        print(f"  {q:<6s} {r['mean_ret']:>8.3f} {r['t_stat']:>7.2f} "
              f"{r['capm_alpha']:>8.3f} {r['ff3_alpha']:>8.3f} "
              f"{r['ff3_tstat']:>7.2f} {r['c4_alpha']:>8.3f} "
              f"{r['c4_tstat']:>7.2f}")

    return res_df, port_ret
