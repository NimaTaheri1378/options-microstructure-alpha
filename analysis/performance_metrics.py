"""
Performance Metrics — Sharpe, Sortino, drawdown, Calmar, etc.

Standalone module for computing strategy performance metrics,
separate from the backtest engine for reuse.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from config.settings import NW_LAGS, TC_BPS_ONE_WAY


def sharpe_ratio(returns, freq=52):
    """Annualized Sharpe ratio."""
    mu = returns.mean() * freq
    vol = returns.std() * np.sqrt(freq)
    return mu / vol if vol > 0 else 0


def sortino_ratio(returns, freq=52):
    """Annualized Sortino ratio (downside deviation)."""
    mu = returns.mean() * freq
    downside = returns[returns < 0].std() * np.sqrt(freq)
    return mu / downside if downside > 0 else 0


def max_drawdown(returns):
    """Maximum drawdown from a return series."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return dd.min()


def calmar_ratio(returns, freq=52):
    """Calmar ratio = annualized return / |max drawdown|."""
    ann_ret = returns.mean() * freq
    mdd = abs(max_drawdown(returns))
    return ann_ret / mdd if mdd > 0 else 0


def information_ratio(returns, benchmark, freq=52):
    """Information ratio vs benchmark."""
    excess = returns - benchmark
    mu = excess.mean() * freq
    te = excess.std() * np.sqrt(freq)
    return mu / te if te > 0 else 0


def hit_rate(returns):
    """Fraction of positive-return periods."""
    return (returns > 0).mean()


def turnover_adjusted_return(returns, turnover_pct=0.25, freq=52):
    """Annualized return after transaction costs."""
    tc_per_period = TC_BPS_ONE_WAY / 10000
    cost = 2 * tc_per_period * turnover_pct  # round-trip × turnover
    net = returns - cost
    return net.mean() * freq


def full_tearsheet(returns, rf=None, factors=None, freq=52):
    """
    Compute complete performance summary.

    Returns dict with all metrics.
    """
    r = returns.dropna()
    if len(r) < 10:
        return {}

    metrics = {
        'ann_return': r.mean() * freq * 100,
        'ann_volatility': r.std() * np.sqrt(freq) * 100,
        'sharpe': sharpe_ratio(r, freq),
        'sortino': sortino_ratio(r, freq),
        'max_drawdown': max_drawdown(r) * 100,
        'calmar': calmar_ratio(r, freq),
        'win_rate': hit_rate(r) * 100,
        'n_periods': len(r),
        'skewness': r.skew(),
        'kurtosis': r.kurtosis(),
        'ann_return_after_tc': turnover_adjusted_return(r) * 100,
    }

    # NW t-stat
    nw = sm.OLS(r.values, np.ones(len(r))).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    metrics['t_stat'] = nw.tvalues[0]

    # Risk-adjusted alphas if factors provided
    if factors is not None and rf is not None:
        common = r.index.intersection(factors.index).intersection(rf.index)
        if len(common) > 30:
            y = r.loc[common] - rf.loc[common]
            f = factors.loc[common]

            for model, cols in [
                ('capm', ['mktrf']),
                ('ff3', ['mktrf', 'smb', 'hml']),
                ('c4', ['mktrf', 'smb', 'hml', 'umd']),
            ]:
                X = sm.add_constant(f[cols].fillna(0).values)
                fit = sm.OLS(y.values, X).fit(
                    cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
                metrics[f'{model}_alpha'] = fit.params[0] * freq * 100
                metrics[f'{model}_tstat'] = fit.tvalues[0]

    return metrics


def format_tearsheet(metrics, name='Strategy'):
    """Pretty-print a performance tearsheet."""
    m = metrics
    lines = [
        f"┌{'─'*55}┐",
        f"│  {name:<51s}│",
        f"├{'─'*55}┤",
        f"│  Ann. Return:      {m.get('ann_return',0):>8.2f}%{' '*33}│",
        f"│  Ann. Volatility:  {m.get('ann_volatility',0):>8.2f}%{' '*33}│",
        f"│  Sharpe:           {m.get('sharpe',0):>8.3f}{' '*34}│",
        f"│  Sortino:          {m.get('sortino',0):>8.3f}{' '*34}│",
        f"│  Max Drawdown:     {m.get('max_drawdown',0):>8.2f}%{' '*33}│",
        f"│  Calmar:           {m.get('calmar',0):>8.3f}{' '*34}│",
        f"│  Win Rate:         {m.get('win_rate',0):>8.1f}%{' '*33}│",
        f"│  t-stat:           {m.get('t_stat',0):>8.2f}{' '*34}│",
    ]
    if 'ff3_alpha' in m:
        lines.append(
            f"│  FF3 Alpha:        {m['ff3_alpha']:>8.2f}% "
            f"(t={m.get('ff3_tstat',0):.2f}){' '*18}│")
    if 'c4_alpha' in m:
        lines.append(
            f"│  Carhart Alpha:    {m['c4_alpha']:>8.2f}% "
            f"(t={m.get('c4_tstat',0):.2f}){' '*18}│")
    lines.append(f"└{'─'*55}┘")
    return '\n'.join(lines)
