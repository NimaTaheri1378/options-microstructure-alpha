"""
Backtest engine and abstract strategy class.

Provides portfolio construction, performance measurement, and
risk-adjusted alpha computation for all strategies.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from abc import ABC, abstractmethod
from config.settings import (
    N_QUINTILES, NW_LAGS, TC_BPS_ONE_WAY, TC_BPS_ROUND, LABELS
)


###############################################################################
# PERFORMANCE METRICS
###############################################################################

def compute_performance(returns: pd.Series, rf: pd.Series = None,
                        freq: int = 52) -> dict:
    """
    Compute full performance metrics for a return series.

    Parameters
    ----------
    returns : pd.Series — strategy returns (gross)
    rf : pd.Series — risk-free rate (same frequency), optional
    freq : int — annualization factor (52 for weekly)
    """
    r = returns.dropna()
    if len(r) < 10:
        return {}

    mu = r.mean()
    vol = r.std()
    ann_ret = mu * freq
    ann_vol = vol * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino (downside deviation)
    downside = r[r < 0].std()
    sortino = (ann_ret) / (downside * np.sqrt(freq)) if downside > 0 else 0

    # Drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (r > 0).mean()

    # NW t-stat for mean != 0
    nw = sm.OLS(r.values, np.ones(len(r))).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    t_stat = nw.tvalues[0]

    # After transaction costs
    tc_per_period = TC_BPS_ONE_WAY / 10000  # one-way
    ann_ret_tc = ann_ret - 2 * tc_per_period * freq * 0.25  # assume 25% turnover

    return {
        'ann_return': ann_ret * 100,
        'ann_volatility': ann_vol * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd * 100,
        'calmar': calmar,
        'win_rate': win_rate * 100,
        't_stat': t_stat,
        'n_periods': len(r),
        'ann_return_after_tc': ann_ret_tc * 100,
    }


def compute_alphas(returns: pd.Series, factors: pd.DataFrame) -> dict:
    """
    Compute CAPM, FF3, and Carhart alphas.

    Parameters
    ----------
    returns : pd.Series — excess returns
    factors : pd.DataFrame — must contain mktrf, smb, hml, umd
    """
    r = returns.dropna()
    f = factors.loc[r.index].dropna()
    common = r.index.intersection(f.index)
    r = r.loc[common]
    f = f.loc[common]
    if len(r) < 30:
        return {}

    results = {}

    # CAPM
    X = sm.add_constant(f[['mktrf']].values)
    fit = sm.OLS(r.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    results['capm_alpha'] = fit.params[0] * 52 * 100  # annualized %
    results['capm_tstat'] = fit.tvalues[0]

    # FF3
    X = sm.add_constant(f[['mktrf', 'smb', 'hml']].values)
    fit = sm.OLS(r.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    results['ff3_alpha'] = fit.params[0] * 52 * 100
    results['ff3_tstat'] = fit.tvalues[0]

    # Carhart
    cols = ['mktrf', 'smb', 'hml', 'umd']
    X = sm.add_constant(f[cols].fillna(0).values)
    fit = sm.OLS(r.values, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    results['c4_alpha'] = fit.params[0] * 52 * 100
    results['c4_tstat'] = fit.tvalues[0]

    return results


###############################################################################
# ABSTRACT STRATEGY
###############################################################################

class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.results = {}

    @abstractmethod
    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Assign portfolio weights to each stock-week.

        Returns DataFrame with columns: permno, week, weight
        (positive = long, negative = short).
        """
        raise NotImplementedError

    def backtest(self, panel: pd.DataFrame, ff_weekly: pd.DataFrame) -> dict:
        """
        Run full backtest: positions → returns → performance → alphas.
        """
        positions = self.generate_positions(panel)
        if positions.empty:
            return {}

        # Merge with forward returns
        merged = positions.merge(
            panel[['permno', 'week', 'ret_lead1', 'me']],
            on=['permno', 'week'], how='inner'
        )
        merged = merged.dropna(subset=['ret_lead1', 'weight'])

        # Portfolio return per week
        port_ret = merged.groupby('week').apply(
            lambda g: (g['weight'] * g['ret_lead1']).sum()
        ).rename('port_ret')

        # Turnover
        turnover = self._compute_turnover(positions)

        # Performance
        perf = compute_performance(port_ret)
        perf['avg_turnover'] = turnover

        # Alphas (merge with FF factors)
        port_ret_df = port_ret.reset_index()
        port_ret_df = port_ret_df.merge(ff_weekly, on='week', how='left')
        port_ret_df = port_ret_df.set_index('week')
        ex_ret = port_ret_df['port_ret'] - port_ret_df['rf'].fillna(0)
        alphas = compute_alphas(ex_ret, port_ret_df[['mktrf', 'smb', 'hml', 'umd']])
        perf.update(alphas)

        # Store
        self.results = perf
        self.returns = port_ret
        return perf

    def _compute_turnover(self, positions: pd.DataFrame) -> float:
        """Average absolute weight change per period."""
        pos = positions.sort_values(['permno', 'week'])
        pos['prev_weight'] = pos.groupby('permno')['weight'].shift(1)
        pos['weight_change'] = (pos['weight'] - pos['prev_weight'].fillna(0)).abs()
        turnover = pos.groupby('week')['weight_change'].sum().mean()
        return turnover

    def tearsheet(self) -> str:
        """Format results as a text tearsheet."""
        r = self.results
        if not r:
            return f"  {self.name}: No results"
        lines = [
            f"┌{'─'*50}┐",
            f"│  STRATEGY: {self.name:<37s}│",
            f"├{'─'*50}┤",
            f"│  Ann. Return:     {r.get('ann_return', 0):>8.2f}%{' '*29}│",
            f"│  Ann. Volatility: {r.get('ann_volatility', 0):>8.2f}%{' '*29}│",
            f"│  Sharpe Ratio:    {r.get('sharpe', 0):>8.2f}{' '*30}│",
            f"│  Sortino Ratio:   {r.get('sortino', 0):>8.2f}{' '*30}│",
            f"│  Max Drawdown:    {r.get('max_drawdown', 0):>8.2f}%{' '*29}│",
            f"│  Calmar Ratio:    {r.get('calmar', 0):>8.2f}{' '*30}│",
            f"│  Win Rate:        {r.get('win_rate', 0):>8.1f}%{' '*29}│",
            f"│  t-stat (≠0):     {r.get('t_stat', 0):>8.2f}{' '*30}│",
            f"│  FF3 Alpha:       {r.get('ff3_alpha', 0):>8.2f}% (t={r.get('ff3_tstat', 0):.2f}){' '*14}│",
            f"│  Carhart Alpha:   {r.get('c4_alpha', 0):>8.2f}% (t={r.get('c4_tstat', 0):.2f}){' '*14}│",
            f"│  After TC (10bps):{r.get('ann_return_after_tc', 0):>8.2f}%{' '*29}│",
            f"└{'─'*50}┘",
        ]
        return '\n'.join(lines)

    def __repr__(self):
        return f"Strategy({self.name})"
