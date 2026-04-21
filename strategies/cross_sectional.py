"""
Cross-Sectional Momentum Strategy — Quintile Long/Short.

For each signal, sort stocks into quintiles each week.
Go long Q1 (lowest signal for bearish signals, highest for bullish),
short Q5, equal- or value-weighted.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class CrossSectionalStrategy(BaseStrategy):
    """Quintile-based long/short strategy on a single signal."""

    # Signals where HIGH value → SHORT (bearish signals)
    BEARISH_SIGNALS = {'os_ratio', 'pc_ratio', 'iv_skew', 'ivol',
                       'abnormal_vol', 'squeeze_signal'}

    def __init__(self, signal_col: str, n_quantiles: int = 5,
                 weighting: str = 'ew', consensus_col: str = None):
        direction = 'bearish' if signal_col in self.BEARISH_SIGNALS else 'bullish'
        name = f'{signal_col}_LS'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description=f'Q1-Q{n_quantiles} {weighting.upper()} on {signal_col}'
        )
        self.signal_col = signal_col
        self.n_quantiles = n_quantiles
        self.weighting = weighting
        self.direction = direction
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        sub = panel.dropna(subset=[self.signal_col, 'ret_lead1']).copy()

        # Optional: condition on consensus signal
        # High consensus = stocks predicted to have higher returns (Chen & Zimmerman)
        # Always keep the HIGH consensus half — signal itself determines L/S direction
        if self.consensus_col and self.consensus_col in sub.columns:
            consensus_med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] >= consensus_med]

        if len(sub) < 100:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        # Assign quintiles
        def assign_q(x):
            n = len(x)
            if n < self.n_quantiles:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'), self.n_quantiles,
                           labels=range(1, self.n_quantiles + 1)).astype(float)

        sub['quintile'] = sub.groupby('week')[self.signal_col].transform(assign_q)
        sub = sub.dropna(subset=['quintile'])
        sub['quintile'] = sub['quintile'].astype(int)

        # Long Q1 (or Q5 for bullish), Short the opposite
        if self.direction == 'bearish':
            long_q, short_q = 1, self.n_quantiles
        else:
            long_q, short_q = self.n_quantiles, 1

        longs = sub[sub['quintile'] == long_q].copy()
        shorts = sub[sub['quintile'] == short_q].copy()

        if self.weighting == 'vw':
            longs['weight'] = longs.groupby('week')['me'].transform(
                lambda x: x / x.sum())
            shorts['weight'] = -shorts.groupby('week')['me'].transform(
                lambda x: x / x.sum())
        else:
            longs['weight'] = longs.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))
            shorts['weight'] = -shorts.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))

        positions = pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)

        return positions
