"""
Regime-Switching Strategy — Adaptive O/S based on structural break.

Your Chow test found a structural break around 2013. This strategy
flips the O/S signal direction pre- vs post-break.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class RegimeSwitchingStrategy(BaseStrategy):
    """O/S signal with regime-dependent direction."""

    def __init__(self, break_year: int = 2013, weighting: str = 'ew',
                 consensus_col: str = None):
        name = 'regime_os'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description=f'O/S with regime flip at {break_year}'
        )
        self.break_year = break_year
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        sub = panel.dropna(subset=['os_ratio', 'ret_lead1']).copy()

        if self.consensus_col and self.consensus_col in sub.columns:
            med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] <= med]

        # Pre-break: standard O/S (low O/S = long)
        # Post-break: flipped (high O/S = long)
        sub['signal'] = sub['os_ratio'].copy()
        post = sub['date'].dt.year >= self.break_year
        sub.loc[post, 'signal'] = -sub.loc[post, 'signal']

        # Quintile sort on the regime-adjusted signal
        def assign_q(x):
            if len(x) < 5:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'), 5,
                           labels=range(1, 6)).astype(float)

        sub['q'] = sub.groupby('week')['signal'].transform(assign_q)
        sub = sub.dropna(subset=['q'])
        sub['q'] = sub['q'].astype(int)

        longs = sub[sub['q'] == 1].copy()
        shorts = sub[sub['q'] == 5].copy()

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

        return pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)
