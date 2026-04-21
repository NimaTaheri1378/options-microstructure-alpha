"""
Constrained Stocks Strategy — O/S signal in low-IO stocks only.

Johnson & So (2012): O/S signal is strongest when short-selling is
costly (proxied by low institutional ownership). This strategy
restricts the trading universe to the bottom tercile of IO.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class ConstrainedStocksStrategy(BaseStrategy):
    """O/S long/short restricted to low institutional ownership stocks."""

    def __init__(self, io_tercile: str = 'low', weighting: str = 'ew',
                 consensus_col: str = None):
        name = 'constrained_os'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description='O/S signal in low-IO stocks only'
        )
        self.io_tercile = io_tercile
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        sub = panel.dropna(subset=['os_ratio', 'io_ratio', 'ret_lead1']).copy()

        if self.consensus_col and self.consensus_col in sub.columns:
            med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] <= med]

        # IO tercile
        sub['io_terc'] = sub.groupby('week')['io_ratio'].transform(
            lambda x: pd.qcut(x.rank(method='first'), 3,
                               labels=['low', 'mid', 'high'],
                               duplicates='drop') if len(x) >= 9 else np.nan
        )
        sub = sub[sub['io_terc'] == self.io_tercile]

        if len(sub) < 50:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        # O/S quintile sort within constrained stocks
        def assign_q(x):
            if len(x) < 5:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'), 5,
                           labels=range(1, 6)).astype(float)

        sub['q'] = sub.groupby('week')['os_ratio'].transform(assign_q)
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
