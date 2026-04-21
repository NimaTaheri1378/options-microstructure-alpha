"""
Volatility Risk Premium Strategy — Sell overpriced options.

Long stocks with high VRP (IV >> realized vol → overpriced puts),
short stocks with low VRP. Captures the variance risk premium.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class VolatilityRiskPremiumStrategy(BaseStrategy):
    """Long high-VRP, short low-VRP stocks."""

    def __init__(self, weighting: str = 'ew', consensus_col: str = None):
        name = 'vrp_strategy'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description='Volatility risk premium long/short'
        )
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        if 'vrp' not in panel.columns:
            # Compute VRP inline
            if 'iv_call' in panel.columns and 'ivol' in panel.columns:
                panel = panel.copy()
                panel['vrp'] = panel['iv_call'] - panel['ivol']
            else:
                return pd.DataFrame(columns=['permno', 'week', 'weight'])

        sub = panel.dropna(subset=['vrp', 'ret_lead1']).copy()

        if self.consensus_col and self.consensus_col in sub.columns:
            med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] >= med]

        def assign_q(x):
            if len(x) < 5:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'), 5,
                           labels=range(1, 6)).astype(float)

        sub['q'] = sub.groupby('week')['vrp'].transform(assign_q)
        sub = sub.dropna(subset=['q'])
        sub['q'] = sub['q'].astype(int)

        # High VRP = overpriced options → short vol → expect mean reversion
        longs = sub[sub['q'] == 5].copy()   # Highest VRP
        shorts = sub[sub['q'] == 1].copy()  # Lowest VRP

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
