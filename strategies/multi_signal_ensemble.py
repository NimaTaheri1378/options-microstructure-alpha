"""
Multi-Signal Ensemble Strategy — Dynamic rank-weighted combination.
Uses decile sorts. Completely avoids pd.qcut to prevent NaN bin crashes.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class MultiSignalEnsemble(BaseStrategy):
    """Combined signal strategy with dynamic rank weights."""

    SIGNAL_WEIGHTS = {
        'os_ratio': (-1, 1.0),
        'iv_spread': (1, 1.0),
        'pc_ratio': (-1, 0.8),
        'ivol': (-1, 0.8),
        'abnormal_vol': (-1, 0.6),
        'vrp': (-1, 0.6),
        'sentiment': (1, 0.5),
        'informed_score': (1, 0.5),
    }

    def __init__(self, weighting: str = 'ew', consensus_col: str = None):
        name = 'multi_ensemble'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description='Rank-weighted multi-signal ensemble (deciles)'
        )
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        sub = panel.dropna(subset=['ret_lead1']).copy()

        if self.consensus_col and self.consensus_col in sub.columns:
            med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] >= med]

        # Build composite signal from available components
        weighted_ranks = []
        total_weight = 0

        for sig, (direction, weight) in self.SIGNAL_WEIGHTS.items():
            if sig not in sub.columns:
                continue
            if sub[sig].notna().mean() < 0.3:
                continue
            ascending = (direction == 1)
            r = sub.groupby('week')[sig].rank(pct=True, ascending=ascending)
            weighted_ranks.append(r * weight)
            total_weight += weight

        if not weighted_ranks or total_weight == 0:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        sub['composite'] = sum(weighted_ranks) / total_weight
        sub = sub.dropna(subset=['composite'])

        # Use numpy percentile-based decile assignment (no qcut/pd.cut)
        def assign_decile(x):
            vals = x.values
            n = len(vals)
            if n < 10:
                return pd.Series(np.nan, index=x.index)
            # Percentile-based: rank → assign to 1-10
            pct_rank = pd.Series(vals, index=x.index).rank(pct=True)
            decile = np.ceil(pct_rank * 10).clip(1, 10)
            return decile

        sub['decile'] = sub.groupby('week')['composite'].transform(assign_decile)
        sub = sub.dropna(subset=['decile'])
        sub['decile'] = sub['decile'].astype(int)

        longs = sub[sub['decile'] == 10].copy()
        shorts = sub[sub['decile'] == 1].copy()

        if len(longs) == 0 or len(shorts) == 0:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

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
