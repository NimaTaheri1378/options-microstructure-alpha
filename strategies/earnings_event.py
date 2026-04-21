"""
Earnings Event Strategy — Trade around earnings using option signals.

Exploits the elevated option activity and IV changes around earnings
announcements. Combines IBES dates with pre-earnings option signals.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class EarningsEventStrategy(BaseStrategy):
    """Event-driven strategy: trade pre-earnings using option signals."""

    def __init__(self, pre_weeks: int = 2, weighting: str = 'ew',
                 consensus_col: str = None):
        name = 'earnings_event'
        if consensus_col:
            name += '_consensus'
        super().__init__(
            name=name,
            description='Pre-earnings option signal strategy'
        )
        self.pre_weeks = pre_weeks
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        # Need earnings dates
        if 'weeks_to_earn' not in panel.columns:
            # Fallback: use SUE as earnings proxy
            if 'sue' not in panel.columns:
                return pd.DataFrame(columns=['permno', 'week', 'weight'])
            return self._sue_based(panel)

        sub = panel.dropna(subset=['iv_spread', 'ret_lead1']).copy()

        if self.consensus_col and self.consensus_col in sub.columns:
            med = sub.groupby('week')[self.consensus_col].transform('median')
            sub = sub[sub[self.consensus_col] >= med]

        # Filter to pre-earnings window
        pre = sub[(sub['weeks_to_earn'] >= 0) & (sub['weeks_to_earn'] <= self.pre_weeks)]

        if len(pre) < 50:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        # Sort on IV spread (bullish signal)
        def assign_q(x):
            if len(x) < 3:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'),
                           min(5, len(x)), labels=False).astype(float)

        pre['q'] = pre.groupby('week')['iv_spread'].transform(assign_q)
        pre = pre.dropna(subset=['q'])

        q_max = pre['q'].max()
        longs = pre[pre['q'] == q_max].copy()
        shorts = pre[pre['q'] == 0].copy()

        longs['weight'] = longs.groupby('week')['permno'].transform(
            lambda x: 1.0 / len(x))
        shorts['weight'] = -shorts.groupby('week')['permno'].transform(
            lambda x: 1.0 / len(x))

        return pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)

    def _sue_based(self, panel):
        """Fallback: sort on recent SUE × IV spread."""
        sub = panel.dropna(subset=['sue', 'iv_spread', 'ret_lead1']).copy()
        sub['earn_signal'] = (
            sub.groupby('week')['sue'].rank(pct=True) *
            sub.groupby('week')['iv_spread'].rank(pct=True)
        )

        def assign_q(x):
            if len(x) < 5:
                return pd.Series(np.nan, index=x.index)
            return pd.qcut(x.rank(method='first'), 5,
                           labels=range(1, 6)).astype(float)

        sub['q'] = sub.groupby('week')['earn_signal'].transform(assign_q)
        sub = sub.dropna(subset=['q'])
        sub['q'] = sub['q'].astype(int)

        longs = sub[sub['q'] == 5].copy()
        shorts = sub[sub['q'] == 1].copy()
        longs['weight'] = longs.groupby('week')['permno'].transform(
            lambda x: 1.0 / len(x))
        shorts['weight'] = -shorts.groupby('week')['permno'].transform(
            lambda x: 1.0 / len(x))

        return pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)
