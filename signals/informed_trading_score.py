"""
Informed Trading Score — Composite of O/S, IV Spread, and IV Skew.

Rank-weighted combination of the three core option-based signals.
Diversifies across different channels of informed trading.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class InformedTradingScore(BaseSignal):
    """Composite informed-trading signal from rank-averaging multiple signals."""

    def __init__(self):
        super().__init__(
            name='informed_score',
            description='Rank-avg of O/S ratio, IV spread, IV skew'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Cross-sectional rank-normalize each component, then average.
        O/S is negated (high O/S → bearish → low rank).
        IV spread is kept as-is (high → bullish → high rank).
        IV skew is negated (high skew → bearish → low rank).
        """
        ranks = []

        if 'os_ratio' in panel.columns:
            # Negate: high O/S = bearish
            r = panel.groupby('week')['os_ratio'].rank(pct=True, ascending=False)
            ranks.append(r)

        if 'iv_spread' in panel.columns:
            r = panel.groupby('week')['iv_spread'].rank(pct=True)
            ranks.append(r)

        if 'iv_skew' in panel.columns:
            # Negate: high skew = bearish
            r = panel.groupby('week')['iv_skew'].rank(pct=True, ascending=False)
            ranks.append(r)
        elif 'iv_put' in panel.columns and 'iv_call' in panel.columns:
            skew_proxy = panel['iv_put'] / panel['iv_call'].replace(0, np.nan)
            r = panel.groupby('week').apply(
                lambda g: skew_proxy.loc[g.index].rank(pct=True, ascending=False)
            ).droplevel(0)
            ranks.append(r)

        if not ranks:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        composite = pd.concat(ranks, axis=1).mean(axis=1)
        return pd.Series(composite.values, index=panel.index, name=self.name)
