"""
Sentiment Composite Signal — Ensemble of all available signals.

Uses cross-sectional rank normalization and equal weighting to combine
all option-based signals into a single composite score.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class SentimentComposite(BaseSignal):
    """Composite sentiment from all available option signals."""

    # Signals and their direction (1 = higher signal → higher return,
    # -1 = higher signal → lower return)
    SIGNAL_DIRECTIONS = {
        'os_ratio': -1,      # High O/S → bearish
        'iv_spread': 1,      # High IV spread → bullish
        'pc_ratio': -1,      # High P/C → bearish
        'iv_skew': -1,       # High skew → bearish
        'abnormal_vol': -1,  # High abnormal vol → bearish (informed selling)
        'vrp': -1,           # High VRP → overpriced options
        'gamma_exp': 1,      # High gamma → stabilizing
        'ivol': -1,          # High IVOL → low returns (anomaly)
    }

    def __init__(self):
        super().__init__(
            name='sentiment',
            description='Rank-averaged composite of all option signals'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """Rank-normalize available signals and average."""
        ranks = []
        for sig, direction in self.SIGNAL_DIRECTIONS.items():
            if sig not in panel.columns:
                continue
            if panel[sig].notna().mean() < 0.3:
                continue  # Skip if < 30% coverage
            ascending = (direction == 1)
            r = panel.groupby('week')[sig].rank(pct=True, ascending=ascending)
            ranks.append(r)

        if not ranks:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        composite = pd.concat(ranks, axis=1).mean(axis=1)
        return pd.Series(composite.values, index=panel.index, name=self.name)
