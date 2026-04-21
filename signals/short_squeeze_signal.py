"""
Short Squeeze Signal — High short interest + rising call volume + low IO.

Stocks with high SIR, low institutional ownership, and spiking call
volume are candidates for short squeezes. Combines multiple data sources.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class ShortSqueezeSignal(BaseSignal):
    """Short squeeze detector: SIR × call volume spike × low IO."""

    def __init__(self):
        super().__init__(
            name='squeeze_signal',
            description='Short squeeze probability signal'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Composite rank of:
        - High SIR (short interest ratio)
        - Rising call volume (vs trailing avg)
        - Low institutional ownership
        """
        ranks = []

        # High short interest
        if 'sir' in panel.columns:
            r_sir = panel.groupby('week')['sir'].rank(pct=True)
            ranks.append(r_sir)

        # Call volume spike
        if 'call_vol' in panel.columns:
            avg_cv = panel.groupby('permno')['call_vol'].transform(
                lambda x: x.shift(1).rolling(4, min_periods=2).mean()
            )
            cv_ratio = np.where(avg_cv > 0, panel['call_vol'] / avg_cv, np.nan)
            cv_series = pd.Series(cv_ratio, index=panel.index)
            r_cv = panel.groupby('week').apply(
                lambda g: cv_series.loc[g.index].rank(pct=True)
            ).droplevel(0)
            ranks.append(r_cv)

        # Low IO (inverted rank — low IO = high rank)
        if 'io_ratio' in panel.columns:
            r_io = panel.groupby('week')['io_ratio'].rank(pct=True, ascending=False)
            ranks.append(r_io)

        if not ranks:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        composite = pd.concat(ranks, axis=1).mean(axis=1)
        return pd.Series(composite.values, index=panel.index, name=self.name)
