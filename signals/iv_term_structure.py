"""
IV Term Structure Signal — Short-term IV / Long-term IV.

When short-term IV is high relative to long-term IV, it signals
imminent expected moves (e.g., pre-earnings). This captures
the urgency of informed options trading.

Requires exdate-based separation in the OM query.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class IVTermStructure(BaseSignal):
    """IV term structure: near-term IV vs far-term IV."""

    def __init__(self):
        super().__init__(
            name='iv_term',
            description='Short-term IV / Long-term IV (ratio)'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        If iv_near and iv_far are pre-computed (from download_enhanced),
        compute their ratio. Otherwise use iv_call as a single-maturity proxy.
        """
        if 'iv_near' in panel.columns and 'iv_far' in panel.columns:
            term = np.where(
                panel['iv_far'] > 0,
                panel['iv_near'] / panel['iv_far'],
                np.nan
            )
        else:
            # Not enough data for term structure — return NaN
            term = np.full(len(panel), np.nan)

        result = pd.Series(term, index=panel.index, name=self.name)
        return self.winsorize(result)
