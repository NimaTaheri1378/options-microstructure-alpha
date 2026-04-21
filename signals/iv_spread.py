"""
IV Spread Signal — Call IV minus Put IV.

Cremers & Weinbaum (2010): Deviations from put-call parity signal
informed trading. Positive IV spread (calls > puts) predicts positive
returns — bullish information embedded in call prices.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class IVSpread(BaseSignal):
    """Implied volatility spread signal (call IV - put IV)."""

    def __init__(self):
        super().__init__(
            name='iv_spread',
            description='ATM call IV minus ATM put IV'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        IV spread = iv_call - iv_put (ATM, |delta| 0.3-0.7).
        These columns are pre-computed in the data pipeline.
        """
        if 'iv_spread' in panel.columns:
            return self.winsorize(panel['iv_spread'].rename(self.name))

        result = panel['iv_call'] - panel['iv_put']
        return self.winsorize(pd.Series(result, index=panel.index, name=self.name))
