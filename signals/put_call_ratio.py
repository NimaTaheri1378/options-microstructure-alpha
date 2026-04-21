"""
Put-Call Ratio Signal.

High put-call ratio reflects bearish sentiment / hedging demand.
Pan & Poteshman (2006): Put-call ratio negatively predicts returns.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class PutCallRatio(BaseSignal):
    """Put-call volume ratio signal."""

    def __init__(self):
        super().__init__(
            name='pc_ratio',
            description='Put volume / (put + call volume)'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        total = panel['call_vol'] + panel['put_vol']
        pc = np.where(total > 0, panel['put_vol'] / total, np.nan)
        result = pd.Series(pc, index=panel.index, name=self.name)
        return self.winsorize(result)
