"""
Gamma Exposure Signal — Net dealer gamma from option open interest.

High dealer gamma → dealers need to buy (sell) stock as price rises
(falls), dampening volatility. Low/negative gamma → amplified moves.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class GammaExposure(BaseSignal):
    """Net dealer gamma exposure proxy."""

    def __init__(self):
        super().__init__(
            name='gamma_exp',
            description='(Call OI - Put OI) / stock volume — gamma proxy'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Proxy: (call_oi - put_oi) normalized by stock volume.
        Positive = dealers long gamma (mean-reverting dynamics).
        """
        if 'call_oi' in panel.columns and 'put_oi' in panel.columns:
            net_oi = panel['call_oi'] - panel['put_oi']
            gamma = np.where(
                panel['vol'] > 0,
                net_oi / panel['vol'],
                np.nan
            )
        else:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        result = pd.Series(gamma, index=panel.index, name=self.name)
        return self.winsorize(result)
