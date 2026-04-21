"""
IV Skew Signal — OTM Put IV / ATM Call IV.

Xing, Zhang & Zhao (2010): Steep IV skew (expensive OTM puts)
predicts negative future returns. Informed traders buy OTM puts.

Requires OptionMetrics data with delta to separate OTM puts and ATM calls.
This signal is computed during the enhanced download step.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class IVSkew(BaseSignal):
    """IV skew: OTM put IV relative to ATM call IV."""

    def __init__(self):
        super().__init__(
            name='iv_skew',
            description='Mean OTM put IV (|delta| 0.1-0.3) / mean ATM call IV (delta 0.4-0.6)'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        If iv_otm_put and iv_atm_call are pre-computed in download_enhanced,
        simply compute their ratio. Otherwise fallback to iv_put / iv_call.
        """
        if 'iv_otm_put' in panel.columns and 'iv_atm_call' in panel.columns:
            skew = np.where(
                panel['iv_atm_call'] > 0,
                panel['iv_otm_put'] / panel['iv_atm_call'],
                np.nan
            )
        else:
            # Fallback: use iv_put / iv_call as proxy
            skew = np.where(
                panel['iv_call'] > 0,
                panel['iv_put'] / panel['iv_call'],
                np.nan
            )
        result = pd.Series(skew, index=panel.index, name=self.name)
        return self.winsorize(result)
