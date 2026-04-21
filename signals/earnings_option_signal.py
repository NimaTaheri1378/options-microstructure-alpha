"""
Earnings Option Signal — Pre-earnings IV run-up + O/S spike.

Options activity intensifies before earnings announcements as informed
traders position themselves. Combines IBES earnings dates with
option volume/IV data.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class EarningsOptionSignal(BaseSignal):
    """Pre-earnings signal: abnormal IV and volume before announcements."""

    def __init__(self, pre_window: int = 2):
        super().__init__(
            name='earnings_opt',
            description='Pre-earnings option activity signal'
        )
        self.pre_window = pre_window  # weeks before earnings

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Flags stocks with upcoming earnings (within pre_window weeks)
        and measures their abnormal option activity.

        Requires: anndats (from IBES, merged to panel), iv_call, total_opt_vol
        """
        if 'weeks_to_earn' not in panel.columns:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        # Pre-earnings flag
        pre_earn = (panel['weeks_to_earn'] >= 0) & (panel['weeks_to_earn'] <= self.pre_window)

        # Abnormal IV: current IV vs 8-week trailing
        if 'iv_call' in panel.columns:
            avg_iv = panel.groupby('permno')['iv_call'].transform(
                lambda x: x.shift(1).rolling(8, min_periods=4).mean()
            )
            iv_ratio = np.where(avg_iv > 0, panel['iv_call'] / avg_iv, np.nan)
        else:
            iv_ratio = np.ones(len(panel))

        # Composite: IV run-up × pre-earnings flag
        signal = np.where(pre_earn, iv_ratio, np.nan)
        result = pd.Series(signal, index=panel.index, name=self.name)
        return self.winsorize(result)
