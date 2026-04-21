"""
Abnormal Option Volume Signal.

Option volume relative to its own trailing average. Spikes indicate
unusual informed activity. Follows Cao, Chen & Griffin (2005).
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class AbnormalOptionVolume(BaseSignal):
    """Abnormal option volume: current vol / 20-day trailing average."""

    def __init__(self, lookback: int = 4):
        super().__init__(
            name='abnormal_vol',
            description='Total option volume / trailing 4-week average'
        )
        self.lookback = lookback

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Compute abnormal volume = current week option vol / trailing avg.
        Panel must be sorted by [permno, week].
        """
        panel = panel.sort_values(['permno', 'week'])
        avg_vol = panel.groupby('permno')['total_opt_vol'].transform(
            lambda x: x.shift(1).rolling(self.lookback, min_periods=2).mean()
        )
        abn = np.where(avg_vol > 0, panel['total_opt_vol'] / avg_vol, np.nan)
        result = pd.Series(abn, index=panel.index, name=self.name)
        return self.winsorize(result)
