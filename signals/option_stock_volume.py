"""
O/S Ratio Signal — Option-to-Stock Volume Ratio.

The O/S ratio captures informed trading in the options market relative
to the stock market. Higher O/S indicates more options activity, which
Johnson & So (2012) show predicts negative future returns (informed
traders use options to express negative views when short-selling is costly).
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class OptionStockVolume(BaseSignal):
    """Option-to-Stock volume ratio signal."""

    def __init__(self):
        super().__init__(
            name='os_ratio',
            description='Total option volume / stock volume (weekly)'
        )

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Compute O/S ratio = total_option_volume / stock_volume.

        Required columns: total_opt_vol (or call_vol + put_vol), vol
        """
        if 'total_opt_vol' in panel.columns:
            opt_vol = panel['total_opt_vol']
        else:
            opt_vol = panel['call_vol'] + panel['put_vol']

        os_ratio = np.where(panel['vol'] > 0, opt_vol / panel['vol'], np.nan)
        result = pd.Series(os_ratio, index=panel.index, name=self.name)
        return self.winsorize(result)
