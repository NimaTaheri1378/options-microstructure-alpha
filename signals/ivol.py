"""
IVOL Signal — Idiosyncratic Volatility.

Ang, Hodrick, Xing & Zhang (2006): Stocks with high IVOL earn
anomalously low returns. IVOL is measured as the standard deviation
of residuals from FF3 factor model over a rolling window.
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class IVOL(BaseSignal):
    """Idiosyncratic volatility signal (rolling std of returns)."""

    def __init__(self, window: int = 8, min_periods: int = 4):
        super().__init__(
            name='ivol',
            description='Rolling 8-week return std (proxy for FF3 residual vol)'
        )
        self.window = window
        self.min_periods = min_periods

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        IVOL = rolling std of weekly returns.
        At weekly frequency, raw std ≈ FF3-residual std.
        """
        if 'ivol' in panel.columns:
            return self.winsorize(panel['ivol'].rename(self.name))

        panel_sorted = panel.sort_values(['permno', 'week'])
        ivol = panel_sorted.groupby('permno')['ret'].transform(
            lambda x: x.rolling(self.window, min_periods=self.min_periods).std()
        )
        result = pd.Series(ivol.values, index=panel.index, name=self.name)
        return self.winsorize(result)
