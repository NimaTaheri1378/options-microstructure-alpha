"""
Volatility Risk Premium Signal — IV minus Realized Volatility.

Stocks where IV significantly exceeds realized vol have overpriced
options. The VRP captures compensation for bearing volatility risk.
Bollerslev, Tauchen & Zhou (2009).
"""
import numpy as np
import pandas as pd
from .base import BaseSignal


class VolatilityRiskPremium(BaseSignal):
    """VRP: implied volatility minus realized volatility."""

    def __init__(self, realized_window: int = 4):
        super().__init__(
            name='vrp',
            description='IV(call) - realized vol (4-week)'
        )
        self.realized_window = realized_window

    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        VRP = ATM call IV - realized volatility (from weekly returns).
        """
        # Realized vol from weekly returns
        panel_sorted = panel.sort_values(['permno', 'week'])
        rvol = panel_sorted.groupby('permno')['ret'].transform(
            lambda x: x.rolling(self.realized_window, min_periods=2).std()
        ) * np.sqrt(52)  # Annualize

        if 'iv_call' in panel_sorted.columns:
            vrp = panel_sorted['iv_call'] - rvol
        elif 'ivol' in panel_sorted.columns:
            vrp = panel_sorted['ivol'] - rvol
        else:
            return pd.Series(np.nan, index=panel.index, name=self.name)

        result = pd.Series(vrp.values, index=panel.index, name=self.name)
        return self.winsorize(result)
