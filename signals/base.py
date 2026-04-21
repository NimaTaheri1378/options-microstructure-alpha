"""
Abstract base class for all option-based trading signals.
Each signal takes raw panel data and produces a signal column.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class BaseSignal(ABC):
    """Base class for cross-sectional signal construction."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, panel: pd.DataFrame) -> pd.Series:
        """
        Compute the signal from the merged panel data.

        Parameters
        ----------
        panel : pd.DataFrame
            Must contain at minimum: permno, date, week, and any
            columns required by the specific signal.

        Returns
        -------
        pd.Series
            Signal values, same index as panel.
        """
        raise NotImplementedError

    def winsorize(self, s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
        """Winsorize a series at the given percentiles."""
        lower = s.quantile(lo)
        upper = s.quantile(hi)
        return s.clip(lower, upper)

    def rank_normalize(self, s: pd.Series) -> pd.Series:
        """Cross-sectional rank normalize to [0, 1]."""
        return s.rank(pct=True)

    def __repr__(self):
        return f"Signal({self.name})"
