"""
Double-Sort Strategy — Consensus × Signal (Signed).

Independent double sort:
  1. Sort into consensus deciles (Chen & Zimmerman predictor_consensus)
  2. Within each consensus bucket, sort by signal (e.g., IVOL)

Key idea (signed):
  Long:  High consensus (D10) + High IVOL (D10) — consensus says go up, IVOL amplifies
  Short: Low consensus (D1)  + High IVOL (D10) — consensus says go down, IVOL amplifies

This exploits the insight that IVOL (and other characteristics) amplify
return dispersion — stocks with extreme characteristics move MORE in the
direction predicted by consensus.
"""
import numpy as np
import pandas as pd
from .base import BaseStrategy


class DoubleSortConsensusStrategy(BaseStrategy):
    """Independent double sort: consensus × signal."""

    def __init__(self, signal_col: str = 'ivol', n_buckets: int = 10,
                 weighting: str = 'ew', consensus_col: str = 'predictor_consensus'):
        name = f'double_{signal_col}_x_consensus'
        super().__init__(
            name=name,
            description=f'D10×D10 vs D1×D10: {signal_col} × consensus ({n_buckets} buckets)'
        )
        self.signal_col = signal_col
        self.n_buckets = n_buckets
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.consensus_col not in panel.columns:
            print(f"    ⚠ {self.consensus_col} not in panel")
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        sub = panel.dropna(subset=[self.signal_col, self.consensus_col, 'ret_lead1']).copy()

        if len(sub) < 500:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        n = self.n_buckets

        # Step 1: Consensus deciles (independent sort)
        def safe_qcut(x, q, labels):
            if len(x) < q:
                return pd.Series(np.nan, index=x.index)
            try:
                return pd.qcut(x.rank(method='first'), q,
                               labels=labels, duplicates='drop').astype(float)
            except ValueError:
                return pd.Series(np.nan, index=x.index)

        sub['cons_dec'] = sub.groupby('week')[self.consensus_col].transform(
            lambda x: safe_qcut(x, n, range(1, n + 1)))

        # Step 2: Signal deciles (independent sort)
        sub['sig_dec'] = sub.groupby('week')[self.signal_col].transform(
            lambda x: safe_qcut(x, n, range(1, n + 1)))

        sub = sub.dropna(subset=['cons_dec', 'sig_dec'])
        sub['cons_dec'] = sub['cons_dec'].astype(int)
        sub['sig_dec'] = sub['sig_dec'].astype(int)

        # SIGNED strategy:
        # Long:  consensus D10 (high expected return) + signal D10 (high IVOL = amplifier)
        # Short: consensus D1  (low expected return)  + signal D10 (high IVOL = amplifier)
        longs  = sub[(sub['cons_dec'] == n) & (sub['sig_dec'] == n)].copy()
        shorts = sub[(sub['cons_dec'] == 1) & (sub['sig_dec'] == n)].copy()

        if len(longs) == 0 or len(shorts) == 0:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        if self.weighting == 'vw':
            longs['weight'] = longs.groupby('week')['me'].transform(
                lambda x: x / x.sum())
            shorts['weight'] = -shorts.groupby('week')['me'].transform(
                lambda x: x / x.sum())
        else:
            longs['weight'] = longs.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))
            shorts['weight'] = -shorts.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))

        return pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)


class DoubleSortSignedStrategy(BaseStrategy):
    """
    Broader signed double sort: use consensus to determine direction,
    signal to select extreme stocks.

    Long:  High consensus + extreme signal
    Short: Low consensus  + extreme signal

    Works for any signal — IVOL, O/S, abnormal vol, etc.
    The signal amplifies dispersion; consensus provides direction.
    """

    def __init__(self, signal_col: str = 'ivol', n_consensus: int = 5,
                 n_signal: int = 5, weighting: str = 'ew',
                 consensus_col: str = 'predictor_consensus'):
        name = f'signed_{signal_col}_x_consensus'
        super().__init__(
            name=name,
            description=f'Signed: high/low consensus × high {signal_col}'
        )
        self.signal_col = signal_col
        self.n_consensus = n_consensus
        self.n_signal = n_signal
        self.weighting = weighting
        self.consensus_col = consensus_col

    def generate_positions(self, panel: pd.DataFrame) -> pd.DataFrame:
        if self.consensus_col not in panel.columns:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        sub = panel.dropna(subset=[self.signal_col, self.consensus_col, 'ret_lead1']).copy()

        if len(sub) < 500:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        nc, ns = self.n_consensus, self.n_signal

        def safe_qcut(x, q, labels):
            if len(x) < q:
                return pd.Series(np.nan, index=x.index)
            try:
                return pd.qcut(x.rank(method='first'), q,
                               labels=labels, duplicates='drop').astype(float)
            except ValueError:
                return pd.Series(np.nan, index=x.index)

        sub['cons_q'] = sub.groupby('week')[self.consensus_col].transform(
            lambda x: safe_qcut(x, nc, range(1, nc + 1)))
        sub['sig_q'] = sub.groupby('week')[self.signal_col].transform(
            lambda x: safe_qcut(x, ns, range(1, ns + 1)))

        sub = sub.dropna(subset=['cons_q', 'sig_q'])
        sub['cons_q'] = sub['cons_q'].astype(int)
        sub['sig_q'] = sub['sig_q'].astype(int)

        # Long: high consensus + high signal (extreme characteristic)
        # Short: low consensus + high signal
        longs  = sub[(sub['cons_q'] == nc) & (sub['sig_q'] == ns)].copy()
        shorts = sub[(sub['cons_q'] == 1)  & (sub['sig_q'] == ns)].copy()

        if len(longs) == 0 or len(shorts) == 0:
            return pd.DataFrame(columns=['permno', 'week', 'weight'])

        if self.weighting == 'vw':
            longs['weight'] = longs.groupby('week')['me'].transform(
                lambda x: x / x.sum())
            shorts['weight'] = -shorts.groupby('week')['me'].transform(
                lambda x: x / x.sum())
        else:
            longs['weight'] = longs.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))
            shorts['weight'] = -shorts.groupby('week')['permno'].transform(
                lambda x: 1.0 / len(x))

        return pd.concat([
            longs[['permno', 'week', 'weight']],
            shorts[['permno', 'week', 'weight']]
        ], ignore_index=True)
