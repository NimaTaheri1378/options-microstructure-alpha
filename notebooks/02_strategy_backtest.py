"""
Strategy Backtest Runner — runs all strategies and generates tearsheets.
Execute after 00_setup and variable construction.
"""
import sys, os
sys.path.insert(0, '/content/drive/MyDrive/Suresh2.github/smart-money-options')

import pandas as pd
import numpy as np
from config.settings import CACHE_DIR, OUTPUT_DIR, PERF_DIR, ensure_dirs

ensure_dirs()

# Load panel
panel = pd.read_parquet(os.path.join(CACHE_DIR, 'analysis_panel.parquet'))
panel_main = panel[panel['is_fin_util'] == 0].copy()
ff_wk = panel_main[['week','mktrf','smb','hml','umd','rf']].drop_duplicates('week')
print(f"Panel: {len(panel_main):,} obs, {panel_main['permno'].nunique():,} stocks")

# ── Import strategies ──
from strategies.cross_sectional import CrossSectionalStrategy
from strategies.regime_switching import RegimeSwitchingStrategy
from strategies.constrained_stocks import ConstrainedStocksStrategy
from strategies.earnings_event import EarningsEventStrategy
from strategies.volatility_risk_premium import VolatilityRiskPremiumStrategy
from strategies.multi_signal_ensemble import MultiSignalEnsemble

# ── Compute additional signals ──
from signals.abnormal_option_volume import AbnormalOptionVolume
from signals.volatility_risk_premium import VolatilityRiskPremium
from signals.informed_trading_score import InformedTradingScore
from signals.sentiment_composite import SentimentComposite

for SignalClass in [AbnormalOptionVolume, VolatilityRiskPremium,
                    InformedTradingScore, SentimentComposite]:
    sig = SignalClass()
    if sig.name not in panel_main.columns:
        panel_main[sig.name] = sig.compute(panel_main)
        cov = panel_main[sig.name].notna().mean() * 100
        print(f"  Computed {sig.name}: {cov:.1f}% coverage")

# Check for consensus
consensus_col = 'predictor_consensus' if 'predictor_consensus' in panel_main.columns else None

# ── Define all strategies ──
strategies = [
    # Tier 1: Core signals
    CrossSectionalStrategy('os_ratio', weighting='ew'),
    CrossSectionalStrategy('iv_spread', weighting='ew'),
    CrossSectionalStrategy('pc_ratio', weighting='ew'),
    CrossSectionalStrategy('ivol', weighting='ew'),
    CrossSectionalStrategy('os_ratio', weighting='vw'),
    CrossSectionalStrategy('iv_spread', weighting='vw'),

    # Tier 2: Advanced
    RegimeSwitchingStrategy(break_year=2013),
    ConstrainedStocksStrategy(io_tercile='low'),
    EarningsEventStrategy(),
    VolatilityRiskPremiumStrategy(),
    MultiSignalEnsemble(),

    # Tier 3: New signals
    CrossSectionalStrategy('abnormal_vol', weighting='ew'),
    CrossSectionalStrategy('vrp', weighting='ew'),
    CrossSectionalStrategy('informed_score', weighting='ew'),
    CrossSectionalStrategy('sentiment', weighting='ew'),
]

# Add consensus-conditioned + double-sort strategies
if consensus_col:
    strategies += [
        CrossSectionalStrategy('os_ratio', consensus_col=consensus_col),
        CrossSectionalStrategy('iv_spread', consensus_col=consensus_col),
        CrossSectionalStrategy('ivol', consensus_col=consensus_col),
        RegimeSwitchingStrategy(consensus_col=consensus_col),
        ConstrainedStocksStrategy(consensus_col=consensus_col),
        MultiSignalEnsemble(consensus_col=consensus_col),
    ]

    # Tier 4: Double-sort signed strategies (consensus × signal)
    from strategies.double_sort_consensus import (
        DoubleSortConsensusStrategy, DoubleSortSignedStrategy
    )
    strategies += [
        # Decile × Decile: Long D10cons×D10sig, Short D1cons×D10sig
        DoubleSortConsensusStrategy('ivol', n_buckets=10, consensus_col=consensus_col),
        DoubleSortConsensusStrategy('os_ratio', n_buckets=10, consensus_col=consensus_col),
        DoubleSortConsensusStrategy('iv_spread', n_buckets=10, consensus_col=consensus_col),
        DoubleSortConsensusStrategy('abnormal_vol', n_buckets=10, consensus_col=consensus_col),

        # Quintile × Quintile signed variants (more stocks per cell)
        DoubleSortSignedStrategy('ivol', n_consensus=5, n_signal=5, consensus_col=consensus_col),
        DoubleSortSignedStrategy('os_ratio', n_consensus=5, n_signal=5, consensus_col=consensus_col),
        DoubleSortSignedStrategy('iv_spread', n_consensus=5, n_signal=5, consensus_col=consensus_col),
        DoubleSortSignedStrategy('vrp', n_consensus=5, n_signal=5, consensus_col=consensus_col),
        DoubleSortSignedStrategy('sentiment', n_consensus=5, n_signal=5, consensus_col=consensus_col),
    ]

# ── Run all backtests ──
print("\n" + "=" * 80)
print("RUNNING ALL STRATEGIES")
print("=" * 80)

all_results = []
all_returns = {}

for strat in strategies:
    print(f"\n{'─'*60}")
    print(f"  {strat.name}: {strat.description}")
    try:
        results = strat.backtest(panel_main, ff_wk)
        if results:
            results['strategy'] = strat.name
            all_results.append(results)
            all_returns[strat.name] = strat.returns
            print(strat.tearsheet())
        else:
            print("  ⚠ No results")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback; traceback.print_exc()

# ── Summary table ──
print("\n" + "=" * 80)
print("STRATEGY PERFORMANCE SUMMARY")
print("=" * 80)

summary = pd.DataFrame(all_results)
cols = ['strategy', 'ann_return', 'sharpe', 'sortino', 'max_drawdown',
        't_stat', 'ff3_alpha', 'ff3_tstat', 'c4_alpha', 'c4_tstat',
        'ann_return_after_tc']
for c in cols:
    if c not in summary.columns:
        summary[c] = np.nan

print(summary[cols].to_string(index=False, float_format='%.3f'))
summary.to_csv(os.path.join(PERF_DIR, 'all_strategies_summary.csv'), index=False)

# ── Visualizations ──
from visualization.tearsheet import (
    plot_cumulative_returns, plot_drawdown, savefig
)

# Top 5 strategies by Sharpe
top5 = summary.nlargest(5, 'sharpe')['strategy'].tolist()
top_returns = {k: v for k, v in all_returns.items() if k in top5}
if top_returns:
    plot_cumulative_returns(top_returns, title='Top 5 Strategies — Cumulative Returns',
                            filename='top5_cumulative')

# Top 5 by Carhart alpha
if 'c4_alpha' in summary.columns:
    top5_alpha = summary.nlargest(5, 'c4_alpha')['strategy'].tolist()
    top_alpha_returns = {k: v for k, v in all_returns.items() if k in top5_alpha}
    if top_alpha_returns:
        plot_cumulative_returns(top_alpha_returns,
                                title='Top 5 by Carhart Alpha — Cumulative Returns',
                                filename='top5_alpha_cumulative')

print(f"\n✓ All done! Results saved to {PERF_DIR}")
