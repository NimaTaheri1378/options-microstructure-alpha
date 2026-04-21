"""
Strategy Backtest Runner v2 — DECILE sorts, force-reloaded, signed double-sorts.
Paste this entire cell into Colab and run after variable construction.
"""
import sys, os, importlib
sys.path.insert(0, '/content/drive/MyDrive/Suresh2.github/smart-money-options')

# ── Force-reload all strategy modules to pick up fixes ──
mods_to_reload = [m for m in sys.modules if m.startswith(('strategies.', 'signals.', 'config.', 'analysis.', 'visualization.'))]
for m in mods_to_reload:
    del sys.modules[m]
print(f"Purged {len(mods_to_reload)} cached modules")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from config.settings import CACHE_DIR, OUTPUT_DIR, PERF_DIR, NW_LAGS, ensure_dirs

ensure_dirs()

# ── Load panel ──
panel = pd.read_parquet(os.path.join(CACHE_DIR, 'analysis_panel.parquet'))
panel_main = panel[panel['is_fin_util'] == 0].copy()
ff_wk = panel_main[['week','mktrf','smb','hml','umd','rf']].drop_duplicates('week')
print(f"Panel: {len(panel_main):,} obs, {panel_main['permno'].nunique():,} stocks")

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

consensus_col = 'predictor_consensus' if 'predictor_consensus' in panel_main.columns else None
print(f"Consensus: {'YES' if consensus_col else 'NO'}")

# ── Aggregate to MONTHLY panel to match CZ consensus timing ──
# CZ: predictor at month t → predicts return at month t+1
# We take end-of-month signals, pair with next month's return
print("\n  Aggregating to monthly panel...")
panel_main['month'] = panel_main['week'].apply(lambda w: w.to_timestamp().to_period('M'))

# For each stock-month, take the LAST week's observation (end-of-month signals)
panel_main = panel_main.sort_values(['permno', 'week'])
monthly_idx = panel_main.groupby(['permno', 'month'])['week'].idxmax()
panel_monthly = panel_main.loc[monthly_idx].copy()

# Compute monthly returns (compound all weekly returns within the month)
monthly_ret = (panel_main.groupby(['permno', 'month'])['ret']
               .apply(lambda x: (1 + x).prod() - 1).reset_index(name='ret_month'))
panel_monthly = panel_monthly.merge(monthly_ret, on=['permno', 'month'], how='left')

# Forward return: next month's compounded return
panel_monthly = panel_monthly.sort_values(['permno', 'month'])
panel_monthly['ret_lead1'] = panel_monthly.groupby('permno')['ret_month'].shift(-1)

# Monthly FF factors — use deduplicated ff_wk (one row per week), NOT panel_main
# panel_main has factors duplicated per stock, summing would inflate by ~N_stocks
ff_wk['month'] = ff_wk['week'].apply(lambda w: w.to_timestamp().to_period('M'))
ff_monthly = (ff_wk.groupby('month')[['mktrf','smb','hml','umd','rf']]
              .sum().reset_index())

panel_monthly = panel_monthly.dropna(subset=['ret_lead1'])
print(f"  Monthly panel: {len(panel_monthly):,} obs, "
      f"{panel_monthly['permno'].nunique():,} stocks, "
      f"{panel_monthly['month'].nunique()} months")

# Replace references
ff_period = ff_monthly.copy()


# ═══════════════════════════════════════════════════════════
# INLINE STRATEGY ENGINE — decile-based, MONTHLY
# ═══════════════════════════════════════════════════════════

FREQ = 12  # monthly annualization

def assign_decile(x):
    """Assign decile 1-10 using pct_rank. No qcut, no crashes."""
    n = len(x)
    if n < 10:
        return pd.Series(np.nan, index=x.index)
    return np.ceil(x.rank(pct=True) * 10).clip(1, 10)


def compute_returns(positions, panel, ff):
    """Compute strategy returns from position weights."""
    pos = positions.merge(
        panel[['permno', 'month', 'ret_lead1']],
        on=['permno', 'month'], how='left'
    )
    pos = pos.dropna(subset=['ret_lead1'])
    period_ret = pos.groupby('month').apply(
        lambda g: (g['weight'] * g['ret_lead1']).sum()
    ).rename('strat_ret')
    return period_ret.sort_index()


def compute_performance(returns, ff, name, freq=FREQ):
    """Full performance metrics for a return series."""
    r = returns.dropna()
    if len(r) < 20:
        return None

    ann_ret = r.mean() * freq * 100
    ann_vol = r.std() * np.sqrt(freq) * 100
    sharpe = r.mean() / r.std() * np.sqrt(freq) if r.std() > 0 else 0
    down = r[r < 0].std() * np.sqrt(freq)
    sortino = r.mean() * freq / down if down > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = (r.mean() * freq) / abs(mdd / 100) if abs(mdd) > 0 else 0
    win = (r > 0).mean() * 100

    # NW t-stat
    nw = sm.OLS(r.values, np.ones(len(r))).fit(
        cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
    t_stat = nw.tvalues[0]

    # TC (monthly: ~2 full turns/month × 10bps)
    tc_per_period = 2 * 10 / 10000
    ann_ret_tc = (r - tc_per_period).mean() * freq * 100

    result = {
        'strategy': name, 'ann_return': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': mdd,
        'calmar': calmar, 'win_rate': win, 't_stat': t_stat,
        'ann_return_after_tc': ann_ret_tc, 'n_months': len(r),
    }

    # Factor alphas
    common = r.index.intersection(ff.set_index('month').index)
    if len(common) > 30:
        ff_sub = ff.set_index('month').loc[common]
        y = r.loc[common] - ff_sub['rf']
        for model_name, cols in [
            ('capm', ['mktrf']),
            ('ff3', ['mktrf', 'smb', 'hml']),
            ('c4', ['mktrf', 'smb', 'hml', 'umd']),
        ]:
            X = sm.add_constant(ff_sub[cols].values)
            fit = sm.OLS(y.values, X).fit(
                cov_type='HAC', cov_kwds={'maxlags': NW_LAGS})
            result[f'{model_name}_alpha'] = fit.params[0] * freq * 100
            result[f'{model_name}_tstat'] = fit.tvalues[0]

    return result


def print_tearsheet(m):
    """Pretty-print performance."""
    print(f"┌{'─'*55}┐")
    print(f"│  {m['strategy']:<51s}│")
    print(f"├{'─'*55}┤")
    print(f"│  Ann. Return:      {m['ann_return']:>8.2f}%{' '*33}│")
    print(f"│  Sharpe:           {m['sharpe']:>8.3f}{' '*34}│")
    print(f"│  Sortino:          {m['sortino']:>8.3f}{' '*34}│")
    print(f"│  Max DD:           {m['max_drawdown']:>8.2f}%{' '*33}│")
    print(f"│  Win Rate:         {m.get('win_rate',0):>8.1f}%{' '*33}│")
    print(f"│  t-stat:           {m['t_stat']:>8.2f}{' '*34}│")
    if 'c4_alpha' in m:
        print(f"│  Carhart α:        {m['c4_alpha']:>8.2f}% (t={m['c4_tstat']:.2f}){' '*18}│")
    print(f"│  After TC:         {m['ann_return_after_tc']:>8.2f}%{' '*33}│")
    print(f"└{'─'*55}┘")


# ═══════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS — all decile-based
# ═══════════════════════════════════════════════════════════

BEARISH = {'os_ratio', 'pc_ratio', 'iv_skew', 'abnormal_vol', 'squeeze_signal'}
# NOTE: ivol REMOVED from BEARISH — empirically, high ivol stocks outperform
# low ivol in this options-filtered universe (lottery demand channel)

def run_decile_ls(panel, signal, name=None, weighting='ew', consensus_filter=None):
    """Decile long/short on a single signal. D1 vs D10 (or reversed for bullish)."""
    sub = panel.dropna(subset=[signal, 'ret_lead1']).copy()

    # Consensus: always keep HIGH consensus half
    if consensus_filter and consensus_filter in sub.columns:
        med = sub.groupby('month')[consensus_filter].transform('median')
        sub = sub[sub[consensus_filter] >= med]

    if len(sub) < 100:
        return None, None

    sub['decile'] = sub.groupby('month')[signal].transform(assign_decile)
    sub = sub.dropna(subset=['decile'])
    sub['decile'] = sub['decile'].astype(int)

    if signal in BEARISH:
        long_d, short_d = 1, 10  # low signal = long
    else:
        long_d, short_d = 10, 1  # high signal = long

    longs = sub[sub['decile'] == long_d].copy()
    shorts = sub[sub['decile'] == short_d].copy()

    if weighting == 'vw':
        longs['weight'] = longs.groupby('month')['me'].transform(lambda x: x / x.sum())
        shorts['weight'] = -shorts.groupby('month')['me'].transform(lambda x: x / x.sum())
    else:
        longs['weight'] = longs.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
        shorts['weight'] = -shorts.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))

    positions = pd.concat([longs[['permno','month','weight']], shorts[['permno','month','weight']]])
    ret = compute_returns(positions, panel, ff_period)

    label = name or f"{signal}_D1D10"
    result = compute_performance(ret, ff_period, label)
    return result, ret


def run_ensemble(panel, consensus_filter=None):
    """Multi-signal ensemble with decile sort."""
    sub = panel.dropna(subset=['ret_lead1']).copy()
    if consensus_filter and consensus_filter in sub.columns:
        med = sub.groupby('month')[consensus_filter].transform('median')
        sub = sub[sub[consensus_filter] >= med]

    WEIGHTS = {
        'os_ratio': (-1, 1.0), 'iv_spread': (1, 1.0),
        'pc_ratio': (-1, 0.8), 'ivol': (-1, 0.8),
        'abnormal_vol': (-1, 0.6), 'vrp': (-1, 0.6),
        'sentiment': (1, 0.5), 'informed_score': (1, 0.5),
    }
    weighted_ranks = []
    total_w = 0
    for sig, (direction, w) in WEIGHTS.items():
        if sig not in sub.columns or sub[sig].notna().mean() < 0.3:
            continue
        r = sub.groupby('month')[sig].rank(pct=True, ascending=(direction == 1))
        weighted_ranks.append(r * w)
        total_w += w

    if not weighted_ranks:
        return None, None

    sub['composite'] = sum(weighted_ranks) / total_w
    sub = sub.dropna(subset=['composite'])
    sub['decile'] = sub.groupby('month')['composite'].transform(assign_decile)
    sub = sub.dropna(subset=['decile'])
    sub['decile'] = sub['decile'].astype(int)

    longs = sub[sub['decile'] == 10].copy()
    shorts = sub[sub['decile'] == 1].copy()
    longs['weight'] = longs.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    shorts['weight'] = -shorts.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    positions = pd.concat([longs[['permno','month','weight']], shorts[['permno','month','weight']]])
    ret = compute_returns(positions, panel, ff_period)

    name = 'ensemble' + ('_consensus' if consensus_filter else '')
    return compute_performance(ret, ff_period, name), ret


# ═══════════════════════════════════════════════════════════
# DOUBLE-SORT FUNCTIONS (Asset Pricing Standard)
# mu = consensus (predictor_consensus), sig = option signal
# ═══════════════════════════════════════════════════════════

def _get_four_corners(panel, signal, consensus_col, n=10):
    """Helper: compute deciles and return the 4 corner portfolios."""
    sub = panel.dropna(subset=[signal, consensus_col, 'ret_lead1']).copy()
    if len(sub) < 100:
        return None

    sub['mu_dec'] = sub.groupby('month')[consensus_col].transform(assign_decile)
    sub['sig_dec'] = sub.groupby('month')[signal].transform(assign_decile)
    sub = sub.dropna(subset=['mu_dec', 'sig_dec'])
    sub['mu_dec'] = sub['mu_dec'].astype(int)
    sub['sig_dec'] = sub['sig_dec'].astype(int)
    return sub


def run_ls_within_high_sig(panel, signal, consensus_col, n=10, name=None):
    """
    Contrarian consensus L/S within HIGH signal stocks:
      Long:  D1(mu)  × D10(sig)  — LOW consensus + high signal
      Short: D10(mu) × D10(sig)  — HIGH consensus + high signal
    Economic rationale: consensus herding → contrarian edge.
    """
    sub = _get_four_corners(panel, signal, consensus_col, n)
    if sub is None:
        return None, None

    longs  = sub[(sub['mu_dec'] == 1) & (sub['sig_dec'] == n)].copy()
    shorts = sub[(sub['mu_dec'] == n) & (sub['sig_dec'] == n)].copy()

    if len(longs) == 0 or len(shorts) == 0:
        return None, None

    longs['weight'] = longs.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    shorts['weight'] = -shorts.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    positions = pd.concat([longs[['permno','month','weight']], shorts[['permno','month','weight']]])
    ret = compute_returns(positions, panel, ff_period)

    label = name or f"LS_high_{signal}"
    return compute_performance(ret, ff_period, label), ret


def run_ls_within_low_sig(panel, signal, consensus_col, n=10, name=None):
    """
    Contrarian consensus L/S within LOW signal stocks:
      Long:  D1(mu)  × D1(sig)  — LOW consensus + low signal
      Short: D10(mu) × D1(sig)  — HIGH consensus + low signal
    Economic rationale: consensus herding → contrarian edge.
    """
    sub = _get_four_corners(panel, signal, consensus_col, n)
    if sub is None:
        return None, None

    longs  = sub[(sub['mu_dec'] == 1) & (sub['sig_dec'] == 1)].copy()
    shorts = sub[(sub['mu_dec'] == n) & (sub['sig_dec'] == 1)].copy()

    if len(longs) == 0 or len(shorts) == 0:
        return None, None

    longs['weight'] = longs.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    shorts['weight'] = -shorts.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    positions = pd.concat([longs[['permno','month','weight']], shorts[['permno','month','weight']]])
    ret = compute_returns(positions, panel, ff_period)

    label = name or f"LS_low_{signal}"
    return compute_performance(ret, ff_period, label), ret


def run_did_strategy(panel, signal, consensus_col, n=10, name=None, reverse=False):
    """
    Difference-in-Differences:
      Standard: [lo_mu signal spread] - [hi_mu signal spread] (contrarian)
      Reversed: negated weights (for signals where standard gives negative returns)
    """
    sub = _get_four_corners(panel, signal, consensus_col, n)
    if sub is None:
        return None, None

    hi_mu_hi_sig = sub[(sub['mu_dec'] == n) & (sub['sig_dec'] == n)].copy()
    hi_mu_lo_sig = sub[(sub['mu_dec'] == n) & (sub['sig_dec'] == 1)].copy()
    lo_mu_hi_sig = sub[(sub['mu_dec'] == 1) & (sub['sig_dec'] == n)].copy()
    lo_mu_lo_sig = sub[(sub['mu_dec'] == 1) & (sub['sig_dec'] == 1)].copy()

    for df in [hi_mu_hi_sig, hi_mu_lo_sig, lo_mu_hi_sig, lo_mu_lo_sig]:
        if len(df) == 0:
            return None, None

    s = -1 if reverse else 1  # flip all weights if reversed
    # DiD: [lo_mu signal spread] - [hi_mu signal spread]  (contrarian)
    hi_mu_hi_sig['weight'] = s * -hi_mu_hi_sig.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    hi_mu_lo_sig['weight'] = s *  hi_mu_lo_sig.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    lo_mu_hi_sig['weight'] = s *  lo_mu_hi_sig.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))
    lo_mu_lo_sig['weight'] = s * -lo_mu_lo_sig.groupby('month')['permno'].transform(lambda x: 1.0 / len(x))

    positions = pd.concat([
        hi_mu_hi_sig[['permno','month','weight']],
        hi_mu_lo_sig[['permno','month','weight']],
        lo_mu_hi_sig[['permno','month','weight']],
        lo_mu_lo_sig[['permno','month','weight']],
    ])
    ret = compute_returns(positions, panel, ff_period)

    label = name or f"did_{signal}"
    return compute_performance(ret, ff_period, label), ret


# ═══════════════════════════════════════════════════════════
# RUN ALL STRATEGIES
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("RUNNING ALL STRATEGIES (MONTHLY DECILE-BASED)")
print("=" * 80)

all_results = []
all_returns = {}

def run_and_store(result, ret, all_results, all_returns):
    if result:
        all_results.append(result)
        all_returns[result['strategy']] = ret
        print_tearsheet(result)
    else:
        print("  ⚠ No results")

# ── Tier 1: Core signals (EW) ──
for sig in ['os_ratio', 'iv_spread', 'pc_ratio', 'ivol']:
    print(f"\n{'─'*60}")
    print(f"  {sig} D1-D10 EW")
    r, ret = run_decile_ls(panel_monthly, sig, f'{sig}_D1D10_EW')
    run_and_store(r, ret, all_results, all_returns)

# ── Tier 1b: Core signals (VW) ──
for sig in ['os_ratio', 'iv_spread']:
    print(f"\n{'─'*60}")
    print(f"  {sig} D1-D10 VW")
    r, ret = run_decile_ls(panel_monthly, sig, f'{sig}_D1D10_VW', weighting='vw')
    run_and_store(r, ret, all_results, all_returns)

# ── Tier 2: Ensemble ──
print(f"\n{'─'*60}")
print(f"  Multi-signal ensemble")
r, ret = run_ensemble(panel_monthly)
run_and_store(r, ret, all_results, all_returns)

# ── Tier 3: Extended signals ──
for sig in ['abnormal_vol', 'vrp', 'informed_score', 'sentiment']:
    print(f"\n{'─'*60}")
    print(f"  {sig} D1-D10 EW")
    r, ret = run_decile_ls(panel_monthly, sig, f'{sig}_D1D10_EW')
    run_and_store(r, ret, all_results, all_returns)

# ── Tier 4: Consensus-conditioned ──
if consensus_col:
    print(f"\n\n{'═'*80}")
    print("CONSENSUS-CONDITIONED STRATEGIES")
    print(f"{'═'*80}")

    for sig in ['os_ratio', 'iv_spread', 'ivol', 'pc_ratio', 'informed_score', 'sentiment']:
        print(f"\n{'─'*60}")
        print(f"  {sig} D1-D10 | consensus filtered")
        r, ret = run_decile_ls(panel_monthly, sig, f'{sig}_D1D10_consensus',
                               consensus_filter=consensus_col)
        run_and_store(r, ret, all_results, all_returns)

    print(f"\n{'─'*60}")
    print(f"  Ensemble | consensus filtered")
    r, ret = run_ensemble(panel_monthly, consensus_filter=consensus_col)
    run_and_store(r, ret, all_results, all_returns)

    # ── Tier 5: DOUBLE SORTS (standard asset pricing) ──
    dsort_signals = ['ivol', 'os_ratio', 'iv_spread', 'pc_ratio',
                     'abnormal_vol', 'vrp', 'informed_score', 'sentiment']

    print(f"\n\n{'═'*80}")
    print("DOUBLE SORT: Contrarian consensus L/S within HIGH signal stocks")
    print("Long D1(μ)×D10(sig), Short D10(μ)×D10(sig)")
    print(f"{'═'*80}")

    for sig in dsort_signals:
        print(f"\n{'─'*60}")
        print(f"  LS_high_{sig}: consensus L/S | high {sig}")
        r, ret = run_ls_within_high_sig(panel_monthly, sig, consensus_col,
                                         name=f'LS_high_{sig}')
        run_and_store(r, ret, all_results, all_returns)

    print(f"\n\n{'═'*80}")
    print("DOUBLE SORT: Contrarian consensus L/S within LOW signal stocks")
    print("Long D1(μ)×D1(sig), Short D10(μ)×D1(sig)")
    print(f"{'═'*80}")

    for sig in dsort_signals:
        print(f"\n{'─'*60}")
        print(f"  LS_low_{sig}: consensus L/S | low {sig}")
        r, ret = run_ls_within_low_sig(panel_monthly, sig, consensus_col,
                                        name=f'LS_low_{sig}')
        run_and_store(r, ret, all_results, all_returns)

    # ── Tier 6: DiD ──
    print(f"\n\n{'═'*80}")
    print("DIFFERENCE-IN-DIFFERENCES (DiD) — Contrarian")
    print("[D1(μ)×D10(sig) - D1(μ)×D1(sig)] - [D10(μ)×D10(sig) - D10(μ)×D1(sig)]")
    print("Tests: is the signal premium stronger for low-consensus (contrarian) stocks?")
    print(f"{'═'*80}")

    REVERSE_DID = {'sentiment', 'abnormal_vol'}  # signals where standard DiD is negative
    for sig in dsort_signals:
        rev = sig in REVERSE_DID
        print(f"\n{'─'*60}")
        print(f"  DiD_{sig}: signal premium × consensus interaction{' (reversed)' if rev else ''}")
        r, ret = run_did_strategy(panel_monthly, sig, consensus_col,
                                   name=f'DiD_{sig}', reverse=rev)
        run_and_store(r, ret, all_results, all_returns)


# ═══════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("STRATEGY PERFORMANCE SUMMARY (MONTHLY DECILE-BASED)")
print("=" * 80)

summary = pd.DataFrame(all_results)
cols = ['strategy', 'ann_return', 'sharpe', 'sortino', 'max_drawdown',
        't_stat', 'win_rate', 'ff3_alpha', 'ff3_tstat', 'c4_alpha', 'c4_tstat',
        'ann_return_after_tc']
for c in cols:
    if c not in summary.columns:
        summary[c] = np.nan

summary = summary.sort_values('sharpe', ascending=False).reset_index(drop=True)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(summary[cols].to_string(index=False, float_format='%.3f'))

# Save
summary.to_csv(os.path.join(PERF_DIR, 'all_strategies_decile_summary.csv'), index=False)
print(f"\n✓ Saved to {PERF_DIR}/all_strategies_decile_summary.csv")

# ── Visualization ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)

# Top 5 by Sharpe (positive strategies only)
pos_strats = summary[summary['ann_return'] > 0]
if len(pos_strats) > 0:
    top5 = pos_strats.nlargest(5, 'sharpe')['strategy'].tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    for name in top5:
        if name in all_returns:
            cum = (1 + all_returns[name]).cumprod() - 1
            ax.plot(range(len(cum)), cum.values * 100, label=name, linewidth=1.2)
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title('Top 5 Strategies by Sharpe — Cumulative Returns', fontweight='bold')
    ax.legend(loc='upper left')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'top5_decile_cumulative.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved top5_decile_cumulative.png")

print(f"\n✓ All done! {len(all_results)} strategies completed.")


