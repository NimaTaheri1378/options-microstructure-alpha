"""
FIXED Double-Sort + DiD Strategies — paste this AFTER running 03_decile_backtest.py
Adds to the existing all_results and all_returns.
"""

# ═══════════════════════════════════════════════════════════
# SIGNED DOUBLE SORTS (DIRECTION-AWARE) + DiD
# ═══════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("SIGNED DOUBLE SORTS — DIRECTION-AWARE")
print("Long: D10(cons)×GOOD_sig | Short: D1(cons)×BAD_sig")
print("=" * 80)

BEARISH_SIGS = {'os_ratio', 'pc_ratio', 'ivol', 'abnormal_vol', 'squeeze_signal'}

def run_double_sort_directional(panel, signal, consensus_col, n=10, name=None):
    """
    Direction-aware double sort:
    For BEARISH signals (high = bad):  Long D10(cons)×D1(sig), Short D1(cons)×D10(sig)
    For BULLISH signals (high = good): Long D10(cons)×D10(sig), Short D1(cons)×D1(sig)
    """
    sub = panel.dropna(subset=[signal, consensus_col, 'ret_lead1']).copy()
    if len(sub) < 500:
        return None, None

    sub['cons_dec'] = sub.groupby('week')[consensus_col].transform(assign_decile)
    sub['sig_dec'] = sub.groupby('week')[signal].transform(assign_decile)
    sub = sub.dropna(subset=['cons_dec', 'sig_dec'])
    sub['cons_dec'] = sub['cons_dec'].astype(int)
    sub['sig_dec'] = sub['sig_dec'].astype(int)

    if signal in BEARISH_SIGS:
        # Bearish: D1(sig) = low signal = GOOD, D10(sig) = high signal = BAD
        longs  = sub[(sub['cons_dec'] == n) & (sub['sig_dec'] == 1)].copy()   # good cons + good sig
        shorts = sub[(sub['cons_dec'] == 1) & (sub['sig_dec'] == n)].copy()   # bad cons + bad sig
    else:
        # Bullish: D10(sig) = high signal = GOOD, D1(sig) = low signal = BAD
        longs  = sub[(sub['cons_dec'] == n) & (sub['sig_dec'] == n)].copy()   # good cons + good sig
        shorts = sub[(sub['cons_dec'] == 1) & (sub['sig_dec'] == 1)].copy()   # bad cons + bad sig

    if len(longs) == 0 or len(shorts) == 0:
        return None, None

    longs['weight'] = longs.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))
    shorts['weight'] = -shorts.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))
    positions = pd.concat([longs[['permno','week','weight']], shorts[['permno','week','weight']]])
    ret = compute_returns(positions, panel, ff_wk)

    label = name or f"dsort_{signal}"
    return compute_performance(ret, ff_wk, label), ret


def run_did_strategy(panel, signal, consensus_col, n=10, name=None):
    """
    Difference-in-Differences portfolio:
    DiD = [D10cons×ExtrSig - D1cons×ExtrSig] - [D10cons×MidSig - D1cons×MidSig]

    Tests whether consensus effect is STRONGER among extreme-signal stocks.
    """
    sub = panel.dropna(subset=[signal, consensus_col, 'ret_lead1']).copy()
    if len(sub) < 500:
        return None, None

    sub['cons_dec'] = sub.groupby('week')[consensus_col].transform(assign_decile)
    sub['sig_dec'] = sub.groupby('week')[signal].transform(assign_decile)
    sub = sub.dropna(subset=['cons_dec', 'sig_dec'])
    sub['cons_dec'] = sub['cons_dec'].astype(int)
    sub['sig_dec'] = sub['sig_dec'].astype(int)

    # Four corners
    if signal in BEARISH_SIGS:
        extr_sig = 1   # extreme = low signal (good for bearish)
        mid_sig = 5    # middle decile
    else:
        extr_sig = n   # extreme = high signal (good for bullish)
        mid_sig = 5

    # Treatment: consensus effect among extreme-signal stocks
    treat_long  = sub[(sub['cons_dec'] == n) & (sub['sig_dec'] == extr_sig)].copy()
    treat_short = sub[(sub['cons_dec'] == 1) & (sub['sig_dec'] == extr_sig)].copy()

    # Control: consensus effect among mid-signal stocks
    ctrl_long  = sub[(sub['cons_dec'] == n) & (sub['sig_dec'] == mid_sig)].copy()
    ctrl_short = sub[(sub['cons_dec'] == 1) & (sub['sig_dec'] == mid_sig)].copy()

    for df in [treat_long, treat_short, ctrl_long, ctrl_short]:
        if len(df) == 0:
            return None, None

    # Weights: +1 for treat_long, -1 for treat_short, -1 for ctrl_long, +1 for ctrl_short
    treat_long['weight']  = treat_long.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))
    treat_short['weight'] = -treat_short.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))
    ctrl_long['weight']   = -ctrl_long.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))
    ctrl_short['weight']  = ctrl_short.groupby('week')['permno'].transform(lambda x: 1.0 / len(x))

    positions = pd.concat([
        treat_long[['permno','week','weight']],
        treat_short[['permno','week','weight']],
        ctrl_long[['permno','week','weight']],
        ctrl_short[['permno','week','weight']],
    ])
    ret = compute_returns(positions, panel, ff_wk)

    label = name or f"did_{signal}"
    return compute_performance(ret, ff_wk, label), ret


# ── Run direction-aware double sorts ──
ds_results = []
ds_returns = {}

for sig in ['ivol', 'os_ratio', 'iv_spread', 'pc_ratio',
            'abnormal_vol', 'vrp', 'informed_score', 'sentiment']:
    print(f"\n{'─'*60}")
    print(f"  DOUBLE SORT: {sig} × consensus (direction-aware)")
    direction = "bearish" if sig in BEARISH_SIGS else "bullish"
    print(f"    Signal type: {direction}")
    r, ret = run_double_sort_directional(panel_main, sig, consensus_col,
                                          name=f'dsort_{sig}')
    if r:
        ds_results.append(r)
        ds_returns[r['strategy']] = ret
        print_tearsheet(r)
    else:
        print("  ⚠ No results")

# ── Run DiD strategies ──
print("\n\n" + "=" * 80)
print("DIFFERENCE-IN-DIFFERENCES (DiD)")
print("Tests: is consensus more predictive among extreme-signal stocks?")
print("=" * 80)

for sig in ['ivol', 'os_ratio', 'iv_spread', 'pc_ratio',
            'informed_score', 'sentiment']:
    print(f"\n{'─'*60}")
    print(f"  DiD: {sig} × consensus")
    r, ret = run_did_strategy(panel_main, sig, consensus_col,
                               name=f'did_{sig}')
    if r:
        ds_results.append(r)
        ds_returns[r['strategy']] = ret
        print_tearsheet(r)
    else:
        print("  ⚠ No results")

# ── Combined summary ──
print("\n\n" + "=" * 80)
print("DOUBLE-SORT + DiD SUMMARY")
print("=" * 80)

ds_summary = pd.DataFrame(ds_results)
cols = ['strategy', 'ann_return', 'sharpe', 'sortino', 'max_drawdown',
        't_stat', 'win_rate', 'c4_alpha', 'c4_tstat', 'ann_return_after_tc']
for c in cols:
    if c not in ds_summary.columns:
        ds_summary[c] = np.nan
print(ds_summary[cols].to_string(index=False, float_format='%.3f'))

# Append to main results
all_results.extend(ds_results)
all_returns.update(ds_returns)

# Save combined
full_summary = pd.DataFrame(all_results)
full_summary.to_csv(os.path.join(PERF_DIR, 'all_strategies_full_summary.csv'), index=False)
print(f"\n✓ Saved {len(full_summary)} strategies to all_strategies_full_summary.csv")

# Updated top 5 chart
top5 = full_summary[full_summary['ann_return'] > 0].nlargest(5, 'sharpe')['strategy'].tolist()
fig, ax = plt.subplots(figsize=(12, 5))
for name in top5:
    if name in all_returns:
        cum = (1 + all_returns[name]).cumprod() - 1
        ax.plot(range(len(cum)), cum.values * 100, label=name, linewidth=1.2)
ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Top 5 Overall Strategies — Cumulative Returns', fontweight='bold')
ax.legend(loc='upper left')
ax.axhline(0, color='black', linewidth=0.5)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'figures', 'top5_overall_cumulative.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print("  ✓ Saved top5_overall_cumulative.png")
