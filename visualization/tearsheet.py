"""
Strategy Tearsheet Generator — Publication-quality performance charts.

Generates matplotlib figures for strategy performance visualization:
- Cumulative returns
- Rolling Sharpe ratios  
- Drawdown charts
- Quintile bar charts with alphas
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from config.settings import FIG_DIR, LABELS

# Journal-quality style
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 13, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150, 'axes.grid': True,
    'grid.alpha': 0.3, 'axes.spines.top': False, 'axes.spines.right': False,
})


def savefig(fig, name, fig_dir=None):
    """Save figure as PNG and PDF."""
    if fig_dir is None:
        fig_dir = FIG_DIR
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(fig_dir, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved {name}.png/.pdf")


def plot_cumulative_returns(returns_dict, title='Cumulative Returns',
                             filename='cumulative_returns'):
    """
    Plot cumulative returns for multiple strategies.

    Parameters
    ----------
    returns_dict : dict[str, pd.Series] — {strategy_name: returns}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Set1(np.linspace(0, 1, len(returns_dict)))

    for (name, ret), color in zip(returns_dict.items(), colors):
        cum = (1 + ret).cumprod() - 1
        ax.plot(range(len(cum)), cum.values * 100, label=name,
                color=color, linewidth=1.2)

    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.axhline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    savefig(fig, filename)


def plot_drawdown(returns, title='Drawdown', filename='drawdown'):
    """Plot drawdown chart for a single strategy."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(range(len(dd)), dd.values, 0,
                     color='#e74c3c', alpha=0.5)
    ax.plot(range(len(dd)), dd.values, color='#c0392b', linewidth=0.8)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(fig, filename)


def plot_quintile_returns(res_df, signal_label, filename):
    """
    Plot quintile mean returns and alphas (from portfolio_sort output).
    """
    q_data = res_df[res_df['Q'] != 'Q1-Q5']
    hedge = res_df[res_df['Q'] == 'Q1-Q5']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Mean returns
    x = range(len(q_data))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(q_data)))
    ax1.bar(x, q_data['mean_ret'].values, color=colors,
            edgecolor='white', linewidth=0.5)
    for i, (m, t) in enumerate(zip(q_data['mean_ret'], q_data['t_stat'])):
        sig = ('***' if abs(t) > 2.576 else '**' if abs(t) > 1.96
               else '*' if abs(t) > 1.645 else '')
        ax1.text(i, m + 0.005 * np.sign(m), f'{m:.3f}{sig}',
                 ha='center', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Q{q}' for q in q_data['Q']])
    ax1.set_ylabel('Weekly Return (%)')
    ax1.set_title(f'Panel A: Mean Returns ({signal_label}, EW)')
    ax1.axhline(0, color='black', linewidth=0.5)

    # Panel B: Alphas
    w2 = 0.25
    for j, (acol, alab, clr) in enumerate([
        ('capm_alpha', 'CAPM', '#3498db'),
        ('ff3_alpha', 'FF3', '#2ecc71'),
        ('c4_alpha', 'Carhart', '#e74c3c'),
    ]):
        offset = (j - 1) * w2
        ax2.bar([i + offset for i in x], q_data[acol].values, w2,
                label=alab, color=clr, alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Q{q}' for q in q_data['Q']])
    ax2.set_ylabel('Alpha (%)')
    ax2.set_title('Panel B: Risk-Adjusted Alphas')
    ax2.legend()
    ax2.axhline(0, color='black', linewidth=0.5)

    if not hedge.empty:
        h = hedge.iloc[0]
        fig.text(0.5, -0.02,
                 f'Hedge (Q1−Q5): Return={h["mean_ret"]:.3f}% '
                 f'(t={h["t_stat"]:.2f}), '
                 f'Carhart α={h["c4_alpha"]:.3f}% (t={h["c4_tstat"]:.2f})',
                 ha='center', fontsize=11, fontweight='bold')

    fig.suptitle(f'{signal_label} Quintile Portfolio Analysis',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(fig, filename)


def plot_structural_break(sub_fm_df, sub_hedge_df, filename='structural_break'):
    """Plot sub-period FM coefficients and hedge returns."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    colors_fm = ['#27ae60' if t > 1.96 else '#e74c3c' if t < -1.96
                 else '#95a5a6' for t in sub_fm_df['tstat']]
    ax1.bar(range(len(sub_fm_df)), sub_fm_df['coef'].values,
            color=colors_fm, edgecolor='white', linewidth=0.5)
    for i, (c, t) in enumerate(zip(sub_fm_df['coef'], sub_fm_df['tstat'])):
        sig = ('***' if abs(t) > 2.576 else '**' if abs(t) > 1.96
               else '*' if abs(t) > 1.645 else '')
        ax1.text(i, c + 0.002 * np.sign(c),
                 f'{c:.3f}\n(t={t:.2f}){sig}', ha='center', fontsize=9)
    ax1.set_xticks(range(len(sub_fm_df)))
    ax1.set_xticklabels(sub_fm_df['period'])
    ax1.set_ylabel('FM Coefficient (×100)')
    ax1.set_title('Panel A: O/S Ratio FM Coefficient')
    ax1.axhline(0, color='black', linewidth=0.8)

    colors_h = ['#27ae60' if t > 1.96 else '#e74c3c' if t < -1.96
                else '#95a5a6' for t in sub_hedge_df['hedge_t']]
    ax2.bar(range(len(sub_hedge_df)), sub_hedge_df['hedge_ret'].values,
            color=colors_h, edgecolor='white', linewidth=0.5)
    for i, (r, t) in enumerate(zip(sub_hedge_df['hedge_ret'],
                                    sub_hedge_df['hedge_t'])):
        ax2.text(i, r + 0.005 * np.sign(r), f'{r:.3f}%\n(t={t:.2f})',
                 ha='center', fontsize=9)
    ax2.set_xticks(range(len(sub_hedge_df)))
    ax2.set_xticklabels(sub_hedge_df['period'])
    ax2.set_ylabel('Weekly Hedge Return (%)')
    ax2.set_title('Panel B: Q1−Q5 Hedge Return (EW)')
    ax2.axhline(0, color='black', linewidth=0.8)

    fig.suptitle('Structural Break — O/S Signal Across Sub-Periods',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    savefig(fig, filename)


def plot_correlation_heatmap(panel, vars_list=None, filename='correlation'):
    """Plot correlation heatmap of main variables."""
    if vars_list is None:
        vars_list = ['os_ratio', 'pc_ratio', 'iv_spread', 'log_me', 'bm',
                      'mom', 'ivol', 'amihud', 'io_ratio', 'beta', 'sir']

    vars_avail = [v for v in vars_list if v in panel.columns]
    corr_labels = [LABELS.get(v, v) for v in vars_avail]
    corr = panel[vars_avail].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(vars_avail)))
    ax.set_yticks(range(len(vars_avail)))
    ax.set_xticklabels(corr_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_labels, fontsize=9)
    for i in range(len(vars_avail)):
        for j in range(len(vars_avail)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    savefig(fig, filename)
