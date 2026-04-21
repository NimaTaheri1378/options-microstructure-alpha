"""
Variable Construction — Part 2 refactored.
Builds the weekly analysis panel with 16+ controls from cached data.
"""
import pandas as pd
import numpy as np
import os, gc, time
import warnings
warnings.filterwarnings('ignore')
from config.settings import (
    CACHE_DIR, BASE_CONTROLS, MIN_PRICE, sic_to_ff48, ensure_dirs
)

T0 = time.time()
def elapsed(): return (time.time() - T0) / 60
def load(name): return pd.read_parquet(os.path.join(CACHE_DIR, f'{name}.parquet'))
def save(df, name):
    df.to_parquet(os.path.join(CACHE_DIR, f'{name}.parquet'), index=False)
    print(f"    ✓ SAVED {name}")


def build_panel(consensus_path=None):
    """
    Build the weekly analysis panel from cached Part 1 data.

    Parameters
    ----------
    consensus_path : str, optional
        Path to Chen & Zimmerman consensus CSV (date, permno, predictor_consensus)
    """
    ensure_dirs()
    print("=" * 80)
    print("VARIABLE CONSTRUCTION")
    print("=" * 80)

    # Load cached data
    print(f"\n[1] Loading... [{elapsed():.1f}m]")
    crsp = load('crsp_daily_final'); opt = load('opt_linked_final')
    iv = load('iv_linked_final');     be = load('be_panel')
    io = load('io_quarterly');        ff = load('ff_daily_final')
    ibes = load('ibes_linked');       si = load('short_interest')
    names = load('crsp_names_compact');  delist = load('crsp_delist')

    for df in [crsp, opt, iv, ff]:
        df['date'] = pd.to_datetime(df['date'])
    for df in [crsp, opt, iv, be, io, ff, ibes, si, delist]:
        for col in df.select_dtypes(include=['Float64','Float32','Int64','Int32']).columns:
            df[col] = df[col].astype(float)

    # Delisting returns
    print(f"\n[2] DELISTING RETURNS [{elapsed():.1f}m]")
    delist = delist.rename(columns={'dlstdt': 'date'})
    delist['date'] = pd.to_datetime(delist['date'])
    delist = delist[delist['dlret'].notna()][['permno','date','dlret']].copy()
    crsp = crsp.merge(delist, on=['permno','date'], how='left')
    crsp['ret'] = np.where(crsp['dlret'].notna(),
                           (1 + crsp['ret'].fillna(0)) * (1 + crsp['dlret']) - 1,
                           crsp['ret'])
    crsp.drop(columns=['dlret'], inplace=True)

    # Merge options + CRSP
    print(f"\n[3] MERGE [{elapsed():.1f}m]")
    df = crsp.merge(opt, on=['permno','date'], how='inner')
    df = df.merge(iv[['permno','date','iv_spread','iv_call','iv_put']],
                  on=['permno','date'], how='left')

    # Option signals
    df['total_opt_vol'] = df['call_vol'] + df['put_vol']
    df['os_ratio'] = np.where(df['vol'] > 0, df['total_opt_vol'] / df['vol'], np.nan)
    df['pc_ratio'] = np.where(df['total_opt_vol'] > 0, df['put_vol'] / df['total_opt_vol'], np.nan)
    for col in ['os_ratio', 'pc_ratio']:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    # Weekly aggregation
    print(f"\n[5] WEEKLY AGG [{elapsed():.1f}m]")
    df['week'] = df['date'].dt.to_period('W')
    df = df.sort_values(['permno', 'date'])
    weekly = df.groupby(['permno', 'week']).agg(
        ret=('ret', lambda x: (1+x).prod()-1),
        vol=('vol', 'sum'), total_opt_vol=('total_opt_vol', 'sum'),
        call_vol=('call_vol', 'sum'), put_vol=('put_vol', 'sum'),
        call_oi=('call_oi', 'sum') if 'call_oi' in df.columns else ('call_vol', 'last'),
        put_oi=('put_oi', 'sum') if 'put_oi' in df.columns else ('put_vol', 'last'),
        prc=('prc', 'last'), me=('me', 'last'), shrout=('shrout', 'last'),
        dollar_vol=('dollar_vol', 'sum'), spread=('spread', 'mean'),
        iv_spread=('iv_spread', 'mean'), iv_call=('iv_call', 'mean'),
        iv_put=('iv_put', 'mean'), date=('date', 'max'),
        exchcd=('exchcd', 'first'), n_days=('ret', 'count'),
    ).reset_index()
    del df; gc.collect()

    weekly['os_ratio'] = np.where(weekly['vol'] > 0, weekly['total_opt_vol'] / weekly['vol'], np.nan)
    weekly['pc_ratio'] = np.where(
        (weekly['call_vol']+weekly['put_vol']) > 0,
        weekly['put_vol'] / (weekly['call_vol']+weekly['put_vol']), np.nan)
    for col in ['os_ratio', 'pc_ratio']:
        lo, hi = weekly[col].quantile([0.01, 0.99])
        weekly[col] = weekly[col].clip(lo, hi)

    # FF factors weekly
    ff['week'] = ff['date'].dt.to_period('W')
    ff_wk = ff.groupby('week').agg(
        mktrf=('mktrf','sum'), smb=('smb','sum'), hml=('hml','sum'),
        rf=('rf','sum'), umd=('umd','sum')).reset_index()
    weekly = weekly.merge(ff_wk, on='week', how='left')
    weekly['exret'] = weekly['ret'] - weekly['rf']

    # Controls
    print(f"\n[7] CONTROLS [{elapsed():.1f}m]")
    weekly = weekly.sort_values(['permno', 'week'])
    g = weekly.groupby('permno')

    weekly['log_me'] = np.log(weekly['me'].clip(lower=0.01))
    weekly['year'] = weekly['date'].dt.year
    be_lag = be.copy(); be_lag['match_year'] = be_lag['fyear_end'] + 1
    weekly = weekly.merge(be_lag[['permno','match_year','be']],
                          left_on=['permno','year'], right_on=['permno','match_year'], how='left')
    weekly['bm'] = np.where(weekly['me'] > 0, weekly['be'] / weekly['me'], np.nan)
    weekly.drop(columns=['match_year','be'], errors='ignore', inplace=True)

    weekly['mom'] = g['ret'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=8).apply(lambda y: (1+y).prod()-1, raw=True))
    weekly['rev'] = g['ret'].shift(1)
    weekly['ivol'] = g['ret'].transform(lambda x: x.rolling(8, min_periods=4).std())
    weekly['amihud'] = np.where(weekly['dollar_vol'] > 0,
                                np.abs(weekly['ret']) / (weekly['dollar_vol'] / 1e6), np.nan)
    weekly['amihud'] = weekly['amihud'].clip(upper=weekly['amihud'].quantile(0.99))
    weekly['turnover'] = np.where(weekly['shrout'] > 0, weekly['vol'] / (weekly['shrout']*1000), np.nan)
    weekly['turnover'] = weekly['turnover'].clip(upper=weekly['turnover'].quantile(0.99))

    weekly['qtr'] = weekly['date'].dt.to_period('Q')
    weekly = weekly.merge(io, on=['permno','qtr'], how='left')
    weekly['io_ratio'] = weekly.groupby('qtr')['io_ratio'].transform(lambda x: x.fillna(x.median()))
    weekly['high_constraint'] = (weekly['io_ratio'] < weekly.groupby('week')['io_ratio'].transform('median')).astype(int)

    def rolling_beta(grp, w=52, mp=26):
        r, m = grp['exret'], grp['mktrf']
        cov = r.rolling(w, min_periods=mp).cov(m)
        var = m.rolling(w, min_periods=mp).var()
        return cov / var.replace(0, np.nan)
    weekly['beta'] = weekly.groupby('permno', group_keys=False).apply(rolling_beta).reset_index(level=0, drop=True)
    weekly['max_ret'] = g['ret'].transform(lambda x: x.rolling(4, min_periods=2).max())

    # IBES
    ibes['statpers'] = pd.to_datetime(ibes['statpers'])
    ibes['month'] = ibes['statpers'].dt.to_period('M')
    weekly['month'] = weekly['date'].dt.to_period('M')
    ibes_m = ibes.groupby(['permno','month']).agg(disp=('disp','last'), sue=('sue','last'), numest=('numest','last')).reset_index()
    weekly = weekly.merge(ibes_m, on=['permno','month'], how='left')
    for col in ['sue', 'disp']:
        lo, hi = weekly[col].quantile([0.01, 0.99])
        weekly[col] = weekly[col].clip(lo, hi)

    # FF48
    sic_map = names[['permno','siccd']].drop_duplicates('permno', keep='last')
    weekly = weekly.merge(sic_map, on='permno', how='left')
    weekly['ff48'] = weekly['siccd'].apply(sic_to_ff48)

    # Short interest
    if len(si) > 0:
        si['datadate'] = pd.to_datetime(si['datadate'])
        si['month'] = si['datadate'].dt.to_period('M')
        si_m = si.groupby(['permno','month'])['shortint'].last().reset_index()
        weekly = weekly.merge(si_m, on=['permno','month'], how='left')
        weekly['sir'] = np.where(weekly['shrout'] > 0, weekly['shortint'] / (weekly['shrout']*1000), np.nan)
        weekly['sir'] = weekly['sir'].clip(0, 1)
    else:
        weekly['sir'] = np.nan

    # Future returns
    weekly['ret_lead1'] = weekly.groupby('permno')['ret'].shift(-1)
    weekly['exret_lead1'] = weekly.groupby('permno')['exret'].shift(-1)
    weekly['ret_lead4'] = weekly.groupby('permno')['ret'].transform(
        lambda x: x.shift(-1).rolling(4, min_periods=3).apply(lambda y: (1+y).prod()-1, raw=True))
    weekly['os_x_constraint'] = weekly['os_ratio'] * weekly['high_constraint']

    # Consensus signal (Chen & Zimmerman)
    if consensus_path and os.path.exists(consensus_path):
        print(f"\n  Loading consensus signal from {consensus_path}")
        cons = pd.read_csv(consensus_path)
        cons['date'] = pd.to_datetime(cons['date'])
        cons['month'] = cons['date'].dt.to_period('M')
        weekly = weekly.merge(cons[['permno','month','predictor_consensus']],
                              on=['permno','month'], how='left')
        cov = weekly['predictor_consensus'].notna().mean()*100
        print(f"  Consensus coverage: {cov:.1f}%")

    # Filters
    print(f"\n[8] FILTERS [{elapsed():.1f}m]")
    weekly = weekly[weekly['prc'] >= MIN_PRICE].copy()
    weekly['is_fin_util'] = ((weekly['siccd'].between(6000,6999)) | (weekly['siccd'].between(4900,4999))).fillna(False).astype(int)

    # Final panel
    essential = ['ret_lead1','os_ratio','log_me','rev','ivol']
    panel = weekly.dropna(subset=essential).copy()
    print(f"\n  Panel: {len(panel):,} obs, {panel['permno'].nunique():,} stocks, "
          f"{panel['week'].nunique():,} weeks")

    save(panel, 'analysis_panel')
    print(f"\n  ✓ Done [{elapsed():.1f}m]")
    return panel


if __name__ == '__main__':
    import sys
    consensus = sys.argv[1] if len(sys.argv) > 1 else None
    build_panel(consensus_path=consensus)
