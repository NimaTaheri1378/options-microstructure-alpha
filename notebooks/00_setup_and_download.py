# -*- coding: utf-8 -*-
"""
SMART MONEY OPTIONS — DATA ACQUISITION
=======================================
Run in Google Colab. Resume-safe: cached steps are skipped automatically.
Estimated runtime: ~2-3 hours (OptionMetrics year-by-year dominates).

Usage:
    # Cell 1 — Install + clone
    !pip install wrds linearmodels pyarrow --quiet
    !git clone https://github.com/NimaTaheri1378/options-microstructure-alpha.git
    %cd options-microstructure-alpha

    # Cell 2 — Set WRDS credentials
    import os, getpass
    os.environ['WRDS_USERNAME'] = 'your_wrds_username'
    os.environ['WRDS_PASSWORD'] = getpass.getpass('WRDS Password: ')

    # Cell 3 — Run
    %run notebooks/00_setup_and_download.py
"""

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'wrds', 'linearmodels', 'pyarrow'])

from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError:
    pass  # already mounted

import wrds
import pandas as pd
import numpy as np
import os, gc, time
import warnings
warnings.filterwarnings('ignore')

# ── Paths from config ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.settings import BASE_DIR, CACHE_DIR, ensure_dirs
ensure_dirs()

START_YEAR = 1996
END_YEAR   = 2024
START_DATE = f'{START_YEAR}-01-01'
END_DATE   = f'{END_YEAR}-12-31'

T0 = time.time()
def elapsed(): return (time.time() - T0) / 60

print("="*80)
print("SMART MONEY OPTIONS — DATA ACQUISITION")
print(f"Sample: {START_DATE} → {END_DATE}")
print(f"Cache:  {CACHE_DIR}")
print("="*80)

# ── Verify Drive writable ─────────────────────────────────────────────────────
_test = os.path.join(CACHE_DIR, '.write_test')
try:
    with open(_test, 'w') as f:
        f.write('ok'); f.flush(); os.fsync(f.fileno())
    assert os.path.getsize(_test) > 0
    os.remove(_test)
    print("✓ Google Drive is mounted and writable")
except Exception as e:
    raise RuntimeError(f"Drive NOT writable: {e}\nTry: Runtime → Disconnect and delete runtime.")

# ── Show existing cache ───────────────────────────────────────────────────────
print("\n  Existing cache files:")
cached_files = sorted(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else []
if cached_files:
    for cf in cached_files:
        if cf.startswith('.'): continue
        sz = os.path.getsize(os.path.join(CACHE_DIR, cf))
        print(f"    {cf:<40s} {sz/1024:>8,.0f} KB")
else:
    print("    (none — fresh start)")

###############################################################################
# WRDS CONNECTION
###############################################################################
def _connect():
    username = os.environ.get('WRDS_USERNAME')
    password = os.environ.get('WRDS_PASSWORD')
    if username and password:
        return wrds.Connection(wrds_username=username, wrds_password=password)
    return wrds.Connection()

db = None
try:
    _ = db.raw_sql("SELECT 1")
    print("✓ Reusing existing WRDS connection")
except:
    db = _connect()
    print("✓ WRDS connection established")

###############################################################################
# CACHING HELPERS (Drive-flush + verify, identical to original)
###############################################################################
def _flush(path):
    for p in [path, os.path.dirname(path)]:
        try:
            fd = os.open(p, os.O_RDONLY); os.fsync(fd); os.close(fd)
        except: pass

def _verify(path, name):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"    ✓ SAVED {name} → {os.path.getsize(path)/1024:.0f} KB")
        return True
    print(f"    ✗ WARNING: {name} may have FAILED — retrying...")
    return False

def save_cache(df, name):
    path = os.path.join(CACHE_DIR, f'{name}.parquet')
    df.to_parquet(path, index=False); _flush(path)
    if not _verify(path, name):
        time.sleep(2); df.to_parquet(path, index=False); _flush(path); _verify(path, name)

def cached_df(name):
    path = os.path.join(CACHE_DIR, f'{name}.parquet')
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"  [CACHE HIT] {name} ({os.path.getsize(path)/1024:.0f} KB)")
        return pd.read_parquet(path)
    return None

def cached_query(query, name, force=False):
    path = os.path.join(CACHE_DIR, f'{name}.parquet')
    if os.path.exists(path) and os.path.getsize(path) > 0 and not force:
        print(f"  [CACHE HIT] {name} ({os.path.getsize(path)/1024:.0f} KB)")
        return pd.read_parquet(path)
    print(f"  [QUERY] {name}...", end=' ', flush=True)
    t = time.time()
    df = db.raw_sql(query)
    print(f"{len(df):,} rows ({time.time()-t:.0f}s)")
    save_cache(df, name)
    return df

def coverage(merged, original, label):
    n_m, n_o = len(merged), len(original)
    pct = n_m / n_o * 100 if n_o > 0 else 0
    print(f"  → {label}: {n_m:,}/{n_o:,} ({pct:.1f}%)")

###############################################################################
# 1. LINKING TABLES
###############################################################################
print(f"\n[1] LINKING TABLES [{elapsed():.1f}m]")

om_link = cached_query("""
    SELECT DISTINCT secid, permno, sdate, edate, score
    FROM wrdsapps.opcrsphist WHERE score <= 2
""", 'link_om_crsp')

ccm_link = cached_query("""
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
""", 'link_ccm')

ibes_link = cached_query("""
    SELECT ticker, permno, sdate, edate
    FROM wrdsapps.ibcrsphist WHERE score <= 2
""", 'link_ibes_crsp')

###############################################################################
# 2. CRSP DAILY
###############################################################################
print(f"\n[2] CRSP DAILY [{elapsed():.1f}m]")

crsp_names = cached_query("""
    SELECT DISTINCT permno, shrcd, exchcd, siccd, namedt, nameendt
    FROM crsp.msenames
    WHERE shrcd IN (10,11) AND exchcd IN (1,2,3)
""", 'crsp_names')
print(f"  Common stocks: {crsp_names['permno'].nunique():,}")

crsp_daily = cached_query(f"""
    SELECT a.permno, a.date, a.ret, a.prc, a.vol, a.shrout,
           a.bid, a.ask, a.cfacpr
    FROM crsp.dsf a
    JOIN crsp.msenames b ON a.permno = b.permno
        AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '2099-12-31')
    WHERE a.date BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND b.shrcd IN (10,11) AND b.exchcd IN (1,2,3)
      AND a.prc IS NOT NULL AND a.vol > 0
""", 'crsp_daily')

crsp_daily['date'] = pd.to_datetime(crsp_daily['date'])
exchcd_map = crsp_names[['permno','namedt','nameendt','exchcd']].copy()
exchcd_map['namedt'] = pd.to_datetime(exchcd_map['namedt'])
exchcd_map['nameendt'] = pd.to_datetime(exchcd_map['nameendt'].fillna('2099-12-31'))
crsp_daily = crsp_daily.merge(exchcd_map, on='permno', how='left')
crsp_daily = crsp_daily[
    (crsp_daily['date'] >= crsp_daily['namedt']) &
    (crsp_daily['date'] <= crsp_daily['nameendt'])
].drop(columns=['namedt','nameendt'])
crsp_daily = crsp_daily.drop_duplicates(subset=['permno','date'], keep='first')

# NASDAQ volume correction (Gao & Ritter 2010)
nq = crsp_daily['exchcd'] == 3
crsp_daily.loc[nq & (crsp_daily['date'] < '2001-02-01'), 'vol'] /= 2.0
crsp_daily.loc[nq & (crsp_daily['date'] >= '2001-02-01') &
               (crsp_daily['date'] < '2002-01-01'), 'vol'] /= 1.8

crsp_daily['prc'] = crsp_daily['prc'].abs()
crsp_daily['me'] = crsp_daily['prc'] * crsp_daily['shrout'] / 1000
crsp_daily['dollar_vol'] = crsp_daily['prc'] * crsp_daily['vol']
_ask = pd.to_numeric(crsp_daily['ask'], errors='coerce').fillna(0)
_bid = pd.to_numeric(crsp_daily['bid'], errors='coerce').fillna(0)
crsp_daily['spread'] = np.where(
    (_ask > 0) & (_bid > 0), (_ask - _bid) / ((_ask + _bid) / 2), np.nan)
del _ask, _bid
print(f"  Rows: {len(crsp_daily):,}, Stocks: {crsp_daily['permno'].nunique():,}")

###############################################################################
# 3. OPTIONMETRICS — year-by-year (~2-3 hours)
###############################################################################
print(f"\n[3] OPTIONMETRICS year-by-year [{elapsed():.1f}m]")
print("  (Resume-safe: each year cached individually)")

agg_cache = cached_df('option_agg_all')
iv_cache  = cached_df('iv_spread_all')

if agg_cache is not None and iv_cache is not None:
    option_agg = agg_cache
    iv_spread  = iv_cache
    print(f"  option_agg: {len(option_agg):,}, iv_spread: {len(iv_spread):,}")
else:
    agg_years, iv_years = [], []
    years_todo = [y for y in range(START_YEAR, END_YEAR + 1)
                  if not (os.path.exists(os.path.join(CACHE_DIR, f'opt_agg_{y}.parquet'))
                          and os.path.exists(os.path.join(CACHE_DIR, f'iv_spread_{y}.parquet')))]
    print(f"  Years remaining: {len(years_todo)} (~{len(years_todo)*4} min)")

    for yi, year in enumerate(range(START_YEAR, END_YEAR + 1)):
        yr_agg = cached_df(f'opt_agg_{year}')
        yr_iv  = cached_df(f'iv_spread_{year}')
        if yr_agg is not None and yr_iv is not None:
            agg_years.append(yr_agg); iv_years.append(yr_iv); continue

        pct = (yi + 1) / (END_YEAR - START_YEAR + 1) * 100
        print(f"\n  ▶ {year} ({pct:.0f}% | {elapsed():.1f}m)...", flush=True)
        t1 = time.time()
        try:
            raw = db.raw_sql(f"""
                SELECT secid, date, cp_flag, volume, open_interest,
                       impl_volatility, delta
                FROM optionm.opprcd{year}
                WHERE volume >= 0
                  AND impl_volatility IS NOT NULL AND impl_volatility > 0
                  AND impl_volatility < 3
                  AND delta IS NOT NULL
                  AND ABS(delta) BETWEEN 0.1 AND 0.9
                  AND cp_flag IN ('C','P')
            """)
            raw['date'] = pd.to_datetime(raw['date'])
            calls = raw[raw['cp_flag'] == 'C']
            puts  = raw[raw['cp_flag'] == 'P']

            c_agg = calls.groupby(['secid','date']).agg(
                call_vol=('volume','sum'), call_oi=('open_interest','sum')).reset_index()
            p_agg = puts.groupby(['secid','date']).agg(
                put_vol=('volume','sum'), put_oi=('open_interest','sum')).reset_index()
            yr_agg = c_agg.merge(p_agg, on=['secid','date'], how='outer').fillna(0)

            atm_c = calls[calls['delta'].between(0.3, 0.7)]
            atm_p = puts[puts['delta'].between(-0.7, -0.3)]
            iv_c = atm_c.groupby(['secid','date'])['impl_volatility'].mean().rename('iv_call')
            iv_p = atm_p.groupby(['secid','date'])['impl_volatility'].mean().rename('iv_put')
            yr_iv = pd.concat([iv_c, iv_p], axis=1).dropna().reset_index()
            yr_iv['iv_spread'] = yr_iv['iv_call'] - yr_iv['iv_put']

            save_cache(yr_agg, f'opt_agg_{year}')
            save_cache(yr_iv, f'iv_spread_{year}')
            agg_years.append(yr_agg); iv_years.append(yr_iv)
            print(f"    {len(raw):,} raw → agg:{len(yr_agg):,} iv:{len(yr_iv):,} ({time.time()-t1:.0f}s)")
            del raw, calls, puts, c_agg, p_agg, atm_c, atm_p; gc.collect()
        except Exception as e:
            print(f"    ERROR on {year}: {e}")

    option_agg = pd.concat(agg_years, ignore_index=True)
    iv_spread  = pd.concat(iv_years, ignore_index=True)
    save_cache(option_agg, 'option_agg_all')
    save_cache(iv_spread,  'iv_spread_all')
    del agg_years, iv_years; gc.collect()
    print(f"\n  TOTAL agg: {len(option_agg):,}  iv: {len(iv_spread):,}")

###############################################################################
# 4. LINK OPTIONS → CRSP
###############################################################################
print(f"\n[4] LINK OPTIONS → CRSP [{elapsed():.1f}m]")

om_link['sdate'] = pd.to_datetime(om_link['sdate'])
om_link['edate'] = pd.to_datetime(om_link['edate'].fillna('2099-12-31'))
option_agg['date'] = pd.to_datetime(option_agg['date'])
iv_spread['date']  = pd.to_datetime(iv_spread['date'])

valid_permnos = set(crsp_names['permno'].dropna().astype(int))
link_filt = om_link[om_link['permno'].isin(valid_permnos)].copy()

opt_linked = option_agg.merge(
    link_filt[['secid','permno','sdate','edate']], on='secid', how='inner')
opt_linked = opt_linked[
    (opt_linked['date'] >= opt_linked['sdate']) &
    (opt_linked['date'] <= opt_linked['edate'])
].drop(columns=['sdate','edate','secid'])
opt_linked = opt_linked.drop_duplicates(subset=['permno','date'], keep='first')
coverage(opt_linked, option_agg, "OM→CRSP (volume)")

iv_linked = iv_spread.merge(
    link_filt[['secid','permno','sdate','edate']], on='secid', how='inner')
iv_linked = iv_linked[
    (iv_linked['date'] >= iv_linked['sdate']) &
    (iv_linked['date'] <= iv_linked['edate'])
].drop(columns=['sdate','edate','secid'])
iv_linked = iv_linked.drop_duplicates(subset=['permno','date'], keep='first')
coverage(iv_linked, iv_spread, "OM→CRSP (IV spread)")

del option_agg, iv_spread; gc.collect()

###############################################################################
# 5. COMPUSTAT (Book Equity)
###############################################################################
print(f"\n[5] COMPUSTAT [{elapsed():.1f}m]")

compustat = cached_query(f"""
    SELECT gvkey, datadate, fyear,
           seq, ceq, at, lt, txditc, pstkrv, pstkl, pstk
    FROM comp.funda
    WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
      AND datadate BETWEEN '{int(START_YEAR)-2}-01-01' AND '{END_DATE}'
""", 'compustat_annual')
compustat['datadate'] = pd.to_datetime(compustat['datadate'])
compustat['ps'] = compustat['pstkrv'].fillna(
    compustat['pstkl'].fillna(compustat['pstk'].fillna(0)))
compustat['be'] = compustat['seq'] + compustat['txditc'].fillna(0) - compustat['ps']
compustat = compustat[compustat['be'] > 0].copy()

ccm_link['linkdt']    = pd.to_datetime(ccm_link['linkdt'])
ccm_link['linkenddt'] = pd.to_datetime(ccm_link['linkenddt'].fillna('2099-12-31'))
comp_ccm = compustat.merge(
    ccm_link[['gvkey','permno','linkdt','linkenddt']], on='gvkey', how='inner')
comp_ccm = comp_ccm[
    (comp_ccm['datadate'] >= comp_ccm['linkdt']) &
    (comp_ccm['datadate'] <= comp_ccm['linkenddt'])
]
comp_ccm['fyear_end'] = comp_ccm['datadate'].dt.year
comp_ccm = comp_ccm.sort_values('datadate').groupby(
    ['permno','fyear_end']).last().reset_index()
coverage(comp_ccm, compustat, "Compustat→CRSP")

###############################################################################
# 6. INSTITUTIONAL OWNERSHIP (13F)
###############################################################################
print(f"\n[6] INSTITUTIONAL OWNERSHIP [{elapsed():.1f}m]")

io_raw = cached_query(f"""
    WITH io AS (
        SELECT SUBSTR(cusip,1,8) AS cusip8, rdate,
               SUM(shares) AS inst_shares
        FROM tfn.s34
        WHERE rdate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND shares IS NOT NULL AND shares > 0
        GROUP BY SUBSTR(cusip,1,8), rdate
    )
    SELECT c.permno, i.rdate, i.inst_shares
    FROM io i
    JOIN (SELECT DISTINCT SUBSTR(ncusip,1,8) AS cusip8, permno
          FROM crsp.msenames WHERE ncusip IS NOT NULL) c
    ON i.cusip8 = c.cusip8
""", 'io_raw')
io_raw['rdate'] = pd.to_datetime(io_raw['rdate'])

msf = cached_query(f"""
    SELECT permno, date, shrout*1000 AS shares_out
    FROM crsp.msf
    WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND shrout IS NOT NULL AND shrout > 0
""", 'crsp_monthly_shrout')
msf['date'] = pd.to_datetime(msf['date'])
msf['qtr']    = msf['date'].dt.to_period('Q')
io_raw['qtr'] = io_raw['rdate'].dt.to_period('Q')
shrout_qtr = msf.groupby(['permno','qtr'])['shares_out'].last().reset_index()
io_merged = io_raw.merge(shrout_qtr, on=['permno','qtr'], how='inner')
io_merged['io_ratio'] = (io_merged['inst_shares'] / io_merged['shares_out']).clip(0, 1)
io_qtr = io_merged.groupby(['permno','qtr'])['io_ratio'].mean().reset_index()
print(f"  IO: {len(io_qtr):,} obs, mean: {io_qtr['io_ratio'].mean():.3f}")

###############################################################################
# 7. FF FACTORS
###############################################################################
print(f"\n[7] FF FACTORS [{elapsed():.1f}m]")

ff_daily = cached_query(f"""
    SELECT date, mktrf, smb, hml, rf, umd
    FROM ff.factors_daily
    WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
""", 'ff_daily')
ff_daily['date'] = pd.to_datetime(ff_daily['date'])
print(f"  FF daily: {len(ff_daily):,} days, UMD coverage: {ff_daily['umd'].notna().mean()*100:.1f}%")

###############################################################################
# 8. IBES (Analyst Dispersion + SUE)
###############################################################################
print(f"\n[8] IBES [{elapsed():.1f}m]")

ibes_stats = cached_query(f"""
    SELECT ticker, statpers, fpedats, meanest, medest, stdev, numest
    FROM ibes.statsumu_epsus
    WHERE statpers BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND measure = 'EPS' AND fpi = '1'
""", 'ibes_summary')
ibes_stats['statpers'] = pd.to_datetime(ibes_stats['statpers'])

ibes_actuals = cached_query(f"""
    SELECT ticker, pends, anndats, value AS actual
    FROM ibes.actu_epsus
    WHERE anndats BETWEEN '{START_DATE}' AND '{END_DATE}'
      AND pdicity = 'QTR' AND measure = 'EPS'
""", 'ibes_actuals')
ibes_actuals['anndats'] = pd.to_datetime(ibes_actuals['anndats'])
ibes_actuals['pends']   = pd.to_datetime(ibes_actuals['pends'])

ibes_stats['fpedats'] = pd.to_datetime(ibes_stats['fpedats'])
ibes_stats = ibes_stats.merge(
    ibes_actuals[['ticker','pends','actual']],
    left_on=['ticker','fpedats'], right_on=['ticker','pends'], how='left'
).drop(columns=['pends'], errors='ignore')

ibes_link['sdate'] = pd.to_datetime(ibes_link['sdate'])
ibes_link['edate'] = pd.to_datetime(ibes_link['edate'].fillna('2099-12-31'))
ibes_stats = ibes_stats.merge(ibes_link, on='ticker', how='inner')
ibes_stats = ibes_stats[
    (ibes_stats['statpers'] >= ibes_stats['sdate']) &
    (ibes_stats['statpers'] <= ibes_stats['edate'])
].drop(columns=['sdate','edate','ticker'])
ibes_stats['disp'] = np.where(
    ibes_stats['meanest'].abs() > 0.01,
    ibes_stats['stdev'] / ibes_stats['meanest'].abs(), np.nan)
ibes_stats['sue'] = np.where(
    ibes_stats['stdev'] > 0,
    (ibes_stats['actual'] - ibes_stats['meanest']) / ibes_stats['stdev'], np.nan)
coverage(ibes_stats, ibes_actuals, "IBES→CRSP")

###############################################################################
# 9. SHORT INTEREST
###############################################################################
print(f"\n[9] SHORT INTEREST [{elapsed():.1f}m]")

try:
    short_int = cached_query(f"""
        SELECT a.gvkey, a.datadate, a.shortint
        FROM comp.sec_shortint a
        WHERE a.datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
          AND a.shortint IS NOT NULL AND a.shortint > 0
    """, 'short_interest')
    short_int['datadate'] = pd.to_datetime(short_int['datadate'])
    short_int = short_int.merge(
        ccm_link[['gvkey','permno','linkdt','linkenddt']], on='gvkey', how='inner')
    short_int = short_int[
        (short_int['datadate'] >= short_int['linkdt']) &
        (short_int['datadate'] <= short_int['linkenddt'])
    ].drop(columns=['linkdt','linkenddt','gvkey'])
    print(f"  Short interest: {len(short_int):,} obs")
except Exception as e:
    print(f"  Short interest not available: {e}")
    short_int = pd.DataFrame(columns=['permno','datadate','shortint'])

###############################################################################
# 10. SAVE ALL FINAL FILES
###############################################################################
print(f"\n[10] SAVING FINAL FILES [{elapsed():.1f}m]")

save_cache(opt_linked,                        'opt_linked_final')
save_cache(iv_linked,                         'iv_linked_final')
save_cache(crsp_daily,                        'crsp_daily_final')
save_cache(crsp_names[['permno','siccd','exchcd']], 'crsp_names')
save_cache(comp_ccm[['permno','fyear_end','be']], 'be_panel')
save_cache(io_qtr,                            'io_quarterly')
save_cache(ff_daily,                          'ff_daily_final')
save_cache(ibes_stats,                        'ibes_linked')
save_cache(short_int,                         'short_interest')

print(f"""
{'='*60}
DATA ACQUISITION COMPLETE — {elapsed():.1f} min total
{'='*60}
  CRSP daily:      {len(crsp_daily):>12,} rows
  Option volume:   {len(opt_linked):>12,} rows
  IV spread:       {len(iv_linked):>12,} rows
  Book equity:     {len(comp_ccm):>12,} rows
  IO quarterly:    {len(io_qtr):>12,} rows
  FF daily:        {len(ff_daily):>12,} rows
  IBES:            {len(ibes_stats):>12,} rows
  Short interest:  {len(short_int):>12,} rows
  Period:          {START_DATE} → {END_DATE}
{'='*60}
→ Next: %run analysis/variable_construction.py
""")
