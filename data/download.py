"""
WRDS Data Acquisition — exact queries from original Suresh_Research2.
Refactored for modularity. Preserves all SQL, filters, and linking logic.
Run in Colab with: python -m data.download
"""
import wrds
import pandas as pd
import numpy as np
import os, gc, time
import warnings
warnings.filterwarnings('ignore')
from config.settings import (
    CACHE_DIR, START_YEAR, END_YEAR, START_DATE, END_DATE, ensure_dirs
)

T0 = time.time()

def elapsed():
    return (time.time() - T0) / 60

class WRDSDownloader:
    """Handles all WRDS data acquisition with caching and reconnection."""

    def __init__(self):
        self.db = None
        ensure_dirs()

    def ensure_connection(self):
        """Test WRDS connection; reconnect if dead."""
        try:
            self.db.raw_sql("SELECT 1")
            return
        except:
            pass
        print("  ↻ Reconnecting to WRDS...")
        self.db = wrds.Connection()
        print("  ✓ Connected")

    def connect(self):
        """Initial WRDS connection."""
        try:
            self.db.raw_sql("SELECT 1")
            print("✓ Reusing existing WRDS connection")
        except:
            self.db = wrds.Connection()
            print("✓ New WRDS connection established")

    def cached_query(self, query, cache_name, force=False):
        path = os.path.join(CACHE_DIR, f'{cache_name}.parquet')
        if os.path.exists(path) and not force:
            print(f"  [CACHE] {cache_name}")
            return pd.read_parquet(path)
        self.ensure_connection()
        print(f"  [QUERY] {cache_name}...", end=' ', flush=True)
        t = time.time()
        df = self.db.raw_sql(query)
        dt = time.time() - t
        print(f"{len(df):,} rows ({dt:.0f}s)")
        df.to_parquet(path, index=False)
        sz = os.path.getsize(path) / 1024
        print(f"    ✓ SAVED {cache_name} → {sz:.0f} KB")
        return df

    def cached_df(self, cache_name):
        path = os.path.join(CACHE_DIR, f'{cache_name}.parquet')
        if os.path.exists(path):
            print(f"  [CACHE] {cache_name}")
            return pd.read_parquet(path)
        return None

    def save_cache(self, df, cache_name):
        path = os.path.join(CACHE_DIR, f'{cache_name}.parquet')
        df.to_parquet(path, index=False)
        sz = os.path.getsize(path) / 1024
        print(f"    ✓ SAVED {cache_name} → {sz:.0f} KB")

    def coverage(self, merged, original, label):
        n_m, n_o = len(merged), len(original)
        pct = n_m / n_o * 100 if n_o > 0 else 0
        print(f"  → {label}: {n_m:,}/{n_o:,} ({pct:.1f}%)")

    # ─── STEP 1: LINKING TABLES ───────────────────────────────────────
    def download_links(self):
        print(f"\n[1] LINKING TABLES [{elapsed():.1f}m]")
        print("-" * 40)
        om_link = self.cached_query("""
            SELECT DISTINCT secid, permno, sdate, edate, score
            FROM wrdsapps.opcrsphist WHERE score <= 2
        """, 'link_om_crsp')
        ccm_link = self.cached_query("""
            SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
            FROM crsp.ccmxpf_linktable
            WHERE linktype IN ('LU','LC') AND linkprim IN ('P','C')
        """, 'link_ccm')
        ibes_link = self.cached_query("""
            SELECT ticker, permno, sdate, edate
            FROM wrdsapps.ibcrsphist WHERE score <= 2
        """, 'link_ibes_crsp')
        cusip_map = self.cached_query("""
            SELECT DISTINCT permno, SUBSTR(ncusip,1,8) AS cusip8
            FROM crsp.msenames WHERE ncusip IS NOT NULL
        """, 'link_cusip')
        return om_link, ccm_link, ibes_link, cusip_map

    # ─── STEP 2: CRSP DAILY ──────────────────────────────────────────
    def download_crsp(self):
        print(f"\n[2] CRSP DAILY [{elapsed():.1f}m]")
        print("-" * 40)
        check = self.cached_df('crsp_daily_final')
        if check is not None:
            crsp_names = self.cached_df('crsp_names')
            print(f"  Rows: {len(check):,}")
            return check, crsp_names

        crsp_names = self.cached_query("""
            SELECT DISTINCT permno, shrcd, exchcd, siccd, namedt, nameendt
            FROM crsp.msenames
            WHERE shrcd IN (10,11) AND exchcd IN (1,2,3)
        """, 'crsp_names')

        crsp_daily = self.cached_query(f"""
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

        # Exchange codes for NASDAQ vol correction
        exchcd_map = crsp_names[['permno','namedt','nameendt','exchcd']].copy()
        exchcd_map['namedt'] = pd.to_datetime(exchcd_map['namedt'])
        exchcd_map['nameendt'] = pd.to_datetime(
            exchcd_map['nameendt'].fillna('2099-12-31'))
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
            (_ask > 0) & (_bid > 0),
            (_ask - _bid) / ((_ask + _bid) / 2), np.nan)

        self.save_cache(crsp_daily, 'crsp_daily_final')
        return crsp_daily, crsp_names

    # ─── STEP 3: OPTIONMETRICS ────────────────────────────────────────
    def download_optionmetrics(self):
        print(f"\n[3] OPTIONMETRICS [{elapsed():.1f}m]")
        print("-" * 40)
        agg_cache = self.cached_df('option_agg_all')
        iv_cache = self.cached_df('iv_spread_all')
        if agg_cache is not None and iv_cache is not None:
            print(f"  agg: {len(agg_cache):,}, iv: {len(iv_cache):,}")
            return agg_cache, iv_cache

        agg_years, iv_years = [], []
        for yi, year in enumerate(range(START_YEAR, END_YEAR + 1)):
            yr_agg = self.cached_df(f'opt_agg_{year}')
            yr_iv = self.cached_df(f'iv_spread_{year}')
            if yr_agg is not None and yr_iv is not None:
                agg_years.append(yr_agg); iv_years.append(yr_iv)
                continue

            pct = (yi+1) / (END_YEAR - START_YEAR + 1) * 100
            print(f"  ▶ {year} ({pct:.0f}%)...", flush=True)
            self.ensure_connection()
            t1 = time.time()
            try:
                raw = self.db.raw_sql(f"""
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
                puts = raw[raw['cp_flag'] == 'P']

                c_agg = calls.groupby(['secid','date']).agg(
                    call_vol=('volume','sum'), call_oi=('open_interest','sum')
                ).reset_index()
                p_agg = puts.groupby(['secid','date']).agg(
                    put_vol=('volume','sum'), put_oi=('open_interest','sum')
                ).reset_index()
                yr_agg = c_agg.merge(p_agg, on=['secid','date'], how='outer').fillna(0)

                atm_c = calls[calls['delta'].between(0.3, 0.7)]
                atm_p = puts[puts['delta'].between(-0.7, -0.3)]
                iv_c = atm_c.groupby(['secid','date'])['impl_volatility'].mean().rename('iv_call')
                iv_p = atm_p.groupby(['secid','date'])['impl_volatility'].mean().rename('iv_put')
                iv_yr = pd.concat([iv_c, iv_p], axis=1).dropna().reset_index()
                iv_yr['iv_spread'] = iv_yr['iv_call'] - iv_yr['iv_put']

                agg_years.append(yr_agg); iv_years.append(iv_yr)
                self.save_cache(yr_agg, f'opt_agg_{year}')
                self.save_cache(iv_yr, f'iv_spread_{year}')
                print(f"    {len(raw):,}→agg:{len(yr_agg):,} ({time.time()-t1:.0f}s)")
                del raw, calls, puts; gc.collect()
            except Exception as e:
                print(f"    ERROR: {e}")

        option_agg = pd.concat(agg_years, ignore_index=True)
        iv_spread = pd.concat(iv_years, ignore_index=True)
        self.save_cache(option_agg, 'option_agg_all')
        self.save_cache(iv_spread, 'iv_spread_all')
        return option_agg, iv_spread

    # ─── STEP 4: LINK OPTIONS → CRSP ─────────────────────────────────
    def link_options(self, option_agg, iv_spread, om_link, crsp_names):
        print(f"\n[4] LINK OPTIONS → CRSP [{elapsed():.1f}m]")
        print("-" * 40)
        check1 = self.cached_df('opt_linked_final')
        check2 = self.cached_df('iv_linked_final')
        if check1 is not None and check2 is not None:
            return check1, check2

        om_link['sdate'] = pd.to_datetime(om_link['sdate'])
        om_link['edate'] = pd.to_datetime(om_link['edate'].fillna('2099-12-31'))
        option_agg['date'] = pd.to_datetime(option_agg['date'])
        iv_spread['date'] = pd.to_datetime(iv_spread['date'])
        valid = set(crsp_names['permno'].dropna().astype(int))
        lf = om_link[om_link['permno'].isin(valid)].copy()

        opt_linked = option_agg.merge(lf[['secid','permno','sdate','edate']], on='secid', how='inner')
        opt_linked = opt_linked[(opt_linked['date'] >= opt_linked['sdate']) &
                                (opt_linked['date'] <= opt_linked['edate'])
        ].drop(columns=['sdate','edate','secid'])
        opt_linked = opt_linked.drop_duplicates(subset=['permno','date'], keep='first')

        iv_linked = iv_spread.merge(lf[['secid','permno','sdate','edate']], on='secid', how='inner')
        iv_linked = iv_linked[(iv_linked['date'] >= iv_linked['sdate']) &
                              (iv_linked['date'] <= iv_linked['edate'])
        ].drop(columns=['sdate','edate','secid'])
        iv_linked = iv_linked.drop_duplicates(subset=['permno','date'], keep='first')

        self.save_cache(opt_linked, 'opt_linked_final')
        self.save_cache(iv_linked, 'iv_linked_final')
        return opt_linked, iv_linked

    # ─── STEP 5: COMPUSTAT ────────────────────────────────────────────
    def download_compustat(self, ccm_link):
        print(f"\n[5] COMPUSTAT [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        comp = self.cached_query(f"""
            SELECT gvkey, datadate, fyear,
                   seq, ceq, at, lt, txditc, pstkrv, pstkl, pstk
            FROM comp.funda
            WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
              AND datadate BETWEEN '{int(START_YEAR)-2}-01-01' AND '{END_DATE}'
        """, 'compustat_annual')
        comp['datadate'] = pd.to_datetime(comp['datadate'])
        for col in ['seq','ceq','at','lt','txditc','pstkrv','pstkl','pstk']:
            comp[col] = pd.to_numeric(comp[col], errors='coerce').astype(float)
        comp['ps'] = comp['pstkrv'].fillna(comp['pstkl'].fillna(comp['pstk'].fillna(0)))
        comp['be'] = comp['seq'] + comp['txditc'].fillna(0) - comp['ps']
        comp = comp[comp['be'] > 0].copy()

        ccm_link['linkdt'] = pd.to_datetime(ccm_link['linkdt'])
        ccm_link['linkenddt'] = pd.to_datetime(ccm_link['linkenddt'].fillna('2099-12-31'))
        ccm = comp.merge(ccm_link[['gvkey','permno','linkdt','linkenddt']], on='gvkey', how='inner')
        ccm = ccm[(ccm['datadate'] >= ccm['linkdt']) & (ccm['datadate'] <= ccm['linkenddt'])]
        ccm['fyear_end'] = ccm['datadate'].dt.year
        ccm = ccm.sort_values('datadate').groupby(['permno','fyear_end']).last().reset_index()
        self.save_cache(ccm[['permno','fyear_end','be']], 'be_panel')
        return ccm

    # ─── STEP 6: INSTITUTIONAL OWNERSHIP ──────────────────────────────
    def download_io(self):
        print(f"\n[6] INSTITUTIONAL OWNERSHIP [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        io_raw = self.cached_query(f"""
            WITH io AS (
                SELECT SUBSTR(cusip,1,8) AS cusip8, rdate, SUM(shares) AS inst_shares
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
        io_raw['inst_shares'] = pd.to_numeric(io_raw['inst_shares'], errors='coerce').astype(float)

        self.ensure_connection()
        msf = self.cached_query(f"""
            SELECT permno, date, shrout*1000 AS shares_out
            FROM crsp.msf
            WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
              AND shrout IS NOT NULL AND shrout > 0
        """, 'crsp_monthly_shrout')
        msf['date'] = pd.to_datetime(msf['date'])
        msf['qtr'] = msf['date'].dt.to_period('Q')
        io_raw['qtr'] = io_raw['rdate'].dt.to_period('Q')
        shrout_qtr = msf.groupby(['permno','qtr'])['shares_out'].last().reset_index()
        io_merged = io_raw.merge(shrout_qtr, on=['permno','qtr'], how='inner')
        io_merged['io_ratio'] = (io_merged['inst_shares'] / io_merged['shares_out']).clip(0, 1)
        io_qtr = io_merged.groupby(['permno','qtr'])['io_ratio'].mean().reset_index()
        self.save_cache(io_qtr, 'io_quarterly')
        return io_qtr

    # ─── STEP 7: FF FACTORS ──────────────────────────────────────────
    def download_ff(self):
        print(f"\n[7] FF FACTORS [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        ff = self.cached_query(f"""
            SELECT date, mktrf, smb, hml, rf, umd
            FROM ff.factors_daily
            WHERE date BETWEEN '{START_DATE}' AND '{END_DATE}'
        """, 'ff_daily')
        ff['date'] = pd.to_datetime(ff['date'])
        self.save_cache(ff, 'ff_daily_final')
        return ff

    # ─── STEP 8: IBES ────────────────────────────────────────────────
    def download_ibes(self, ibes_link):
        print(f"\n[8] IBES [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        stats = self.cached_query(f"""
            SELECT ticker, statpers, fpedats, meanest, medest, stdev, numest
            FROM ibes.statsumu_epsus
            WHERE statpers BETWEEN '{START_DATE}' AND '{END_DATE}'
              AND measure = 'EPS' AND fpi = '1'
        """, 'ibes_summary')
        stats['statpers'] = pd.to_datetime(stats['statpers'])

        self.ensure_connection()
        actuals = self.cached_query(f"""
            SELECT ticker, pends, anndats, value AS actual
            FROM ibes.actu_epsus
            WHERE anndats BETWEEN '{START_DATE}' AND '{END_DATE}'
              AND pdicity = 'QTR' AND measure = 'EPS'
        """, 'ibes_actuals')
        actuals['anndats'] = pd.to_datetime(actuals['anndats'])
        actuals['pends'] = pd.to_datetime(actuals['pends'])
        stats['fpedats'] = pd.to_datetime(stats['fpedats'])
        stats = stats.merge(actuals[['ticker','pends','actual']],
                            left_on=['ticker','fpedats'], right_on=['ticker','pends'],
                            how='left').drop(columns=['pends'], errors='ignore')

        ibes_link['sdate'] = pd.to_datetime(ibes_link['sdate'])
        ibes_link['edate'] = pd.to_datetime(ibes_link['edate'].fillna('2099-12-31'))
        stats = stats.merge(ibes_link, on='ticker', how='inner')
        stats = stats[(stats['statpers'] >= stats['sdate']) &
                      (stats['statpers'] <= stats['edate'])
        ].drop(columns=['sdate','edate','ticker'])

        for col in ['meanest','medest','stdev','numest','actual']:
            if col in stats.columns:
                stats[col] = pd.to_numeric(stats[col], errors='coerce').astype(float)

        stats['disp'] = np.where(stats['meanest'].abs() > 0.01,
                                  stats['stdev'] / stats['meanest'].abs(), np.nan)
        stats['sue'] = np.where(stats['stdev'] > 0,
                                 (stats['actual'] - stats['meanest']) / stats['stdev'], np.nan)
        self.save_cache(stats, 'ibes_linked')
        return stats

    # ─── STEP 9: SHORT INTEREST ──────────────────────────────────────
    def download_short_interest(self, ccm_link):
        print(f"\n[9] SHORT INTEREST [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        try:
            si = self.cached_query(f"""
                SELECT a.gvkey, a.datadate, a.shortint, a.shortintadj
                FROM comp.sec_shortint a
                WHERE a.datadate BETWEEN '{START_DATE}' AND '{END_DATE}'
                  AND a.shortint IS NOT NULL AND a.shortint > 0
            """, 'short_interest_raw')
            si['datadate'] = pd.to_datetime(si['datadate'])
            for col in ['shortint','shortintadj']:
                si[col] = pd.to_numeric(si[col], errors='coerce').astype(float)
            si = si.merge(ccm_link[['gvkey','permno','linkdt','linkenddt']],
                          on='gvkey', how='inner')
            si = si[(si['datadate'] >= si['linkdt']) &
                    (si['datadate'] <= si['linkenddt'])
            ].drop(columns=['linkdt','linkenddt','gvkey'])
        except Exception as e:
            print(f"  Short interest not available: {e}")
            si = pd.DataFrame(columns=['permno','datadate','shortint','shortintadj'])
        self.save_cache(si, 'short_interest')
        return si

    # ─── STEP 10: DELISTING RETURNS ──────────────────────────────────
    def download_delist(self):
        print(f"\n[10] DELISTING RETURNS [{elapsed():.1f}m]")
        print("-" * 40)
        self.ensure_connection()
        dl = self.cached_query(f"""
            SELECT permno, dlstdt, dlstcd, dlret
            FROM crsp.dsedelist
            WHERE dlstdt BETWEEN '{START_DATE}' AND '{END_DATE}'
        """, 'crsp_delist')
        dl['dlstdt'] = pd.to_datetime(dl['dlstdt'])
        for col in ['dlstcd','dlret']:
            dl[col] = pd.to_numeric(dl[col], errors='coerce')
        perf = dl['dlstcd'].between(500, 599)
        dl.loc[perf & dl['dlret'].isna(), 'dlret'] = -0.30
        self.save_cache(dl, 'crsp_delist')
        return dl

    # ─── MAIN ─────────────────────────────────────────────────────────
    def run_all(self):
        """Execute full data pipeline."""
        print("=" * 80)
        print(f"SMART MONEY OPTIONS — DATA ACQUISITION")
        print(f"Sample: {START_DATE} → {END_DATE}")
        print("=" * 80)
        self.connect()

        om_link, ccm_link, ibes_link, cusip_map = self.download_links()
        crsp, crsp_names = self.download_crsp()
        option_agg, iv_spread = self.download_optionmetrics()
        opt_linked, iv_linked = self.link_options(option_agg, iv_spread, om_link, crsp_names)
        del option_agg, iv_spread; gc.collect()
        self.download_compustat(ccm_link)
        self.download_io()
        self.download_ff()
        self.download_ibes(ibes_link)
        self.download_short_interest(ccm_link)
        self.download_delist()

        # Save compact names
        self.save_cache(
            crsp_names[['permno','siccd','exchcd']].drop_duplicates('permno', keep='last'),
            'crsp_names_compact')

        print(f"\n{'='*60}")
        print(f"DATA ACQUISITION COMPLETE — {elapsed():.1f} min")
        print(f"{'='*60}")


if __name__ == '__main__':
    dl = WRDSDownloader()
    dl.run_all()
