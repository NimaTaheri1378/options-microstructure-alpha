"""
Global configuration for the Smart Money Options project.
All paths, date ranges, and constants are defined here.
"""
import os

###############################################################################
# PATHS — Adjust BASE_DIR for your environment
###############################################################################
# Google Colab + Drive
BASE_DIR   = '/content/drive/MyDrive/Suresh2.github'
CACHE_DIR  = os.path.join(BASE_DIR, 'data', 'cache')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR  = os.path.join(OUTPUT_DIR, 'tables')
PERF_DIR   = os.path.join(OUTPUT_DIR, 'performance')

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [CACHE_DIR, OUTPUT_DIR, FIG_DIR, TABLE_DIR, PERF_DIR]:
        os.makedirs(d, exist_ok=True)

###############################################################################
# SAMPLE PERIOD
###############################################################################
START_YEAR = 1996
END_YEAR   = 2024
START_DATE = f'{START_YEAR}-01-01'
END_DATE   = f'{END_YEAR}-12-31'

###############################################################################
# FILTERS (Johnson & So 2012, standard in literature)
###############################################################################
MIN_PRICE       = 5.0          # Minimum stock price ($)
SHRCD_FILTER    = (10, 11)     # Common stocks only
EXCHCD_FILTER   = (1, 2, 3)   # NYSE, AMEX, NASDAQ
IV_MAX          = 3.0          # Max implied volatility (300%)
DELTA_MIN       = 0.1          # Min |delta| for options
DELTA_MAX       = 0.9          # Max |delta| for options
ATM_DELTA_RANGE = (0.3, 0.7)  # ATM delta range for IV spread

###############################################################################
# STRATEGY PARAMETERS
###############################################################################
N_QUINTILES     = 5            # Number of portfolio buckets
WEEKLY_FREQ     = True         # Rebalance weekly
NW_LAGS         = 6            # Newey-West lag length
MIN_OBS_FM      = 30           # Min obs per FM cross-section
ROLLING_IVOL_W  = 8            # Rolling window for IVOL (weeks)
ROLLING_BETA_W  = 52           # Rolling window for beta (weeks)
WINSORIZE_PCT   = (0.01, 0.99) # Winsorization percentiles

###############################################################################
# TRANSACTION COSTS (for realistic backtesting)
###############################################################################
TC_BPS_ONE_WAY  = 10           # 10 bps one-way transaction cost
TC_BPS_ROUND    = 2 * TC_BPS_ONE_WAY

###############################################################################
# CONTROL VARIABLES (literature-standard)
###############################################################################
BASE_CONTROLS = [
    'log_me', 'bm', 'mom', 'rev', 'ivol', 'amihud',
    'turnover', 'io_ratio', 'beta', 'max_ret', 'sir'
]

###############################################################################
# FF48 INDUSTRY MAPPING (SIC → Fama-French 48)
###############################################################################
FF48_MAPPING = [
    (100,999,1),(1000,1499,2),(1500,1799,3),(2000,2099,4),
    (2100,2199,5),(2200,2399,6),(2400,2499,7),(2500,2599,8),
    (2600,2699,9),(2700,2799,10),(2800,2899,11),(2900,2999,12),
    (3000,3099,13),(3100,3199,14),(3200,3299,15),(3300,3399,16),
    (3400,3499,17),(3500,3599,18),(3600,3699,19),(3700,3799,20),
    (3800,3899,21),(3900,3999,22),(4000,4099,23),(4100,4199,24),
    (4200,4299,25),(4400,4499,26),(4500,4599,27),(4600,4699,28),
    (4700,4799,29),(4800,4899,30),(4900,4999,31),(5000,5199,32),
    (5200,5399,33),(5400,5499,34),(5500,5599,35),(5600,5699,36),
    (5700,5799,37),(5800,5999,38),(6000,6199,39),(6200,6299,40),
    (6300,6411,41),(6500,6599,42),(6700,6799,43),(7000,7299,44),
    (7300,7399,45),(7400,7499,46),(7500,7999,47),(8000,8999,48),
]

def sic_to_ff48(sic):
    """Map SIC code to Fama-French 48 industry."""
    import pandas as pd
    if pd.isna(sic) or sic <= 0:
        return 0
    sic = int(sic)
    for lo, hi, ind in FF48_MAPPING:
        if lo <= sic <= hi:
            return ind
    return 0

###############################################################################
# DISPLAY LABELS
###############################################################################
LABELS = {
    'ret_lead1':'Ret(t+1)', 'os_ratio':'O/S Ratio', 'pc_ratio':'Put-Call',
    'iv_spread':'IV Spread', 'log_me':'log(ME)', 'bm':'B/M',
    'mom':'Momentum', 'rev':'Reversal', 'ivol':'IVOL', 'amihud':'Amihud',
    'turnover':'Turnover', 'io_ratio':'Inst. Own.', 'beta':'Beta',
    'max_ret':'MAX', 'sir':'Short Int.', 'disp':'Dispersion',
    'sue':'SUE', 'spread':'Bid-Ask', 'iv_skew':'IV Skew',
    'iv_term':'IV Term Struct.', 'abnormal_vol':'Abn. Opt. Vol',
    'vrp':'Vol Risk Prem.', 'gamma_exp':'Gamma Exp.',
}
