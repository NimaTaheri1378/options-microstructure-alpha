"""
Colab Setup & Full Pipeline Runner.
Run this in Google Colab to:
1. Install dependencies
2. Mount Google Drive
3. Copy cached data from Research.Suresh2 → Suresh2.github
4. Run data download (or use cache)
5. Build analysis panel
6. Run all strategies
7. Generate figures and tearsheets
"""
# ── 1. Install ────────────────────────────────────────────────────────────
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                       'wrds', 'linearmodels', 'pyarrow', 'plotly', 'scikit-learn'])

# ── 2. Mount Drive ────────────────────────────────────────────────────────
from google.colab import drive
try:
    drive.mount('/content/drive')
except ValueError:
    pass  # already mounted

import os, shutil

# ── 3. Setup directories & copy cache ─────────────────────────────────────
GITHUB_DIR = '/content/drive/MyDrive/Suresh2.github'
OLD_CACHE  = '/content/drive/MyDrive/Research.Suresh2/inputs/cache'
NEW_CACHE  = os.path.join(GITHUB_DIR, 'data', 'cache')

os.makedirs(NEW_CACHE, exist_ok=True)
os.makedirs(os.path.join(GITHUB_DIR, 'results', 'figures'), exist_ok=True)
os.makedirs(os.path.join(GITHUB_DIR, 'results', 'tables'), exist_ok=True)
os.makedirs(os.path.join(GITHUB_DIR, 'results', 'performance'), exist_ok=True)

# Copy cached parquet files from old project
if os.path.exists(OLD_CACHE):
    existing = os.listdir(NEW_CACHE) if os.path.exists(NEW_CACHE) else []
    old_files = [f for f in os.listdir(OLD_CACHE) if f.endswith('.parquet')]
    copied = 0
    for f in old_files:
        if f not in existing:
            shutil.copy2(os.path.join(OLD_CACHE, f), os.path.join(NEW_CACHE, f))
            copied += 1
    print(f"✓ Copied {copied} new cache files ({len(old_files)} total in old cache)")
else:
    print("⚠ Old cache not found — will download from WRDS")

# List cache contents
print("\nCache files:")
for f in sorted(os.listdir(NEW_CACHE)):
    sz = os.path.getsize(os.path.join(NEW_CACHE, f)) / 1024
    print(f"  {f:<40s} {sz:>10,.0f} KB")

# ── 4. Add project to path ────────────────────────────────────────────────
PROJECT_DIR = os.path.join(GITHUB_DIR, 'smart-money-options')
if not os.path.exists(PROJECT_DIR):
    print(f"\n⚠ Project code not found at {PROJECT_DIR}")
    print("  Upload the smart-money-options/ folder to your Drive first.")
else:
    sys.path.insert(0, PROJECT_DIR)
    print(f"\n✓ Project path: {PROJECT_DIR}")

# ── 5. Verify ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
print(f"""
Next steps:
  1. If cache is empty, run data download:
       from data.download import WRDSDownloader
       dl = WRDSDownloader()
       dl.run_all()

  2. Build analysis panel:
       from analysis.variable_construction import build_panel
       panel = build_panel(
           consensus_path='/content/drive/MyDrive/Suresh2.github/consensus.csv'
       )

  3. Run strategies:
       from strategies.cross_sectional import CrossSectionalStrategy
       strat = CrossSectionalStrategy('os_ratio')
       ff_wk = panel[['week','mktrf','smb','hml','umd','rf']].drop_duplicates('week')
       results = strat.backtest(panel, ff_wk)
       print(strat.tearsheet())

  4. Run ALL strategies (see notebooks/02_strategy_backtest.py)
""")
