# Smart Money in Options Markets

**Cross-sectional equity return prediction using options market microstructure signals, conditioned on institutional consensus.**

> 8 option-based signals × 42 strategy variants × 29 years (1996–2024) × monthly rebalancing × Carhart risk-adjustment.  
> **All results are net of 20bps round-trip transaction costs.**

---

## ⚡ Headline Results — Top 10 Strategies by Sharpe Ratio

All strategies use **monthly decile sorts** with a strict no-look-ahead timing convention: signals observed at end of month *t* predict returns realized in month *t+1*.

| # | Strategy | Ann. Return | Sharpe | Sortino | Carhart α | α t-stat | Max DD | Win Rate |
|---|----------|-------------|--------|---------|-----------|----------|--------|----------|
| 1 | **LS_high_informed_score** | 36.5% | 1.63 | 3.30 | 32.4% | 8.91 | −19.5% | 66.9% |
| 2 | **LS_high_pc_ratio** | 34.7% | 1.63 | 3.47 | 32.5% | 8.86 | −15.6% | 65.7% |
| 3 | **LS_low_pc_ratio** | 31.7% | 1.56 | 3.01 | 29.9% | 7.95 | −20.8% | 65.7% |
| 4 | **VRP D1-D10 (EW)** | 19.0% | 1.53 | 2.73 | 17.2% | 7.84 | −14.9% | 69.7% |
| 5 | **LS_low_abnormal_vol** | 30.7% | 1.51 | 3.48 | 28.1% | 6.72 | −26.5% | 65.3% |
| 6 | **LS_high_sentiment** | 28.4% | 1.50 | 2.89 | 26.1% | 7.29 | −25.2% | 66.0% |
| 7 | **LS_high_vrp** | 51.5% | 1.49 | 2.85 | 48.4% | 9.22 | −28.6% | 67.4% |
| 8 | **IV Spread D1-D10 (EW)** | 14.4% | 1.37 | 2.52 | 12.0% | 5.00 | −12.8% | 66.6% |
| 9 | **LS_low_ivol** | 22.1% | 1.37 | 2.14 | 19.6% | 7.35 | −20.9% | 66.0% |
| 10 | **LS_low_os_ratio** | 26.6% | 1.34 | 2.36 | 23.5% | 6.85 | −35.4% | 63.7% |

> **All 10 strategies carry Carhart alpha t-statistics above 5.0**, indicating strong statistical significance after controlling for market, size, value, and momentum factors.

**Reading the table:** Strategies prefixed `LS_high_*` / `LS_low_*` are **double-sort** strategies (Tier 3) — they condition jointly on both a signal and the consensus measure and hold ~70 stocks per leg. Strategies named `*_D1D10_*` are **single-signal decile sorts** (Tier 1) running on the full options universe (~690 stocks per leg). Both are reported together ranked by Sharpe for comparability; see [Strategy Types](#-strategy-types-explained) for the distinction.

---

## 🏆 Best Risk-Adjusted Strategies (Practitioner Picks)

These strategies combine high Sharpe ratios with manageable drawdowns and survive transaction costs:

### 1. VRP Long/Short (Sharpe 1.53, α = 17.2%)
The **volatility risk premium** — buying stocks where implied vol exceeds realized vol (high VRP, D10) and shorting the opposite (low VRP, D1). This is the cleanest single-signal strategy: highest win rate (69.7%), lowest max drawdown (−14.9%), and a massive *t* = 7.84.

### 2. IV Spread Long/Short (Sharpe 1.37, α = 12.0%)
Long stocks where call IV exceeds put IV (bullish informed flow) and short the reverse. Max drawdown of only −12.8% makes this the most defensive strategy in the set.

### 3. LS_high_pc_ratio (Sharpe 1.63, α = 32.5%)
A **double-sort** strategy: among stocks with the highest put-call ratios (D10), go long those with low institutional consensus (contrarian) and short those with high consensus. Max DD = −15.6%.
> ⚠️ *Concentrated portfolio (~70 stocks per leg). Returns are academically robust but require careful liquidity assessment before scaling.*

### 4. LS_high_informed_score (Sharpe 1.63, α = 32.4%)
Same double-sort logic applied to the composite informed trading score. Strong Sortino (3.30) indicates minimal left-tail risk.
> ⚠️ *Concentrated portfolio (~70 stocks per leg). The high alpha reflects both genuine signal interaction and the inherent concentration of double-sort portfolios.*

### 5. PC Ratio Long/Short (Sharpe 1.17, α = 6.9%)
Simple decile sort on put-call ratio. Win rate of 68.0% — the highest among single-signal strategies.

---

## 📊 Strategy Types Explained

### Tier 1: Single-Signal Decile Sorts
For each signal, stocks are sorted into **deciles (D1–D10)** each month. A long/short portfolio goes long the extreme decile predicted to outperform and short the opposite. Equal-weighted (EW) and value-weighted (VW) variants are tested.

### Tier 2: Consensus-Conditioned Sorts
Same as Tier 1, but restricted to stocks where **institutional predictor consensus** (from Chen & Zimmerman, 2022) is above the monthly cross-sectional median. This filters for stocks where institutional forecasters agree on direction, testing whether options signals add value beyond consensus.

### Tier 3: Double-Sort (Consensus × Signal)
Stocks are **independently sorted** into deciles on both the consensus measure (μ) and an options signal. The strategy goes:
- **Long**: D1(consensus) × D10(signal) — contrarian positions in high-signal stocks
- **Short**: D10(consensus) × D10(signal) — consensus-aligned positions in the same signal bucket

This tests whether **disagreeing with institutional consensus** when options signals are strong generates alpha. Both "within HIGH signal" and "within LOW signal" variants are reported.

### Tier 4: Difference-in-Differences (DiD)
A four-corner portfolio that isolates the **interaction effect** between signal strength and consensus level:

```
DiD = [D1(μ)×D10(sig) − D1(μ)×D1(sig)] − [D10(μ)×D10(sig) − D10(μ)×D1(sig)]
```

Tests whether the options signal premium is **differentially stronger** among contrarian (low-consensus) stocks.

---

## 🔬 Signals (8)

| Signal | Description | Reference |
|--------|-------------|-----------|
| **O/S Ratio** | Option volume / stock volume. High → bearish informed trading. | Johnson & So (2012) |
| **IV Spread** | Call IV − Put IV (ATM). Positive → bullish private info. | Cremers & Weinbaum (2010) |
| **Put-Call Ratio** | Put volume / total option volume. High → bearish. | Pan & Poteshman (2006) |
| **IVOL** | Idiosyncratic volatility (rolling return std). High → lottery demand. | Ang et al. (2006) |
| **Abnormal Volume** | Current option volume / 4-week trailing average. | Cao, Chen & Griffin (2005) |
| **VRP** | Implied volatility − realized volatility. | Bollerslev, Tauchen & Zhou (2009) |
| **Informed Score** | Rank-average composite of O/S + IV spread + skew. | Composite |
| **Sentiment** | Ensemble of all signal z-scores. | Composite |

---

## 🔐 Predictor Consensus Measure

### What it is
The **predictor consensus** (μ) aggregates agreement across institutional forecasters using the Chen & Zimmerman (2022) cross-sectional predictor database. For each stock-month, μ measures the degree to which a broad set of academic return predictors point in the same direction — a high μ means the institutional consensus strongly agrees on expected return; a low μ means forecasters disagree or diverge.

### Why it matters — the core research finding
The central result of this project is **not** that options signals predict returns (that is well-established). It is that **options signals are substantially more informative when they conflict with institutional consensus**:

- Among high-consensus stocks (μ = D10), options signals add modest incremental alpha (~5–12%)
- Among low-consensus stocks (μ = D1), the same options signals generate **3–5× larger alpha** (28–48%)

This is consistent with the hypothesis that **options-informed traders possess private information that is most valuable precisely when it contradicts the prevailing institutional view.** When everyone agrees, the option market's edge is already priced. When the option market disagrees with the consensus, that disagreement is a stronger signal.

This interaction — options informativeness × consensus divergence — is the novel contribution of this research relative to the existing options microstructure literature.

> **The consensus variable is proprietary and not included in this repository.**  
> For access to the construction methodology or data, please contact:  
> 📧 **nimataheri1378@gmail.com**

---

## ✅ Look-Ahead Bias Audit

The backtest pipeline has been rigorously audited for information leakage:

| Component | Timing | Status |
|-----------|--------|--------|
| **Signal observation** | End-of-month *t* (last weekly obs) | ✅ No leak |
| **Return realization** | Month *t+1* compounded return | ✅ Forward-looking via `shift(-1)` |
| **Consensus predictor** | Month *t* (CZ convention: predictor at *t* → return at *t+1*) | ✅ Matches CZ timing |
| **Decile sorting** | Cross-sectional within month *t* only | ✅ No future data |
| **FF factor alignment** | Monthly factors matched to return month (*t+1*) | ✅ Correctly timed |
| **Strategy direction** | Ex-ante hardcoded (no auto-flip on realized returns) | ✅ No data snooping |

---

## 🗂 Data Sources

| Source | Description |
|--------|-------------|
| **OptionMetrics** (WRDS) | Daily option-level data, 1996–2024. ~4B rows processed year-by-year. |
| **CRSP** | Daily stock returns, prices, volumes. Common stocks only (share codes 10, 11). |
| **Compustat** | Annual fundamentals (book equity, Fama-French definition). |
| **Thomson 13F** | Quarterly institutional ownership. |
| **Fama-French** | Daily factors (MKT-RF, SMB, HML, UMD, RF) aggregated to monthly. |

All data accessed via [WRDS](https://wrds-www.wharton.upenn.edu/). Intermediate results cached as parquet files.

---

## 🚀 Quick Start (Google Colab)

```bash
# 1. Upload project to Google Drive under /Suresh2.github/smart-money-options/

# 2. Run data download (requires WRDS account)
# Execute: notebooks/00_setup_and_download.py

# 3. Build analysis panel
# Execute: analysis/variable_construction.py

# 4. Run all 42 strategies
# Execute: notebooks/03_decile_backtest.py
# → Outputs: results/performance/all_strategies_decile_summary.csv
# → Outputs: results/figures/top5_decile_cumulative.png
```

---

## 📁 Project Structure

```
smart-money-options/
├── config/settings.py              # Constants, paths, parameters
├── data/download.py                # WRDS data acquisition
├── signals/                        # 8 signal implementations
│   ├── base.py                     # Abstract signal class
│   ├── option_stock_volume.py      # O/S ratio
│   ├── iv_spread.py                # IV spread
│   ├── ivol.py                     # Idiosyncratic volatility
│   ├── abnormal_option_volume.py   # Abnormal option volume
│   ├── volatility_risk_premium.py  # VRP
│   ├── informed_trading_score.py   # Composite informed score
│   └── sentiment_composite.py      # Sentiment ensemble
├── strategies/                     # Strategy engine
│   ├── base.py                     # Backtest engine + performance metrics
│   └── cross_sectional.py          # Decile L/S + consensus conditioning
├── analysis/                       # Econometric analysis
│   ├── variable_construction.py    # Weekly → monthly panel builder
│   ├── fama_macbeth.py             # FM regressions with NW
│   └── portfolio_sorts.py          # Decile sorts with alphas
├── visualization/tearsheet.py      # Publication-quality charts
├── notebooks/                      # Colab-ready execution scripts
│   ├── 00_setup_and_download.py    # Data acquisition
│   └── 03_decile_backtest.py       # Main backtest runner (42 strategies)
└── results/                        # Output figures, tables, CSVs
```

---

## 📈 Full Results (42 Strategies, Sorted by Sharpe)

<details>
<summary>Click to expand full performance table</summary>

| Strategy | Ann. Return | Sharpe | Sortino | Max DD | t-stat | Win % | Carhart α | α t-stat | After TC |
|----------|-------------|--------|---------|--------|--------|-------|-----------|----------|----------|
| LS_high_informed_score | 36.5% | 1.63 | 3.30 | −19.5% | 10.64 | 66.9% | 32.4% | 8.91 | 34.1% |
| LS_high_pc_ratio | 34.7% | 1.63 | 3.47 | −15.6% | 10.66 | 65.7% | 32.5% | 8.86 | 32.3% |
| LS_low_pc_ratio | 31.7% | 1.56 | 3.01 | −20.8% | 8.49 | 65.7% | 29.9% | 7.95 | 29.3% |
| vrp_D1D10_EW | 19.0% | 1.53 | 2.73 | −14.9% | 8.87 | 69.7% | 17.2% | 7.84 | 16.6% |
| LS_low_abnormal_vol | 30.7% | 1.51 | 3.48 | −26.5% | 7.93 | 65.3% | 28.1% | 6.72 | 28.3% |
| LS_high_sentiment | 28.4% | 1.50 | 2.89 | −25.2% | 8.34 | 66.0% | 26.1% | 7.29 | 26.0% |
| LS_high_vrp | 51.5% | 1.49 | 2.85 | −28.6% | 9.64 | 67.4% | 48.4% | 9.22 | 49.1% |
| iv_spread_D1D10_EW | 14.4% | 1.37 | 2.52 | −12.8% | 5.64 | 66.6% | 12.0% | 5.00 | 12.0% |
| LS_low_ivol | 22.1% | 1.37 | 2.14 | −20.9% | 8.65 | 66.0% | 19.6% | 7.35 | 19.7% |
| LS_low_os_ratio | 26.6% | 1.34 | 2.36 | −35.4% | 8.90 | 63.7% | 23.5% | 6.85 | 24.2% |
| LS_low_sentiment | 45.4% | 1.33 | 2.58 | −40.1% | 8.41 | 62.2% | 42.5% | 7.71 | 43.0% |
| LS_high_iv_spread | 39.9% | 1.29 | 2.39 | −34.7% | 8.71 | 64.3% | 36.0% | 7.20 | 37.5% |
| LS_high_abnormal_vol | 28.1% | 1.23 | 2.21 | −48.9% | 7.50 | 64.5% | 27.6% | 7.00 | 25.7% |
| LS_low_iv_spread | 31.1% | 1.19 | 2.41 | −32.6% | 7.42 | 62.2% | 29.9% | 7.26 | 28.7% |
| pc_ratio_D1D10_EW | 9.1% | 1.17 | 1.50 | −23.3% | 5.57 | 68.0% | 6.9% | 3.77 | 6.7% |
| LS_low_informed_score | 33.1% | 1.14 | 2.05 | −37.2% | 6.62 | 63.1% | 31.0% | 6.28 | 30.7% |
| iv_spread_D1D10_consensus | 11.8% | 1.04 | 1.72 | −19.4% | 4.37 | 60.8% | 9.9% | 3.91 | 9.4% |
| LS_high_ivol | 42.2% | 1.04 | 1.77 | −66.0% | 7.23 | 62.0% | 40.2% | 6.72 | 39.8% |
| LS_high_os_ratio | 31.8% | 0.95 | 1.38 | −70.6% | 5.22 | 61.7% | 28.4% | 4.18 | 29.4% |
| DiD_vrp | 31.2% | 0.94 | 1.53 | −43.1% | 5.06 | 61.4% | 27.1% | 4.62 | 28.8% |
| ivol_D1D10_EW | 23.2% | 0.88 | 1.43 | −72.3% | 4.31 | 60.2% | 22.1% | 3.74 | 20.8% |
| informed_score_D1D10_consensus | 13.1% | 0.85 | 1.14 | −46.9% | 3.95 | 65.7% | 11.4% | 3.59 | 10.7% |
| informed_score_D1D10_EW | 11.9% | 0.78 | 1.03 | −46.7% | 3.23 | 62.5% | 10.6% | 3.02 | 9.5% |
| os_ratio_D1D10_consensus | 12.5% | 0.75 | 0.96 | −65.8% | 3.46 | 66.3% | 11.3% | 3.07 | 10.1% |
| pc_ratio_D1D10_consensus | 7.8% | 0.75 | 0.97 | −29.7% | 3.80 | 64.0% | 5.9% | 2.70 | 5.4% |
| iv_spread_D1D10_VW | 12.1% | 0.74 | 1.18 | −33.2% | 4.11 | 58.2% | 9.4% | 3.07 | 9.7% |
| LS_low_vrp | 20.3% | 0.69 | 1.09 | −63.9% | 4.55 | 60.8% | 19.1% | 4.40 | 17.9% |
| ivol_D1D10_consensus | 16.7% | 0.68 | 1.00 | −78.2% | 3.13 | 57.9% | 16.1% | 2.76 | 14.3% |
| DiD_sentiment | 17.0% | 0.56 | 0.94 | −60.6% | 3.26 | 59.1% | 14.2% | 2.68 | 14.6% |
| DiD_ivol | 20.1% | 0.55 | 0.88 | −84.9% | 3.53 | 59.1% | 18.4% | 3.21 | 17.7% |
| os_ratio_D1D10_EW | 7.1% | 0.45 | 0.57 | −69.0% | 1.98 | 57.6% | 6.0% | 1.70 | 4.7% |
| DiD_iv_spread | 8.8% | 0.31 | 0.47 | −60.0% | 1.80 | 53.9% | 3.9% | 0.74 | 6.4% |
| ensemble_consensus | 5.5% | 0.29 | 0.36 | −66.9% | 1.33 | 57.8% | 3.2% | 0.80 | 3.1% |
| DiD_pc_ratio | 5.2% | 0.26 | 0.42 | −46.0% | 1.40 | 47.8% | 2.6% | 0.65 | 2.8% |
| sentiment_D1D10_consensus | 4.0% | 0.22 | 0.27 | −70.4% | 1.00 | 57.1% | 1.6% | 0.42 | 1.6% |
| DiD_os_ratio | 5.2% | 0.17 | 0.23 | −89.9% | 0.81 | 51.6% | 2.8% | 0.39 | 2.8% |
| os_ratio_D1D10_VW | 2.6% | 0.17 | 0.23 | −75.7% | 0.69 | 54.5% | 1.8% | 0.47 | 0.2% |
| DiD_informed_score | 3.3% | 0.13 | 0.21 | −56.2% | 0.75 | 50.7% | −0.8% | −0.18 | 0.9% |
| DiD_abnormal_vol | 2.5% | 0.12 | 0.20 | −70.9% | 0.64 | 49.7% | −1.7% | −0.37 | 0.1% |
| ensemble | 1.8% | 0.10 | 0.13 | −73.5% | 0.48 | 49.7% | 0.1% | 0.04 | −0.6% |
| sentiment_D1D10_EW | −1.3% | −0.07 | −0.09 | −78.5% | −0.34 | 48.1% | −3.0% | −0.84 | −3.7% |
| abnormal_vol_D1D10_EW | −0.8% | −0.10 | −0.16 | −42.4% | −0.47 | 47.7% | −2.4% | −1.44 | −3.2% |

</details>

---

## 📐 Methodology

### Timing Convention
Following Chen & Zimmerman (2022): predictors observed at month *t* → sorted into decile portfolios → held through month *t+1*. Weekly data is aggregated to monthly by taking the last observation of each stock-month.

### Risk Adjustment
- **CAPM**, **Fama-French 3-Factor**, and **Carhart 4-Factor** alphas reported
- All *t*-statistics use **Newey-West HAC** standard errors (6 lags)

### Transaction Costs
All strategies report returns net of **20bps round-trip** (10bps each way per monthly rebalance).

### Win Rate (Batting Average)
Win Rate = **% of months where the L/S portfolio return > 0**. This is a strategy-level metric — a month "wins" if the aggregate long book outperformed the aggregate short book that calendar month, regardless of magnitude. Random chance = 50%. This metric is also known as the *batting average* in practitioner literature.

### Controls (16 variables)
Size, B/M, momentum (12-month skip 1), reversal, IVOL, Amihud illiquidity, turnover, institutional ownership, beta, MAX, analyst dispersion, SUE, bid-ask spread, FF48 industry, short interest ratio, high-constraint flag.

---

## 📚 References

- Ang, A., Hodrick, R., Xing, Y., & Zhang, X. (2006). The cross-section of volatility and expected returns. *Journal of Finance*.
- Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*.
- Cao, C., Chen, Z., & Griffin, J. (2005). Informational content of option volume prior to takeovers. *Journal of Business*.
- Chen, A. & Zimmerman, T. (2022). Open source cross-sectional asset pricing. *Critical Finance Review*.
- Cremers, M. & Weinbaum, D. (2010). Deviations from put-call parity and stock return predictability. *JFQA*.
- Johnson, T. & So, E. (2012). The option to stock volume ratio and future returns. *Journal of Financial Economics*.
- Pan, J. & Poteshman, A. (2006). The information in option volume for future stock prices. *Review of Financial Studies*.
- Xing, Y., Zhang, X., & Zhao, R. (2010). What does the individual option volatility smirk tell us about future equity returns? *JFQA*.

---

## 📬 Contact

**Nima Taheri** — nimataheri1378@gmail.com

For questions about the predictor consensus measure, data access, or collaboration inquiries.

---

## License

MIT
