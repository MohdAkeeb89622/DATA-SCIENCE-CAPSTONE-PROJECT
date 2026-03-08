# Stock Market Anomaly Detection – Capstone Report

## 1. Goal
Detect unusual **market days** and **stock days** using only daily price and volume data. “Unusual” means extreme behavior relative to recent history (rolling windows; no look-ahead).

## 2. Dataset
- Source: Kaggle “stock-market-dataset”
- Files: one CSV per ticker with columns: Date, Open, High, Low, Close, Adj Close, Volume
- Universe used: QQQ, AAPL, MSFT, NVDA, AMZN, META

## 3. Splits (as required)
- Train: 2018
- Validation: 2019
- Test: 2020 Q1 (Jan–Mar)

All results below focus on **2018-01-02 to 2020-03-31**.

## 4. Feature Engineering (leakage-safe)
We compute features per ticker/day using **past-only rolling windows**:

- Return (Adj Close):
  - r[t] = (AdjClose[t] - AdjClose[t-1]) / AdjClose[t-1]
- Return z-score (63 trading days):
  - ret_z[t] = (r[t] - mean(r[t-63..t-1])) / std(r[t-63..t-1])
- Log-volume z-score (21 days):
  - volz[t] = (log(Vol[t]) - mean(log(Vol[t-21..t-1]))) / std(log(Vol[t-21..t-1]))
- Intraday range percentile (63 days):
  - range = (High - Low) / Close
  - range_pct[t] = percentile of range[t] compared to range[t-63..t-1]

Warm-up: a day is scored only after at least `max(63, 21, 63)=63` prior observations exist.

## 5. Rule-based Detector (stock-day)
A stock-day is flagged as unusual if **any trigger fires**:
- |ret_z| > 2.5  OR  volz > 2.5  OR  range_pct > 95

Type label:
- crash: r[t] < 0 and |ret_z| > 2.5
- spike: r[t] > 0 and |ret_z| > 2.5
- volume_shock: volz > 2.5 (can co-exist with crash/spike)

## 6. Market-day Context
Using the same day across the universe:
- market_ret[t] = mean of r_i[t] across tickers available that day
- breadth[t] = fraction of tickers with r_i[t] > 0

A market-day is flagged if:
- |market_ret[t]| is above its own rolling 95th percentile (past 63 days), OR
- breadth[t] < 0.3

## 7. Outputs Generated
- `outputs/daily_anomaly_card.csv`
- `outputs/market_day_table.csv`
- `outputs/features_and_flags.csv`

## 8. Results (2018-01 to 2020-03)
**Counts**
- Total scored rows (all tickers): 3,390
- Stock-day anomalies flagged: 436
- Market days: 565
- Market-day anomalies flagged: 192
- Overlap (market-day flagged AND at least one stock anomaly that day): 102

### 8.1 Anomalies by ticker
| ticker | anomaly_days |
| --- | --- |
| AMZN | 77 |
| META | 76 |
| NVDA | 72 |
| AAPL | 71 |
| MSFT | 70 |
| QQQ | 70 |

### 8.2 Anomalies by type
| type | count |
| --- | --- |
| range_spike | 156 |
| volume_shock | 103 |
| crash | 66 |
| spike | 58 |
| crash + volume_shock | 39 |
| spike + volume_shock | 14 |

### 8.3 Top 10 most severe flagged stock-days (example evidence)
| date | ticker | type | ret | ret_z | volz | range_pct | severity | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2018-07-26 | META | crash + volume_shock | -0.19 | -12.091 | 6.525 | 100.0 | 99.975 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2019-06-03 | META | crash + volume_shock | -0.075 | -4.678 | 6.374 | 100.0 | 99.886 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-02-05 | QQQ | crash + volume_shock | -0.039 | -5.765 | 3.922 | 100.0 | 99.734 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-10-10 | QQQ | crash + volume_shock | -0.044 | -5.782 | 3.401 | 100.0 | 99.613 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-02-05 | NVDA | crash + volume_shock | -0.085 | -4.073 | 3.793 | 100.0 | 99.596 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-02-02 | AAPL | crash + volume_shock | -0.043 | -4.237 | 3.461 | 100.0 | 99.543 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-10-10 | AMZN | crash + volume_shock | -0.062 | -4.616 | 3.292 | 100.0 | 99.522 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-10-10 | MSFT | crash + volume_shock | -0.054 | -5.113 | 3.114 | 100.0 | 99.473 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2019-06-03 | AMZN | crash + volume_shock | -0.046 | -3.472 | 3.367 | 100.0 | 99.383 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |
| 2018-03-27 | NVDA | crash + volume_shock | -0.078 | -3.092 | 3.683 | 100.0 | 99.355 | |ret_z| > 2.5; volz > 2.5; range_pct > 95 |

## 9. Figures
The following plots are generated in the Jupyter notebook (`Capstone_Project_(Stock_Market_Anomaly_Detection).ipynb`):
- Anomalies by ticker (bar chart)
- Anomalies per month (time series)
- Market anomalies per month (time series)

## 10. How to Run
All commands below should be run from the `stock_anomaly_capstone/` directory:
```bash
cd stock_anomaly_capstone
python -m src.walkforward --data-dir data/raw --universe QQQ,AAPL,MSFT,NVDA,AMZN,META --out-dir outputs
python -m src.query --out-dir outputs --date 2020-02-27
python -m src.monthly --out-dir outputs --month 2020-02
```

## 11. Limitations & Improvements
- Rule thresholds are fixed; they can be tuned using validation year (2019).
- Events like earnings or macro-news are not modeled (intentionally).
- Market breadth with a small universe can be noisy; increasing tickers can stabilize it.
- Optional next step: clustering (KMeans/DBSCAN) on [ret_z, volz, range_pct] for unsupervised outlier detection.

