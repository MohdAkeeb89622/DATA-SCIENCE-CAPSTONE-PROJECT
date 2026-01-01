# Stock Market Anomaly Detection (Capstone)

This project implements the capstone spec from the provided PDF: leakage-safe rolling features,
simple anomaly flags, a date query, and a monthly mini-report. fileciteturn0file0

## What you build (per PDF)
- Features per ticker-day (computed using **past-only rolling windows**):
  - `ret_z`: return z-score using Adj Close, rolling 63-day mean/std
  - `volz`: log-volume z-score, rolling 21-day mean/std
  - `range_pct`: intraday range percentile vs prior 63 days
- Detector (rule-based + optional clustering):
  - Rule-based triggers: `|ret_z| > 2.5` OR `volz > 2.5` OR `range_pct > 95`
  - (Optional) KMeans / DBSCAN as described in the PDF
- Outputs:
  - `outputs/daily_anomaly_card.csv`
  - `outputs/market_day_table.csv`
  - `outputs/monthly_report_YYYY-MM.csv`
- CLI:
  - `python -m src.walkforward --universe QQQ,AAPL,MSFT,NVDA,AMZN,META`
  - `python -m src.query --date 2020-02-27`
  - `python -m src.monthly --month 2020-02`

## 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset (Kaggle)
Download the "stock-market-dataset" Kaggle dataset mentioned in the PDF and place it like:

```
data/
  raw/
    stocks/   (one CSV per ticker)
    etfs/     (one CSV per ETF like QQQ)
```

Each CSV should have columns: Date, Open, High, Low, Close, Adj Close, Volume.

## 3) Run (recommended flow)
### A) Build features + detect anomalies + write CSVs
```bash
python -m src.walkforward --data-dir data/raw --universe QQQ,AAPL,MSFT,NVDA,AMZN,META --out-dir outputs
```

Optional: include clustering detectors:
```bash
python -m src.walkforward --data-dir data/raw --universe QQQ,AAPL,MSFT,NVDA,AMZN,META --out-dir outputs --methods rule,kmeans,dbscan
```

### B) Query a date (prints market status + anomalous tickers)
```bash
python -m src.query --out-dir outputs --date 2020-02-27
```

### C) Monthly report
```bash
python -m src.monthly --out-dir outputs --month 2020-02
```

## Notes (important)
- Leakage-safe rolling stats: when scoring day `t`, we only use data from `[t-W, t-1]` (shifted windows). fileciteturn0file0
- Warm-up: scoring starts only after enough history exists for the largest window. fileciteturn0file0
- Splits per PDF:
  - Train=2018, Val=2019, Test=2020-Q1 (Jan–Mar). fileciteturn0file0
