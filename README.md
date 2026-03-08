# Stock Market Anomaly Detection (Capstone)

This project implements the capstone spec from the provided PDF: leakage-safe rolling features,
simple anomaly flags, a date query, and a monthly mini-report.

## Project Structure

```
DATA-SCIENCE-CAPSTONE-PROJECT/
├── README.md
├── requirements.txt
├── capstone_report.md              # Detailed write-up with results & tables
├── capstone_report.pdf             # PDF version of the report
├── Capstone_Project_(...).ipynb    # Jupyter notebook (EDA + analysis)
├── capstone project anamoly detection.pdf   # Original assignment spec
└── stock_anomaly_capstone/
    ├── src/
    │   ├── __init__.py
    │   ├── config.py               # Windows, Thresholds, DEFAULT_UNIVERSE
    │   ├── features.py             # Leakage-safe rolling feature engineering
    │   ├── detectors_rule.py       # Rule-based anomaly detector
    │   ├── detectors_kmeans.py     # KMeans clustering detector (optional)
    │   ├── detectors_dbscan.py     # DBSCAN clustering detector (optional)
    │   ├── market.py               # Market-day context (breadth, market_ret)
    │   ├── reporting.py            # Daily anomaly card + monthly mini-report
    │   ├── io_utils.py             # CSV loading from Kaggle dataset layout
    │   ├── walkforward.py          # Main pipeline CLI entry point
    │   ├── query.py                # Date query CLI
    │   └── monthly.py              # Monthly report CLI
    ├── data/                       # (not tracked in git)
    │   └── raw/
    │       ├── stocks/             # One CSV per stock ticker
    │       └── etfs/               # One CSV per ETF (e.g. QQQ)
    └── outputs/                    # (generated, not tracked in git)
        ├── daily_anomaly_card.csv
        ├── features_and_flags.csv
        └── market_day_table.csv
```

## What It Does

- **Features per ticker-day** (computed using **past-only rolling windows**):
  - `ret_z`: return z-score using Adj Close, rolling 63-day mean/std
  - `volz`: log-volume z-score, rolling 21-day mean/std
  - `range_pct`: intraday range percentile vs prior 63 days
- **Detector** (rule-based + optional clustering):
  - Rule-based triggers: `|ret_z| > 2.5` OR `volz > 2.5` OR `range_pct > 95`
  - (Optional) KMeans / DBSCAN as described in the PDF
- **Outputs**:
  - `outputs/daily_anomaly_card.csv`
  - `outputs/market_day_table.csv`
  - `outputs/monthly_report_YYYY-MM.csv`

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Dataset (Kaggle)

Download the [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset) from Kaggle and place it like:

```
stock_anomaly_capstone/data/raw/
  stocks/   (one CSV per ticker)
  etfs/     (one CSV per ETF like QQQ)
```

Each CSV should have columns: `Date, Open, High, Low, Close, Adj Close, Volume`.

## 3) Run

All commands should be run from the `stock_anomaly_capstone/` directory:

```bash
cd stock_anomaly_capstone
```

### A) Build features + detect anomalies + write CSVs

```bash
python -m src.walkforward --data-dir data/raw --universe QQQ,AAPL,MSFT,NVDA,AMZN,META --out-dir outputs
```

Optional — include clustering detectors:

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

## Notes

- **Leakage-safe** rolling stats: when scoring day `t`, we only use data from `[t-W, t-1]` (shifted windows).
- **Warm-up**: scoring starts only after enough history exists for the largest window (63 days).
- **Splits per PDF**:
  - Train = 2018, Validation = 2019, Test = 2020-Q1 (Jan–Mar).

## License

This project is developed as a capstone assignment for educational purposes.
