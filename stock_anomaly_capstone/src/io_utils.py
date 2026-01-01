from __future__ import annotations
import os
import pandas as pd

REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _read_one_csv(path: str, ticker: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing columns {missing} in {path}")
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    df["ticker"] = ticker
    # normalize column names to snake_case
    df = df.rename(columns={
        "Adj Close": "adj_close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "Date": "date",
    })
    return df[["date","ticker","open","high","low","close","adj_close","volume"]]

def load_universe(data_dir: str, tickers: list[str]) -> pd.DataFrame:
    """Load CSVs from Kaggle dataset structure: data_dir/stocks or data_dir/etfs."""
    frames = []
    for t in tickers:
        candidates = [
            os.path.join(data_dir, "stocks", f"{t}.csv"),
            os.path.join(data_dir, "etfs", f"{t}.csv"),
            os.path.join(data_dir, f"{t}.csv"),  # allow flat layout
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if not path:
            raise FileNotFoundError(
                f"Could not find CSV for {t}. Looked in: " + ", ".join(candidates)
            )
        frames.append(_read_one_csv(path, t))
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["ticker","date"]).reset_index(drop=True)
