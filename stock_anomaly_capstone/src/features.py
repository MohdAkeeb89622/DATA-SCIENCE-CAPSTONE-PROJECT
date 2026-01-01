from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Windows

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret"] = df.groupby("ticker")["adj_close"].pct_change()
    return df

def _rolling_zscore(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    # shift(1) ensures we only use past data when scoring current day
    mu = series.shift(1).rolling(window=window, min_periods=window).mean()
    sd = series.shift(1).rolling(window=window, min_periods=window).std(ddof=0)
    z = (series - mu) / sd.replace(0, np.nan)
    return z, mu, sd

def _range_percentile_per_ticker(range_series: pd.Series, window: int) -> pd.Series:
    # range_series is already (high-low)/close
    x = range_series.to_numpy(dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        start = i - window
        end = i  # up to i-1
        if start < 0:
            continue
        past = x[start:end]
        if np.any(np.isnan(past)) or np.isnan(x[i]):
            continue
        out[i] = (past < x[i]).mean() * 100.0
    return pd.Series(out, index=range_series.index)

def compute_features(df: pd.DataFrame, windows: Windows | None = None) -> pd.DataFrame:
    """Compute leakage-safe rolling features per spec."""
    if windows is None:
        windows = Windows()

    df = df.copy().sort_values(["ticker","date"]).reset_index(drop=True)
    df = add_returns(df)

    # ret_z
    df["ret_z"], df["ret_mu"], df["ret_sd"] = (np.nan, np.nan, np.nan)
    df["ret_z"] = df.groupby("ticker", group_keys=False)["ret"].apply(
        lambda s: _rolling_zscore(s, windows.w_return)[0]
    )

    # volz (log volume)
    df["log_volume"] = np.log(df["volume"].replace(0, np.nan))
    df["volz"] = df.groupby("ticker", group_keys=False)["log_volume"].apply(
        lambda s: _rolling_zscore(s, windows.w_volume)[0]
    )

    # intraday range and percentile vs past window
    df["range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["range_pct"] = df.groupby("ticker", group_keys=False)["range"].apply(
        lambda s: _range_percentile_per_ticker(s, windows.w_range)
    )

    # warm-up filter marker
    min_obs = max(windows.w_return, windows.w_volume, windows.w_range)
    df["has_history"] = df.groupby("ticker").cumcount() >= min_obs
    return df
