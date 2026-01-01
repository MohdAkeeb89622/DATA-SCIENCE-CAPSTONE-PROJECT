from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Windows, Thresholds

def compute_market_table(df_feat: pd.DataFrame, windows: Windows | None = None, thr: Thresholds | None = None) -> pd.DataFrame:
    if windows is None:
        windows = Windows()
    if thr is None:
        thr = Thresholds()

    # use only rows where returns exist (ret can be NaN on first day of each ticker)
    tmp = df_feat.dropna(subset=["ret"]).copy()
    # aggregate per date
    g = tmp.groupby("date")
    market_ret = g["ret"].mean()
    breadth = g.apply(lambda x: (x["ret"] > 0).mean())
    out = pd.DataFrame({"date": market_ret.index, "market_ret": market_ret.values, "breadth": breadth.values}).sort_values("date")

    # rolling threshold for |market_ret| 95th percentile using past window only
    abs_mkt = out["market_ret"].abs()
    roll_thr = abs_mkt.shift(1).rolling(window=windows.w_return, min_periods=windows.w_return).quantile(thr.market_ret_pct/100.0)
    out["market_anomaly_flag"] = ((abs_mkt > roll_thr) | (out["breadth"] < thr.market_breadth)).astype(int)

    return out
