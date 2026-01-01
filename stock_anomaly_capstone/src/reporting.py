from __future__ import annotations
import pandas as pd

def build_daily_anomaly_card(df: pd.DataFrame, method: str = "rule") -> pd.DataFrame:
    """Return the per-PDF Daily Anomaly Card columns."""
    if method == "rule":
        flag_col = "anomaly_flag_rule"
        type_col = "type_rule"
        why_col = "why_rule"
    else:
        flag_col = f"anomaly_flag_{method}"
        type_col = f"type_{method}"
        why_col = f"why_{method}"

    cols = ["date","ticker",flag_col,"ret","ret_z","volz","range_pct",type_col,why_col]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out = out.rename(columns={
        flag_col: "anomaly_flag",
        type_col: "type",
        why_col: "why",
    })
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.sort_values(["date","ticker"])

def monthly_mini_report(daily_card: pd.DataFrame, market_table: pd.DataFrame, month: str) -> pd.DataFrame:
    """Monthly mini-report: join market flag onto daily anomalies for given month (YYYY-MM)."""
    dc = daily_card.copy()
    dc["date_dt"] = pd.to_datetime(dc["date"])
    mt = market_table.copy()
    mt["date_dt"] = pd.to_datetime(mt["date"])
    mt = mt[["date_dt","market_anomaly_flag","market_ret","breadth"]]

    m = dc[dc["date_dt"].dt.strftime("%Y-%m") == month].copy()
    m = m[m["anomaly_flag"] == 1].copy()
    m = m.merge(mt, on="date_dt", how="left")

    out = m[["date","ticker","type","ret_z","volz","market_anomaly_flag","why","market_ret","breadth"]].copy()
    out = out.rename(columns={"market_anomaly_flag":"mkt_flag"})
    return out.sort_values(["date","ticker"])
