from __future__ import annotations
import argparse
import os
import pandas as pd

def main():
    p = argparse.ArgumentParser(description="Date query: show market status + anomalous tickers for that day.")
    p.add_argument("--out-dir", default="outputs", help="Folder containing market_day_table.csv and daily_anomaly_card.csv")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    args = p.parse_args()

    mt_path = os.path.join(args.out_dir, "market_day_table.csv")
    dc_path = os.path.join(args.out_dir, "daily_anomaly_card.csv")

    mt = pd.read_csv(mt_path)
    dc = pd.read_csv(dc_path)

    date = args.date
    mrow = mt[mt["date"] == date]
    if mrow.empty:
        print(f"No market row found for date={date}. Did you run walkforward?")
        return

    m = mrow.iloc[0].to_dict()
    print("=== Market Status ===")
    print(f"date: {m['date']}")
    print(f"market_ret: {m['market_ret']:.6f}")
    print(f"breadth: {m['breadth']:.3f}")
    print(f"market_anomaly_flag: {int(m['market_anomaly_flag'])}")

    print("\n=== Anomalous Tickers (rule-based) ===")
    rows = dc[(dc["date"] == date) & (dc["anomaly_flag"] == 1)].copy()
    if rows.empty:
        print("None")
        return
    # show key fields
    show = rows[["ticker","type","ret","ret_z","volz","range_pct","why"]].sort_values("ticker")
    print(show.to_string(index=False))

if __name__ == "__main__":
    main()
