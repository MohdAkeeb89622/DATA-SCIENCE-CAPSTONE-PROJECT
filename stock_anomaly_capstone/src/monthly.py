from __future__ import annotations
import argparse
import os
import pandas as pd
from .reporting import monthly_mini_report

def main():
    p = argparse.ArgumentParser(description="Monthly mini-report for YYYY-MM.")
    p.add_argument("--out-dir", default="outputs", help="Folder containing market_day_table.csv and daily_anomaly_card.csv")
    p.add_argument("--month", required=True, help="YYYY-MM")
    args = p.parse_args()

    mt = pd.read_csv(os.path.join(args.out_dir, "market_day_table.csv"))
    dc = pd.read_csv(os.path.join(args.out_dir, "daily_anomaly_card.csv"))

    rep = monthly_mini_report(dc, mt, args.month)
    out_path = os.path.join(args.out_dir, f"monthly_report_{args.month}.csv")
    rep.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
