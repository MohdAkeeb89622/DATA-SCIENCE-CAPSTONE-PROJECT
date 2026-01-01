from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_UNIVERSE, Windows, Thresholds
from .io_utils import load_universe
from .features import compute_features
from .detectors_rule import detect_rule_based
from .market import compute_market_table
from .reporting import build_daily_anomaly_card

# Optional clustering
from .detectors_kmeans import fit_kmeans_train, kmeans_distance_to_centroid, per_cluster_thresholds, flag_kmeans
from .detectors_dbscan import fit_dbscan

def _parse_methods(s: str) -> list[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser(description="Compute features + detect anomalies + write required CSVs.")
    p.add_argument("--data-dir", default="data/raw", help="Folder containing stocks/ and etfs/ subfolders.")
    p.add_argument("--out-dir", default="outputs", help="Output folder for CSVs.")
    p.add_argument("--universe", default=",".join(DEFAULT_UNIVERSE), help="Comma-separated tickers.")
    p.add_argument("--methods", default="rule", help="Comma-separated: rule,kmeans,dbscan (optional).")
    p.add_argument("--k", type=int, default=8, help="KMeans k (optional).")
    p.add_argument("--q", type=float, default=97.5, help="KMeans cluster distance percentile threshold (optional).")
    p.add_argument("--eps", type=float, default=0.9, help="DBSCAN eps (optional starting point).")
    p.add_argument("--min-samples", type=int, default=15, help="DBSCAN min_samples.")
    args = p.parse_args()

    universe = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    methods = _parse_methods(args.methods)

    os.makedirs(args.out_dir, exist_ok=True)

    raw = load_universe(args.data_dir, universe)
    feat = compute_features(raw, windows=Windows())

    # Market table is computed from features (uses per-ticker returns)
    market_table = compute_market_table(feat, windows=Windows(), thr=Thresholds())
    market_table.to_csv(os.path.join(args.out_dir, "market_day_table.csv"), index=False)

    # Rule-based detector (required)
    det = detect_rule_based(feat, thr=Thresholds())

    # Start with rule outputs as baseline
    out_df = det.copy()

    # ---- Optional clustering detectors (per PDF methodology section) ----
    if any(m in methods for m in ["kmeans","dbscan"]):
        # build design matrix using only non-null feature rows
        Xdf = out_df.dropna(subset=["ret_z","volz","range_pct"]).copy()
        Xdf["date"] = pd.to_datetime(Xdf["date"])
        Xdf["year"] = Xdf["date"].dt.year

        # Split: Train=2018, Val=2019, Test=2020-Q1 (Jan–Mar)
        train_mask = Xdf["date"].dt.strftime("%Y") == "2018"
        val_mask   = Xdf["date"].dt.strftime("%Y") == "2019"
        test_mask  = (Xdf["date"] >= "2020-01-01") & (Xdf["date"] <= "2020-03-31")

        feats = Xdf[["ret_z","volz","range_pct"]].to_numpy(dtype=float)
        scaler = StandardScaler()
        scaler.fit(feats[train_mask.values])
        X_scaled = scaler.transform(feats)

        # KMeans
        if "kmeans" in methods:
            km = fit_kmeans_train(X_scaled[train_mask.values], k=args.k)
            train_labels, train_d = kmeans_distance_to_centroid(km, X_scaled[train_mask.values])
            thr_by_cluster = per_cluster_thresholds(train_labels, train_d, q=args.q)

            labels_all, d_all = kmeans_distance_to_centroid(km, X_scaled)
            flags_all = flag_kmeans(labels_all, d_all, thr_by_cluster)

            Xdf["kmeans_cluster"] = labels_all
            Xdf["kmeans_dist"] = d_all
            Xdf["anomaly_flag_kmeans"] = flags_all

            # Simple "why" and "type" for clustering output (still use rule labels for crash/spike direction)
            Xdf["why_kmeans"] = np.where(flags_all==1, f"dist > cluster_p{args.q}", "")
            Xdf["type_kmeans"] = np.where(flags_all==1, Xdf["type_rule"], "")

        # DBSCAN (simple monthly walk-forward on expanding window)
        if "dbscan" in methods:
            Xdf = Xdf.sort_values("date").reset_index(drop=True)
            flags = np.zeros(len(Xdf), dtype=int)
            labels = np.full(len(Xdf), -999, dtype=int)

            # score months in Val+Test; fit on expanding history up to prior day
            score_mask = val_mask | test_mask
            months = sorted(Xdf.loc[score_mask, "date"].dt.to_period("M").unique().tolist())
            for m in months:
                month_mask = (Xdf["date"].dt.to_period("M") == m) & score_mask
                if not month_mask.any():
                    continue
                first_day = Xdf.loc[month_mask, "date"].min()
                hist_mask = Xdf["date"] < first_day
                if hist_mask.sum() < 200:
                    # need some history for DBSCAN to behave reasonably
                    continue

                X_hist = X_scaled[hist_mask.values]
                X_block = X_scaled[month_mask.values]

                model = fit_dbscan(X_hist, eps=args.eps, min_samples=args.min_samples)
                # DBSCAN has no predict; approximate by fitting on hist+block to label the block
                X_combo = np.vstack([X_hist, X_block])
                combo_model = fit_dbscan(X_combo, eps=args.eps, min_samples=args.min_samples)
                combo_labels = combo_model.labels_
                block_labels = combo_labels[-len(X_block):]

                labels[month_mask.values] = block_labels
                flags[month_mask.values] = (block_labels == -1).astype(int)

            Xdf["dbscan_label"] = labels
            Xdf["anomaly_flag_dbscan"] = flags
            Xdf["why_dbscan"] = np.where(flags==1, "DBSCAN label = -1 (noise)", "")
            Xdf["type_dbscan"] = np.where(flags==1, Xdf["type_rule"], "")

        # merge back into out_df on (date,ticker)
        out_df = out_df.merge(
            Xdf[["date","ticker"] + [c for c in Xdf.columns if c.startswith("anomaly_flag_") or c.endswith("_dist") or c.endswith("_label") or c.endswith("_cluster") or c.startswith("why_") or c.startswith("type_")]],
            on=["date","ticker"], how="left"
        )

    # daily anomaly card: by default, write rule-based. You can switch method in code if needed.
    daily_card = build_daily_anomaly_card(out_df, method="rule")
    daily_card.to_csv(os.path.join(args.out_dir, "daily_anomaly_card.csv"), index=False)

    # also store a richer parquet/csv for convenience
    out_df.to_csv(os.path.join(args.out_dir, "features_and_flags.csv"), index=False)

    print(f"Wrote: {os.path.join(args.out_dir, 'daily_anomaly_card.csv')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'market_day_table.csv')}")
    print(f"Wrote: {os.path.join(args.out_dir, 'features_and_flags.csv')}")
    print("Next: python -m src.query --out-dir outputs --date 2020-02-27")

if __name__ == "__main__":
    main()
