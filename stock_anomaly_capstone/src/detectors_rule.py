from __future__ import annotations
import numpy as np
import pandas as pd
from .config import Thresholds

def detect_rule_based(df_feat: pd.DataFrame, thr: Thresholds | None = None) -> pd.DataFrame:
    """Rule-based detector per PDF:
    anomaly if |ret_z|>2.5 or volz>2.5 or range_pct>95, with type labels crash/spike/volume_shock.
    """
    if thr is None:
        thr = Thresholds()

    df = df_feat.copy()
    # Only score rows with full history
    df = df[df["has_history"]].copy()

    trig_ret = df["ret_z"].abs() > thr.ret_z
    trig_vol = df["volz"] > thr.volz
    trig_rng = df["range_pct"] > thr.range_pct

    df["anomaly_flag_rule"] = (trig_ret | trig_vol | trig_rng).astype(int)

    # type label
    types = []
    for r, rz, vz, fr in zip(df["ret"], df["ret_z"], df["volz"], df["anomaly_flag_rule"]):
        t = []
        if fr != 1:
            types.append("")
            continue
        if pd.notna(rz) and abs(rz) > thr.ret_z:
            t.append("crash" if (r is not None and r < 0) else "spike")
        if pd.notna(vz) and vz > thr.volz:
            t.append("volume_shock")
        types.append(" + ".join(t) if t else "range_spike")
    df["type_rule"] = types

    # why
    why = []
    for a, b, c in zip(trig_ret, trig_vol, trig_rng):
        reasons = []
        if a: reasons.append("|ret_z| > 2.5")
        if b: reasons.append("volz > 2.5")
        if c: reasons.append("range_pct > 95")
        why.append("; ".join(reasons))
    df["why_rule"] = why

    return df
