"""Time aggregation of the filled stream with circular statistics for directions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import P_MIN_OPERATING

LINEAR = ["GenSpeed", "WindSpeed", "PitchAngle", "RotSpeed", "Power", "vane"]


def circ_mean_deg(sin_mean, cos_mean):
    return np.degrees(np.arctan2(sin_mean, cos_mean)) % 360


def aggregate(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Bin the filled 12-s stream to `freq`.

    Output columns: <sig>_mean for linear signals, <sig>_std for Power/WindSpeed/vane,
    vane_med, NacDir/WindDir circular means, n_rows, n_nac_updates, op_frac
    (fraction of rows with Power above the operating threshold).
    """
    g = df.groupby(df["ts"].dt.floor(freq), sort=True)
    lin = [c for c in LINEAR if c in df.columns]
    out = g[lin].mean().add_suffix("_mean")
    for c in ("Power", "WindSpeed", "vane"):
        if c in df.columns:
            out[f"{c}_std"] = g[c].std()
    if "vane" in df.columns:
        out["vane_med"] = g["vane"].median()
    for d in ("NacDir", "WindDir"):
        if d in df.columns:
            rad = np.radians(df[d])
            s = pd.Series(np.sin(rad), index=df.index).groupby(df["ts"].dt.floor(freq)).mean()
            c = pd.Series(np.cos(rad), index=df.index).groupby(df["ts"].dt.floor(freq)).mean()
            out[d] = circ_mean_deg(s, c)
            out[f"{d}_R"] = np.hypot(s, c)   # mean resultant length: 1 = steady, 0 = all over
    out["n_rows"] = g.size()
    if "nac_update" in df.columns:
        out["n_nac_updates"] = g["nac_update"].sum()
    if "Power" in df.columns:
        out["op_frac"] = g["Power"].apply(lambda s: (s > P_MIN_OPERATING).mean())
    out.index.name = "ts"
    return out.reset_index()
