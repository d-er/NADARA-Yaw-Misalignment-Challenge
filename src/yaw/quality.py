"""Daily quality table and flags derived from the 1-minute aggregates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import P_MAX_PARTIAL, P_MIN_OPERATING, PITCH_MAX_PARTIAL
from .aggregate import circ_mean_deg


def minute_flags(m: pd.DataFrame) -> pd.DataFrame:
    """Add boolean operating-regime flags to a 1-minute frame (in place, returned)."""
    m["operating"] = m["Power_mean"] > P_MIN_OPERATING
    m["partial_load"] = m["operating"] & (m["Power_mean"] < P_MAX_PARTIAL) & (m["PitchAngle_mean"] < PITCH_MAX_PARTIAL)
    return m


def daily_table(m: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day summarising coverage, regime and sensor health."""
    m = minute_flags(m.copy())
    day = m["ts"].dt.normalize()
    g = m.groupby(day)
    op = m[m.operating]
    gop = op.groupby(op["ts"].dt.normalize())
    out = pd.DataFrame({
        "n_minutes": g.size(),
        "n_rows": g["n_rows"].sum(),
        "n_nac_updates": g["n_nac_updates"].sum(),
        "op_minutes": g["operating"].sum(),
        "partial_minutes": g["partial_load"].sum(),
        "power_mean": g["Power_mean"].mean(),
        "ws_mean": g["WindSpeed_mean"].mean(),
    })
    out["vane_med"] = gop["vane_med"].median()
    out["vane_iqr"] = gop["vane_med"].quantile(0.75) - gop["vane_med"].quantile(0.25)
    out["vane_std_within"] = gop["vane_std"].mean()
    out["pitch_med"] = gop["PitchAngle_mean"].median()
    out["rot_med"] = gop["RotSpeed_mean"].median()
    for d in ("NacDir", "WindDir"):
        rad = np.radians(op[d])
        s = pd.Series(np.sin(rad), index=op.index).groupby(op["ts"].dt.normalize()).mean()
        c = pd.Series(np.cos(rad), index=op.index).groupby(op["ts"].dt.normalize()).mean()
        out[f"{d}_op"] = circ_mean_deg(s, c)
    # sensor health flags
    out["flag_frozen_vane"] = (out["vane_std_within"].fillna(0) < 0.05) & (out["op_minutes"] > 60)
    out["flag_no_yaw"] = (out["n_nac_updates"] == 0) & (out["op_minutes"] > 60)
    out["flag_sparse"] = out["n_rows"] < 1000
    out["flag_ok"] = ~(out["flag_frozen_vane"] | out["flag_no_yaw"] | out["flag_sparse"])
    out.index.name = "date"
    return out.reset_index()
