"""Loading and reconstruction of the high-resolution SCADA stream.

The stream is change-based: a null means "unchanged since the previous update".
`prepare` forward-fills per turbine and derives the vane angle. It also keeps a
`nac_update` flag (raw NacDir was written this row) because the number of yaw
controller updates is itself a useful signal and is lost after filling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import DATA, SIGNALS, SPLIT_OF


def wrap180(x):
    """Wrap an angle (deg) to [-180, 180)."""
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def load_raw(tid: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Rows of one turbine straight from its split file, unsorted and unfilled."""
    path = DATA / f"{SPLIT_OF[tid]}.parquet"
    cols = None if columns is None else ["turbine_id", "ts", *columns]
    table = pq.read_table(path, columns=cols, filters=[("turbine_id", "=", tid)])
    return table.to_pandas()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, forward-fill per turbine, add vane angle and calendar date."""
    cols = [c for c in SIGNALS if c in df.columns]
    out = df.sort_values(["turbine_id", "ts"]).reset_index(drop=True)
    if "NacDir" in out.columns:
        out["nac_update"] = out["NacDir"].notna()
    out[cols] = out.groupby("turbine_id", sort=False)[cols].ffill()
    if {"WindDir", "NacDir"} <= set(out.columns):
        out["vane"] = wrap180(out["WindDir"] - out["NacDir"])
    out["date"] = out["ts"].dt.normalize()
    return out


def load_turbine(tid: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Forward-filled stream of one turbine with `vane`, `date`, `nac_update`."""
    return prepare(load_raw(tid, columns))
