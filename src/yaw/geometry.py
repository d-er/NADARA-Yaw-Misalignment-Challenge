"""Farm layout: neighbours and bearings.

Coordinates carry an unknown scale factor (same on both axes), so distances are
proportional and bearings exact. Distances are therefore only used relative to
each other (nearest-neighbour ratios), never as metres.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA, SITE_OF

_layout: pd.DataFrame | None = None


def layout() -> pd.DataFrame:
    global _layout
    if _layout is None:
        frames = [pd.read_csv(DATA / f"turbine_locations_{s}.csv").assign(site=s) for s in ("PPP", "SSS")]
        _layout = pd.concat(frames, ignore_index=True).set_index("turbine_id")
    return _layout


def neighbours(tid: str, k: int | None = None, max_dist: float | None = None) -> pd.DataFrame:
    """Same-site turbines sorted by distance, with bearing (deg clockwise from north)."""
    lay = layout()
    here = lay[lay.site == SITE_OF[tid]]
    o = here.loc[tid]
    dx, dy = here.x_m - o.x_m, here.y_m - o.y_m
    out = pd.DataFrame({
        "distance": np.hypot(dx, dy),
        "bearing": np.degrees(np.arctan2(dx, dy)) % 360,
        "role": here.role,
    }).drop(index=tid).sort_values("distance")
    if max_dist is not None:
        out = out[out.distance <= max_dist]
    return out.head(k) if k else out

