"""Farm layout: neighbours, bearings and wake sectors.

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


def waked_mask(tid: str, wind_dir, half_width: float = 20.0, max_dist_ratio: float = 4.0) -> np.ndarray:
    """True where the wind comes from the direction of a neighbour close enough to wake `tid`.

    `max_dist_ratio` is relative to the nearest published neighbour, since absolute
    distances are unknown. Only published turbines are considered, so the mask is
    a lower bound on real waking.
    """
    nb = neighbours(tid)
    nb = nb[nb.distance <= max_dist_ratio * nb.distance.min()]
    wd = np.asarray(wind_dir, dtype=float)
    mask = np.zeros(wd.shape, dtype=bool)
    for b in nb.bearing:
        mask |= np.abs((wd - b + 180) % 360 - 180) <= half_width
    return mask
