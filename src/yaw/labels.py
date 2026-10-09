"""Ground-truth handling: daily labels, label-derived states and transition windows.

The scoring rule counts only labelled days outside a transition window. The
organisers do not publish the window definition, so we reconstruct states from the
label series itself with a change-point search and exclude days near each
breakpoint or far from their state's level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt

from .config import DATA


def daily_labels() -> pd.DataFrame:
    """turbine_id, date, yaw_misalignment_deg for every labelled turbine-day."""
    t = pd.read_parquet(DATA / "train.parquet", columns=["turbine_id", "ts", "yaw_misalignment_deg"])
    t = t.dropna(subset=["yaw_misalignment_deg"])
    t["date"] = t["ts"].dt.normalize()
    return (t.groupby(["turbine_id", "date"])["yaw_misalignment_deg"].first()
             .reset_index().sort_values(["turbine_id", "date"]).reset_index(drop=True))


def segment_series(values: np.ndarray, pen: float = 20.0, min_size: int = 7) -> np.ndarray:
    """Piecewise-constant state id per element via PELT (L2 cost)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2 * min_size:
        return np.zeros(len(values), dtype=int)
    bkps = rpt.Pelt(model="l2", min_size=min_size, jump=1).fit(values).predict(pen=pen)
    state = np.zeros(len(values), dtype=int)
    start = 0
    for i, b in enumerate(bkps):
        state[start:b] = i
        start = b
    return state


def add_states(labels: pd.DataFrame, pen: float = 20.0, guard_days: int = 7, max_dev: float = 0.75) -> pd.DataFrame:
    """Add `state`, `state_level` (median), `transition` and `scored` per turbine-day.

    A day is a transition if it lies within `guard_days` of a state breakpoint
    (calendar days, so gaps count) or deviates more than `max_dev` degrees from
    its state's median level. `scored` = labelled and not a transition.
    """
    out = []
    for tid, g in labels.groupby("turbine_id", sort=False):
        g = g.sort_values("date").copy()
        g["state"] = segment_series(g["yaw_misalignment_deg"].values, pen=pen)
        g["state_level"] = g.groupby("state")["yaw_misalignment_deg"].transform("median")
        change_dates = g.loc[g["state"].diff().fillna(0) != 0, "date"]
        near = np.zeros(len(g), dtype=bool)
        for d in change_dates:
            near |= (g["date"] - d).abs().dt.days.values <= guard_days
        dev = (g["yaw_misalignment_deg"] - g["state_level"]).abs() > max_dev
        g["transition"] = near | dev.values
        g["scored"] = ~g["transition"]
        out.append(g)
    return pd.concat(out, ignore_index=True)
