"""Step 3 — turn SCADA states into a submission.

Model (per turbine, per day):

    theta_day = level - x_day,      x_day = r_level(state) - anchor(segment)

* `r_level` is the consensus heading residual of the day's SCADA state (step 2).
  Sensor algebra gives dr = -dtheta, so the slope on x is -1 in physical degrees.
* Segments are runs of states between encoder-typed boundaries; r is not
  comparable across an encoder re-reference, so each segment gets its own anchor
  (median or day-weighted mean of r_level over the segment's days).
* `level` is the misalignment of the anchor state: the vane setpoint, i.e. the
  turbine's median daily vane in operation — the controller nulls the vane at
  this value, so with an unbiased vane it *is* the static misalignment.

No parameter is fitted to labels. Labels are read only by `sign_check` and by the
local scoring in `evaluate`.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from .config import CACHE, DATA
from .io import wrap180


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """(states_scada, labels_daily, vane setpoint per turbine)."""
    states = pd.read_parquet(CACHE / "states_scada.parquet")
    labels = pd.read_parquet(CACHE / "labels_daily.parquet")
    daily = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(CACHE / "daily" / "*.parquet")))])
    vane = daily[daily.flag_ok].groupby("turbine_id")["vane_med"].median()
    return states, labels, vane


def features(states: pd.DataFrame, anchor: str = "mean") -> pd.DataFrame:
    """Add `seg` (encoder-free segment id) and `x` = r_level - anchor(segment) per turbine-day."""
    if anchor not in ("mean", "median"):
        raise ValueError(anchor)
    out = []
    for _, g in states.groupby("turbine_id", sort=False):
        g = g.sort_values("date").copy()
        enc = (g["boundary_type"] == "encoder") & (g["state"].diff().fillna(0) != 0)
        g["seg"] = enc.cumsum().astype(int)
        g["x"] = g["r_level"] - g.groupby("seg")["r_level"].transform(anchor)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def predict(feat: pd.DataFrame, vane: pd.Series) -> pd.DataFrame:
    """turbine_id, date, yaw_misalignment_deg, cluster. Level = vane setpoint; slope -1 on x."""
    level = feat["turbine_id"].map(vane).astype(float)
    if level.isna().any():
        raise ValueError(f"no vane setpoint for {sorted(feat.loc[level.isna(), 'turbine_id'].unique())}")
    return pd.DataFrame({
        "turbine_id": feat["turbine_id"].values,
        "date": feat["date"].values,
        "yaw_misalignment_deg": (level - feat["x"]).values,
        "cluster": feat["state"].astype(int).values,
    })


def sign_check(feat: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Two independent checks of the sign convention; raises if either fails.

    1. Slope of label on x over the scored train days must be negative
       (r moves opposite to the label: dr = -dtheta).
    2. At the WTG13 event of mid-January 2023 the derived vane
       median moved down by ~2.5° while the label moved up (-11.5 -> -8.1).
       The vane and the label must move in opposite directions.
    """
    d = feat.merge(labels.loc[labels.scored, ["turbine_id", "date", "yaw_misalignment_deg"]], on=["turbine_id", "date"])
    slope = float(np.polyfit(d["x"], d["yaw_misalignment_deg"], 1)[0])

    t = pd.read_parquet(DATA / "train.parquet", columns=["turbine_id", "ts", "NacDir", "WindDir", "Power", "yaw_misalignment_deg"])
    t = t[(t.turbine_id == "PPP_WTG13") & (t.ts >= "2023-01-01") & (t.ts < "2023-02-01")].sort_values("ts")
    t[["NacDir", "WindDir", "Power"]] = t[["NacDir", "WindDir", "Power"]].ffill()
    t = t[t.Power > 100]
    t["vane"] = wrap180(t.WindDir - t.NacDir)
    before, after = t[t.ts < "2023-01-13"], t[t.ts >= "2023-01-19"]
    d_vane = float(after.vane.median() - before.vane.median())
    d_label = float(after.yaw_misalignment_deg.median() - before.yaw_misalignment_deg.median())

    out = {"slope_label_on_x": slope, "wtg13_jan2023_d_vane": d_vane, "wtg13_jan2023_d_label": d_label}
    if not slope < 0:
        raise AssertionError(f"sign check 1 failed: label-on-x slope {slope:+.2f} should be negative")
    if not (d_vane < -1 and d_label > 1):
        raise AssertionError(f"sign check 2 failed: WTG13 Jan-2023 d_vane {d_vane:+.2f}, d_label {d_label:+.2f}")
    return out


def write_submission(pred: pd.DataFrame, template: pd.DataFrame, path) -> pd.DataFrame:
    """Align predictions with a round template (all rows, template order), write CSV."""
    p = pred.copy()
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    tmpl = template[["turbine_id", "date"]].astype(str)
    out = tmpl.merge(p, on=["turbine_id", "date"], how="left")
    if out.yaw_misalignment_deg.isna().any():
        # days without a SCADA state (gaps at the record edges): nearest state's value
        out = out.sort_values("date")
        out[["yaw_misalignment_deg", "cluster"]] = out[["yaw_misalignment_deg", "cluster"]].ffill().bfill()
        out = out.loc[tmpl.index]
    out["cluster"] = out["cluster"].astype(int)
    out["yaw_misalignment_deg"] = out["yaw_misalignment_deg"].round(3)
    out.to_csv(path, index=False)
    return out
