"""Step 3: per-window yaw-misalignment estimates (C1, C2, C4) for the five target turbines.

Writes cache/state_estimates.parquet (one row per turbine × window kind × window ×
estimator) and cache/manoeuvre_events/<tid>.parquet. Windows: label states
(train, scored days), SCADA states (all, non-transition days), calendar quarters,
and the whole record.
"""
import sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from yaw.config import CACHE, SPLITS
from yaw.consensus import reference_neighbours
from yaw.estimators import load_minute, power_vane_fit, manoeuvres, manoeuvre_events, manoeuvre_fit
from yaw.io import load_turbine

TARGETS = SPLITS["train"] + SPLITS["validate"] + SPLITS["test"]


def windows_for(tid: str) -> list[tuple[str, str, pd.DatetimeIndex]]:
    out = []
    lab = pd.read_parquet(CACHE / "labels_daily.parquet")
    lab = lab[(lab.turbine_id == tid) & lab.scored]
    for st, g in lab.groupby("state"):
        out.append(("label", f"L{st}", pd.DatetimeIndex(g.date)))
    sc = pd.read_parquet(CACHE / "states_scada.parquet")
    sc = sc[(sc.turbine_id == tid) & ~sc.transition]
    for st, g in sc.groupby("state"):
        out.append(("scada", f"S{st}", pd.DatetimeIndex(g.date)))
    days = pd.date_range("2023-01-01", "2024-12-31")
    for q, g in pd.Series(days, index=days).groupby(days.to_period("Q")):
        out.append(("quarter", str(q), pd.DatetimeIndex(g.values)))
    out.append(("all", "all", days))
    return out


def run_turbine(tid: str) -> pd.DataFrame:
    t0 = time.time()
    cands = list(reference_neighbours(tid, 3))
    best, m = None, None
    for c in cands:
        mm = load_minute(tid, c)
        if best is None or mm.good_ref.sum() > best[1]:
            best, m = (c, int(mm.good_ref.sum())), mm
    ref = best[0]
    df = load_turbine(tid, ["NacDir"])
    runs = manoeuvres(df)
    ev = manoeuvre_events(runs, m)
    (CACHE / "manoeuvre_events").mkdir(exist_ok=True)
    ev.assign(turbine_id=tid, reference=ref).to_parquet(CACHE / "manoeuvre_events" / f"{tid}.parquet")
    rows = []
    for kind, name, days in windows_for(tid):
        sel = m.date.isin(days)
        e = ev[ev.date.isin(days)]
        for est, res in [("C1_k2", power_vane_fit(m[sel])),
                         ("C1_kfree", power_vane_fit(m[sel], k=None)),
                         ("C2_k2", power_vane_fit(m[sel], ref="Power_ref", ref_bin=150.0)),
                         ("C4", manoeuvre_fit(e))]:
            rows.append({"turbine_id": tid, "kind": kind, "window": name, "start": days.min(), "end": days.max(),
                         "n_days": len(days), "estimator": est, "reference": ref, **res})
    print(f"{tid}: ref={ref} good_ref_minutes={best[1]} manoeuvres={len(runs)} events={len(ev)} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        frames = list(ex.map(run_turbine, TARGETS))
    est = pd.concat(frames, ignore_index=True)
    est.to_parquet(CACHE / "state_estimates.parquet")
    lab = pd.read_parquet(CACHE / "labels_daily.parquet")
    lvl = lab[lab.scored].groupby(["turbine_id", "state"]).state_level.first().reset_index()
    lvl["window"] = "L" + lvl.state.astype(str)
    pd.set_option("display.width", 200)
    for kind in ("label", "scada", "quarter", "all"):
        t = est[est.kind == kind].pivot_table(index=["turbine_id", "window", "start", "n_days"], columns="estimator", values="theta")
        se = est[est.kind == kind].pivot_table(index=["turbine_id", "window", "start", "n_days"], columns="estimator", values="theta_se")
        t = t.join(se.add_suffix("_se"))
        if kind == "label":
            t = t.reset_index().merge(lvl[["turbine_id", "window", "state_level"]], on=["turbine_id", "window"]).set_index(["turbine_id", "window", "start", "n_days"])
        print(f"\n=== {kind}\n", t.round(2).to_string())
