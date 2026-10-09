"""Step 2 — encoder de-stepping, consensus heading residual, SCADA-only states.

    .venv/bin/python scripts/segment_states.py [--pen 40]

Outputs under cache/:
  encoder_steps.parquet        large shared jumps credited to a turbine's encoder
  frame_plan.parquet           per turbine: frames, excursions (dropped) and verified re-references
  heading_residual_daily.parquet  r_i(t) per turbine-day after de-stepping and sector correction
  pair_residual_daily.parquet  the per-pair daily residuals behind r_i
  states_scada.parquet         PELT states on r_i with transition flags
Prints: ARI vs label states on train, and Δr vs Δlabel at every label step.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yaw.config import CACHE  # noqa: E402
from yaw.consensus import destep, encoder_steps, frame_plan, heading_residual, load_10min, segment_residual  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pen", type=float, default=400.0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-jump", type=float, default=25.0)
    args = ap.parse_args()

    frames = load_10min()
    steps = encoder_steps(frames, k=args.k, min_jump=args.min_jump)
    steps.to_parquet(CACHE / "encoder_steps.parquet", index=False)
    print("encoder steps credited:\n", steps.to_string(index=False) if len(steps) else "none")

    plan = frame_plan(frames, steps, k=args.k)
    plan.to_parquet(CACHE / "frame_plan.parquet", index=False)
    summ = plan.assign(days=(plan.end - plan.start).dt.days).groupby("turbine_id").agg(
        excursions=("kind", lambda x: (x == "excursion").sum()), excursion_days=("days", lambda d: d[plan.loc[d.index, "kind"] == "excursion"].sum()),
        rereferences=("rereference", "sum"))
    print("\nframe plan:\n", summ.to_string())
    print("\nre-references kept:\n", plan[plan.rereference].to_string(index=False) if plan.rereference.any() else "none")

    frames = destep(frames, plan)
    res, pairs = heading_residual(frames, k=args.k)
    res.to_parquet(CACHE / "heading_residual_daily.parquet", index=False)
    pairs.to_parquet(CACHE / "pair_residual_daily.parquet", index=False)
    print("\nneighbours kept:", res.groupby("turbine_id")["neighbours"].first().to_dict())

    states = segment_residual(res, pen=args.pen, plan=plan)
    states.to_parquet(CACHE / "states_scada.parquet", index=False)
    summ = states.groupby(["turbine_id", "state"]).agg(start=("date", "min"), end=("date", "max"), days=("date", "size"), r_level=("r_level", "first"), boundary=("boundary_type", "first"))
    print("\nSCADA states:\n", summ.round(2).to_string())

    # validation on train
    lab = pd.read_parquet(CACHE / "labels_daily.parquet")
    j = lab[lab.scored].merge(states[["turbine_id", "date", "state", "r"]].rename(columns={"state": "state_scada"}), on=["turbine_id", "date"], how="inner")
    for t, g in j.groupby("turbine_id"):
        print(f"\n{t}: ARI(label states, SCADA states) = {adjusted_rand_score(g.state, g.state_scada):.3f} on {len(g)} scored days")
        lvl = g.groupby("state").agg(label=("yaw_misalignment_deg", "median"), r=("r", "median"), n=("date", "size"))
        lvl["d_label"] = lvl.label.diff(); lvl["d_r"] = lvl.r.diff()
        print(lvl.round(2).to_string())


if __name__ == "__main__":
    main()
