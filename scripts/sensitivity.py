"""Sensitivity of the submitted states to the hand-set thresholds.

    .venv/bin/python scripts/sensitivity.py [--workers 8]

Varies one threshold at a time around the submitted defaults, reruns steps 2 and 3
and records, for PPP_WTG17 and SSS_WTG06, the states and their predicted levels.
The train RMSE / ARI are added as a diagnostic (they use labels; nothing is
selected on them here). Needs the caches of scripts/build_cache.py.

Outputs: cache/sensitivity.csv, and a summary table on stdout.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yaw.config import CACHE, SPLITS  # noqa: E402
from yaw.consensus import destep, encoder_steps, frame_plan, heading_residual, load_10min, segment_residual  # noqa: E402
from yaw.evaluate import score  # noqa: E402
from yaw.submission import features, load_inputs, predict  # noqa: E402

DEFAULTS = dict(pen=400.0, min_size=14, smooth_days=7, min_jump=25.0, tol=15.0, long_days=45,
                max_excursion=10.0, excursion_window=91, k=4, n_sectors=12)
GRID = dict(pen=[100, 200, 300, 600, 800, 1200, 1600], min_size=[7, 28], smooth_days=[1, 15],
            min_jump=[15, 20, 35], tol=[10, 20], long_days=[30, 60],
            max_excursion=[7, 15], excursion_window=[61, 121], k=[3, 5], n_sectors=[8, 18])
SEGMENT_ONLY = ("pen", "min_size", "smooth_days")
TARGETS = {"ppp17": SPLITS["validate"][0], "sss06": SPLITS["test"][0]}


def residual(p: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 2 up to the daily residual: (residual table, frame plan)."""
    frames = load_10min()
    steps = encoder_steps(frames, k=p["k"], min_jump=p["min_jump"])
    plan = frame_plan(frames, steps, k=p["k"], long_days=p["long_days"], tol=p["tol"])
    res, _ = heading_residual(destep(frames, plan), k=p["k"], n_sectors=p["n_sectors"],
                              max_excursion=p["max_excursion"], excursion_window=p["excursion_window"])
    return res, plan


def summarise(res: pd.DataFrame, plan: pd.DataFrame, p: dict) -> dict:
    """Step 2 segmentation and step 3 for one setting; states of the two targets, train score."""
    states = segment_residual(res, pen=p["pen"], min_size=p["min_size"], smooth_days=p["smooth_days"], plan=plan)
    _, labels, vane = load_inputs()
    pred = predict(features(states), vane)
    out = {}
    for name, tid in TARGETS.items():
        g = pred[pred.turbine_id == tid].groupby("cluster").agg(start=("date", "min"), days=("date", "size"),
                                                                 deg=("yaw_misalignment_deg", "first"))
        out[f"{name}_states"] = len(g)
        out[f"{name}_boundaries"] = " ".join(d.strftime("%Y-%m-%d") for d in g.start.iloc[1:])
        out[f"{name}_levels"] = " ".join(f"{v:+.1f}" for v in g.deg)
        out[f"{name}_max_step"] = float(g.deg.diff().abs().max()) if len(g) > 1 else 0.0
    s = score(pred[pred.turbine_id.isin(SPLITS["train"])], labels)
    out["train_rmse"], out["train_ari"] = s["rmse"], s["ari"]
    return out


def run(job: tuple[str, float]) -> dict:
    name, value = job
    p = {**DEFAULTS, name: value}
    return {"parameter": name, "value": value, "default": DEFAULTS[name], **summarise(*residual(p), p)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    res, plan = residual(DEFAULTS)
    rows = [{"parameter": "submitted", "value": float("nan"), "default": float("nan"), **summarise(res, plan, DEFAULTS)}]
    for name in SEGMENT_ONLY:
        for value in GRID[name]:
            p = {**DEFAULTS, name: value}
            rows.append({"parameter": name, "value": value, "default": DEFAULTS[name], **summarise(res, plan, p)})
    jobs = [(name, value) for name in GRID if name not in SEGMENT_ONLY for value in GRID[name]]
    with ProcessPoolExecutor(args.workers) as ex:
        rows += list(ex.map(run, jobs))

    out = pd.DataFrame(rows)
    out.to_csv(CACHE / "sensitivity.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 200)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
