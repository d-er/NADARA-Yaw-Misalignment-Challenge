"""Step 1 — build per-turbine caches from the raw 12-s stream.

    .venv/bin/python scripts/build_cache.py [--workers 8] [--turbines PPP_WTG13 ...]

Outputs under cache/:
  agg1min/<tid>.parquet    1-minute aggregates (circular means for directions)
  agg10min/<tid>.parquet   10-minute aggregates, same schema
  daily/<tid>.parquet      daily coverage / regime / sensor-health table
  labels_daily.parquet     labelled turbine-days with states and transition flags
  pair_offsets_daily.parquet  daily median NacDir/WindDir offset for every same-site pair
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yaw.aggregate import aggregate  # noqa: E402
from yaw.config import CACHE, SITE_OF, TURBINES  # noqa: E402
from yaw.io import load_turbine, wrap180  # noqa: E402
from yaw.labels import add_states, daily_labels  # noqa: E402
from yaw.quality import daily_table  # noqa: E402


def build_turbine(tid: str) -> str:
    t0 = time.time()
    df = load_turbine(tid)
    m1 = aggregate(df, "1min")
    m10 = aggregate(df, "10min")
    d = daily_table(m1)
    for name, frame in (("agg1min", m1), ("agg10min", m10), ("daily", d)):
        (CACHE / name).mkdir(parents=True, exist_ok=True)
        frame.assign(turbine_id=tid).to_parquet(CACHE / name / f"{tid}.parquet", index=False)
    return f"{tid}: {len(df):,} rows -> {len(m1):,} min, {len(d)} days, {time.time() - t0:.0f}s"


def pair_offsets() -> pd.DataFrame:
    """Daily median of wrap(dir_a - dir_b) over 10-min bins where both turbines operate."""
    frames = {tid: pd.read_parquet(CACHE / "agg10min" / f"{tid}.parquet") for tid in TURBINES}
    rows = []
    for a in TURBINES:
        for b in TURBINES:
            if a >= b or SITE_OF[a] != SITE_OF[b]:
                continue
            fa, fb = frames[a], frames[b]
            j = fa.merge(fb, on="ts", suffixes=("_a", "_b"))
            j = j[(j.op_frac_a > 0.9) & (j.op_frac_b > 0.9)]
            if j.empty:
                continue
            j["date"] = j["ts"].dt.normalize()
            for d in ("NacDir", "WindDir"):
                j[f"d{d}"] = wrap180(j[f"{d}_a"] - j[f"{d}_b"])
            g = j.groupby("date")
            rows.append(pd.DataFrame({
                "a": a, "b": b, "n_bins": g.size(),
                "dNacDir": g["dNacDir"].median(), "dWindDir": g["dWindDir"].median(),
                "dNacDir_iqr": g["dNacDir"].quantile(0.75) - g["dNacDir"].quantile(0.25),
            }).reset_index())
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--turbines", nargs="*", default=TURBINES)
    ap.add_argument("--skip-turbines", action="store_true", help="only rebuild the derived tables")
    args = ap.parse_args()
    CACHE.mkdir(exist_ok=True)
    if not args.skip_turbines:
        with ProcessPoolExecutor(args.workers) as ex:
            for msg in ex.map(build_turbine, args.turbines):
                print(msg, flush=True)
    labels = add_states(daily_labels())
    labels.to_parquet(CACHE / "labels_daily.parquet", index=False)
    print("labels:", labels.groupby("turbine_id").agg(days=("date", "size"), states=("state", "nunique"), scored=("scored", "sum")).to_string())
    po = pair_offsets()
    po.to_parquet(CACHE / "pair_offsets_daily.parquet", index=False)
    print(f"pair offsets: {len(po):,} rows, {po[['a', 'b']].drop_duplicates().shape[0]} pairs")


if __name__ == "__main__":
    main()
