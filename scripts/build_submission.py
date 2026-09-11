"""Step 4 — build submission files from the SCADA states.

    .venv/bin/python scripts/build_submission.py --participant 42 [--anchor mean] [--number 0]

Writes submissions/Results_<ID>_T0_<n>.csv, _T3_<n>.csv (validate round, PPP_WTG17)
and _T0_final.csv, _T3_final.csv (SSS_WTG06), cache/predictions_daily.parquet
(all turbines, both tiers), and runs submission_kit/validate_submission.py on each.
Prints the train scores (T0 is label-free, so its train score is an honest
estimate), the leave-one-turbine-out T3 fits, and the sign checks. Aborts if a
sign check fails.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from yaw.config import CACHE, SPLITS  # noqa: E402
from yaw.evaluate import score  # noqa: E402
from yaw.submission import features, fit_offset, load_inputs, predict, sign_check, write_submission  # noqa: E402

KIT = ROOT / "submission_kit"
TRAIN = SPLITS["train"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participant", required=True, help="participant ID (digits)")
    ap.add_argument("--anchor", default="mean", choices=["mean", "median"], help="segment anchor for r")
    ap.add_argument("--number", default="0", help="submission counter for the validate round")
    ap.add_argument("--out", default=str(ROOT / "submissions"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True)

    states, labels, vane = load_inputs()
    feat = features(states, anchor=args.anchor)
    print("vane setpoints (deg):", vane.round(2).to_dict())

    checks = sign_check(feat, labels)
    print("sign checks passed:", {k: round(v, 2) for k, v in checks.items()})

    # T0: physics only
    p0 = predict(feat, vane, a=0.0, b=-1.0)
    s0 = score(p0[p0.turbine_id.isin(TRAIN)], labels)
    print(f"\nT0 on train (label-free): rmse {s0['rmse']:.2f} mae {s0['mae']:.2f} bias {s0['bias']:+.2f} ari {s0['ari']:.2f} ci {tuple(round(c, 2) for c in s0['rmse_ci'])}")
    for t in TRAIN:
        s = score(p0[p0.turbine_id == t], labels); print(f"   {t}: rmse {s['rmse']:.2f} bias {s['bias']:+.2f} ari {s['ari']:.2f}")

    # T3: offset + slope fitted on the train turbines, leave-one-turbine-out reported
    parts = []
    for t in TRAIN:
        a, b = fit_offset(feat, labels, vane, exclude=t)
        pt = predict(feat[feat.turbine_id == t], vane, a=a, b=b); parts.append(pt)
        s = score(pt, labels); print(f"   LOTO {t}: a {a:+.2f} b {b:+.2f} -> rmse {s['rmse']:.2f} bias {s['bias']:+.2f}")
    s3 = score(pd.concat(parts), labels)
    a, b = fit_offset(feat, labels, vane)
    print(f"T3 LOTO on train: rmse {s3['rmse']:.2f} mae {s3['mae']:.2f} bias {s3['bias']:+.2f}; full fit a {a:+.2f} b {b:+.2f}")
    p3 = predict(feat, vane, a=a, b=b)

    pd.concat([p0.assign(tier="T0"), p3.assign(tier="T3")]).to_parquet(CACHE / "predictions_daily.parquet", index=False)

    files = []
    for tier, pred in [("T0", p0), ("T3", p3)]:
        for rnd, tid, suffix in [("validate", SPLITS["validate"][0], args.number), ("final", SPLITS["test"][0], "final")]:
            tmpl = pd.read_csv(KIT / f"submission_template_{rnd}.csv")
            path = out_dir / f"Results_{args.participant}_{tier}_{suffix}.csv"
            w = write_submission(pred[pred.turbine_id == tid], tmpl, path)
            g = w.groupby("cluster").agg(start=("date", "min"), days=("date", "size"), deg=("yaw_misalignment_deg", "first"))
            print(f"\n{path.name}: mean {w.yaw_misalignment_deg.mean():+.2f}\n{g.to_string()}")
            files.append(path)
    print()
    subprocess.run([sys.executable, str(KIT / "validate_submission.py"), *map(str, files)], check=True)


if __name__ == "__main__":
    main()
