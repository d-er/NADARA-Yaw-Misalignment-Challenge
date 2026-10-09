"""Format check for submissions. Runs on every PR; needs no labels.

    python validate_submission.py Submissions/Results_42_T0_0.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# TIER is TK: K = how many labelled turbines the method needed. T0 none,
# T3 all three train turbines, T4 also the scored turbine.
RESULTS = re.compile(r"^Results_(\d+)_(T[0-4])_(\d+|final)\.csv$")
TEMPLATES = Path(__file__).parent  # copied from the WeDoWind Submissions repo
REQUIRED = ["turbine_id", "date", "yaw_misalignment_deg"]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


def check(path: Path) -> None:
    match = RESULTS.match(path.name)
    if match is None:
        fail(f"{path.name} does not match Results_NN_TIER_x.csv "
             "(TIER is T0-T4: T followed by how many labelled turbines your "
             "method needed)")

    # a numbered entry is scored on the public validation turbine, "final" on the
    # held-out one -- each round has its own template, so neither carries the other's rows
    round_name = "final" if match.group(3) == "final" else "validate"
    template_path = TEMPLATES / f"submission_template_{round_name}.csv"
    if not template_path.exists():
        fail(f"missing {template_path.name} — copy it from the challenge data")

    df = pd.read_csv(path)
    if list(df.columns[:3]) != REQUIRED:
        fail(f"first three columns are {list(df.columns[:3])}, expected {REQUIRED}")
    extra_cols = [c for c in df.columns[3:] if c != "cluster"]
    if extra_cols:
        fail(f"unexpected columns {extra_cols}; only an optional 'cluster' may follow")

    if df[["turbine_id", "date"]].duplicated().any():
        fail("duplicate turbine_id/date rows")

    template = pd.read_csv(template_path)
    want = set(zip(template.turbine_id, template.date.astype(str)))
    got = set(zip(df.turbine_id, df.date.astype(str)))
    if missing := want - got:
        fail(f"{len(missing)} rows from the template are missing, e.g. {sorted(missing)[:3]}")
    if extra := got - want:
        fail(f"{len(extra)} rows are not in the template, e.g. {sorted(extra)[:3]}")

    values = pd.to_numeric(df.yaw_misalignment_deg, errors="coerce")
    if values.isna().any():
        fail(f"{int(values.isna().sum())} rows have a non-finite yaw_misalignment_deg")
    if values.abs().max() > 90:
        fail(f"values outside +/-90 degrees (max |value| {values.abs().max():.1f})")

    # the cluster column is optional, but a partly filled one is almost certainly a mistake
    note = ""
    if "cluster" in df.columns:
        filled = df.cluster.notna().sum()
        if filled == 0:
            note = ", no clustering entered"
        elif filled < len(df):
            fail(f"cluster is filled on {filled} of {len(df)} rows — "
                 "fill every row to enter the tie-breaker, or leave the column empty")
        else:
            note = f", {df.cluster.nunique()} clusters"

    print(f"OK: {path.name} — {round_name} round, {len(df):,} rows, "
          f"{df.turbine_id.nunique()} turbine{note}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    for path in args.paths:
        check(path)


if __name__ == "__main__":
    main()
