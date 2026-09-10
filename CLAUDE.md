# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Working folder for the Nadara / WeDoWind **static yaw misalignment challenge**
(Energy Data Hackdays 2026). Goal: predict the daily WindFit yaw misalignment
(degrees) per turbine-day for PPP_WTG17 (public leaderboard) and SSS_WTG06
(final round) from ~12-second SCADA. `PLAN.md` holds the strategy, measured data
facts and the timeline — read it before starting any modelling work.

## Environment and commands

System pip is blocked (PEP 668); always install into `.venv` with uv:

```bash
uv venv .venv                                   # once
uv pip install --python .venv/bin/python <pkgs> # e.g. pandas pyarrow polars scipy ruptures
.venv/bin/python -m pytest -q                   # unit tests (pyproject sets pythonpath=src)
.venv/bin/python scripts/build_cache.py         # step 1: rebuild cache/ (~1 min, 8 workers)
.venv/bin/jupyter notebook notebooks/read_croissant_data.ipynb   # kernel "download (.venv)"
```

Code lives in `src/yaw/` (import with `sys.path.insert(0, "src")` in scripts, or
run pytest which sets it). Each pipeline step is a script in `scripts/` that reads
and writes `cache/`, and is documented in `docs/NN_*.md` with figures in
`figures/`. Keep that pattern: one script + one doc per step, tables cached to
parquet, nothing downstream depends on a notebook.

`requirements.txt` is a conda export from another machine — do not pip-install it.

Submission format check (repo cloned to a scratch dir, templates live at its root):

```bash
git clone https://github.com/WeDoWind/NADARA-Static-Yaw-Misalignment-Submissions.git
python validate_submission.py Submissions/Results_NN_T0_0.csv
```

Files are `Results_<participantID>_<T0..T4>_<n|final>.csv` with columns
`turbine_id,date,yaw_misalignment_deg,cluster` (731 rows from the round's
template; `cluster` all-filled or absent). Submissions go in by PR (Git LFS) and
are immutable once merged.

## Data (`data/`, ~1.7 GB, not in git)

| File | Rows | Turbines | Labels |
|---|---|---|---|
| `train.parquet` | 17.2 M | PPP_WTG12/13/14 | `yaw_misalignment_deg`, null on unlabelled days (labels only 2023-01..2023-11) |
| `validate.parquet` | 5.7 M | PPP_WTG17 | none (public LB, 76 scored days) |
| `test.parquet` | 4.7 M | SSS_WTG06 | none (final, 185 scored days) |
| `context.parquet` | 59.1 M | 11 neighbours | none — read with `columns=` and filter by turbine |
| `turbine_locations_{PPP,SSS}.csv` | | layout | bearings exact, distances scaled |

Rules that are easy to get wrong:
- **A null means "unchanged since last update", not missing.** Sort by
  `turbine_id, ts` and forward-fill per turbine before anything else. `NacDir`
  updates on only ~6 % of rows.
- Vane angle is not shipped: `vane = (WindDir - NacDir + 180) % 360 - 180`.
  `WindDir` already equals `NacDir + vane`, so it carries the nacelle encoder offset.
- Power, WindSpeed, GenSpeed, RotSpeed, PitchAngle carry an undisclosed constant
  scale factor (same for all turbines); directions are unscaled. Never interpret
  absolute values (Cp, TSR); ratios and relationships are fine.
- Nacelle encoders have no north reference and get re-referenced (e.g. WTG12 jumps
  ~138° vs neighbours from Aug 2023). Use circular means for directions.
- Labels are a smoothed piecewise-constant series; days within ~a week of a
  step are not scored. Score locally on labelled days outside those windows and
  use ARI against label-derived states, mirroring the leaderboard.
- Sign matters: all train labels are negative. Verify the convention on the
  train turbines before writing any submission file (see PLAN.md §3.4).

`metadata.croissant.json` documents every field. Canonical loaders are
`yaw.io.load_turbine` (ffill + vane + `nac_update` flag) and the cached
aggregates in `cache/agg1min`, `cache/agg10min`, `cache/daily`,
`cache/labels_daily.parquet` (states + `scored` flag) and
`cache/pair_offsets_daily.parquet`, `cache/states_scada.parquet` (SCADA-only states,
`transition`, `boundary_type`) and `cache/encoder_steps.parquet`; schemas in docs/01 and docs/02.

## Machine

32 cores, 60 GB RAM, RTX 5090 (24 GB). Loading all four parquet files fully takes
several GB; cache per-turbine 1-minute aggregates to parquet rather than
re-reading `context.parquet`.
