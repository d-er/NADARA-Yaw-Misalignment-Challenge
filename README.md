# Nadara / WeDoWind static yaw misalignment challenge

Predict the daily WindFit static yaw misalignment of PPP_WTG17 (public
leaderboard) and SSS_WTG06 (final round) from ~12-second SCADA. See
[PLAN.md](PLAN.md) for the strategy and the evaluation against already-scored
approaches, and `docs/` for the step-by-step lab log.

## Layout

```
data/            raw parquet + layout CSVs (not in git)
cache/           per-turbine aggregates and derived tables, built by scripts/build_cache.py (not in git)
src/yaw/         package: io, aggregate, quality, geometry, labels, evaluate
scripts/         one CLI per pipeline step (01 build_cache, ...)
docs/            lab log, one markdown file per step, with figures in figures/
notebooks/       exploration only; nothing downstream depends on them
tests/           pytest unit tests for the reconstruction rules
```

## Setup and run

```bash
uv venv .venv
uv pip install --python .venv/bin/python pandas pyarrow polars scipy ruptures scikit-learn matplotlib pytest jupyter
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_cache.py --workers 8      # ~1 min, writes cache/
```

## Lab log
1. [Data preparation and cleaning](docs/01_data_preparation.md)
2. [Encoder de-stepping and state segmentation](docs/02_segmentation.md)
