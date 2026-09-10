# Step 1 — Data preparation and cleaning

*2026-09-11. Code: `src/yaw/{io,aggregate,quality,labels,geometry}.py`, `scripts/build_cache.py`. Run time ≈ 1 min on 8 workers.*

## What was built

| Cache file | Content | Size |
|---|---|---|
| `cache/agg1min/<tid>.parquet` | 1-minute aggregates of the filled 12-s stream: means of linear signals, std of Power/WindSpeed/vane, vane median, **circular** means of NacDir/WindDir with resultant length R, row count, number of raw NacDir updates, operating fraction | ~1.0 M rows × 16 turbines |
| `cache/agg10min/<tid>.parquet` | same schema at 10 min (for neighbour comparisons, per the leader's finding that 10-min is enough for everything except cleaning) | |
| `cache/daily/<tid>.parquet` | per-day coverage, regime minutes, vane statistics, operating-hours circular mean headings, sensor-health flags | 727–731 rows each |
| `cache/labels_daily.parquet` | labelled turbine-days with PELT states, state level, transition flag, `scored` flag | 702 rows |
| `cache/pair_offsets_daily.parquet` | daily median of wrap(NacDir_a − NacDir_b) and of WindDir for every same-site pair, using 10-min bins where both turbines operate >90 % of the bin | 65 pairs |

## Reconstruction rules (verified, see `tests/test_basics.py`)

1. Sort by `turbine_id, ts`, forward-fill the seven signals per turbine. A raw
   non-null `NacDir` is kept as `nac_update` *before* filling: it counts yaw
   controller actions (≈ 300–650 per day) and is lost otherwise.
2. `vane = wrap(WindDir − NacDir)` to [−180, 180). WindDir is NacDir + vane in this
   SCADA, so it carries the encoder offset; only the vane is encoder-free.
3. Directions are always averaged as unit vectors (sin/cos). A naive mean of
   350° and 10° is 180°; several turbines spend a lot of time near north.
4. Regime flags (scaled units): `operating` = Power > 100 (≈ 2 % of the ~4800
   rated), `partial_load` = operating & Power < 0.85·rated & pitch < 0.5°.
   Pitch sits at −3…−1.7 in partial load, so this cleanly removes above-rated
   time where yaw has no power signature.

## Quality findings

- **Coverage** (`figures/01_coverage.png`): every turbine has 727–731 days; a few
  days per turbine have <1000 rows (flagged `flag_sparse`; 1–8 per turbine). Median
  operating time ≈ 1200 min/day on PPP, ≈ 1250–1300 on SSS.
- **Frozen vane** — PPP_WTG13 reports vane = −34.0 with zero variance and no yaw
  updates on 8–16 Feb 2024, and +39.1 on 9–12 Oct 2024 (flagged `flag_frozen_vane`,
  `flag_no_yaw`; 11 and 7 days). No other turbine has this. These days must be
  excluded from any vane-based estimator.
- **Encoder faults** (`figures/01_vane_and_offsets.png`, right column): PPP_WTG12's
  heading offset against every neighbour jumps to ≈ +138° from Aug 2023 (with
  excursions in Feb and May 2023) — a north re-reference, not a physical rotation
  (its label does not move). SSS_WTG06's offsets vs its four neighbours wander by
  ±10° over months, much more than any PPP pair. Both need the de-stepping /
  drift handling that the leading team reported as their single largest gain.
  → Step 2.
- **PPP_WTG17** shows one clean −8…−10° heading step against WTG15/18/33 in the
  first week of January 2024. WTG16 is too noisy to serve as a reference (as the
  leader also found).
- **Vane setpoint**: all eleven PPP turbines hold a daily median vane of −2.7…−3.7°
  in operation, i.e. the controller nulls the vane at ≈ −3°, not 0. SSS turbines
  differ: WTG04/07 sit at −0.9, WTG05/16 at −3.1, and WTG06 *wanders* between 0
  and −4 over weeks. Whatever produces the label is on top of that setpoint, so
  the vane setpoint must be handled per turbine (and per period on SSS_WTG06).

## Labels (`figures/01_labels_states.png`)

PELT (L2, `pen=20`, `min_size=7`) on each turbine's daily label series gives:

| Turbine | States (level °, days) | Scored days after ±7-day guard and ±0.75° deviation rule |
|---|---|---|
| PPP_WTG12 | −1.73 (93), −2.80 (102) | 180 of 195 |
| PPP_WTG13 | −11.35 (17), −8.40 (87), −7.85 (101), −4.37 (38), −6.01 (39), −7.23 (30) | 217 of 312 |
| PPP_WTG14 | −1.45 (121), −2.16 (74) | 180 of 195 |

The organisers' own transition rule is unknown; the validate round scores 76 of
PPP_WTG17's labelled days, so theirs is probably stricter than this. The
penalty is a judgement call (pen=30 merges WTG14 into one state; pen=4 splits
WTG13 into 13). All evaluation code takes the label table as input, so the rule
can be changed in one place (`yaw.labels.add_states`).

## Sign / convention notes carried forward
- All labels negative; PPP vane setpoint ≈ −3°.
- WTG13 Jan-2023: vane median moves from ≈ −0.5 to ≈ −3 on 14–18 Jan while the
  label goes −11.5 → −8.1 (see PLAN.md §2.8). To be turned into an automated
  sign assertion in the estimator step.

## Next step
Step 2 — encoder de-stepping and the state segmenter: detect steps in each
turbine's heading offset against a robust neighbour consensus, remove
re-reference jumps, then run multivariate change-point detection on the daily
features to produce candidate states for PPP_WTG17 and SSS_WTG06 and an ARI
check on the three train turbines.
