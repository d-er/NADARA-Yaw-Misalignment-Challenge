# Slide figures

Charts for the 5-minute pitch deck "Label-free yaw misalignment". All of them are
bare SVGs: axis labels and annotations are placed on the slide, using the pixel
positions in `figures/slides/ticks.json`.

Source: `scripts/plot_slides.py`. Run `build_cache.py` and `segment_states.py`
first, then

    .venv/bin/python scripts/plot_slides.py

`v1_sss06.svg` also needs the first version's step-2 output; see the script's
docstring for the worktree recipe.

## Used in the deck

| Slide | Figure | Shows |
|---|---|---|
| 2 Starting point | `vane_vs_label.svg` | PPP_WTG13, 2023: daily median vane flat at about −3° while the label moves by 8° |
| 3 Change detection | `raw_pairs.svg` | PPP_WTG17: raw nacelle-heading offset to its 4 neighbours |
| 3 Change detection | `clean_states.svg` | PPP_WTG17: residual per pair, their median r, and the PELT state levels |
| 5 Thresholds | `gap_size.svg`, `gap_duration.svg`, `gap_reref.svg` | Each encoder threshold against the data: it sits in a gap |
| 7 Measuring the level | `level_merged.svg` | PPP_WTG17: daily vane medians, L = their median (−2.8°), the neighbour step pattern centred on 0, the same steps shifted onto L (one arrow per period), and the shaded areas between the placed steps and L, equal above and below (day-weighted mean = L) |
| 8 Results | `result_ppp17.svg`, `result_sss06.svg` | Prediction over the daily evidence for the two leaderboard turbines |

Slide 6 (straight versus crooked vane) is a schematic drawn directly in the slide
and has no data figure.

## Not used in the current deck

`sss_raw.svg`, `sss_clean.svg` (SSS_WTG06 before and after masking excursions),
`nl_jump.svg`, `nl_excursion.svg`, `nl_reref.svg` (narrow versions of the gap
charts), `v1_sss06.svg`, `v2_sss06.svg` (first and present version on SSS_WTG06).

## Numbers quoted on slides 6–8

Written by the script to `ticks.json` under `_level` and `level_merged.marks`:

- L for PPP_WTG17 = −2.8°: median over 721 healthy days of the daily median vane
  in operation (power > 100 kW); 727 days had data.
- Periods: 370 days at −6.62°, 361 days at +1.12°; day-weighted mean −2.80° = L.
- Balance areas: −1413 and +1413 degree-days.

The training-error split on slide 8 (59 % of the squared error from the level;
RMSE 2.18° → 1.40° with the per-turbine mean error removed; mean errors −0.96,
+2.33, −1.22° on PPP_WTG12/13/14) is computed by the same script from
`yaw.submission.predict` against the training labels, under `_training_split`.
