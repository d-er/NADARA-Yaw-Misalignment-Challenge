# Step 4 — Submission builder

*2026-09-11. Code: `src/yaw/submission.py`, `scripts/build_submission.py` (≈ 10 s),
figure `figures/04_submission.png` from `scripts/plot_04_submission.py`. Outputs:
`submissions/Results_<ID>_{T0,T3}_{0,final}.csv`, `cache/predictions_daily.parquet`
(turbine × day × tier: `yaw_misalignment_deg`, `cluster`). `submission_kit/` holds the
organisers' `validate_submission.py` and the two round templates (copied from the
Submissions repo, 2026-09-11).*

## Model

Per turbine-day, `θ = level + b · x`, with `x = r_level(state) − anchor(segment)`:

| Piece | Source | T0 | T3 |
|---|---|---|---|
| states / `cluster` | step 2 PELT states on the consensus heading residual r | same | same |
| deltas between states | Δr with `Δθ = −Δr` (sensor algebra, docs/02) | b = −1 | b fitted |
| segments | runs of states between `encoder`-typed boundaries; r is not comparable across an encoder re-reference, so each segment is anchored separately (anchor = day-weighted mean of `r_level`, `--anchor median` optional) | same | same |
| level of the anchor | **vane setpoint**: median daily vane in operation over `flag_ok` days. The controller nulls the vane at this value; with an unbiased vane it *is* the static misalignment. PPP turbines sit at −2.7…−3.7°, SSS at −0.8…−3.3° | as is | + offset a |

T3 fits `label − vane_setpoint = a + b·x` (Huber) on the scored train days.
Full fit: a = −0.73, b = −0.93 — the label data confirm the slope of −1 and
barely move the level, so T3 is structurally T0 with a 0.7° shift.

Days without a SCADA state (record edges) take the nearest state's value.

Honour rule (no calendar adjacency between scored days): clusters are the step-2
states, computed from the SCADA residual over all 731 days without any knowledge
of the template or of which days are scored. The only calendar prior is the
segmenter's minimum state length of 14 days on the SCADA series.

## Sign checks (both must pass, the script aborts otherwise)

1. Slope of label on x over scored train days: **−0.74** (< 0 ✓; r moves opposite to the label).
2. PLAN.md §2.8, PPP_WTG13 mid-January 2023: derived vane median moved **−2.6°**,
   label moved **+3.0°** (opposite ✓). Consistent with `θ = vane setpoint − (r − anchor)`.

## Scores on the train turbines (organisers' rule, scored days only)

| Model | RMSE | MAE | bias | ARI | state-bootstrap 95 % CI |
|---|---|---|---|---|---|
| T0, mean anchor (label-free → honest) | **2.48** | 1.96 | +0.54 | 0.54 | 1.8–3.3 |
| T0, median anchor | 2.18 | 1.39 | −0.11 | 0.54 | 1.2–3.3 |
| T3 leave-one-turbine-out | 4.43 | 3.89 | +1.06 | 0.54 | |
| best constant (LOTO) | 3.1 / 6.1 / 3.7 per turbine | | | | |

Per turbine (T0, mean anchor): WTG12 2.79 (bias −0.45), WTG13 2.89 (+2.48), WTG14 1.35 (−0.81).
WTG13's level is 4.5° below its vane setpoint — the case a label-free prior cannot
see; its two large label steps are what make the slope identifiable, which is why
the LOTO fit without WTG13 collapses to b ≈ 0 (LOTO T3 is *worse* than T0; the
three-turbine fit is too small to learn anything beyond the sign).

## The anchor choice matters for PPP_WTG17

Its two states are 370 and 361 days long, so "the typical state" is a coin flip:

| anchor | 2023 state | 2024 state | mean over all days |
|---|---|---|---|
| mean (submitted as `_0`) | −6.8 | +1.3 | −2.8 |
| median | −2.8 | +5.3 | +1.2 |

The all-zero entry's bias on the board implies a true mean of ≈ −1.4 on the 76
scored days and the constant baseline (4.80) implies two states ~8–9° apart, which
matches Δr = −8.05 → Δθ = +8. The mean anchor is the minimum-expected-error hedge
under ignorance of which state is typical; on train it costs 0.3 RMSE against the
median anchor. Submit the mean-anchor file first; the median-anchor file is the
natural `_1` if the board reports a bias near +4.

## SSS_WTG06 (final round) — do not submit yet

13 states in 4 encoder-free segments; the r deltas imply swings of −10…+12°,
while the constant baseline (3.35) says the label spread is ≈ ±3°. Either the
turbine really changes that often or r drifts on this site (docs/02 flagged it).
The `_final` files are produced for completeness, but the final is scored once
and is immutable, so they should not be filed until SSS-specific evidence (power
response per state, or a shrunk slope validated on the SSS neighbours) supports
the state levels.

## What comes next

1. File `Results_<ID>_T0_0.csv` and `Results_<ID>_T3_0.csv` (validate round).
2. Read the board's `bias` for T0: near 0 → keep; near +4 → median anchor as `_1`; large negative → the level prior is off, revisit.
3. SSS_WTG06: test the slope on the four SSS neighbours' r series (do their states co-move? — then it is site flow, not misalignment) before any final file.
