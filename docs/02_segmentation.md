# Step 2 — Encoder de-stepping and SCADA-only state segmentation

*2026-09-11. Code: `src/yaw/consensus.py`, `scripts/segment_states.py` (≈ 3 min). Figures `figures/02_residuals.png`, `figures/02_states.png` from `scripts/plot_02_states.py`.*

## Idea

Reported heading `Nr = Nt + δn` (encoder offset δn) and true misalignment
`θ = Wt − Nt`, so against a neighbour consensus of the true wind,
`Nr − consensus = −θ + δn + const`. **Changes** of that residual are −Δθ in
physical degrees as long as δn is constant; encoder re-references change δn by
arbitrary amounts and must be removed first. This is the "consensus anchor" the
leading team built their model on; we use it to *propose states*, not to fix
levels.

## Pipeline

1. **Pair series.** For each turbine and its 4 nearest same-site references
   (excluding `PPP_WTG12`, `PPP_WTG16`, whose headings are too unstable to serve
   as references — the same two the leader rejected), 10-min bins where both
   operate > 90 % of the bin; heading difference wrapped; daily circular median.
2. **Encoder steps.** Each pair series is unwrapped and segmented with PELT (L1
   cost, `min_size=10`, `pen=150`). Level shifts ≥ 25° that appear within ±3 days,
   same sign, in a majority (≥ 60 %, ≥ 2) of a turbine's pairs are credited to
   that turbine's encoder and subtracted from its NacDir/WindDir
   (`cache/encoder_steps.parquet`: 67 steps on 10 turbines; none on PPP_WTG17).
3. **Excursion masking.** After de-stepping, days whose pair offset sits > 30°
   from its 31-day rolling median are dropped (multi-day frame excursions shorter
   than `min_size`).
4. **Sector correction.** Per pair, subtract the circular median offset in the
   neighbour's 30° wind-direction sector (terrain / wake dependence), then take
   the daily median residual (≥ 12 bins). Pairs with an unusually noisy residual
   (daily IQR > 2× the turbine's median) are dropped.
5. **r_i(t)** = median over the surviving pairs; daily noise σ ≈ 2° (MAD).
6. **States.** PELT (L2) on the 7-day rolling median of r_i, `pen=400`,
   `min_size=14`; ±7-day transition guard; a state boundary whose level jump
   exceeds 15° is typed `encoder`, otherwise `candidate` (a possible misalignment
   change). Output `cache/states_scada.parquet`.

## What the data showed on the way

- The naive daily pair offsets have std ≈ 40° for the WTG12/13/14 cluster and
  the SSS turbines, because on 10–17 % of days a turbine's heading sits
  100–180° away from every neighbour for several days *while producing normally*
  (e.g. WTG14 on 2023-01-02: heading 160°, neighbours 0°, both at ~500 power,
  resultant length 1.0). These are multi-day encoder-frame excursions, not flow;
  a power threshold does not remove them. PPP_WTG17 has almost none (0.4 % of
  days). De-stepping + masking handles them.
- After the fix, r tracks the inverted label at WTG13's two big events (20 Jan
  and 27 Jul 2023). At the July event Δr ≈ −9° for a label step of +3.65°: the
  service visit also re-referenced the encoder by a few degrees. **A small
  encoder re-reference and a misalignment change are indistinguishable in r**;
  r gives boundaries reliably, magnitudes only as a soft prior.
- Label steps below ~1° (WTG12, WTG14) are under the daily noise floor; they cost
  almost nothing in RMSE but cap the ARI achievable from r alone.

## Results

| Turbine | SCADA states (start → level of r) | ARI vs label states (scored days) |
|---|---|---|
| PPP_WTG12 | 7 states, 3 encoder-typed; Jul–Oct 2023 is a −30° frame excursion the de-stepping only partly removed | 0.47 |
| PPP_WTG13 | 2: 2023-01-01 (+3.6) → **2023-07-27** (−1.2) | 0.37 (labels have 6 states; the 0.5–2° ones are invisible) |
| PPP_WTG14 | boundaries 2023-01-23, **2023-09-23**, 2024-02-23, 2024-03-08, 2024-11-25 (encoder) | 0.79 (label step 2023-09-05) |
| PPP_WTG17 | **2 states: 2023-01-01 → 2024-01-05 (r = +4.8), 2024-01-06 → end (r = −3.25); Δr = −8.05°** | — (leader dated the change 2024-01-06 and found −9°) |
| SSS_WTG06 | 13 states; r wanders between −16 and +11 with three encoder-typed jumps (2023-11-03, 2024-05-22, 2024-09-14) | — |

The PELT penalty was chosen on the train turbines (sweep in the log): lower
penalties split WTG17's 2023 into up to 13 states and SSS_WTG06 into 50 — noise,
not physics.

## Consequences for step 3 (estimators)

- PPP_WTG17: estimate one level per state; if the physics estimators agree with
  Δθ ≈ +8° between the two states, the sign convention is confirmed independently
  of the labels.
- SSS_WTG06 is the hard case: either its encoder drifts (then the r-states are
  partly artefacts and the final answer should be smoother) or the turbine really
  changes often. The power-based estimators must decide; do not trust r levels
  there.
- Use `transition` to exclude ±7 days around boundaries when pooling data per
  state; use `boundary_type == 'encoder'` to forbid using Δr as a prior across
  that boundary.

## Next step
Step 3 — per-state angle estimators: C4 (yaw-step natural experiments on the
12-s stream) and C5 (vane gain) first, then C1/C2 (power-vs-vane fits) as
baselines; evaluation harness on the train turbines with `yaw.evaluate.score`.
