# Step 3 — Per-window angle estimators (C1, C2, C4): mostly a negative result

*2026-09-11. Code: `src/yaw/estimators.py`, `scripts/estimate_angles.py` (≈ 5 min, 5 workers),
figure `figures/03_estimators.png` from `scripts/plot_03_estimators.py`. Outputs:
`cache/state_estimates.parquet` (turbine × window kind × window × estimator:
`theta`, `theta_se`, `k`, `n`, `reference`), `cache/manoeuvre_events/<tid>.parquet`.*

## What was built

| Estimator | Implementation | Units |
|---|---|---|
| C1 `power_vane_fit` | 1-min aggregates, partial load (300 < P < 3600, pitch < 0.5, op_frac > 0.99), un-waked by layout sector on de-stepped WindDir, healthy days only. Power divided by its 0.5 m/s wind-speed-bin median, binned by 1-min vane mean, bin medians fitted with A·cos^k(v − θ₀), k = 2 fixed (also free). | vane degrees |
| C2 | same, but the wind reference is the nearest good neighbour's simultaneous power (150-unit bins) instead of the nacelle anemometer | vane degrees |
| C4 `manoeuvres` + `manoeuvre_events` + `manoeuvre_fit` | Yaw manoeuvres = runs of NacDir updates < 60 s apart (≈ 100–150 k per turbine; ramps of ~0.5°/s, 2–30°). For each: ln(P_after/P_before) − same for the reference, windows −6…−2 min and +2…+6 min, both windows fully in partial load / un-waked. Huber slope of lnR vs signed rotation Δ; θ = atan(slope/k) (sign per `docs/estimators_theory.tex` Prop. C4; an earlier build had the sign flipped). | physical degrees (no vane involved) |

Windows evaluated: label states (train, scored days), SCADA states from step 2,
calendar quarters, whole record. 9–26 k usable manoeuvre events per turbine.

A detail that matters for anything at 12-s resolution: the derived vane
(`WindDir − NacDir`) jumps by exactly −ΔNacDir at every NacDir update until the
next WindDir update, because both are change-based and asynchronous. The
"instantaneous vane jump after a step" idea for the vane gain (C5 in PLAN.md) is
therefore an artefact of the encoding and was dropped.

## Results (figure row 1: quarterly estimates vs label on the train turbines)

| | PPP_WTG12 (label −1.7 / −2.8) | PPP_WTG13 (−11 … −4) | PPP_WTG14 (−1.4 / −2.2) |
|---|---|---|---|
| C1 k=2, whole record | −7.9 ± 0.5 | +2.4 ± 0.5 | −9.6 ± 0.4 |
| C2 k=2, whole record | −5.0 ± 3.3 | −18 ± 2.9 | −18 ± 5.6 |
| C4, whole record | +1.5 ± 0.5 | −0.4 ± 0.7 | −3.1 ± 0.4 |
| quarter-to-quarter swing of C1 inside one label state | 4° | 9° | 5° |

None of the three ranks the turbines correctly (WTG13 should be far below the
other two) and none reproduces the label levels; the quoted standard errors are
5–10× smaller than the real quarter-to-quarter scatter, i.e. all three are
dominated by systematic effects, not sampling noise.

### Why C1/C2 fail here (figure row 2)
- The controller keeps the vane inside a ±8° deadband (occupancy panel), and over
  that range cos² varies by only 2 %. The measured normalised-power curves vary by
  ±3 % and **change shape between quarters of the same label state** (WTG13:
  monotone rising in 2023Q1, monotone falling in 2023Q2, peaked at +1° in
  2023Q4). A 1 % sector- or season-dependent effect (terrain, turbulence,
  wind-direction change rate correlating with vane deviation) moves the fitted
  peak by several degrees. The free-k fits confirm it: k wanders 0.2 → 6.
- C2 removes the nacelle anemometer but inherits the same x-axis problem and adds
  the neighbour's own sector dependence; its uncertainty is 3–6°.
- This is the failure mode the README warned about and the leading team's T0
  reproduces (bias −3°, ARI 0.84). Fixing it means modelling the sector
  dependence explicitly — not a hackday task.

### Why C4 is not the clean estimator hoped for (figure row 3)
- The response exists and is large on some turbine-years (PPP_WTG17 2023:
  +10 % power gain after a −20° rotation, nothing after +20°), but it is not
  consistent across turbines: SSS_WTG06 has the opposite slope, PPP_WTG12 gives
  +1.5° while its label is −2, and WTG13 (label −8) comes out near zero.
- **Null test:** PPP_WTG17's manoeuvre times applied to the power ratio of two
  unrelated turbines (WTG14/WTG08) give slope −0.0010 ± 0.0002 per degree, a third
  of the "real" slope. Clockwise (veering) wind shifts systematically coincide
  with a relative power drop between neighbours — a meteorological confounder
  the reference-turbine normalisation does not remove. Also, the misalignment
  before a manoeuvre is only a fraction f of Δ (the wind shifts gradually), which
  scales any level estimate by an unknown f.
- Different references disagree by 2–4° (WTG17 vs WTG15/18/33: 1.9, 4.3, 4.5).
- C4's implied 2023 → 2024 change on PPP_WTG17 (≈ +5°) agrees in sign with the
  consensus residual's Δr = −8° (→ Δθ ≈ +8°) but is only ~60 % of it (the
  unknown fraction f, see the theory note). Until the confounder is differenced
  out (e.g. against the null distribution per manoeuvre direction sector), C4
  must not be used for levels.

### What does carry information (figure row 2, right)
The consensus heading residual r (step 2) separates WTG13's label states
cleanly (−8.4 at r ≈ +3.5, −4.4 at r ≈ +0.6, −11.4 at r ≈ +4), but its *level*
per turbine is arbitrary (WTG12 sits at r ≈ −30 with label −2.5). Leave-one-
turbine-out Huber fit of label on r − median(r):

| Predictor | LOTO RMSE (scored train days) | Constant baseline |
|---|---|---|
| r − own median (7-day) | **2.6** | 4.0 |

which matches the leading team's reported LOTO of 3.17 for their anchor.

## Consequences for step 4 (submission builder)

1. **States** come from step 2 (SCADA residual), **deltas between states** from
   Δr with sign θ = −r (sensor algebra), **level** from a fleet prior fitted on
   the train turbines (T3) — the same structure the leaderboard leader used. For
   T0 the level has to be a physical prior with no labels; the physics
   estimators above cannot provide it at better than ±5°, so T0 will be the T3
   structure with the level replaced by the fleet-wide vane-setpoint argument
   (all PPP turbines null the vane at ≈ −3°), and should be expected to land
   near the existing 3.9 T0 score, not beat it.
2. Keep `cache/state_estimates.parquet` as *features* only; do not put C1/C2/C4
   levels into a submission.
3. The sign check from PLAN.md §2.8 (WTG13 Jan-2023 vane recalibration) remains
   the only independent sign evidence; implement it in step 4 before writing
   any file.
4. If time remains after a first PR: difference C4 against its null per
   wind-direction sector, which is the one path to a label-free level.
