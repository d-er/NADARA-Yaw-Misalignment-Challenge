# Plan — Nadara / WeDoWind Static Yaw Misalignment Challenge

Written 2026-09-10 (Energy Data Hackdays are 10–11 Sep 2026; the WeDoWind final
round is scored once at its own deadline). Everything below is grounded in probes
run on the actual data in `data/` — numbers quoted are measured, not assumed.

## 0. Strategy in one paragraph

The task decomposes into two sub-problems with very different difficulty:
**(a) find the misalignment *states*** (when did the turbine change) and
**(b) pin the absolute angle of each state** (sign + magnitude). The public
leaderboard proves (a) is solvable from SCADA alone (participant 50: ARI 1.0 in
every tier, including T0) and that (b) is the whole game: the same participant
scores RMSE 3.92 with bias −3.1 at T0 but 0.62 at T1 — one labelled turbine fixes
the calibration. Our plan is therefore: build a robust, anemometer-independent
state segmenter; build several *physics* estimators of the per-state angle (T0);
then calibrate them against the train turbines with leave-one-turbine-out CV
(T1–T3). Submit early to validate, iterate on the 76 scored days, and keep the
final (SSS_WTG06, different site, 185 scored days) as the real target — do not
overfit the PPP-17 board.

## 1. Facts that constrain the plan

| Item | Value |
|---|---|
| Train (labelled) | PPP_WTG12 (195 days, −2.3±0.6°), WTG13 (312 days, −7.5±1.7°, 9 steps), WTG14 (195 days, −1.7±0.4°). All labels 2023-01..2023-11-20 only. |
| Validate (public LB) | PPP_WTG17, 731 rows, **76 scored days**; constant baseline RMSE 4.80 → the label spread on those days is large (~±5°); true mean ≈ −1.4 (from the all-zero bias). |
| Final (private) | SSS_WTG06, 731 rows, **185 scored days**; constant baseline 3.35; true mean ≈ −3.7; the constant is *in the same band as all-zero* → need a real model. |
| Metric | RMSE (paired bootstrap over *states*, submissions in the same band are tied) → ties broken by ARI on `cluster`. Days inside ~7-day transition windows are not scored. |
| Current LB (validate) | Band 1: 0.62 / 0.67 / 0.70 (all participant 50, ARI 1.0). Best T0: 3.92 (ARI 0.84). Best other team: 4.3–4.9. |
| Tiers | T0 no labels (strongest claim) … T3 all three train turbines. T4 penalised and not open. Counter per tier, best per tier kept → enter **T0 and T3** at minimum. |
| Format | `Results_NN_TIER_x.csv`, columns `turbine_id,date,yaw_misalignment_deg,cluster`, 731 rows from the round's template, PR into the Submissions repo (Git LFS). Validate locally with `validate_submission.py`. |
| Sign | Labels are negative on all three PPP turbines. A sign flip scores worse than a constant. Verify empirically (§3.4) before every submission. |

## 2. What the data probes showed (drives every design choice)

1. **Null = unchanged.** Per-signal update fraction is 0.5–0.8 except `NacDir`
   (6 %). Forward-fill per turbine; median cadence 12 s. 727/731 days present on
   validate and test; a handful of days have <1000 rows.
2. **The vane cannot see the misalignment.** Daily median vane sits at ≈ −3° on
   every PPP turbine regardless of label (corr with label −0.33). The controller
   drives the vane to its setpoint; the static error *is* the vane's bias. Signal
   must come from *power/rotor response vs vane* or from *neighbours*.
3. **OpenOA-style cos^k fit tracks changes but not levels.** Monthly θ₀ on WTG13
   moves −4 → −1 across the Aug-2023 step (label −7.3 → −4.2): the *delta* is
   right, the level is off by ~4° and the ranking of the three turbines is wrong
   (README warns of exactly this). Cause: nacelle anemometer/vane sit in the
   rotor wake, so the vane reading is a damped (gain <1), offset function of the
   true inflow angle, and nacelle wind speed is itself biased under yaw. The fit
   is also unstable when the exponent k is free (k ranges 0.3–13 month to month) —
   fix k per turbine type or constrain it.
4. **Neighbour-consensus nacelle heading is a state detector.** Daily circular-mean
   `NacDir(WTG13) − NacDir(WTG11)` is +13° Jan–Jul 2023 and ≈ 0–3° from Aug 2023,
   coincident with the label step. PPP_WTG17 vs WTG15/18/33 all shift by −7…−10°
   at Dec-2023 → Jan-2024, so PPP-17 has at least two states split at the turn of
   the year. `WindDir` = `NacDir` + vane, so it carries the same encoder offset.
5. **Caveats for that detector.** Encoders get re-referenced: WTG12 jumps by ≈ +138°
   vs all neighbours from Aug 2023 (no label change), SSS_WTG06 shows ±130° months.
   A jump is an *event*, not necessarily a *state change*; treat large jumps as
   encoder re-references and estimate states from power physics inside each
   segment. Restrict comparisons to un-waked sectors and use per-sector offsets.
6. **Vane freezes.** WTG13 vane is constant at −34.0 (8–16 Feb 2024) and +39.1
   (9–12 Oct 2024) with zero NacDir updates: sensor stuck / turbine parked. Filter
   any day with vane std ≈ 0 or no NacDir updates.
7. **Labels are smoothed.** WindFit (Sereema) is a nacelle box with its own wind
   sensor that fits the power-optimal heading over several days; the label drifts
   for ~a week before each step and is otherwise piecewise-constant. Our output
   should be piecewise-constant per state, then optionally smoothed with the same
   ~7-day kernel — but only scored days *outside* transitions count, so the
   per-state constant is what matters.
8. **Jan-2023 WTG13 event pins the sign convention.** Vane median shifted from
   ≈ −0.5 to ≈ −3 around 14–18 Jan (vane recalibrated by −2.5°) while the label
   went −11.5 → −8.1. The controller yawed the nacelle by the recalibration amount
   and the misalignment *shrank*, which fixes the geometric sign of the label
   relative to our derived vane (`WindDir − NacDir`). Reproduce this check in code
   and assert it before submitting.

## 3. Method architecture

Build as a small package (`src/yaw/`), each stage cached to parquet so the
59 M-row context file is read once.

### 3.1 Data engine
- Load with pyarrow/polars, per turbine: ffill, `vane = wrap(WindDir − NacDir)`,
  resample to 1-min (mean, circular mean for directions, std, count) and keep the
  12-s stream for §3.3-C4 only.
- Quality flags per minute: operating (`Power > ~100`, `PitchAngle < 0` for
  partial load), not rated, not frozen, not curtailed (power vs rotor-speed
  outliers), not waked (sector mask from layout: bearing to each neighbour ± ~20°
  within ~5 D — layout distances are scaled but bearings are exact).
- Neighbour tables: for every target, the ≤4 nearest neighbours' 1-min power,
  wind speed, circular-mean NacDir/WindDir joined on the minute.

### 3.2 State segmentation (gives `cluster`, and the windows for estimation)
Multivariate change-point detection (ruptures, PELT / binary segmentation, or a
Bayesian online CPD) on **daily** features:
- neighbour-consensus heading offset (median over un-waked sectors of
  `NacDir_target − NacDir_j`, per neighbour, plus a robust consensus);
- daily median/IQR vane, share of days with no NacDir updates;
- rolling 10–14-day power-vs-vane θ₀ (§3.3-C1/C2) as a slower channel;
- rolling power-ratio to neighbours at matched wind speed.
Post-process: merge segments < 7 days, mark ±7 days around each change as
transition, cap the number of states with a BIC-like penalty. Validate on the
train turbines: ARI vs true states (derive true states by segmenting the label
series itself). Never use calendar adjacency of *scored* days as a feature —
segments come from SCADA, which is allowed.

### 3.3 Per-state angle estimators (T0 physics — build several, ensemble)
- **C1 OpenOA cos^k fit**, but with k fixed (fit one k on all turbines, ≈ 3–5),
  wind-speed bins in partial load, `P/ws³` normalisation, and *bin-median* not
  mean; bootstrap CI. Known to underestimate → treated as a feature, not the answer.
- **C2 Anemometer-free power ratio.** In each vane bin, use `P_target / P_ref`
  (or `P_target − f(P_ref)`) with a neighbour's power at the same minute as the
  wind reference, un-waked sectors only; fit the cos^k peak. Removes the
  nacelle-wind-speed-under-yaw bias that corrupts C1.
- **C3 Rotor-speed method (Castellani et al.).** In below-rated operation rotor
  speed at a given power is minimal when aligned; regress `RotSpeed` on power and
  vane, find the vane at optimum. No anemometer involved.
- **C4 Yaw-step natural experiments (the 12-s-data edge).** Each yaw manoeuvre
  (300–1000 NacDir updates/day) is a step change of misalignment by a known Δ.
  Compare power immediately after vs before (30–120 s windows, wind held constant
  via the neighbour's power) as a function of the pre-step vane. The sign of
  ΔP/Δθ flips at the true optimum → a robust, curve-shape-free estimate of θ₀.
  Genuinely novel for this dataset; expect it to be the cleanest T0 estimator.
- **C5 Vane transfer-function gain.** Regress short-term changes of the vane
  against neighbour-consensus wind-direction changes (or against NacDir steps
  while wind is steady) to estimate the vane gain b < 1. Divide C1–C4 angle
  deltas by b to convert vane-degrees into physical degrees; this is the likely
  fix for the 2–4× underestimate.
Priority after §8: C4 + C5 first, C1/C2 as baselines, C3 optional.
Each estimator returns (θ̂, CI, n) per state; combine by inverse-variance
weighting.

### 3.4 Calibration (T1–T3) and sign check
- Model: `label_state = a + b·θ̂_state` (+ optional per-estimator weights),
  fitted at the *state* level on train turbines (≈ 12–15 states total), with
  leave-one-turbine-out CV. Report CV RMSE/MAE/ARI on train days outside
  transitions using the organisers' rule. Use Huber loss; the effective sample
  size is tiny, so keep ≤ 3 parameters.
- Sign assertion: (i) recovered slope b must be positive; (ii) reproduce the
  Jan-2023 WTG13 relation between the vane recalibration and the label; (iii)
  predicted validate mean near −1.4 and final mean near −3.7 (from all-zero biases)
  is a sanity check, not a target.
- T0 submission uses the physics ensemble with no fitted a, b (only physical
  constants); T3 uses the calibrated mapping. Submit both; they can't hurt each
  other.

### 3.5 Output
Piecewise-constant per state, both turbines, 731 rows each, `cluster` = state id.
Unscored days (transitions, gaps) get the neighbouring state's value. Run
`validate_submission.py` on every file.

### 3.6 Evaluation harness (build first, it drives everything)
`scripts/evaluate.py`: given predictions for a train turbine, compute RMSE/MAE on
labelled days outside ±7-day windows around label steps, ARI vs label-derived
states, and a state-level paired bootstrap — mirrors the leaderboard so local
gains are real.

## 4. Timeline for the hack days

| When | Deliverable |
|---|---|
| Day 1, hrs 0–3 | Data engine + caches; evaluation harness; register participant ID; fork Submissions repo, `git lfs install`. |
| Day 1, hrs 3–6 | Segmenter (§3.2) on all 5 turbines; check ARI on train; look at PPP-17/SSS-06 state plots. |
| Day 1, hrs 6–9 | C1 + C2 estimators; first T0 file for validate → PR (**first score by end of day 1**). Also a T3 file with the linear calibration. |
| Day 2, hrs 0–4 | C3, C4, C5; ensemble; LOTO-CV numbers; decide k, b. |
| Day 2, hrs 4–7 | Iterate on validate board (max 2–3 more numbered submissions — the band structure means small RMSE moves are noise; ARI matters). |
| Day 2, last 2 hrs | Freeze method, produce `_final` files for SSS_WTG06 in T0 and T3, write the methodology note (interpretability + reproducibility are judged), pitch slides with state plots and the yaw-step figure. |

If the final deadline is later than the hackdays, spend the extra time on §5.

## 5. Stretch ideas (only after §3 is scored)
- Hierarchical Bayesian model: shared cos^k shape and vane gain across the fleet,
  per-turbine-state offsets; posterior gives calibrated CIs and a principled T3.
- Transductive use of SSS context turbines: estimate SSS-site vane gain and wake
  sectors from the four SSS neighbours so the calibration transfers across sites.
- Daily-feature gradient boosting (LightGBM) on top of the physics features with
  LOTO-CV — only if it beats the 3-parameter map; with ~15 states it probably won't.
- Detect controller-offset changes vs vane-mount changes separately (C4 response
  differs) — useful for the interpretability part of the judging.

## 6. Risks / open questions
- Sign convention of the label vs our derived vane (resolved empirically in §3.4;
  do not submit without it).
- Encoder re-references confound the heading-offset detector (§2.5): keep it as
  an event channel, confirm each state with a power-based estimator.
- Site SSS is different terrain/hub heights; the vane gain b and wake sectors must
  be re-estimated there, not copied from PPP.
- Few states (~15) for calibration → keep the T3 map tiny, report CIs.
- Rated-power and curtailment periods carry no yaw information — filter hard;
  a bad filter is the usual reason cos-fits go wild (k → 0.3 or 13).

## 7. Sources
- Submissions repo, scoring & tiers: https://github.com/WeDoWind/NADARA-Static-Yaw-Misalignment-Submissions
- Leaderboard: https://wedowind.github.io/NADARA-Static-Yaw-Misalignment-Submissions/
- OpenOA static yaw method: https://openoa.readthedocs.io/en/latest/examples/07_static_yaw_misalignment.html
- Rotor-speed / nacelle-anemometer diagnosis (Castellani et al.): https://www.sciencedirect.com/science/article/pii/S2352467723000796 , https://www.mdpi.com/1996-1073/17/24/6381
- Yawed-turbine transfer functions from SCADA (2025 preprint): https://wes.copernicus.org/preprints/wes-2025-223/
- Data-driven yaw correction (arXiv 2109.08998): https://arxiv.org/abs/2109.08998
- WindFit (label source): https://www.sereema.com/solutions/windfit-yaw-static-misalignment

## 8. Evaluation against the already-scored approaches (added 2026-09-11)

Source: participant 50's two write-ups on the WeDoWind solutions page, plus the
paper their T0 entry reproduces (Jacquet, Marx Hermoso, Pulikollu 2026, JPCS
3224 062049, open access).

### What has been tried
| Entry | Method | Validate score | Weakness they report |
|---|---|---|---|
| P50 T0 (3.92, bias −3.1, ARI 0.84) | Jacquet 2026: ADMM consensus *wind-speed* correction from neighbours + sparse-GP power curve with a multiplicative yaw-loss factor h(γ)=cos^m(γ−γ₀) or squared-exp; γ₀ = vane angle at max power. Fitted per **calendar year** (needed the samples). | Yearly-static → ARI < 1; level biased ≈ −3° | "Detected misalignment assumed static through the year" |
| P50 T1–T3 (0.62–0.70, ARI 1.0) | 48 k-param GNN on daily graphs. Dominant signal = **consensus anchor**: reported WindDir minus neighbour consensus (≤1.5 km) minus own long-run median; a *change* in true misalignment shows as an equal-and-opposite change of that residual (slope fixed at 1 from sensor algebra). Level comes from a Huber loss on the 702 labelled train days. Biggest single gain: **de-stepping encoders** that lose north (1.16 → 0.70). Plain MLP ties the GNN. | **Leave-one-turbine-out RMSE 3.17** vs 4.74 for the best constant | "Level not identifiable from SCADA alone"; needs a stable encoder and a trustworthy reference; peak-of-power-vs-vane is precise but biased, and the bias grows with the misalignment |

### How the plan maps onto that
- **§2.4/§3.2 (neighbour-consensus heading, de-stepping)** — identical in spirit to
  their consensus anchor. Not novel; necessary. ARI 1.0 is reproducible; keep it.
- **C1, C2, C3 (power/rotor vs vane, with or without neighbour wind reference)** —
  the same family as Jacquet 2026. C2's neighbour power reference is a cheaper
  stand-in for their ADMM wind-speed correction. Expect these to land near their
  T0 result (bias ≈ −3°), *because they keep the vane as the x-axis*: the rotor
  wake compresses the vane reading (gain b<1), so the peak location is
  systematically too small. Their own ablation says exactly this. → **demoted to
  baselines / features**, not the answer.
- **C4 (yaw-step natural experiments) + C5 (vane gain)** — not in any scored
  entry. They are the only components that measure in *physical* degrees: a
  NacDir step Δ is a true rotation regardless of encoder offset, so the power
  response to Δ locates the optimum in physical units; and the instantaneous vane
  jump after a step (−b·Δ before the wind moves) gives b directly from thousands
  of events per turbine. Physical misalignment of a state is then
  (vane setpoint − θ₀_vane)/b, which also explains why the naive peak underestimates
  more for larger misalignments. → **promoted to the core; build right after the
  segmenter.**
- Their "level is not identifiable from SCADA" claim is derived from direction
  signals only (Wr = Wt + δn + δv). Power response *does* identify δv; the open
  question is only the vane transfer function, which C5 estimates.

### Can it beat them?
- **Validate, T3:** band 1 is 0.62–0.70 with ARI 1.0 and only ~2 states. The
  paired bootstrap over states makes a *provable* improvement nearly impossible;
  realistic outcome is joining band 1 and edging on RMSE. Don't spend the hackdays
  chasing it.
- **Validate, T0:** the bar is 3.92 with a year-static model. A daily segmenter
  (ARI 1.0) plus a physics level within ~2° gives RMSE ≈ 2 → clear new best T0,
  and T0 is the tier the organisers weight most for the final winner.
- **Final (SSS_WTG06):** their transferable performance is the LOTO number, 3.17,
  vs a 3.35 constant that ties all-zero — nothing yet works on an unseen turbine.
  A method whose level comes from physics rather than labels is the only kind
  that can win this round, and it needs SSS-specific b and wake sectors from the
  four SSS neighbours (all within ~600 m — good for the consensus anchor).

### Changes to the plan
1. Build order: data engine → de-stepping + segmenter → **C4/C5** → C1/C2 as
   cross-checks → C3 only if time.
2. Use 10-min aggregates for everything except C4/C5 and encoder cleaning (their
   finding: sufficient and 10× cheaper).
3. No GNN. Calibration in T3 should fit **fleet-shared physical constants** (k, b,
   vane setpoint handling) on the train turbines, not a level offset — that is
   what transfers to SSS.
4. Sign: their sensor algebra gives θ ≈ −δv; combine with the Jan-2023 WTG13 check
   (§2.8) before any submission.
5. Their T0 implementation only worked per year for lack of samples; our
   segmenter + 12-s data lets C4 work on ~2–4-week windows, so per-state levels
   are feasible where theirs were not.
