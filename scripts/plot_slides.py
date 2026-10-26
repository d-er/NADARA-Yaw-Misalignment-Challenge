"""Charts for the 5-minute committee deck, as bare SVG (no text: labels sit on the slide).

Reads cache/ (run build_cache.py and segment_states.py first) and writes
figures/slides/*.svg plus figures/slides/ticks.json, the pixel positions of the
axis ticks so the slide can place its own labels.

  vane_vs_label.svg   PPP_WTG13: daily vane median vs label (the vane cannot see theta)
  raw_pairs.svg       PPP_WTG17: raw nacelle-heading offset to its 4 neighbours
  clean_states.svg    PPP_WTG17: per-pair residual, their median r, PELT state levels
  sss_raw.svg         SSS_WTG06: raw heading offset to its neighbours, excursions shaded
  sss_clean.svg       SSS_WTG06: residual r after masking the excursions
  result_ppp17.svg    PPP_WTG17: prediction and the daily evidence mapped the same way
  result_sss06.svg    SSS_WTG06: same
  nl_jump.svg         jump sizes vs the 25° attribution threshold and the largest misalignment change
  nl_excursion.svg    interval lengths vs the 45-day excursion threshold
  nl_reref.svg        re-reference size vs the 15° threshold
  gap_size.svg        wide version of nl_jump (thresholds slide)
  gap_duration.svg    wide version of nl_excursion (thresholds slide)
  gap_reref.svg       wide version of nl_reref (thresholds slide)
  v1_sss06.svg        SSS_WTG06 residual and states with the first version (de-stepping)
  v2_sss06.svg        SSS_WTG06 residual and states with the present version (masking)

ticks.json also holds the numbers quoted on the level and results slides (_level, _training_split).
  level_merged.svg    PPP_WTG17: daily median vane (healthy days), the level L (their median), the step
                      pattern from the neighbours centred on 0, the same steps shifted onto L (one arrow
                      per period), and the balance areas between the placed steps and L (equal above/below)

v1_sss06.svg needs the first version's step-2 output. Produce it in a separate
worktree so the main cache is not touched:
  git worktree add --detach ../yaw_v1 540fa11
  mkdir ../yaw_v1/cache && ln -s $PWD/cache/agg10min ../yaw_v1/cache/ && cp cache/labels_daily.parquet ../yaw_v1/cache/
  ln -s $PWD/data ../yaw_v1/data && (cd ../yaw_v1 && $PWD/.venv/bin/python scripts/segment_states.py)
  python scripts/plot_slides.py --v1-cache ../yaw_v1/cache
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"
OUT = ROOT / "figures" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#14232E"
GRID = "#D9D6CE"
BLUE = "#2F6F9F"
ORANGE = "#D9622B"
GREY = "#8C969E"
PAIR_COLS = ["#2F6F9F", "#6A9FC4", "#8C969E", "#B4BCC2"]

T0, T1 = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-01-01")
ticks = {}


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


class Chart:
    def __init__(self, name, w, h, ymin, ymax, x0=T0, x1=T1, yticks=(), pad=8):
        self.name, self.w, self.h = name, w, h
        self.ymin, self.ymax, self.x0, self.x1, self.pad = ymin, ymax, x0, x1, pad
        self.parts = []
        months = pd.date_range(x0, x1, freq="3MS") if (x1 - x0).days > 400 else pd.date_range(x0, x1, freq="MS")
        ticks[name] = {
            "w": w, "h": h,
            "y": {str(v): round(self.Y(v), 1) for v in yticks},
            "x": {m.strftime("%Y-%m"): round(self.X(m), 1) for m in months},
        }
        for v in yticks:
            self.parts.append(f'<path d="M0 {self.Y(v):.1f}H{w}" stroke="{GRID}" stroke-width="2"/>')
        self.parts.append(f'<path d="M0 {h - 1}H{w}" stroke="{INK}" stroke-width="2"/>')

    def X(self, d):
        return (pd.Timestamp(d) - self.x0).days / (self.x1 - self.x0).days * self.w

    def Y(self, v):
        v = min(max(v, self.ymin), self.ymax)
        return self.pad + (self.ymax - v) / (self.ymax - self.ymin) * (self.h - 2 * self.pad)

    def dots(self, dates, vals, color, size=6, opacity=1.0):
        d = "".join(f"M{self.X(t):.1f} {self.Y(v):.1f}h0" for t, v in zip(dates, vals) if np.isfinite(v))
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{size}" stroke-linecap="round" opacity="{opacity}" fill="none"/>')

    def line(self, dates, vals, color, width=3, gap_days=3, opacity=1.0):
        """Polyline that breaks across gaps longer than gap_days or NaNs."""
        segs, cur, last = [], [], None
        for t, v in zip(pd.to_datetime(dates), vals):
            if not np.isfinite(v) or (last is not None and (t - last).days > gap_days):
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
            if np.isfinite(v):
                cur.append((self.X(t), self.Y(v)))
                last = t
        if len(cur) > 1:
            segs.append(cur)
        d = "".join("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in s) for s in segs)
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linejoin="round" opacity="{opacity}"/>')

    def hline(self, v, color, width=2, dash=None, opacity=1.0):
        dd = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="M0 {self.Y(v):.1f}H{self.w}" stroke="{color}" stroke-width="{width}"{dd} opacity="{opacity}"/>')

    def steps(self, spans, color, width=6, opacity=1.0, dash=None):
        """spans: list of (start, end_inclusive, level); vertical joins between adjacent spans."""
        d, prev = "", None
        for s, e, v in spans:
            xs, xe, y = self.X(s), self.X(pd.Timestamp(e) + pd.Timedelta(days=1)), self.Y(v)
            d += (f"L{xs:.1f} {y:.1f}" if prev is not None else f"M{xs:.1f} {y:.1f}") + f"L{xe:.1f} {y:.1f}"
            prev = y
        self.parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linejoin="round" opacity="{opacity}"/>')

    def band(self, start, end, color, opacity=0.18):
        xs, xe = self.X(start), self.X(end)
        self.parts.append(f'<rect x="{xs:.1f}" y="0" width="{max(xe - xs, 2):.1f}" height="{self.h}" fill="{color}" opacity="{opacity}"/>')

    def vline(self, d, color, width=2, dash="8 8"):
        x = self.X(d)
        self.parts.append(f'<path d="M{x:.1f} 0V{self.h}" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')

    def save(self):
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}">' + "".join(self.parts) + "</svg>")
        (OUT / f"{self.name}.svg").write_text(svg)
        print(f"{self.name}.svg  {len(svg) / 1024:.1f} KB")


daily = {t: pd.read_parquet(CACHE / "daily" / f"{t}.parquet") for t in ["PPP_WTG13", "PPP_WTG17"]}
labels = pd.read_parquet(CACHE / "labels_daily.parquet")
pairs = pd.read_parquet(CACHE / "pair_offsets_daily.parquet")
pres = pd.read_parquet(CACHE / "pair_residual_daily.parquet")
states = pd.read_parquet(CACHE / "states_scada.parquet")
preds = pd.read_parquet(CACHE / "predictions_daily.parquet")
plan = pd.read_parquet(CACHE / "frame_plan.parquet")
hres = pd.read_parquet(CACHE / "heading_residual_daily.parquet")


def raw_offsets(target):
    nb = hres[hres.turbine_id == target].neighbours.mode().iloc[0].split(",")
    out = {}
    for n in nb:
        a = pairs[(pairs.a == target) & (pairs.b == n)].set_index("date").dNacDir
        b = -pairs[(pairs.a == n) & (pairs.b == target)].set_index("date").dNacDir
        s = pd.concat([a, b]).sort_index()
        out[n] = pd.Series(wrap(s.values), s.index)
    return out


def state_spans(target):
    m = preds[preds.turbine_id == target].merge(states[states.turbine_id == target], on=["turbine_id", "date"], how="left")
    g = m.groupby("cluster").agg(s=("date", "min"), e=("date", "max"), pred=("yaw_misalignment_deg", "first"),
                                 lvl=("r_level", "first"))
    g = g.sort_values("s")
    const = (m.yaw_misalignment_deg + m.r_level).dropna().iloc[0]  # L + alpha (one encoder-free segment)
    return g, const, m


# 1. The vane cannot see theta: PPP_WTG13 in the labelled period
d = daily["PPP_WTG13"]
lab = labels[labels.turbine_id == "PPP_WTG13"]
c = Chart("vane_vs_label", 1040, 600, -14, 2, pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-01"),
          yticks=(-12, -8, -4, 0))
ok = d[d.flag_ok & (d.date < "2023-12-01")]
c.dots(ok.date, ok.vane_med, BLUE, size=8, opacity=0.85)
c.line(lab.date, lab.yaw_misalignment_deg, INK, width=6)
c.save()

# 2. PPP_WTG17 raw offsets and cleaned residual with states
raw = raw_offsets("PPP_WTG17")
c = Chart("raw_pairs", 1160, 300, -30, 30, yticks=(-20, 0, 20))
for col, (n, s) in zip(PAIR_COLS, raw.items()):
    c.dots(s.index, s.values, col, size=5, opacity=0.8)
c.save()

g, const, m = state_spans("PPP_WTG17")
c = Chart("clean_states", 1160, 300, -12, 14, yticks=(-10, 0, 10))
pr = pres[pres.turbine_id == "PPP_WTG17"]
for col, (n, s) in zip(PAIR_COLS, pr.groupby("neighbour")):
    c.dots(s.date, s.resid, col, size=4, opacity=0.35)
st = states[states.turbine_id == "PPP_WTG17"]
c.line(st.date, st.r_smooth, INK, width=3)
c.steps(list(zip(g.s, g.e, g.lvl)), ORANGE, width=7)
c.save()

# 3. SSS_WTG06 raw offsets with excursions, and r after masking
raw = raw_offsets("SSS_WTG06")
c = Chart("sss_raw", 1160, 300, -180, 60, yticks=(-150, -100, -50, 0, 50))
ex = plan[(plan.turbine_id == "SSS_WTG06") & (plan.kind == "excursion")]
for _, e in ex.iterrows():
    c.band(e.start, e.end, ORANGE, opacity=0.22)
for col, (n, s) in zip(PAIR_COLS, raw.items()):
    c.dots(s.index, s.values, col, size=5, opacity=0.8)
c.save()
excursion_days = int(sum((e.end - e.start).days for _, e in ex.iterrows()))

g6, const6, m6 = state_spans("SSS_WTG06")
c = Chart("sss_clean", 1160, 300, -15, 15, yticks=(-10, 0, 10))
for _, e in ex.iterrows():
    c.band(e.start, e.end, ORANGE, opacity=0.22)
st6 = states[states.turbine_id == "SSS_WTG06"]
r6 = hres[hres.turbine_id == "SSS_WTG06"]
c.dots(r6.date, r6.r, GREY, size=5, opacity=0.8)
c.save()

# 4. Results: prediction and daily evidence (L + alpha - r), same mapping as the model
for tid, name, gg, cc in [("PPP_WTG17", "result_ppp17", g, const), ("SSS_WTG06", "result_sss06", g6, const6)]:
    ch = Chart(name, 780, 440, -16, 10, yticks=(-15, -10, -5, 0, 5))
    hr = hres[hres.turbine_id == tid]
    stt = states[states.turbine_id == tid]
    ch.dots(hr.date, cc - hr.r, GREY, size=5, opacity=0.55)
    ch.line(stt.date, cc - stt.r_smooth, "#5E6A73", width=2.5)
    spans = list(zip(gg.s, gg.e, gg.pred))
    if tid == "SSS_WTG06":  # 2024 states depend on the thresholds: drawn lighter
        firm = [sp for sp in spans if sp[0] < pd.Timestamp("2024-05-01")]
        weak = [sp for sp in spans if sp[0] >= pd.Timestamp("2024-05-01")]
        ch.steps(firm, ORANGE, width=8)
        ch.steps([firm[-1]] + weak, ORANGE, width=6, opacity=0.4)
    else:
        ch.steps(spans, ORANGE, width=8)
    ch.save()

# 5. Encoder thresholds against the data: each one sits in a gap.
#    Misalignment changes = jumps between neighbouring SCADA state levels of r, all turbines.
lv = states.groupby(["turbine_id", "state"]).r_level.first()
max_change = float(lv.groupby(level=0).diff().abs().max())
steps_all = pd.read_parquet(CACHE / "encoder_steps.parquet")
spans_all = plan.assign(days=(plan.end - plan.start).dt.days)
rereference = spans_all[spans_all.rereference]


def number_line(name, vmax, threshold, band, dots, dot_color, extra=None):
    w, h, y = 476, 64, 40
    X = lambda v: min(v, vmax) / vmax * w
    p = [f'<rect x="0" y="22" width="{X(band):.1f}" height="36" fill="{BLUE}" opacity="0.35"/>',
         f'<path d="M0 {y}H{w}" stroke="{INK}" stroke-width="2"/>']
    for v, col in ([(v, dot_color) for v in dots] + (extra or [])):
        p.append(f'<path d="M{X(v):.1f} {y}h0" stroke="{col}" stroke-width="12" stroke-linecap="round" opacity="0.55"/>')
    p.append(f'<path d="M{X(threshold):.1f} 0V{h}" stroke="{INK}" stroke-width="3" stroke-dasharray="6 5"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">' + "".join(p) + "</svg>"
    (OUT / f"{name}.svg").write_text(svg)
    ticks[name] = {"w": w, "threshold_x": round(X(threshold), 1)}
    print(f"{name}.svg  {len(svg) / 1024:.1f} KB")


number_line("nl_jump", 180, 25, max_change, steps_all.jump.abs().values, INK)
number_line("nl_excursion", 120, 45, 0, spans_all[spans_all.kind == "excursion"].days.values, ORANGE,
            extra=[(v, INK) for v in spans_all[(spans_all.kind == "frame") & (spans_all.days < 120)].days.values])
number_line("nl_reref", 180, 15, max_change, rereference["corr"].abs().values, INK)

# 6. Wide gap charts for the thresholds slide
def gap_chart(name, vmax, threshold, band, dots, extra=None, w=1400, h=72):
    y = h / 2
    X = lambda v: min(v, vmax) / vmax * w
    p = [f'<rect x="0" y="{y - 20}" width="{X(band):.1f}" height="40" fill="{BLUE}" opacity="0.35"/>' if band else "",
         f'<path d="M0 {y}H{w}" stroke="{INK}" stroke-width="2"/>']
    for v, col in dots + (extra or []):
        p.append(f'<path d="M{X(v):.1f} {y}h0" stroke="{col}" stroke-width="15" stroke-linecap="round" opacity="0.5"/>')
    p.append(f'<path d="M{X(threshold):.1f} 0V{h}" stroke="{INK}" stroke-width="4" stroke-dasharray="8 6"/>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">' + "".join(p) + "</svg>"
    (OUT / f"{name}.svg").write_text(svg)
    ticks[name] = {"w": w, "h": h, "threshold_x": round(X(threshold), 1), "x_per_unit": round(w / vmax, 3)}
    print(f"{name}.svg  {len(svg) / 1024:.1f} KB")


gap_chart("gap_size", 180, 25, max_change, [(v, INK) for v in steps_all.jump.abs().values])
gap_chart("gap_duration", 150, 45, 0,
          [(v, ORANGE) for v in spans_all[spans_all.kind == "excursion"].days.values],
          [(v, INK) for v in spans_all[spans_all.kind == "frame"].days.values])
gap_chart("gap_reref", 180, 15, max_change, [(v, INK) for v in rereference["corr"].abs().values])

# 7. First version (de-stepping) against the present one on SSS_WTG06, same axes
ap = argparse.ArgumentParser()
ap.add_argument("--v1-cache", type=Path, default=None)
args = ap.parse_args()


def version_chart(name, st):
    x = st[st.turbine_id == "SSS_WTG06"].sort_values("date")
    ch = Chart(name, 800, 300, -20, 20, yticks=(-15, 0, 15))
    ch.dots(x.date, x.r, GREY, size=5, opacity=0.6)
    g = x.groupby("state").agg(s=("date", "min"), e=("date", "max"), lvl=("r_level", "first")).sort_values("s")
    ch.steps(list(zip(g.s, g.e, g.lvl)), ORANGE if name.startswith("v2") else INK, width=6)
    ch.save()
    ticks[name]["n_states"] = int(len(g))
    ticks[name]["level_range"] = [round(float(g.lvl.min()), 1), round(float(g.lvl.max()), 1)]


version_chart("v2_sss06", states)
if args.v1_cache is not None:
    version_chart("v1_sss06", pd.read_parquet(args.v1_cache / "states_scada.parquet"))

# 8. The level L: how it is measured and how it places the step pattern (PPP_WTG17)
#    L = median over healthy days of the daily median vane in operation (as in yaw.submission.load_inputs).
d17 = daily["PPP_WTG17"]
ok17 = d17[d17.flag_ok & d17.vane_med.notna()]
L17 = float(ok17.vane_med.median())
spans17 = list(zip(g.s, g.e, g.pred))
days17 = np.array([(pd.Timestamp(e) - pd.Timestamp(s0)).days + 1 for s0, e, _ in spans17])
lv17 = np.array([v for *_, v in spans17])
c = Chart("level_merged", 1040, 460, -9, 6, yticks=(4, 0, -4, -8))
yL = c.Y(L17)
# balance: area between the placed steps and L; equal above and below <=> day-weighted mean = L
shade = []
for s0, e, v in spans17:
    xs, xe, yv = c.X(s0), c.X(pd.Timestamp(e) + pd.Timedelta(days=1)), c.Y(v)
    top, hgt = min(yL, yv), abs(yv - yL)
    c.parts.append(f'<rect x="{xs:.1f}" y="{top:.1f}" width="{xe - xs:.1f}" height="{hgt:.1f}" fill="{ORANGE}" opacity="0.14"/>')
    shade.append([round(xs, 1), round(top, 1), round(xe - xs, 1), round(hgt, 1)])
c.dots(ok17.date, ok17.vane_med, BLUE, size=5, opacity=0.2)
c.hline(L17, ORANGE, width=3, dash="10 8")
c.steps([(s0, e, v - L17) for s0, e, v in spans17], GREY, width=5, dash="12 8")
c.steps(spans17, ORANGE, width=8)
# the same rigid shift in every period: one arrow grey -> orange, 120 px after each period start
xa = []
for s0, _, v in spans17:
    x, y0, y1 = c.X(s0) + 120, c.Y(v - L17), c.Y(v)
    sg = 1 if y1 > y0 else -1
    c.parts.append(f'<path d="M{x:.1f} {y0 + 6 * sg:.1f}V{y1 - 14 * sg:.1f}" stroke="{INK}" stroke-width="3"/>'
                   f'<path d="M{x - 9:.1f} {y1 - 18 * sg:.1f}L{x:.1f} {y1 - 4 * sg:.1f}L{x + 9:.1f} {y1 - 18 * sg:.1f}Z" fill="{INK}"/>')
    xa.append(round(x, 1))
c.save()
ticks["level_merged"]["marks"] = {
    "x_arrows": xa, "x_change": round(c.X(spans17[1][0]), 1),
    "y_L": round(yL, 1),
    "y_grey": [round(c.Y(v - L17), 1) for v in lv17], "y_orange": [round(c.Y(v), 1) for v in lv17],
    "shade_xywh": shade,
    "area_deg_days": [round(float(n * (v - L17)), 0) for n, v in zip(days17, lv17)],
}
ticks["_level"] = {
    "ppp17_L": round(L17, 2),
    "ppp17_minutes": int(d17.n_minutes.sum()),
    "ppp17_operating_minutes": int(d17.op_minutes.sum()),
    "ppp17_days": int(len(d17)),
    "ppp17_healthy_days": int(len(ok17)),
    "ppp17_vane_iqr": [round(float(ok17.vane_med.quantile(q)), 2) for q in (0.25, 0.75)],
    "ppp17_periods": [[str(s0.date()), int(n), round(float(v), 2)] for (s0, _, v), n in zip(spans17, days17)],
    "ppp17_day_weighted_mean": round(float((days17 * lv17).sum() / days17.sum()), 2),
}
print(json.dumps(ticks["_level"], indent=1))

# 9. Training-error split for the results slide: share of the squared error that the level carries.
#    Per training turbine, the mean error is the level error; the rest comes from the steps.
import sys
sys.path.insert(0, str(ROOT / "src"))
from yaw.submission import features, load_inputs, predict  # noqa: E402

st_, lab_, vane_ = load_inputs()
pr_ = predict(features(st_), vane_)
d_ = pr_.merge(lab_.loc[lab_.scored, ["turbine_id", "date", "yaw_misalignment_deg"]].rename(columns={"yaw_misalignment_deg": "y"}),
               on=["turbine_id", "date"])
err = d_.yaw_misalignment_deg - d_.y
bias = err.groupby(d_.turbine_id).transform("mean")
ticks["_training_split"] = {
    "rmse": round(float(np.sqrt((err ** 2).mean())), 2),
    "rmse_without_level_error": round(float(np.sqrt(((err - bias) ** 2).mean())), 2),
    "level_share_of_mse": round(float((bias ** 2).mean() / (err ** 2).mean()), 2),
    "level_error_per_turbine": err.groupby(d_.turbine_id).mean().round(2).to_dict(),
}
print(json.dumps(ticks["_training_split"], indent=1))

ticks["_facts"] = {
    "max_misalignment_change_in_r": round(max_change, 1),
    "jumps": [len(steps_all), round(float(steps_all.jump.abs().min()), 1), round(float(steps_all.jump.abs().max()), 1)],
    "excursion_days": [int(spans_all[spans_all.kind == "excursion"].days.min()), int(spans_all[spans_all.kind == "excursion"].days.max())],
    "shortest_kept_interval": int(spans_all[spans_all.kind == "frame"].days.min()),
    "rereferences": rereference[["turbine_id", "start", "corr"]].astype(str).values.tolist(),"sss06_excursion_days": excursion_days,
                   "sss06_excursions": int(len(ex)),
                   "ppp17_states": g[["s", "e", "pred"]].astype(str).values.tolist(),
                   "sss06_states": g6[["s", "e", "pred"]].astype(str).values.tolist()}
(OUT / "ticks.json").write_text(json.dumps(ticks, indent=1))
print(json.dumps(ticks["_facts"], indent=1))
