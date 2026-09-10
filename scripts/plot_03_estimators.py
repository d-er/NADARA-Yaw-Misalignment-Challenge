"""Figure for step 3: what the physics estimators (C1, C2, C4) do against the labels.

Reads cache/state_estimates.parquet, cache/manoeuvre_events/*.parquet and
cache/agg1min; writes figures/03_estimators.png. Run after scripts/estimate_angles.py.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from yaw.config import CACHE, FIGURES
from yaw.estimators import load_minute

est = pd.read_parquet(CACHE / "state_estimates.parquet")
lab = pd.read_parquet(CACHE / "labels_daily.parquet")
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

# Row 1: quarterly estimates vs label, train turbines
for ax, tid in zip(axes[0], ["PPP_WTG12", "PPP_WTG13", "PPP_WTG14"]):
    q = est[(est.turbine_id == tid) & (est.kind == "quarter")]
    for e, c in [("C1_k2", "C0"), ("C2_k2", "C2"), ("C4", "C3")]:
        s = q[q.estimator == e]
        ax.errorbar(s.start + pd.Timedelta(days=45), s.theta, yerr=s.theta_se, fmt="o-", color=c, label=e, capsize=2)
    l = lab[lab.turbine_id == tid]
    ax.plot(l.date, l.yaw_misalignment_deg, "k-", lw=2, label="label")
    ax.set_title(f"{tid}: quarterly estimates vs label"); ax.set_ylim(-30, 15); ax.grid(alpha=.3); ax.legend(fontsize=8)

# Row 2: WTG13 normalised power vs vane per quarter (why C1 fails: flat, shape drifts)
m = load_minute("PPP_WTG13")
ax = axes[1, 0]
for q in ["2023Q1", "2023Q2", "2023Q4", "2024Q2"]:
    d = m[m.good & (m.ts.dt.to_period("Q") == q) & (m.vane_mean.abs() < 15) & (m.vane_std < 6)].copy()
    d["rb"] = (d.WindSpeed_mean / 0.5).round()
    d["pn"] = d.Power_mean / d.groupby("rb").Power_mean.transform("median")
    b = d.groupby(d.vane_mean.round()).pn.agg(["median", "size"]); b = b[b["size"] >= 30]
    ax.plot(b.index, b["median"], "o-", label=q)
v = np.arange(-15, 16); ax.plot(v, np.cos(np.radians(v)) ** 2 / np.cos(np.radians(-3)) ** 2, "k--", label="cos² peak at −3°")
ax.set_xlabel("vane (1-min mean, deg)"); ax.set_ylabel("power / wind-speed-bin median"); ax.set_title("PPP_WTG13: C1 curves per quarter")
ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(0.9, 1.08)
# vane occupancy
ax = axes[1, 1]
for tid in ["PPP_WTG13", "PPP_WTG17", "SSS_WTG06"]:
    mm = load_minute(tid); h, e = np.histogram(mm.loc[mm.good, "vane_mean"].dropna(), bins=np.arange(-20, 20.5, 1), density=True)
    ax.step(e[:-1], h, where="post", label=tid)
ax.set_xlabel("vane (1-min mean, deg)"); ax.set_title("vane occupancy in partial load (controller deadband)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
# label vs consensus residual per state (train)
ax = axes[1, 2]
sc = pd.read_parquet(CACHE / "states_scada.parquet")
j = lab[lab.scored].merge(sc[["turbine_id", "date", "r_smooth"]], on=["turbine_id", "date"])
for tid, g in j.groupby("turbine_id"):
    ax.scatter(g.r_smooth, g.yaw_misalignment_deg, s=6, label=tid, alpha=.6)
ax.set_xlabel("consensus heading residual r (7-day median, deg)"); ax.set_ylabel("label (deg)"); ax.set_title("label vs r on scored train days"); ax.legend(fontsize=8); ax.grid(alpha=.3)

# Row 3: C4 binned lnR vs Δ for the three turbines with the most events, plus the null
def binned(ev):
    b = pd.cut(ev.delta, np.arange(-20, 21, 2.5)); g = ev.groupby(b, observed=True).lnR
    return g.mean(), g.sem(), np.array([i.mid for i in g.mean().index])
for ax, tid in zip(axes[2], ["PPP_WTG14", "PPP_WTG17", "SSS_WTG06"]):
    ev = pd.read_parquet(CACHE / "manoeuvre_events" / f"{tid}.parquet")
    ev["year"] = ev.t0.dt.year
    for y, g in ev.groupby("year"):
        mu, se, x = binned(g); ax.errorbar(x, mu, yerr=se, fmt="o-", capsize=2, label=f"{y} (n={len(g)})")
    ax.axhline(0, color="k", lw=.5); ax.set_xlabel("signed yaw rotation Δ (deg)"); ax.set_ylabel("ln(P_after/P_before) − ref")
    ax.set_title(f"C4 manoeuvre response, {tid} vs {ev.reference.iloc[0]}"); ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(-0.12, 0.12)
fig.tight_layout(); fig.savefig(FIGURES / "03_estimators.png", dpi=110)
print("saved figures/03_estimators.png")
