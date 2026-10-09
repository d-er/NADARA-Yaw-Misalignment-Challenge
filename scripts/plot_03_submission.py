"""Figure for step 3: the submitted daily predictions against the consensus residual
and (on train) the labels. Reads cache/predictions_daily.parquet, cache/states_scada.parquet,
cache/labels_daily.parquet; writes figures/03_submission.png. Run after scripts/build_submission.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from yaw.config import CACHE, FIGURES

pred = pd.read_parquet(CACHE / "predictions_daily.parquet")
st = pd.read_parquet(CACHE / "states_scada.parquet")
lab = pd.read_parquet(CACHE / "labels_daily.parquet")
tids = ["PPP_WTG12", "PPP_WTG13", "PPP_WTG14", "PPP_WTG17", "SSS_WTG06"]
fig, axes = plt.subplots(len(tids), 1, figsize=(15, 3.2 * len(tids)), sharex=True)
for ax, tid in zip(axes, tids):
    s = st[st.turbine_id == tid]
    ax.plot(s.date, s.r, ".", color="0.7", ms=3, label="r (daily consensus residual)")
    ax.plot(s.date, s.r_smooth, "-", color="0.4", lw=1, label="r (7-day median)")
    p = pred[pred.turbine_id == tid].sort_values("date")
    ax.plot(p.date, p.yaw_misalignment_deg, "-", color="C0", lw=2, label="prediction T0")
    l = lab[lab.turbine_id == tid]
    if len(l):
        ax.plot(l.date, l.yaw_misalignment_deg, "k-", lw=2, label="label")
        ax.plot(l[l.scored].date, l[l.scored].yaw_misalignment_deg, "k.", ms=3)
    for d in s.loc[(s.state.diff().fillna(0) != 0) & (s.boundary_type == "encoder"), "date"]:
        ax.axvline(d, color="r", ls="--", lw=1)
    ax.set_ylim(-25, 20); ax.set_ylabel("deg"); ax.grid(alpha=.3); ax.set_title(tid, loc="left")
    ax.legend(fontsize=7, ncol=5, loc="upper right")
axes[0].set_title("SCADA states -> submission: level = vane setpoint, slope -1 on r, re-anchored at encoder boundaries (red)", loc="right", fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "03_submission.png", dpi=110)
print("saved figures/03_submission.png")
