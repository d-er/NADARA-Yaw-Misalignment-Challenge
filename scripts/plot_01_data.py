"""Figures for step 1, all from cache/:

  figures/01_coverage.png          operating minutes per day, all 16 turbines
  figures/01_labels_states.png     WindFit labels with PELT states and transition days
  figures/01_vane_and_offsets.png  daily median vane (flagged days) and raw heading offsets vs 4 nearest neighbours

Run after scripts/build_cache.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from yaw.config import CACHE, FIGURES, SPLITS, TURBINES
from yaw.geometry import neighbours
from yaw.io import wrap180

TARGETS = SPLITS["train"] + SPLITS["validate"] + SPLITS["test"]
daily = pd.concat([pd.read_parquet(CACHE / "daily" / f"{t}.parquet") for t in TURBINES], ignore_index=True)
days = pd.date_range("2023-01-01", "2024-12-31")

# --- coverage heatmap
cov = daily.pivot(index="turbine_id", columns="date", values="op_minutes").reindex(index=TURBINES, columns=days)
fig, ax = plt.subplots(figsize=(15, 5))
im = ax.imshow(cov.values, aspect="auto", cmap="viridis", vmin=0, vmax=1440, interpolation="nearest")
ax.set_yticks(range(len(TURBINES))); ax.set_yticklabels(TURBINES, fontsize=8)
ticks = np.arange(0, len(days), 61); ax.set_xticks(ticks); ax.set_xticklabels(days[ticks].strftime("%Y-%m-%d"), rotation=45, fontsize=8)
fig.colorbar(im, ax=ax, label="operating minutes / day"); ax.set_title("Coverage: operating minutes per day")
fig.tight_layout(); fig.savefig(FIGURES / "01_coverage.png", dpi=110); plt.close(fig)

# --- labels with states and transition days
lab = pd.read_parquet(CACHE / "labels_daily.parquet")
fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
for ax, tid in zip(axes, SPLITS["train"]):
    g = lab[lab.turbine_id == tid]
    for st, s in g.groupby("state"):
        ax.plot(s.date, s.yaw_misalignment_deg, ".", color=f"C{st % 10}")
    t = g[g.transition]; ax.plot(t.date, t.yaw_misalignment_deg, "kx", ms=4, label="transition (unscored)")
    ax.set_ylabel(f"{tid}\nlabel (deg)"); ax.legend(loc="upper right", fontsize=8)
fig.suptitle("WindFit labels, PELT states (colour) and reconstructed transition days")
fig.tight_layout(); fig.savefig(FIGURES / "01_labels_states.png", dpi=110); plt.close(fig)

# --- daily vane median and raw pair offsets
pairs = pd.read_parquet(CACHE / "pair_offsets_daily.parquet")
fig, axes = plt.subplots(len(TARGETS), 2, figsize=(16, 3.2 * len(TARGETS)), sharex=True)
for (axl, axr), tid in zip(axes, TARGETS):
    d = daily[daily.turbine_id == tid]
    axl.plot(d.date, d.vane_med, ".", ms=3)
    f = d[~d.flag_ok]; axl.plot(f.date, f.vane_med.clip(-10, 5), "rx", ms=4, label="flagged day")
    axl.set_ylim(-10, 5); axl.set_ylabel(f"{tid}\nvane median (deg)"); axl.legend(loc="upper right", fontsize=8)
    for nb in neighbours(tid, 4).index:
        p = pairs[(pairs.a == tid) & (pairs.b == nb)]; sign = 1.0
        if p.empty:
            p = pairs[(pairs.a == nb) & (pairs.b == tid)]; sign = -1.0
        axr.plot(p.date, wrap180(sign * p.dNacDir), ".", ms=2, label=nb)
    axr.set_ylim(-40, 40); axr.set_ylabel("NacDir − neighbour (deg)"); axr.legend(loc="upper left", ncol=4, fontsize=8)
axes[0, 0].set_title("Daily median vane angle (operating)"); axes[0, 1].set_title("Daily median nacelle-heading offset vs 4 nearest neighbours")
fig.tight_layout(); fig.savefig(FIGURES / "01_vane_and_offsets.png", dpi=110); plt.close(fig)
print("saved figures/01_coverage.png, 01_labels_states.png, 01_vane_and_offsets.png")
