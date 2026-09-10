"""Figures for step 2 (docs/02_segmentation.md), all from cache/:

  figures/02_residuals.png  per-pair sector-corrected residuals, their median r, PELT state levels, labels inverted
  figures/02_states.png     r daily + 7-day median, state levels, credited encoder steps (cyan), encoder-typed boundaries (magenta)

Run after scripts/segment_states.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from yaw.config import CACHE, FIGURES, SPLITS

TARGETS = SPLITS["train"] + SPLITS["validate"] + SPLITS["test"]
pairs = pd.read_parquet(CACHE / "pair_residual_daily.parquet")
states = pd.read_parquet(CACHE / "states_scada.parquet")
steps = pd.read_parquet(CACHE / "encoder_steps.parquet")
lab = pd.read_parquet(CACHE / "labels_daily.parquet")


def label_axis(ax, tid):
    g = lab[lab.turbine_id == tid]
    if g.empty:
        return
    ax2 = ax.twinx(); ax2.plot(g.date, g.yaw_misalignment_deg, ".", color="green", ms=3)
    ax2.invert_yaxis(); ax2.set_ylabel("label (deg, inverted)", color="green")


def state_steps(ax, s):
    ax.step(s.date, s.r_level, where="post", color="red", lw=2, label="state level")


# --- residuals
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(15, 3.6 * len(TARGETS)), sharex=True)
for ax, tid in zip(axes, TARGETS):
    for nb, p in pairs[pairs.turbine_id == tid].groupby("neighbour"):
        ax.plot(p.date, p.resid, ".", ms=2, alpha=.5, label=f"pair {nb}")
    s = states[states.turbine_id == tid]
    ax.plot(s.date, s.r, "k-", lw=.8, label="r (median of pairs)"); state_steps(ax, s)
    ax.set_ylim(-20, 20); ax.set_ylabel(f"{tid}\nheading residual (deg)"); ax.legend(loc="upper left", ncol=6, fontsize=7)
    label_axis(ax, tid)
fig.suptitle("Step 2: neighbour-consensus heading residual r_i(t) after de-stepping; label axis inverted (Δr ≈ −Δθ expected)")
fig.tight_layout(); fig.savefig(FIGURES / "02_residuals.png", dpi=110); plt.close(fig)

# --- states
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(15, 3.6 * len(TARGETS)), sharex=True)
for ax, tid in zip(axes, TARGETS):
    s = states[states.turbine_id == tid]
    ax.plot(s.date, s.r, ".", color="grey", ms=2, label="r daily"); ax.plot(s.date, s.r_smooth, "k-", lw=.8, label="r 7-day median")
    state_steps(ax, s)
    for d in steps[steps.turbine_id == tid].date:
        ax.axvline(d, color="cyan", ls=":", lw=1)
    b = s[(s.state.diff().fillna(0) != 0) & (s.boundary_type == "encoder")]
    for d in b.date:
        ax.axvline(d, color="magenta", ls="--", lw=1)
    ax.set_ylim(-20, 20); ax.set_ylabel(f"{tid}\nr (deg)"); ax.legend(loc="upper left", fontsize=7)
    label_axis(ax, tid)
fig.suptitle("Step 2: consensus heading residual, PELT states (red), encoder-typed boundaries (magenta), credited encoder steps (cyan); labels green, inverted")
fig.tight_layout(); fig.savefig(FIGURES / "02_states.png", dpi=110); plt.close(fig)
print("saved figures/02_residuals.png, 02_states.png")
