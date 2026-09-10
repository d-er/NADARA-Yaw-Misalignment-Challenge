"""Per-window estimators of the static yaw misalignment.

Every estimator takes a set of days (a state, a quarter, the whole record) and
returns a point estimate with a standard error and a sample size:

* C1 `power_vane_fit`   – classic OpenOA-style fit: within wind-speed bins,
  normalised partial-load power vs vane angle, peak of cos^k(vane − θ0).
  Measured in *vane* degrees, hence biased towards small |θ| (rotor-wake gain).
* C2 `power_vane_fit(..., ref="Power_ref")` – same, but the wind reference is a
  neighbour's simultaneous power instead of the nacelle anemometer.
* C4 `manoeuvre_events` + `manoeuvre_fit` – natural experiments: the power ratio
  after/before each yaw manoeuvre (relative to a neighbour) as a function of the
  signed rotation Δ. If the nacelle is at static misalignment θ after the
  manoeuvre and at θ+fΔ before it (f ≤ 1: the wind shifted gradually), then
  ln R = k[ln cos θ − ln cos(θ + fΔ)] ≈ −k·f·tan(θ)·Δ, so the slope's sign is the
  sign of θ in the *physical* convention (wind − nacelle, clockwise positive) and
  is independent of the vane and of the encoder offset.

The derived vane jumps by exactly −ΔNacDir at every NacDir update until WindDir
refreshes (WindDir is change-based and asynchronous), so nothing at the 12-s
scale may use `vane` directly around a manoeuvre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn.linear_model import HuberRegressor

from .config import CACHE
from .geometry import waked_mask
from .io import wrap180

K_FIXED = 2.0          # cos^k exponent used for the fixed-k fits
P_LO, P_HI = 300.0, 3600.0
PITCH_LO = 0.5


# ---------------------------------------------------------------- helpers
def destep(series: pd.Series, ts: pd.Series, steps: pd.DataFrame) -> pd.Series:
    """Subtract the credited encoder re-references (cumulative from their date)."""
    if steps.empty:
        return series
    off = np.zeros(len(series))
    for _, s in steps.iterrows():
        off[ts.values >= np.datetime64(s.date)] += s.jump
    return (series - off) % 360


def load_minute(tid: str, ref: str | None = None) -> pd.DataFrame:
    """1-min aggregates of `tid` with regime/wake flags, optionally joined to a reference turbine."""
    m = pd.read_parquet(CACHE / "agg1min" / f"{tid}.parquet")
    steps = pd.read_parquet(CACHE / "encoder_steps.parquet")
    m["WindDir_ds"] = destep(m["WindDir"], m["ts"], steps[steps.turbine_id == tid])
    day = pd.read_parquet(CACHE / "daily" / f"{tid}.parquet")[["date", "flag_ok"]]
    m["date"] = m["ts"].dt.normalize()
    m = m.merge(day, on="date", how="left")
    m["partial"] = m.Power_mean.between(P_LO, P_HI) & (m.PitchAngle_mean < PITCH_LO) & (m.op_frac > 0.99)
    m["unwaked"] = ~waked_mask(tid, m["WindDir_ds"].values)
    m["good"] = m.partial & m.unwaked & m.flag_ok.fillna(False)
    if ref is not None:
        r = pd.read_parquet(CACHE / "agg1min" / f"{ref}.parquet")
        r["WindDir_ds"] = destep(r["WindDir"], r["ts"], steps[steps.turbine_id == ref])
        r["partial_ref"] = r.Power_mean.between(P_LO, P_HI) & (r.PitchAngle_mean < PITCH_LO) & (r.op_frac > 0.99)
        r["unwaked_ref"] = ~waked_mask(ref, r["WindDir_ds"].values)
        m = m.merge(r[["ts", "Power_mean", "partial_ref", "unwaked_ref"]].rename(columns={"Power_mean": "Power_ref"}),
                    on="ts", how="left")
        m["good_ref"] = m.good & m.partial_ref.fillna(False) & m.unwaked_ref.fillna(False)
    return m


def _cosk(v, k, th):
    c = np.cos(np.radians(v - th))
    return np.where(c > 0, c, 1e-6) ** k


def fit_peak(v: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, k: float | None = K_FIXED) -> dict:
    """Least-squares peak of A·cos^k(v − θ0) on binned data; k fixed or free (k=None)."""
    w = np.ones_like(v) if w is None else np.sqrt(w)
    if k is None:
        f = lambda p: w * (p[0] * _cosk(v, p[1], p[2]) - y)
        p0, lo, hi = [1.0, K_FIXED, 0.0], [0, 0.2, -40], [10, 15, 40]
    else:
        f = lambda p: w * (p[0] * _cosk(v, k, p[1]) - y)
        p0, lo, hi = [1.0, 0.0], [0, -40], [10, 40]
    res = least_squares(f, p0, bounds=(lo, hi))
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J) * (res.fun ** 2).sum() / max(len(v) - len(p0), 1)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(p0), np.nan)
    if k is None:
        return {"theta": res.x[2], "theta_se": se[2], "k": res.x[1], "k_se": se[1]}
    return {"theta": res.x[1], "theta_se": se[1], "k": k, "k_se": 0.0}


def power_vane_fit(m: pd.DataFrame, ref: str = "WindSpeed_mean", ref_bin: float = 0.5,
                   vane_bin: float = 1.0, vane_max: float = 20.0, min_rows: int = 300, k=K_FIXED) -> dict:
    """C1 (ref = nacelle wind speed) or C2 (ref = 'Power_ref').

    Within each reference bin power is divided by the bin median, then the
    normalised power is binned by vane (1-min mean) and the bin medians are fitted
    with a cos^k peak. Returns theta (vane deg), se, k, n.
    """
    good = m["good_ref"] if ref == "Power_ref" else m["good"]
    d = m.loc[good & m[ref].notna() & (m.vane_mean.abs() < vane_max) & (m.vane_std < 6),
              ["Power_mean", ref, "vane_mean"]].copy()
    if len(d) < min_rows:
        return {"theta": np.nan, "theta_se": np.nan, "k": np.nan, "k_se": np.nan, "n": len(d)}
    d["rb"] = (d[ref] / ref_bin).round()
    d["pn"] = d.Power_mean / d.groupby("rb").Power_mean.transform("median")
    d = d[d.groupby("rb").Power_mean.transform("size") >= 30]
    d["vb"] = (d.vane_mean / vane_bin).round() * vane_bin
    b = d.groupby("vb").pn.agg(["median", "size"])
    b = b[b["size"] >= 10]
    if len(b) < 8:
        return {"theta": np.nan, "theta_se": np.nan, "k": np.nan, "k_se": np.nan, "n": len(d)}
    out = fit_peak(b.index.values, b["median"].values, b["size"].values, k=k)
    out["n"] = int(len(d))
    return out


# ---------------------------------------------------------------- C4
def manoeuvres(df: pd.DataFrame, gap_s: float = 60.0, min_step: float = 0.05) -> pd.DataFrame:
    """Yaw manoeuvres from the filled 12-s stream (needs NacDir, nac_update).

    A manoeuvre is a run of NacDir updates separated by < `gap_s`. Columns:
    t0, t1 (first/last update), delta (signed total rotation from the heading
    before the first update), n_updates, duration_s.
    """
    u = df.loc[df.nac_update, ["ts", "NacDir"]].copy()
    u["dNac"] = wrap180(u.NacDir.diff())
    u = u[u.dNac.abs() > min_step]
    gap = u.ts.diff().dt.total_seconds().fillna(np.inf)
    u["run"] = (gap > gap_s).cumsum()
    first_idx = u.groupby("run").head(1).index
    pos = df.index.get_indexer(first_idx) - 1
    start_heading = df.NacDir.values[np.clip(pos, 0, None)]
    runs = u.groupby("run").agg(t0=("ts", "first"), t1=("ts", "last"), n1=("NacDir", "last"), n_updates=("ts", "size"))
    runs["delta"] = wrap180(runs.n1.values - start_heading)
    runs["duration_s"] = (runs.t1 - runs.t0).dt.total_seconds()
    return runs.drop(columns="n1").reset_index(drop=True)


def manoeuvre_events(runs: pd.DataFrame, m: pd.DataFrame, before=(-6, -2), after=(2, 6),
                     max_delta: float = 30.0, min_delta: float = 2.0, max_duration: float = 300.0) -> pd.DataFrame:
    """ln(P_after/P_before) − ln(Pref_after/Pref_before) per manoeuvre, windows in minutes.

    Both windows must be entirely 'good_ref' minutes (partial load, unwaked,
    healthy day, reference likewise).
    """
    mi = m.set_index("ts")
    ok = mi["good_ref"].astype(bool)
    lp = np.log(mi.Power_mean.where(ok)); lr = np.log(mi.Power_ref.where(ok))
    # rolling window means aligned on the window END; missing values poison the window
    nb, na = before[1] - before[0], after[1] - after[0]
    def win_mean(s, n):
        return s.rolling(n, min_periods=n).mean()
    idx = pd.date_range(mi.index.min(), mi.index.max(), freq="1min")
    lp, lr = lp.reindex(idx), lr.reindex(idx)
    pb, rb = win_mean(lp, nb + 1), win_mean(lr, nb + 1)
    pa, ra = win_mean(lp, na + 1), win_mean(lr, na + 1)
    r = runs[(runs.delta.abs() >= min_delta) & (runs.delta.abs() <= max_delta) & (runs.duration_s <= max_duration)].copy()
    tb = r.t0.dt.floor("min") + pd.Timedelta(minutes=before[1])
    ta = r.t1.dt.ceil("min") + pd.Timedelta(minutes=after[1])
    r["lnP_before"] = pb.reindex(tb).values; r["lnR_before"] = rb.reindex(tb).values
    r["lnP_after"] = pa.reindex(ta).values; r["lnR_after"] = ra.reindex(ta).values
    r = r.dropna(subset=["lnP_before", "lnR_before", "lnP_after", "lnR_after"])
    r["lnR"] = (r.lnP_after - r.lnP_before) - (r.lnR_after - r.lnR_before)
    r["date"] = r.t0.dt.normalize()
    return r[["t0", "t1", "date", "delta", "n_updates", "duration_s", "lnR"]].reset_index(drop=True)


def manoeuvre_fit(ev: pd.DataFrame, k: float = K_FIXED, f: float = 1.0, delta_max: float = 20.0,
                  n_boot: int = 200, seed: int = 0) -> dict:
    """Robust slope of lnR vs Δ and the implied θ = atan(−slope / (k f)) in physical degrees."""
    e = ev[ev.delta.abs() <= delta_max]
    if len(e) < 50:
        return {"slope": np.nan, "slope_se": np.nan, "theta": np.nan, "theta_se": np.nan, "n": len(e)}
    X, y = e[["delta"]].values, e.lnR.values
    slope = HuberRegressor().fit(X, y).coef_[0]
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(e), len(e))
        bs.append(HuberRegressor().fit(X[i], y[i]).coef_[0])
    se = float(np.std(bs))
    th = lambda s: np.degrees(np.arctan(-s * 180 / np.pi / (k * f)))   # slope per degree → per radian
    return {"slope": float(slope), "slope_se": se, "theta": float(th(slope)),
            "theta_se": float(abs(th(slope + se) - th(slope - se)) / 2), "n": int(len(e))}
