"""Neighbour-consensus heading residual and encoder de-stepping.

Sensor algebra (after the leading team's write-up): the reported nacelle heading
is Nr = Nt + δn with δn the encoder's mounting offset, and the turbine settles at
true misalignment θ = Wt − Nt. Against a neighbour consensus of the true wind,
Nr − consensus = −θ + δn + const, so *changes* of the residual are −Δθ in
physical degrees, while the level is unknown (δn). Encoder re-references change
δn abruptly by arbitrary amounts; those must be removed before the residual is
read as a misalignment track.

Most shifts of the encoder frame are not re-references but excursions: the
heading reads tens of degrees off for a few weeks and then returns. Subtracting
an estimated jump out and another one back leaves their difference in the
residual, which is as large as the misalignment signal. `frame_plan` therefore
drops excursion days and applies no correction when the frame came back; only a
shift that persists between two long intervals is corrected.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt

from .config import CACHE, SITE_OF, TURBINES
from .geometry import neighbours
from .io import wrap180
from .labels import segment_series

# turbines whose heading is too unstable to serve as a reference for others
# (they still get their own residual from the remaining neighbours)
REFERENCE_BLACKLIST = ("PPP_WTG12", "PPP_WTG16")


def reference_neighbours(tid: str, k: int) -> pd.Index:
    nb = neighbours(tid)
    nb = nb[~nb.index.isin(REFERENCE_BLACKLIST)]
    return nb.head(k).index


def circ_median_deg(x: np.ndarray) -> float:
    """Median of angles via the circular mean as reference (robust to ±180 wrap)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    ref = np.degrees(np.arctan2(np.sin(np.radians(x)).mean(), np.cos(np.radians(x)).mean()))
    return ref + np.median(wrap180(x - ref))


def load_10min(tids=TURBINES) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(CACHE / "agg10min" / f"{t}.parquet") for t in tids}


def pair_bins(fa: pd.DataFrame, fb: pd.DataFrame, min_op: float = 0.9) -> pd.DataFrame:
    """10-min bins where both turbines operate; heading difference a − b."""
    j = fa.merge(fb, on="ts", suffixes=("_a", "_b"))
    j = j[(j.op_frac_a > min_op) & (j.op_frac_b > min_op)].copy()
    j["dNac"] = wrap180(j.NacDir_a - j.NacDir_b)
    j["date"] = j.ts.dt.normalize()
    return j


def daily_pair_series(frames: dict[str, pd.DataFrame], a: str, b: str) -> pd.Series:
    j = pair_bins(frames[a], frames[b])
    if j.empty:
        return pd.Series(dtype=float)
    return j.groupby("date")["dNac"].apply(circ_median_deg)


def segment_pair(s: pd.Series, pen: float = 150.0, min_size: int = 10) -> pd.DataFrame:
    """Level shifts in a daily pair-offset series (date of shift, wrapped jump size).

    The series is unwrapped so that ±180 wrap-around does not look like noise, then
    segmented with an L1 (median) cost so that a few outlying days do not create
    segments. Excursions shorter than `min_size` days are left to the masking stage.
    """
    s = s.dropna()
    if len(s) < 2 * min_size + 1:
        return pd.DataFrame(columns=["date", "jump"])
    u = np.degrees(np.unwrap(np.radians(s.values)))
    bkps = rpt.Pelt(model="l1", min_size=min_size, jump=1).fit(u).predict(pen=pen)
    levels, start = [], 0
    for b in bkps:
        levels.append(np.median(u[start:b])); start = b
    rows = [(s.index[b], float(wrap180(levels[k + 1] - levels[k]))) for k, b in enumerate(bkps[:-1])]
    return pd.DataFrame(rows, columns=["date", "jump"])


def encoder_steps(frames: dict[str, pd.DataFrame], k: int = 4, min_share: float = 0.6, min_jump: float = 25.0,
                  pen: float = 150.0, min_size: int = 10) -> pd.DataFrame:
    """Attribute large shared level shifts to a turbine's own encoder frame.

    A shift on date d in the series (i − j) is credited to i when shifts of the
    same sign and |jump| ≥ min_jump within ±3 days appear in at least `min_share`
    of i's k nearest pairs (and in at least two of them). Returns turbine_id,
    date, jump = amount by which i's reported heading moved.
    """
    rows = []
    for i in frames:
        nb = reference_neighbours(i, k)
        cand = pd.concat([segment_pair(daily_pair_series(frames, i, j), pen=pen, min_size=min_size) for j in nb],
                         ignore_index=True)
        cand = cand[cand.jump.abs() >= min_jump]
        cand["date"] = pd.to_datetime(cand["date"]); cand["jump"] = cand["jump"].astype(float)
        cand = cand.sort_values("date").reset_index(drop=True)
        used = np.zeros(len(cand), dtype=bool)
        need = max(2, int(np.ceil(min_share * len(nb))))
        for idx in range(len(cand)):
            if used[idx]:
                continue
            d, jmp = cand.date.iloc[idx], cand.jump.iloc[idx]
            close = (~used) & ((cand.date - d).abs().dt.days <= 3) & (np.sign(cand.jump) == np.sign(jmp))
            if close.sum() >= need:
                rows.append((i, cand.date[close].min(), float(np.median(cand.jump[close]))))
                used |= close.values
    return pd.DataFrame(rows, columns=["turbine_id", "date", "jump"]).sort_values(["turbine_id", "date"]).reset_index(drop=True)


def frame_plan(frames: dict[str, pd.DataFrame], steps: pd.DataFrame, k: int = 4, long_days: int = 45,
               tol: float = 15.0, margin: int = 3, window: int = 40, guard: int = 25) -> pd.DataFrame:
    """Decide per turbine which days share an encoder frame, which are excursions.

    The record is cut at the credited step dates. Intervals shorter than
    `long_days` are excursions and are dropped. Between two consecutive long
    intervals the shift is measured locally: raw pair offset over the first
    `window` days after minus the last `window` days before, median over the
    neighbours, skipping days within `guard` days of a neighbour's own steps.
    Below `tol` the frame came back and no correction is applied; above it the
    shift is a re-reference and is subtracted. Returns one row per interval:
    turbine_id, start, end (exclusive), kind ('frame' | 'excursion'), corr,
    rereference (True on the first interval of a new frame).
    """
    t0 = min(f.ts.min() for f in frames.values()).normalize()
    t1 = max(f.ts.max() for f in frames.values()).normalize() + pd.Timedelta(days=1)
    pad = pd.Timedelta(days=margin)
    step_dates = {t: pd.DatetimeIndex(g.date) for t, g in steps.groupby("turbine_id")}
    offsets: dict[tuple[str, str], pd.Series] = {}

    def offset(i: str, j: str) -> pd.Series:
        if (i, j) not in offsets:
            s = daily_pair_series(frames, i, j)
            for d in step_dates.get(j, []):
                s = s[(s.index < d - pd.Timedelta(days=guard)) | (s.index > d + pd.Timedelta(days=guard))]
            offsets[(i, j)] = s
        return offsets[(i, j)]

    def local_jump(i: str, before: tuple, after: tuple) -> float:
        jumps = []
        for j in reference_neighbours(i, k):
            s = offset(i, j)
            b = s[(s.index >= before[0] + pad) & (s.index < before[1] - pad)].tail(window)
            a = s[(s.index >= after[0] + pad) & (s.index < after[1] - pad)].head(window)
            if len(a) >= 7 and len(b) >= 7:
                jumps.append(float(wrap180(circ_median_deg(a.values) - circ_median_deg(b.values))))
        return float(np.median(jumps)) if jumps else np.nan

    rows = []
    for t in frames:
        edges = [t0, *sorted(step_dates.get(t, [])), t1]
        spans = list(zip(edges[:-1], edges[1:]))
        days = np.array([(b - a).days for a, b in spans])
        long = days >= long_days
        if not long.any():
            long[np.argmax(days)] = True
        corr, prev = 0.0, None
        for n, span in enumerate(spans):
            if not long[n]:
                rows.append((t, *span, "excursion", np.nan, False))
                continue
            reref = False
            if prev is not None:
                jump = local_jump(t, prev, span)
                if not np.isnan(jump) and abs(jump) >= tol:
                    corr += jump
                    reref = True
            rows.append((t, *span, "frame", corr, reref))
            prev = span
    return pd.DataFrame(rows, columns=["turbine_id", "start", "end", "kind", "corr", "rereference"])


def destep(frames: dict[str, pd.DataFrame], plan: pd.DataFrame, margin: int = 3) -> dict[str, pd.DataFrame]:
    """Apply `frame_plan`: drop excursion bins and `margin` days around every
    interval edge, subtract each frame's correction from NacDir and WindDir."""
    out = {}
    pad = pd.Timedelta(days=margin)
    for t, f in frames.items():
        f = f.copy()
        pl = plan[plan.turbine_id == t].sort_values("start").reset_index(drop=True)
        corr = np.zeros(len(f))
        keep = np.ones(len(f), dtype=bool)
        for n, r in pl.iterrows():
            inside = ((f.ts >= r.start) & (f.ts < r.end)).values
            if r.kind == "excursion":
                keep &= ~inside
            else:
                corr[inside] = r["corr"]
            if n > 0:
                keep &= ~((f.ts >= r.start - pad) & (f.ts < r.start + pad + pd.Timedelta(days=1))).values
        for d in ("NacDir", "WindDir"):
            f[d] = (f[d] - corr) % 360
        f["encoder_corr"] = corr
        out[t] = f[keep].reset_index(drop=True)
    return out


def heading_residual(frames: dict[str, pd.DataFrame], k: int = 4, n_sectors: int = 12,
                     iqr_ratio: float = 2.0, max_excursion: float = 10.0, excursion_window: int = 91,
                     min_bins: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily consensus residual r_i(t) for every turbine.

    For each of the k nearest pairs: 10-min heading difference minus that pair's
    median in the neighbour's 30° wind-direction sector (removes terrain/wake
    dependence), then the daily median. Before that, days more than
    `max_excursion` from the pair's `excursion_window`-day rolling median are
    dropped: a rolling median keeps a level shift of any size but not a pulse
    shorter than half the window, so this removes the short frame excursions,
    of either turbine, that are too small to be credited as encoder steps. Pairs whose residual is unusually noisy
    (daily IQR > iqr_ratio × the turbine's median pair IQR) are dropped. r_i is
    the median across the remaining pairs. Returns (residual table, pair table).
    """
    res_rows, pair_rows = [], []
    for i in frames:
        nb = reference_neighbours(i, k)
        per = {}
        for j in nb:
            b = pair_bins(frames[i], frames[j])
            if b.empty:
                continue
            # transient removal: days whose circular-median offset sits
            # > `max_excursion` from the rolling median
            day_off = b.groupby("date")["dNac"].apply(circ_median_deg)
            roll = day_off.rolling(excursion_window, center=True, min_periods=7).median()
            bad_days = day_off.index[(wrap180(day_off - roll)).__abs__() > max_excursion]
            b = b[~b.date.isin(bad_days)]
            sector = (b.WindDir_b // (360 / n_sectors)).astype(int)
            b["resid"] = wrap180(b.dNac - b.groupby(sector)["dNac"].transform(circ_median_deg))
            daily = b.groupby("date")["resid"].agg(["median", "size", lambda x: x.quantile(0.75) - x.quantile(0.25)])
            daily.columns = ["resid", "n_bins", "iqr"]
            daily = daily[daily.n_bins >= min_bins]
            per[j] = daily
            pair_rows.append(daily.assign(turbine_id=i, neighbour=j).reset_index())
        if not per:
            continue
        iqrs = pd.Series({j: d.iqr.median() for j, d in per.items()})
        keep = iqrs[iqrs <= iqr_ratio * iqrs.median()].index
        wide = pd.concat({j: per[j].resid for j in keep}, axis=1)
        r = pd.DataFrame({"r": wide.median(axis=1), "n_neighbours": wide.notna().sum(axis=1)})
        res_rows.append(r.assign(turbine_id=i, neighbours=",".join(keep)).reset_index())
    return (pd.concat(res_rows, ignore_index=True), pd.concat(pair_rows, ignore_index=True))


def segment_residual(res: pd.DataFrame, pen: float = 400.0, min_size: int = 14, guard_days: int = 7,
                     smooth_days: int = 7, encoder_jump: float = 15.0, plan: pd.DataFrame | None = None,
                     reref_days: int = 10) -> pd.DataFrame:
    """PELT states on the `smooth_days` rolling median of r_i(t), per turbine.

    Adds r_smooth, state, r_level, transition (±guard_days around a breakpoint)
    and boundary_type for the first day of each state: 'encoder' when the level
    jump exceeds `encoder_jump` (too large for a misalignment change), else
    'candidate'. With a `plan` (from `frame_plan`), a boundary within
    `reref_days` of a verified re-reference is 'encoder' whatever its size. Days
    missing from r are filled by the rolling median so that states are
    contiguous in calendar time.
    """
    reref = {} if plan is None else {t: pd.DatetimeIndex(g.start) for t, g in plan[plan.rereference].groupby("turbine_id")}
    out = []
    for t, g in res.groupby("turbine_id", sort=False):
        s = g.set_index("date")["r"].asfreq("D")
        sm = s.rolling(smooth_days, center=True, min_periods=max(1, smooth_days // 2)).median()
        sm = sm.interpolate(limit_direction="both")
        df = pd.DataFrame({"r": s, "r_smooth": sm}).reset_index()
        df["state"] = segment_series(df.r_smooth.values, pen=pen, min_size=min_size)
        df["r_level"] = df.groupby("state")["r_smooth"].transform("median")
        first = df.state.diff().fillna(0) != 0
        jump = df.r_level.diff().where(first)
        enc = jump.abs() > encoder_jump
        for d in reref.get(t, []):
            enc |= first & ((df.date - d).abs().dt.days <= reref_days)
        df["boundary_type"] = np.where(first, np.where(enc, "encoder", "candidate"), "")
        near = np.zeros(len(df), dtype=bool)
        for d in df.loc[first, "date"]:
            near |= (df.date - d).abs().dt.days.values <= guard_days
        df["transition"] = near
        df["turbine_id"] = t
        out.append(df)
    return pd.concat(out, ignore_index=True)
