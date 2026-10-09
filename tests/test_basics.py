import numpy as np
import pandas as pd
from yaw.io import wrap180, prepare
from yaw.aggregate import aggregate
from yaw.labels import segment_series


def test_wrap():
    assert np.allclose(wrap180([190, -190, 180, 0, 359.9]), [-170, 170, -180, 0, -0.1])


def test_prepare_ffill_and_vane():
    df = pd.DataFrame({
        "turbine_id": ["a"] * 3, "ts": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 00:00:12", "2023-01-01 00:00:24"]),
        "NacDir": [350.0, np.nan, np.nan], "WindDir": [np.nan, 5.0, 340.0], "Power": [1.0, np.nan, 2.0],
    })
    out = prepare(df)
    assert out.NacDir.tolist() == [350.0] * 3
    assert out.nac_update.tolist() == [True, False, False]
    assert np.allclose(out.vane.iloc[1:].tolist(), [15.0, -10.0])


def test_aggregate_circular_mean_across_north():
    ts = pd.date_range("2023-01-01", periods=4, freq="12s")
    df = pd.DataFrame({"turbine_id": "a", "ts": ts, "NacDir": [350.0, 10.0, 355.0, 5.0], "WindDir": 0.0, "Power": 500.0})
    df["vane"] = wrap180(df.WindDir - df.NacDir); df["nac_update"] = True
    a = aggregate(df, "1min")
    assert abs(a.NacDir.iloc[0]) < 1e-6 or abs(a.NacDir.iloc[0] - 360) < 1e-6


def test_segment_series_finds_step():
    x = np.r_[np.full(30, -8.0), np.full(30, -4.0)] + np.random.default_rng(0).normal(0, 0.2, 60)
    s = segment_series(x, pen=20.0)
    assert s[:30].max() == 0 and s[30:].min() == 1 and s.max() == 1


def test_submission_features_reanchor_at_encoder_boundary():
    from yaw.submission import features, predict
    dates = pd.date_range("2023-01-01", periods=6)
    st = pd.DataFrame({
        "turbine_id": "a", "date": dates, "state": [0, 0, 1, 1, 2, 2], "r": 0.0, "r_smooth": 0.0,
        "r_level": [4.0, 4.0, 0.0, 0.0, 30.0, 30.0], "transition": False,
        "boundary_type": ["", "", "candidate", "candidate", "encoder", "encoder"],
    })
    f = features(st, anchor="mean")
    assert f.seg.tolist() == [0, 0, 0, 0, 1, 1]
    assert np.allclose(f.x.tolist(), [2, 2, -2, -2, 0, 0])      # segment 1 re-anchored, jump of 30 not propagated
    p = predict(f, pd.Series({"a": -3.0}))
    assert np.allclose(p.yaw_misalignment_deg.tolist(), [-5, -5, -1, -1, -3, -3])  # dtheta = -dr
    assert p.cluster.tolist() == [0, 0, 1, 1, 2, 2]


def _frames(shifts: dict[str, list[tuple[int, int, float]]], days: int = 300) -> dict[str, pd.DataFrame]:
    """Hourly synthetic frames; `shifts[t]` = (first day, last day exclusive, degrees added to the heading)."""
    ts = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    day = np.arange(len(ts)) // 24
    wind = 180 + 40 * np.sin(np.arange(len(ts)) / 50)
    out = {}
    for t in ("a", "b", "c", "d", "e"):
        nac = wind.copy()
        for d0, d1, deg in shifts.get(t, []):
            nac[(day >= d0) & (day < d1)] += deg
        out[t] = pd.DataFrame({"ts": ts, "NacDir": nac % 360, "WindDir": nac % 360, "op_frac": 1.0})
    return out


def test_frame_plan_masks_excursion_and_keeps_rereference(monkeypatch):
    from yaw import consensus as C
    monkeypatch.setattr(C, "reference_neighbours", lambda tid, k: pd.Index([t for t in "abcde" if t != tid][:k]))
    # a: 15-day excursion of -100 deg that comes back 4 deg off (a real change, not an encoder event)
    # b: permanent re-reference of +60 deg
    frames = _frames({"a": [(100, 115, -100.0), (115, 300, 4.0)], "b": [(200, 300, 60.0)]})
    d = lambda n: pd.Timestamp("2023-01-01") + pd.Timedelta(days=n)
    steps = pd.DataFrame({"turbine_id": ["a", "a", "b"], "date": [d(100), d(115), d(200)], "jump": [-97.0, 108.0, 55.0]})
    plan = C.frame_plan(frames, steps)
    pa, pb = plan[plan.turbine_id == "a"], plan[plan.turbine_id == "b"]
    assert pa.kind.tolist() == ["frame", "excursion", "frame"]
    assert not pa.rereference.any() and (pa["corr"].dropna() == 0).all()     # came back: nothing subtracted
    assert pb.rereference.tolist() == [False, True] and abs(pb["corr"].iloc[1] - 60) < 1
    out = C.destep(frames, plan)
    gone = out["a"].ts.dt.normalize()
    assert not ((gone >= d(97)) & (gone <= d(118))).any()                    # excursion and margins dropped
    after = out["a"][out["a"].ts >= d(150)]
    assert np.allclose(C.wrap180(after.NacDir - frames["c"].set_index("ts").NacDir.reindex(after.ts).values), 4.0)  # the 4 deg survive
    late = out["b"][out["b"].ts >= d(210)]
    assert np.abs(C.wrap180(late.NacDir - frames["c"].set_index("ts").NacDir.reindex(late.ts).values)).max() < 1  # re-reference removed
