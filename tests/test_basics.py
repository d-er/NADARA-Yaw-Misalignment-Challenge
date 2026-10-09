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
