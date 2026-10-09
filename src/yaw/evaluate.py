"""Local scoring that mirrors the leaderboard: RMSE / MAE / bias on scored days,
ARI on states, and a state-level bootstrap for the RMSE confidence interval."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def score(pred: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """`pred`: turbine_id, date, yaw_misalignment_deg[, cluster]. `truth`: output of labels.add_states."""
    t = truth[truth.scored][["turbine_id", "date", "yaw_misalignment_deg", "state"]]
    p = pred.rename(columns={"yaw_misalignment_deg": "pred"})
    p["date"] = pd.to_datetime(p["date"])
    j = t.merge(p, on=["turbine_id", "date"], how="left")
    missing = j["pred"].isna().sum()
    j = j.dropna(subset=["pred"])
    err = j["pred"] - j["yaw_misalignment_deg"]
    out = {
        "n_days": int(len(j)), "missing": int(missing),
        "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(err.abs().mean()), "bias": float(err.mean()),
    }
    if "cluster" in j.columns and j["cluster"].notna().all():
        out["ari"] = float(np.mean([adjusted_rand_score(g["state"], g["cluster"]) for _, g in j.groupby("turbine_id")]))
    else:
        out["ari"] = 0.0
    out["rmse_ci"] = state_bootstrap(j, err)
    return out


def state_bootstrap(j: pd.DataFrame, err: pd.Series, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of RMSE resampling whole (turbine, state) blocks, as the organisers do."""
    rng = np.random.default_rng(seed)
    blocks = [(err.values[idx]) for _, idx in j.groupby(["turbine_id", "state"]).indices.items()]
    if len(blocks) < 2:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        e = np.concatenate([blocks[i] for i in pick])
        vals.append(np.sqrt((e ** 2).mean()))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
