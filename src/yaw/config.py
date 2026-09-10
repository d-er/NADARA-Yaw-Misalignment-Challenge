"""Paths, turbine roster and physical constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
FIGURES = ROOT / "figures"

SIGNALS = ["GenSpeed", "NacDir", "WindSpeed", "WindDir", "PitchAngle", "RotSpeed", "Power"]
DIRECTIONS = ["NacDir", "WindDir"]

SPLITS = {
    "train": ["PPP_WTG12", "PPP_WTG13", "PPP_WTG14"],
    "validate": ["PPP_WTG17"],
    "test": ["SSS_WTG06"],
    "context": ["PPP_WTG07", "PPP_WTG08", "PPP_WTG11", "PPP_WTG15", "PPP_WTG16",
                "PPP_WTG18", "PPP_WTG33", "SSS_WTG04", "SSS_WTG05", "SSS_WTG07", "SSS_WTG16"],
}
SPLIT_OF = {tid: split for split, tids in SPLITS.items() for tid in tids}
TURBINES = sorted(SPLIT_OF)
SITE_OF = {tid: tid[:3] for tid in TURBINES}

# Scaled units (undisclosed factor). Rated power is ~4800 in these units
# (99.9th percentile of Power); pitch sits at -3..-1.7 in partial load and rises
# above rated. Thresholds are deliberately loose; tighten per estimator.
P_RATED = 4800.0
P_MIN_OPERATING = 100.0          # ~2% of rated: turbine producing
P_MAX_PARTIAL = 0.85 * P_RATED   # below this, power still responds to inflow
PITCH_MAX_PARTIAL = 0.5          # pitch is negative in partial load

DATE_START, DATE_END = "2023-01-01", "2024-12-31"
