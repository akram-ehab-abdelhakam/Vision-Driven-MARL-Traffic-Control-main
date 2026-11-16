"""
cv_loader.py
Path: desrutge/src/machine_learning/dataset/cv_loader.py

cv_stats.json now contains vehicle_type_distribution ONLY.
Demand magnitude (entry/exit weights, scenario totals) comes from
the verified field-survey workbook constants in SIMULA.py
(DEMAND_SCENARIOS dict), NOT from this JSON file.

This module therefore provides exactly one function:
load_cv_type_distribution() — consumed by route_postprocessor.py
to assign SUMO vTypes.

REMOVED: load_cv_trainloader() — was reading
  summary["hourly_demand_profile"], intervals[*]["smoothed_flow_veh_per_hour"],
  intervals[*]["flow_rate_veh_per_hour"]
These fields are no longer read anywhere in the pipeline.
"""

import json


def load_cv_type_distribution(cv_stats_path: str) -> dict:
    """
    Load the vehicle type distribution from cv_stats.json.

    Returns a dict like:
        {"car": 0.67, "bus": 0.10, "truck": 0.09,
         "motorcycle": 0.12, "emergency": 0.02}

    Values sum to 1.0. Used by route_postprocessor.py to assign SUMO
    vTypes when post-processing the DesRUTGe-generated routes.rou.xml.

    Resolution priority:
      1. summary["overall_type_distribution"]  — preferred
      2. unweighted mean of intervals[*]["vehicle_type_distribution"]
         (no flow-based weighting — cv_stats.json no longer carries
         flow magnitude fields to weight by)

    Args:
        cv_stats_path: Absolute or relative path to cv_stats.json.

    Returns:
        Dict[str, float] with keys: car, bus, truck, motorcycle, emergency.
    """
    _KEYS = ["car", "bus", "truck", "motorcycle", "emergency"]

    with open(cv_stats_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # ── Priority 1: summary overall distribution ───────────────────────────
    summary = data.get("summary", {})
    overall = summary.get("overall_type_distribution", {})
    if overall and all(k in overall for k in _KEYS):
        total = sum(overall.get(k, 0.0) for k in _KEYS) or 1.0
        return {k: round(overall[k] / total, 4) for k in _KEYS}

    # ── Priority 2: unweighted mean across intervals ────────────────────────
    intervals = data.get("intervals", [])
    if not intervals:
        # Safe default: all cars
        return {"car": 1.0, "bus": 0.0, "truck": 0.0,
                "motorcycle": 0.0, "emergency": 0.0}

    agg = {k: 0.0 for k in _KEYS}
    n = len(intervals)
    for iv in intervals:
        dist = iv.get("vehicle_type_distribution", {})
        for k in _KEYS:
            agg[k] += float(dist.get(k, 0.0)) / n

    s = sum(agg.values()) or 1.0
    return {k: round(agg[k] / s, 4) for k in _KEYS}