"""xml_metrics.py

Offline post-processing / validation module for the SUMO RL traffic-control
framework.  Parses the official SUMO XML outputs (tripinfo.xml, emissions.xml,
fcd.xml) and computes metrics that are consistent -- in definition and units --
with the metrics collected online via TraCI in ``sumo_env.py``.

    Units (internal storage is mg; all display/export converts to kg/g):
        CO2, NOx, CO, HC, PMx   -> mg  (SUMO tripinfo/emission attrs are "_abs"
                                          totals in mg, or instantaneous mg/s rates
                                          in emissions.xml)
        fuel                    -> mg  (SUMO reports fuel mass, not volume)
        time                    -> s

This module is intentionally dependency-light (stdlib ElementTree + pandas)
so it can be used both online (during/after an RL episode when SUMO wrote its
XML outputs) and fully offline (pointing at a folder of already-generated
XML files, e.g. from the SUMO GUI).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

__all__ = [
    "parse_tripinfo",
    "parse_emissions",
    "parse_fcd",
    "parse_fcd_for_metrics",
    "compute_all_metrics",
    "validate_traci_vs_xml",
    "load_metrics_from_xml",
]

PathLike = Union[str, Path]

# EV keyword set mirrors config.EV_TYPE_KEYWORDS so offline and online
# classification never disagree. Duplicated here (not imported) to keep this
# module standalone / importable without pulling in the full config stack
# (torch, matplotlib, traci, etc.), which matters for lightweight offline use.
_DEFAULT_EV_KEYWORDS = {"ambulance", "fire_truck", "police", "emergency"}


# ---------------------------------------------------------------------------
# tripinfo.xml
# ---------------------------------------------------------------------------
def parse_tripinfo(
    xml_path: PathLike,
    ev_keywords: Optional[set] = None,
) -> pd.DataFrame:
    """Parse tripinfo.xml into a DataFrame with one row per completed trip.

    Each row combines the <tripinfo> attributes with its nested <emissions>
    sub-element (if the emissions device was enabled). Missing/absent fields
    default to 0.0 so downstream aggregation never raises on partial data.

    Returns an empty (but correctly-columned) DataFrame if the file has no
    <tripinfo> children (e.g. an episode with zero completed trips).
    """
    ev_keywords = ev_keywords or _DEFAULT_EV_KEYWORDS
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"tripinfo XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    records: List[dict] = []
    for trip in root.findall("tripinfo"):
        em = trip.find("emissions")

        def _f(el, key, default=0.0):
            if el is None:
                return default
            v = el.get(key)
            return float(v) if v not in (None, "") else default

        record = {
            "id":            trip.get("id"),
            "depart":        _f(trip, "depart"),
            "departDelay":   _f(trip, "departDelay"),
            "arrival":       _f(trip, "arrival"),
            "duration":      _f(trip, "duration"),
            "waitingTime":   _f(trip, "waitingTime"),
            "waitingCount":  _f(trip, "waitingCount"),
            "stopTime":      _f(trip, "stopTime"),
            "timeLoss":      _f(trip, "timeLoss"),
            "routeLength":   _f(trip, "routeLength"),
            "rerouteNo":     _f(trip, "rerouteNo"),
            "departSpeed":   _f(trip, "departSpeed"),
            "arrivalSpeed":  _f(trip, "arrivalSpeed"),
            "speedFactor":   _f(trip, "speedFactor"),
            "vType":         trip.get("vType", ""),
            "vaporized":     trip.get("vaporized", "") not in ("", None),
            # -- emissions sub-element (absolute totals for the whole trip) --
            "CO_abs":            _f(em, "CO_abs"),
            "CO2_abs":           _f(em, "CO2_abs"),
            "HC_abs":            _f(em, "HC_abs"),
            "PMx_abs":           _f(em, "PMx_abs"),
            "NOx_abs":           _f(em, "NOx_abs"),
            "fuel_abs":          _f(em, "fuel_abs"),
            "electricity_abs":   _f(em, "electricity_abs"),
        }
        records.append(record)

    columns = [
        "id", "depart", "departDelay", "arrival", "duration", "waitingTime",
        "waitingCount", "stopTime", "timeLoss", "routeLength", "rerouteNo",
        "departSpeed", "arrivalSpeed", "speedFactor", "vType", "vaporized",
        "CO_abs", "CO2_abs", "HC_abs", "PMx_abs", "NOx_abs", "fuel_abs",
        "electricity_abs",
    ]
    df = pd.DataFrame(records, columns=columns)

    if df.empty:
        df["ev"] = pd.Series(dtype=bool)
        df["finished"] = pd.Series(dtype=bool)
        return df

    vtype_lower = df["vType"].fillna("").str.lower()
    pattern = "|".join(sorted(ev_keywords))
    df["ev"] = vtype_lower.str.contains(pattern, case=False, regex=True, na=False)
    df["finished"] = df["arrival"] >= 0
    return df


# ---------------------------------------------------------------------------
# emissions.xml
# ---------------------------------------------------------------------------
def parse_emissions(
    xml_path: PathLike,
    aggregate: bool = True,
) -> Union[Dict[str, float], pd.DataFrame]:
    """Parse emissions.xml.

    If ``aggregate`` (default): returns a dict of network-wide totals,
    computed as sum(rate_i * STEP_LENGTH) over all (timestep, vehicle) rows --
    i.e. the same Riemann-sum integration TraCI-side collection performs in
    ``sumo_env._subscribe_vehicles_bulk``. Assumes uniform per-timestep
    spacing; pass ``step_length`` explicitly if it differs from 1.0s.

    If not aggregate: returns the raw per-(timestep, vehicle) DataFrame for
    detailed offline analysis (e.g. time-series plots, per-vehicle profiles).
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"emissions XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    if not aggregate:
        rows: List[dict] = []
        for ts in root.findall("timestep"):
            t = float(ts.get("time", 0.0))
            for veh in ts.findall("vehicle"):
                rows.append({
                    "time":   t,
                    "id":     veh.get("id"),
                    "type":   veh.get("type"),
                    "lane":   veh.get("lane"),
                    "speed":  float(veh.get("speed", 0.0)),
                    "CO2":    float(veh.get("CO2", 0.0)),
                    "CO":     float(veh.get("CO", 0.0)),
                    "HC":     float(veh.get("HC", 0.0)),
                    "NOx":    float(veh.get("NOx", 0.0)),
                    "PMx":    float(veh.get("PMx", 0.0)),
                    "fuel":   float(veh.get("fuel", 0.0)),
                    "electricity": float(veh.get("electricity", 0.0)),
                    "waiting": float(veh.get("waiting", 0.0)),
                })
        return pd.DataFrame(rows)

    # -- aggregate branch: sum instantaneous rates * step_length -------------
    #
    # SUMO's emission-output writes one <timestep> element per SIMULATED
    # step (even when empty of vehicles), so consecutive <timestep time="..">
    # values are normally exactly STEP_LENGTH apart. We infer that spacing
    # from the file itself (rather than hardcoding 1.0) so this still works
    # correctly if a run used --step-length != 1.0.
    #
    # IMPORTANT: if the file is truncated or has genuinely irregular gaps
    # (e.g. a partial/corrupted export), silently taking the median and
    # applying it everywhere would produce a total that is wrong by a
    # constant, hard-to-detect factor across the whole file. We guard
    # against that by checking the gaps are actually uniform (within a
    # small tolerance) before trusting the inferred spacing; if they are
    # not, we fall back to STEP_LENGTH-agnostic per-gap integration instead.
    timesteps = root.findall("timestep")
    times = sorted({float(ts.get("time", 0.0)) for ts in timesteps})
    diffs = np.diff(times) if len(times) >= 2 else np.array([])

    irregular = False
    if len(diffs) == 0:
        step_length = 1.0
    else:
        step_length = float(np.median(diffs))
        # Flag files where the spacing is not effectively constant -- more
        # than 1% of gaps deviate from the median by >1e-6, which would make
        # a single scalar step_length an inaccurate integrator.
        if step_length <= 0 or np.any(np.abs(diffs - step_length) > max(1e-6, step_length * 1e-3)):
            irregular = True

    totals = defaultdict(float)
    n_vehicle_steps = 0

    if not irregular:
        # Fast path: uniform spacing -- scale every row by the same constant.
        for ts in timesteps:
            for veh in ts.findall("vehicle"):
                totals["CO2"]  += float(veh.get("CO2", 0.0))  * step_length
                totals["CO"]   += float(veh.get("CO",  0.0))  * step_length
                totals["HC"]   += float(veh.get("HC",  0.0))  * step_length
                totals["NOx"]  += float(veh.get("NOx", 0.0))  * step_length
                totals["PMx"]  += float(veh.get("PMx", 0.0))  * step_length
                totals["fuel"] += float(veh.get("fuel", 0.0)) * step_length
                totals["electricity"] += float(veh.get("electricity", 0.0)) * step_length
                n_vehicle_steps += 1
    else:
        # Irregular / truncated file: integrate using the ACTUAL gap
        # preceding each timestep (trapezoid-free, still a left-Riemann sum,
        # but per-gap rather than one global constant). The gap before the
        # first timestep is unknown, so it falls back to the median as a
        # best-effort estimate for that single step only.
        time_to_gap = {}
        prev_t = None
        for t in times:
            time_to_gap[t] = (t - prev_t) if prev_t is not None else step_length
            prev_t = t
        for ts in timesteps:
            t = float(ts.get("time", 0.0))
            gap = time_to_gap.get(t, step_length)
            for veh in ts.findall("vehicle"):
                totals["CO2"]  += float(veh.get("CO2", 0.0))  * gap
                totals["CO"]   += float(veh.get("CO",  0.0))  * gap
                totals["HC"]   += float(veh.get("HC",  0.0))  * gap
                totals["NOx"]  += float(veh.get("NOx", 0.0))  * gap
                totals["PMx"]  += float(veh.get("PMx", 0.0))  * gap
                totals["fuel"] += float(veh.get("fuel", 0.0)) * gap
                totals["electricity"] += float(veh.get("electricity", 0.0)) * gap
                n_vehicle_steps += 1

    totals["step_length"]     = step_length
    totals["step_length_irregular"] = irregular
    totals["n_vehicle_steps"] = n_vehicle_steps
    return dict(totals)


# ---------------------------------------------------------------------------
# fcd.xml
# ---------------------------------------------------------------------------
def parse_fcd(xml_path: PathLike) -> pd.DataFrame:
    """Parse fcd.xml into a flat per-(timestep, vehicle) DataFrame."""
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"fcd XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows: List[dict] = []
    for ts in root.findall("timestep"):
        t = float(ts.get("time", 0.0))
        for veh in ts.findall("vehicle"):
            rows.append({
                "time":  t,
                "id":    veh.get("id"),
                "x":     float(veh.get("x", 0.0)),
                "y":     float(veh.get("y", 0.0)),
                "angle": float(veh.get("angle", 0.0)),
                "type":  veh.get("type"),
                "speed": float(veh.get("speed", 0.0)),
                "pos":   float(veh.get("pos", 0.0)),
                "lane":  veh.get("lane"),
                "slope": float(veh.get("slope", 0.0)),
            })
    return pd.DataFrame(rows)


def parse_fcd_for_metrics(
    xml_path: PathLike,
    stopped_speed_threshold: float = 0.1,
) -> Dict:
    """Extract throughput, average-speed, and congestion indicators from FCD.

    Returns a dict:
        total_vehicles   -- distinct vehicle IDs observed
        avg_speed        -- mean speed (m/s) across all (timestep, vehicle) rows
        max_queue        -- max simultaneous stopped-vehicle count on any single lane
        per_lane_max_queue -- {lane_id: max stopped count observed on that lane}
        n_timesteps      -- number of distinct timesteps with >=1 vehicle
    """
    df = parse_fcd(xml_path)
    if df.empty:
        return {
            "total_vehicles": 0,
            "avg_speed": 0.0,
            "max_queue": 0,
            "per_lane_max_queue": {},
            "n_timesteps": 0,
        }

    df["stopped"] = df["speed"] <= stopped_speed_threshold
    per_ts_lane_stopped = (
        df[df["stopped"]]
        .groupby(["time", "lane"])
        .size()
        .reset_index(name="stopped_count")
    )
    per_lane_max_queue = (
        per_ts_lane_stopped.groupby("lane")["stopped_count"].max().to_dict()
        if not per_ts_lane_stopped.empty else {}
    )
    max_queue = max(per_lane_max_queue.values()) if per_lane_max_queue else 0

    return {
        "total_vehicles":     int(df["id"].nunique()),
        "avg_speed":          float(df["speed"].mean()),
        "max_queue":          int(max_queue),
        "per_lane_max_queue": {k: int(v) for k, v in per_lane_max_queue.items()},
        "n_timesteps":        int(df["time"].nunique()),
    }


# ---------------------------------------------------------------------------
# Combined / high-level API
# ---------------------------------------------------------------------------
def compute_all_metrics(
    sumo_output_dir: PathLike,
    tripinfo_name:  str = "tripinfo.xml",
    emissions_name: str = "emissions.xml",
    fcd_name:       str = "fcd.xml",
    use_fcd:        bool = False,
    ev_keywords: Optional[set] = None,
) -> Dict:
    """Compute a full metrics dictionary from a directory of SUMO XML outputs.

    ``use_fcd`` is False by default because fcd.xml is typically much larger
    than tripinfo/emissions and is not required for the core metric set;
    enable it explicitly when queue/congestion detail from FCD is wanted.

    Returns a dict shaped to mirror ``_run_evaluation_episode`` / the online
    TraCI collection path, so results are directly comparable:
        avg_wait, avg_time_loss, total_duration, total_route_length,
        cars_passed, ev_cleared, ev_avg_wait,
        co2_total, nox_total, fuel_total, co_total, hc_total, pmx_total,
        emissions_xml_totals (cross-check from emissions.xml),
        fcd_metrics (optional).
    """
    sumo_output_dir = Path(sumo_output_dir)
    trip_df = parse_tripinfo(sumo_output_dir / tripinfo_name, ev_keywords=ev_keywords)

    if trip_df.empty:
        base = {
            "cars_passed": 0, "avg_wait": 0.0, "avg_time_loss": 0.0,
            "total_duration": 0.0, "total_route_length": 0.0,
            "ev_cleared": 0, "ev_avg_wait": None,
            "co2_total": 0.0, "nox_total": 0.0, "fuel_total": 0.0,
            "co_total": 0.0, "hc_total": 0.0, "pmx_total": 0.0,
        }
    else:
        # Only count FINISHED trips for cars_passed / ev_cleared to match
        # TraCI semantics (which only credits vehicles that actually arrived
        # during the simulation steps).  Emissions totals use ALL trips
        # (including unfinished ones) because partial emissions from
        # still-running vehicles were already counted by SUMO's accumulator.
        finished_df = trip_df[trip_df["finished"]]
        ev_df = finished_df[finished_df["ev"]]
        base = {
            "cars_passed":        int(len(finished_df)),
            "avg_wait":           float(finished_df["waitingTime"].mean()),
            "avg_time_loss":      float(finished_df["timeLoss"].mean()),
            "total_duration":     float(finished_df["duration"].sum()),
            "total_route_length": float(finished_df["routeLength"].sum()),
            "ev_cleared":         int(len(ev_df)),
            "ev_avg_wait":        (float(ev_df["waitingTime"].mean())
                                    if not ev_df.empty else None),
            "co2_total":  float(trip_df["CO2_abs"].sum()),
            "nox_total":  float(trip_df["NOx_abs"].sum()),
            "fuel_total": float(trip_df["fuel_abs"].sum()),
            "co_total":   float(trip_df["CO_abs"].sum()),
            "hc_total":   float(trip_df["HC_abs"].sum()),
            "pmx_total":  float(trip_df["PMx_abs"].sum()),
        }

    em_path = sumo_output_dir / emissions_name
    if em_path.exists():
        base["emissions_xml_totals"] = parse_emissions(em_path, aggregate=True)
    else:
        base["emissions_xml_totals"] = None

    if use_fcd:
        fcd_path = sumo_output_dir / fcd_name
        base["fcd_metrics"] = parse_fcd_for_metrics(fcd_path) if fcd_path.exists() else None
    else:
        base["fcd_metrics"] = None

    return base


def validate_traci_vs_xml(
    traci_stats: Dict,
    xml_metrics: Dict,
    rel_tolerance: float = 0.05,
    abs_tolerance: float = 1.0,
) -> Dict:
    """Compare TraCI-collected episode stats against XML-derived metrics.

    ``traci_stats`` -- as returned by ``sumo_env._snapshot_ep_stats()`` /
        ``_run_evaluation_episode`` (keys: co2_sum/co2_total,
        nox_sum/nox_total, fuel_sum/fuel_total, cars_passed, ev_cleared,
        ev_wait_sum).
    ``xml_metrics``  -- as returned by ``compute_all_metrics``.

    A field is "consistent" when it agrees within
    ``max(abs_tolerance, rel_tolerance * xml_value)``. TraCI and tripinfo.xml
    integrate emissions slightly differently (per-step subscription poll vs.
    SUMO's internal emission device), so exact equality is not expected --
    the tolerance defaults are chosen to catch real discrepancies (e.g. a
    broken accumulator) while allowing normal numerical drift.

    Returns a dict of {field: {"traci": v, "xml": v, "diff": d,
                                 "diff_pct": p, "consistent": bool}}
    plus an "all_consistent" boolean summary key.
    """
    def _get(d, *keys, default=0.0):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    checks = {
        "co2":  (_get(traci_stats, "co2_sum", "co2_total"),
                 _get(xml_metrics, "co2_total")),
        "nox":  (_get(traci_stats, "nox_sum", "nox_total"),
                 _get(xml_metrics, "nox_total")),
        "fuel": (_get(traci_stats, "fuel_sum", "fuel_total"),
                 _get(xml_metrics, "fuel_total")),
        "cars_passed": (_get(traci_stats, "cars_passed"),
                         _get(xml_metrics, "cars_passed")),
        "ev_cleared":  (_get(traci_stats, "ev_cleared"),
                         _get(xml_metrics, "ev_cleared")),
    }

    result: Dict[str, dict] = {}
    all_ok = True
    for field, (t_val, x_val) in checks.items():
        t_val = float(t_val or 0.0)
        x_val = float(x_val or 0.0)
        diff = t_val - x_val
        diff_pct = (abs(diff) / abs(x_val) * 100.0) if x_val != 0 else (
            0.0 if t_val == 0 else 100.0)
        tol = max(abs_tolerance, rel_tolerance * abs(x_val))
        consistent = abs(diff) <= tol
        all_ok = all_ok and consistent
        result[field] = {
            "traci": t_val,
            "xml":   x_val,
            "diff":  diff,
            "diff_pct": round(diff_pct, 2),
            "consistent": consistent,
        }

    result["all_consistent"] = all_ok

    # Surface a warning if emissions.xml had non-uniform timestep spacing --
    # this affects only the emissions_xml_totals cross-check field (not the
    # co2/nox/fuel checks above, which use tripinfo.xml's own accumulated
    # totals), but the person should know that secondary cross-check may be
    # less precise for this episode.
    em_totals = xml_metrics.get("emissions_xml_totals")
    if em_totals and em_totals.get("step_length_irregular"):
        result["warnings"] = result.get("warnings", []) + [
            "emissions.xml had non-uniform timestep spacing; "
            "emissions_xml_totals used per-gap integration as a fallback "
            "and may be less precise than the tripinfo.xml-based totals."
        ]

    return result


def load_metrics_from_xml(
    output_dir: PathLike,
    fallback_when_missing: bool = True,
    verbose: bool = True,
) -> Optional[Dict]:
    """Convenience loader used by sumo_env / metrics_and_eval as a fallback
    when TraCI-side data is incomplete or unavailable (e.g. resuming an
    offline analysis from XML files alone, without ever running TraCI).

    Returns None (instead of raising) if the tripinfo file is missing and
    ``fallback_when_missing`` is True, so callers can gracefully skip XML
    validation without crashing an RL run that never enabled XML output.

    When ``verbose`` (default True), prints the absolute path it checked --
    this is deliberately noisy on failure because a silent None return here
    previously made it impossible to tell whether XML output was disabled,
    misconfigured, or simply written somewhere else (e.g. a Windows
    relative-path / working-directory mismatch).
    """
    output_dir = Path(output_dir)
    tripinfo_path = (output_dir / "tripinfo.xml").resolve()
    if not tripinfo_path.exists():
        if verbose:
            print(f"    [xml_metrics] tripinfo.xml not found at: {tripinfo_path}", flush=True)
            print(f"    [xml_metrics] (resolved from output_dir={output_dir!r}, "
                  f"cwd={Path.cwd()})", flush=True)
            if output_dir.exists():
                siblings = sorted(p.name for p in output_dir.iterdir())
                print(f"    [xml_metrics] contents of {output_dir}: {siblings}", flush=True)
            else:
                print(f"    [xml_metrics] output_dir does not exist: {output_dir.resolve()}", flush=True)
        if fallback_when_missing:
            return None
        raise FileNotFoundError(f"tripinfo.xml not found in {tripinfo_path}")
    if verbose:
        print(f"    [xml_metrics] loading tripinfo.xml from: {tripinfo_path}", flush=True)
    return compute_all_metrics(output_dir)
