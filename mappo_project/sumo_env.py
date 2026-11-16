from __future__ import annotations
import sys
import os
import time
import csv
import json
import random
import signal
import warnings
import multiprocessing as mp
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

try:
    import torch_geometric
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    from rich.progress import (
        Progress, BarColumn, TextColumn,
        TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn,
    )
    try:
        from rich.progress import MofNCompleteColumn, TaskProgressColumn
    except ImportError:
        class MofNCompleteColumn:
            def __init__(self, *a, **kw): pass
        class TaskProgressColumn:
            def __init__(self, *a, **kw): pass
    try:
        from rich.layout import Layout
        from rich.columns import Columns
        from rich.style import Style
        from rich.progress import ProgressColumn
    except ImportError:
        Layout = Columns = Style = ProgressColumn = None
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Live = Table = Panel = Text = Rule = box = None
    Progress = BarColumn = TextColumn = None
    TimeElapsedColumn = TimeRemainingColumn = SpinnerColumn = None
    MofNCompleteColumn = TaskProgressColumn = None

try:
    from torch.utils.tensorboard import SummaryWriter as TBWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    TBWriter = None

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .config import *

__all__ = ['_reset_ep_stats', '_snapshot_ep_stats', '_collect_final_emissions', '_get_reward_denom', '_StepCache', '_subscribe_lanes', '_subscribe_detectors', '_subscribe_vehicles_bulk', '_setup_all_subscriptions', '_is_emergency_vehicle', '_get_ev_features', '_ev_potential', '_moving_stopped_counts', '_lane_queue', '_get_tls_queue_from_detectors', '_get_tls_mean_wait', '_estimate_lane_flow', '_shock_wave_speed', '_get_lanes_for_tls', '_get_green_phases_and_links', '_build_local_edge_index', '_build_global_edge_index', '_build_flat_edge_index', '_build_adj_mask', '_start_sumo', '_kill_sumo', '_close_sumo', '_null_ctx', '_fmt_time', '_STEP_CACHE', '_xml_output_args', '_load_xml_metrics_for_episode']


def _reset_ep_stats() -> None:
    global _active_evs, _ev_wait_tracker, _emission_tracker
    _EP_STATS["ev_cleared"]  = 0
    _EP_STATS["ev_wait_sum"] = 0.0
    _EP_STATS["cars_passed"] = 0
    # Reset emission totals for the new episode.
    _EP_STATS["co2_sum"]     = 0.0
    _EP_STATS["nox_sum"]     = 0.0
    _EP_STATS["fuel_sum"]    = 0.0
    # Clear per-vehicle trackers so data never bleeds across episodes.
    _active_evs       = set()
    _ev_wait_tracker  = {}
    _emission_tracker = {}

def _snapshot_ep_stats() -> dict:
    """Return a copy of the current episode stats."""
    return dict(_EP_STATS)

def _collect_final_emissions() -> None:
    """Flush remaining emissions of active vehicles to episode totals at the end of an episode."""
    global _emission_tracker
    for veh_id, em in _emission_tracker.items():
        _EP_STATS["co2_sum"]  += em["co2"]
        _EP_STATS["nox_sum"]  += em["nox"]
        _EP_STATS["fuel_sum"] += em["fuel"]
    _emission_tracker = {}

_xml_output_printed: bool = False

def _xml_output_args(output_dir: str | None = None) -> list[str]:
    """Return extra sumo_cmd flags to enable tripinfo/emissions/fcd XML output.

    Returns [] (no-op) when config.ENABLE_XML_OUTPUT is False, so callers can
    unconditionally splice this into sumo_cmd without an if-check at every
    call site. When enabled, paths are placed under XML_OUTPUT_DIR (or the
    explicit ``output_dir`` override) and the directory is created if needed.

    NOTE: this module does `from .config import *`, which copies config's
    names into this module's namespace at IMPORT time. If something later
    does `config.ENABLE_XML_OUTPUT = True` at runtime (e.g. a verification
    script, a notebook, or a monkeypatch), that assignment only updates
    config's own namespace -- it does NOT update the stale copy that would
    otherwise be read here via the bare name `ENABLE_XML_OUTPUT`. To make
    runtime toggling actually work, this function re-reads the live value
    from the `config` module object itself rather than trusting the
    star-imported copy.
    """
    from . import config as _cfg
    if not getattr(_cfg, "ENABLE_XML_OUTPUT", False):
        return []
    out_dir = os.path.abspath(output_dir or getattr(_cfg, "XML_OUTPUT_DIR", "."))
    os.makedirs(out_dir, exist_ok=True)
    tripinfo_path = os.path.join(out_dir, getattr(_cfg, "XML_TRIPINFO_FILE", "tripinfo.xml"))
    emission_path = os.path.join(out_dir, getattr(_cfg, "XML_EMISSIONS_FILE", "emissions.xml"))
    fcd_path      = os.path.join(out_dir, getattr(_cfg, "XML_FCD_FILE", "fcd.xml"))
    global _xml_output_printed
    if not _xml_output_printed:
        print(f"    [xml_metrics] XML output enabled -> {out_dir}", flush=True)
        print(f"    [xml_metrics]   tripinfo: {tripinfo_path}", flush=True)
        print(f"    [xml_metrics]   emission: {emission_path}", flush=True)
        print(f"    [xml_metrics]   fcd     : {fcd_path}", flush=True)
        _xml_output_printed = True
    return [
        "--tripinfo-output",  tripinfo_path,
        "--emission-output",  emission_path,
        "--fcd-output",       fcd_path,
        "--tripinfo-output.write-unfinished", "true",
    ]

def _load_xml_metrics_for_episode(output_dir: str | None = None) -> dict | None:
    """Best-effort load of offline XML-derived metrics for the episode that
    just finished (SUMO has already closed and flushed its output files by
    the time this is called). Returns None if XML output was not enabled or
    the files are missing/unparseable -- never raises, so it is safe to call
    unconditionally from the eval/fixed-time episode wrappers.

    Reads config.ENABLE_XML_OUTPUT / config.XML_OUTPUT_DIR live from the
    `config` module (see _xml_output_args docstring for why this matters).
    """
    from . import config as _cfg
    if not getattr(_cfg, "ENABLE_XML_OUTPUT", False):
        return None
    out_dir = os.path.abspath(output_dir or getattr(_cfg, "XML_OUTPUT_DIR", "."))
    try:
        from . import xml_metrics
        return xml_metrics.load_metrics_from_xml(out_dir, fallback_when_missing=True, verbose=True)
    except Exception as e:
        print(f"    [xml_metrics] Could not load XML metrics from {out_dir}: {e}", flush=True)
        return None

def _get_reward_denom(tls_id: str) -> float:
    n  = _TLS_LANE_COUNTS.get(tls_id, 4)
    qc = _QUEUE_CAPACITY_PER_TLS.get(tls_id, QUEUE_CAPACITY)
    return max(qc * LAMBDA * n, 1.0)

class _StepCache:
    __slots__ = ("_detector_halting", "_lane_veh_ids",
                 "_lane_halting", "_lane_vehicles",
                 "_veh_speed", "_veh_wait")

    def __init__(self):
        self._detector_halting: dict[str, float] = {}
        self._lane_veh_ids:     dict[str, tuple]  = {}
        self._lane_halting:     dict[str, float]  = {}
        self._lane_vehicles:    dict[str, int]    = {}
        self._veh_speed:        dict[str, float]  = {}
        self._veh_wait:         dict[str, float]  = {}

    def clear(self):
        self._detector_halting.clear()
        self._lane_veh_ids.clear()
        self._lane_halting.clear()
        self._lane_vehicles.clear()
        self._veh_speed.clear()
        self._veh_wait.clear()

    def bulk_load_lanes(self):
        if not _ALL_SUBSCRIBED_LANES:
            return
        try:
            results = traci.lane.getAllSubscriptionResults()
            for lane_id, data in results.items():
                if data is None:
                    continue
                if tc.LAST_STEP_VEHICLE_ID_LIST in data:
                    self._lane_veh_ids[lane_id] = data[tc.LAST_STEP_VEHICLE_ID_LIST]
                if tc.LAST_STEP_HALTING_NUMBER in data:
                    self._lane_halting[lane_id] = float(data[tc.LAST_STEP_HALTING_NUMBER])
                if tc.LAST_STEP_VEHICLE_NUMBER in data:
                    self._lane_vehicles[lane_id] = int(data[tc.LAST_STEP_VEHICLE_NUMBER])
        except Exception:
            pass

    def bulk_load_detectors(self):
        if not _ALL_SUBSCRIBED_DETECTORS:
            return
        try:
            results = traci.lanearea.getAllSubscriptionResults()
            for det_id, data in results.items():
                if data and tc.LAST_STEP_VEHICLE_HALTING_NUMBER in data:
                    self._detector_halting[det_id] = float(
                        data[tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
        except Exception:
            pass

    def bulk_load_vehicles(self):
        try:
            veh_ids = traci.vehicle.getIDList()
            if not veh_ids:
                return
            results = traci.vehicle.getAllSubscriptionResults()
            for veh_id, data in results.items():
                if data is None:
                    continue
                if tc.VAR_SPEED in data:
                    self._veh_speed[veh_id] = float(data[tc.VAR_SPEED])
                if tc.VAR_ACCUMULATED_WAITING_TIME in data:
                    self._veh_wait[veh_id] = float(data[tc.VAR_ACCUMULATED_WAITING_TIME])
        except Exception:
            pass

    def detector_halting(self, det_id: str) -> float:
        v = self._detector_halting.get(det_id)
        if v is None:
            try:
                v = float(traci.lanearea.getLastStepHaltingNumber(det_id))
            except Exception:
                v = 0.0
            self._detector_halting[det_id] = v
        return v

    def lane_veh_ids(self, lane_id: str) -> tuple:
        v = self._lane_veh_ids.get(lane_id)
        if v is None:
            try:
                v = traci.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                v = ()
            self._lane_veh_ids[lane_id] = v
        return v

    def lane_halting(self, lane_id: str) -> float:
        v = self._lane_halting.get(lane_id)
        if v is None:
            try:
                v = float(traci.lane.getLastStepHaltingNumber(lane_id))
            except Exception:
                v = 0.0
            self._lane_halting[lane_id] = v
        return v

    def lane_vehicles(self, lane_id: str) -> int:
        v = self._lane_vehicles.get(lane_id)
        if v is None:
            try:
                v = traci.lane.getLastStepVehicleNumber(lane_id)
            except Exception:
                v = 0
            self._lane_vehicles[lane_id] = v
        return v

    def veh_speed(self, veh_id: str) -> float:
        v = self._veh_speed.get(veh_id)
        if v is None:
            try:
                v = float(traci.vehicle.getSpeed(veh_id))
            except Exception:
                v = 0.0
            self._veh_speed[veh_id] = v
        return v

    def veh_wait(self, veh_id: str) -> float:
        v = self._veh_wait.get(veh_id)
        if v is None:
            try:
                v = float(traci.vehicle.getAccumulatedWaitingTime(veh_id))
            except Exception:
                v = 0.0
            self._veh_wait[veh_id] = v
        return v

def _subscribe_lanes(lane_ids: list[str]):
    for lane_id in lane_ids:
        if lane_id not in _ALL_SUBSCRIBED_LANES:
            try:
                traci.lane.subscribe(lane_id, _SUBSCRIPTION_VARS)
                _ALL_SUBSCRIBED_LANES.add(lane_id)
            except Exception:
                pass

def _subscribe_detectors(det_ids: list[str]):
    for det_id in det_ids:
        if det_id not in _ALL_SUBSCRIBED_DETECTORS:
            try:
                traci.lanearea.subscribe(det_id, _DETECTOR_SUB_VARS)
                _ALL_SUBSCRIBED_DETECTORS.add(det_id)
            except Exception:
                pass

def _subscribe_vehicles_bulk():
    """Subscribe newly-departed vehicles and unsubscribe arrived ones.
    Updates _EP_STATS with throughput, EV clearing, and CO2/NOx/fuel totals.

    THREE-PHASE approach -- avoids any TraCI call on already-removed vehicles:

      PHASE 1 DEPARTURE  : subscribe to speed, wait, and emission vars;
                           classify EVs via getTypeID() (reliable at departure);
                           initialise per-vehicle emission and EV-wait trackers.
      PHASE 2 ACTIVE     : every step, accumulate emission rates * STEP_LENGTH
                           into the per-vehicle tracker and refresh EV wait.
      PHASE 3 ARRIVAL    : read accumulated totals from trackers and add to
                           _EP_STATS -- zero TraCI calls on removed vehicles.
    """
    global _active_evs, _ev_wait_tracker, _emission_tracker
    try:
        # -- PHASE 1: subscribe newly-departed vehicles ------------------------
        new_vehs = traci.simulation.getDepartedIDList()
        for veh_id in new_vehs:
            try:
                # One subscribe call covers kinematics + all three emission vars.
                traci.vehicle.subscribe(veh_id, [
                    tc.VAR_SPEED,
                    tc.VAR_ACCUMULATED_WAITING_TIME,
                    tc.VAR_CO2EMISSION,      # CO2  rate (mg/s)
                    tc.VAR_NOXEMISSION,      # NOx  rate (mg/s)
                    tc.VAR_FUELCONSUMPTION,  # Fuel rate (ml/s)
                ])
            except Exception:
                pass
            # Initialise emission accumulator for every vehicle entering the network.
            _emission_tracker[veh_id] = {"co2": 0.0, "nox": 0.0, "fuel": 0.0}
            # Classify EV at departure -- getTypeID() is reliable here because
            # the vehicle has just been inserted and is still active in SUMO.
            try:
                _vtype = traci.vehicle.getTypeID(veh_id).lower()
                if any(kw in _vtype for kw in EV_TYPE_KEYWORDS):
                    _active_evs.add(veh_id)
                    _ev_wait_tracker[veh_id] = 0.0
            except Exception:
                pass

        # -- PHASE 2: update trackers for vehicles still in the network --------
        # Emission vars are instantaneous rates (mg/s); multiply by
        # STEP_LENGTH (seconds) to convert to per-step totals before summing.
        for veh_id, acc in list(_emission_tracker.items()):
            try:
                sub = traci.vehicle.getSubscriptionResults(veh_id)
                if sub:
                    acc["co2"]  += max(0.0, sub.get(tc.VAR_CO2EMISSION,     0.0)) * STEP_LENGTH
                    acc["nox"]  += max(0.0, sub.get(tc.VAR_NOXEMISSION,     0.0)) * STEP_LENGTH
                    acc["fuel"] += max(0.0, sub.get(tc.VAR_FUELCONSUMPTION, 0.0)) * STEP_LENGTH
            except Exception:
                pass
        # Refresh EV accumulated-wait from the subscription cache.
        for veh_id in list(_active_evs):
            try:
                sub = traci.vehicle.getSubscriptionResults(veh_id)
                if sub and tc.VAR_ACCUMULATED_WAITING_TIME in sub:
                    _ev_wait_tracker[veh_id] = sub[tc.VAR_ACCUMULATED_WAITING_TIME]
            except Exception:
                pass

        # -- PHASE 3: process arrived vehicles ---------------------------------
        # SUMO has already removed these vehicles before this code runs, so
        # ALL data comes from the tracker dicts -- no TraCI calls on them.
        arrived_vehs = traci.simulation.getArrivedIDList()
        _EP_STATS["cars_passed"] += len(arrived_vehs)
        for veh_id in arrived_vehs:
            # Add this vehicle's accumulated emissions to episode totals.
            em = _emission_tracker.pop(veh_id, None)
            if em:
                _EP_STATS["co2_sum"]  += em["co2"]
                _EP_STATS["nox_sum"]  += em["nox"]
                _EP_STATS["fuel_sum"] += em["fuel"]
            # Credit EV stats using the tracker (no getTypeID / getWaitingTime call).
            if veh_id in _active_evs:
                _EP_STATS["ev_cleared"]  += 1
                _EP_STATS["ev_wait_sum"] += _ev_wait_tracker.pop(veh_id, 0.0)
                _active_evs.discard(veh_id)
            else:
                _ev_wait_tracker.pop(veh_id, None)  # clean up non-EV entry if any
            # Unsubscribe (best-effort; SUMO may have already cleaned up).
            try:
                traci.vehicle.unsubscribe(veh_id)
            except Exception:
                pass
    except Exception:
        pass

def _setup_all_subscriptions(agents: dict):
    all_lanes = set()
    for agent in agents.values():
        all_lanes.update(agent.incoming_lanes)
        all_lanes.update(agent.outgoing_lanes)
    for tls_id in INTERSECTIONS:
        all_lanes.update(UPSTREAM_LANES.get(tls_id, []))
    _subscribe_lanes(list(all_lanes))

    all_dets = set()
    for dets in TLS_DETECTOR_MAP.values():
        all_dets.update(dets)
    _subscribe_detectors(list(all_dets))

def _is_emergency_vehicle(veh_id: str) -> bool:
    """Return True if veh_id is an ambulance / fire-truck / police car."""
    if not EV_PRIORITY_ENABLED:
        return False
    try:
        if traci.vehicle.getVehicleClass(veh_id) in EV_CLASSES:
            return True
        vtype = traci.vehicle.getTypeID(veh_id).lower()
        return any(kw in vtype for kw in EV_TYPE_KEYWORDS)
    except Exception:
        return False

def _get_ev_features(tls_id: str, incoming_lanes) -> np.ndarray:
    """
    Returns EV feature vector for one agent (length = 2·n_in + 2):
      [ev_count_norm[0..n_in-1],
       ev_priority[0..n_in-1],
       ev_total_priority,
       ev_closest_dist]
    All values in [0, 1]. Returns zeros when EV_PRIORITY_ENABLED=False.
    """
    n_in = len(incoming_lanes)
    zero = np.zeros(2 * n_in + 2, dtype=np.float32)
    if not EV_PRIORITY_ENABLED:
        return zero

    ev_count    = np.zeros(n_in, dtype=np.float32)
    ev_priority = np.zeros(n_in, dtype=np.float32)
    closest_dist = 0.0

    for k, lane_id in enumerate(incoming_lanes):
        try:
            lane_len = max(traci.lane.getLength(lane_id), 1.0)
        except Exception:
            lane_len = 100.0
        for veh_id in _STEP_CACHE.lane_veh_ids(lane_id):
            if not _is_emergency_vehicle(veh_id):
                continue
            try:
                pos = traci.vehicle.getLanePosition(veh_id)   # 0=entry, L=stop-line
            except Exception:
                continue
            # proximity weight: 1.0 at stop-line, ~0.05 at far end
            w = float(np.exp(-EV_BETA * (1.0 - pos / lane_len)))
            ev_count[k]    += 1.0
            ev_priority[k]  = max(ev_priority[k], w)
            closest_dist    = max(closest_dist, pos / lane_len)

    ev_count_norm = np.minimum(ev_count / 5.0, 1.0)   # cap at 5 EVs
    total_p       = float(1.0 - np.exp(-ev_priority.sum()))  # soft saturation
    return np.concatenate([ev_count_norm, ev_priority,
                           [total_p], [closest_dist]]).astype(np.float32)

def _ev_potential(incoming_lanes, current_action_idx: int,
                  action_phases: list) -> float:
    """
    Potential Φ(s) = Σ_l  max_v[w_prox(v,l)] × is_green(l)
    High when EVs already have a green phase -> encourages holding green for EVs.
    Returns 0.0 when EV_PRIORITY_ENABLED=False.
    """
    if not EV_PRIORITY_ENABLED:
        return 0.0
    # lanes that are currently green
    try:
        green_lanes = set(action_phases[current_action_idx]) if action_phases else set()
    except (IndexError, TypeError):
        green_lanes = set()

    phi = 0.0
    for lane_id in incoming_lanes:
        if lane_id not in green_lanes:
            continue
        try:
            lane_len = max(traci.lane.getLength(lane_id), 1.0)
        except Exception:
            lane_len = 100.0
        for veh_id in _STEP_CACHE.lane_veh_ids(lane_id):
            if not _is_emergency_vehicle(veh_id):
                continue
            try:
                pos = traci.vehicle.getLanePosition(veh_id)
            except Exception:
                continue
            phi += float(np.exp(-EV_BETA * (1.0 - pos / lane_len)))
    return phi

def _moving_stopped_counts(lane_id: str):
    moving  = 0
    stopped = 0
    for veh_id in _STEP_CACHE.lane_veh_ids(lane_id):
        if _STEP_CACHE.veh_speed(veh_id) <= STOPPED_SPEED_THRESHOLD:
            stopped += 1
        else:
            moving += 1
    return moving, stopped

def _lane_queue(lane_id: str) -> float:
    det_id = LANE_DETECTOR_MAP.get(lane_id)
    if det_id:
        return _STEP_CACHE.detector_halting(det_id)
    return _STEP_CACHE.lane_halting(lane_id)

def _get_tls_queue_from_detectors(tls_id: str) -> float:
    dets = TLS_DETECTOR_MAP.get(tls_id)
    if dets:
        return sum(_STEP_CACHE.detector_halting(d) for d in dets)
    return 0.0

def _get_tls_mean_wait(tls_id: str) -> float:
    lanes = UPSTREAM_LANES.get(tls_id, [])
    if not lanes:
        return 0.0
    total_wait = 0.0
    total_veh  = 0
    for lane_id in lanes:
        for veh_id in _STEP_CACHE.lane_veh_ids(lane_id):
            total_wait += _STEP_CACHE.veh_wait(veh_id)
            total_veh  += 1
    return total_wait / total_veh if total_veh > 0 else 0.0

def _estimate_lane_flow(lane_id: str, lane_length: float) -> float:
    n_vehicles = _STEP_CACHE.lane_vehicles(lane_id)
    density    = n_vehicles / lane_length
    flow       = V_FREE * density - _V_FREE_OVER_KJAM * (density ** 2)
    return max(flow, 0.0)

def _shock_wave_speed(lane_up: str, lane_down: str,
                      lane_length: float = 100.0) -> float:
    """Rankine-Hugoniot shock speed between two adjacent lanes.

    v_shock = (q_down - q_up) / (k_down - k_up)

    Returns 0.0 when the density difference is negligible (no shock).
    Positive value  -> shock propagates downstream (flow moving forward).
    Negative value  -> shock propagates upstream  (queue growing backward).
    """
    q_up   = _estimate_lane_flow(lane_up,   lane_length)
    q_down = _estimate_lane_flow(lane_down, lane_length)
    k_up   = _STEP_CACHE.lane_vehicles(lane_up)   / max(lane_length, 1.0)
    k_down = _STEP_CACHE.lane_vehicles(lane_down) / max(lane_length, 1.0)
    dk = k_down - k_up
    if abs(dk) < 1e-4:          # no meaningful density gradient -> no shock
        return 0.0
    v_shock = (q_down - q_up) / dk
    # Clip to physically meaningful range [-V_FREE, V_FREE]
    return float(np.clip(v_shock, -V_FREE, V_FREE))

def _get_lanes_for_tls(tls_id: str):
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    incoming = set()
    outgoing = set()
    for link_group in controlled_links:
        for link in link_group:
            if link:
                in_lane, out_lane, _ = link
                if in_lane:
                    incoming.add(in_lane)
                if out_lane:
                    outgoing.add(out_lane)
    return sorted(incoming), sorted(outgoing)

def _get_green_phases_and_links(tls_id: str):
    logic            = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    controlled_links = traci.trafficlight.getControlledLinks(tls_id)
    action_phases    = []
    phase_links      = {}

    for phase_idx, phase in enumerate(logic.phases):
        state_str = phase.state
        if ("G" not in state_str and "g" not in state_str) or ("y" in state_str):
            continue
        active_links = []
        for link_idx, signal in enumerate(state_str):
            if signal not in ("G", "g"):
                continue
            for link in controlled_links[link_idx]:
                in_lane, out_lane, _ = link
                if in_lane and out_lane:
                    active_links.append((in_lane, out_lane))
        if active_links:
            action_phases.append(phase_idx)
            phase_links[phase_idx] = active_links

    return action_phases, phase_links

def _build_local_edge_index(cluster: list) -> "torch.Tensor":
    n   = len(cluster)
    idx = {tls: i for i, tls in enumerate(cluster)}
    src, dst = [], []
    for tls_id in cluster:
        i = idx[tls_id]
        for nb in cluster:
            if nb != tls_id:
                src.append(i); dst.append(idx[nb])
    return torch.tensor([src, dst], dtype=torch.long)

def _build_global_edge_index() -> "torch.Tensor":
    n_clusters = len(LOCAL_CLUSTERS)
    src = list(range(n_clusters)) * n_clusters
    dst = [j for j in range(n_clusters) for _ in range(n_clusters)]
    src = [s for s, d in zip(src, dst) if s != d]
    dst = [d for s, d in zip(
        list(range(n_clusters)) * n_clusters,
        [j for j in range(n_clusters) for _ in range(n_clusters)]
    ) if s != d]
    return torch.tensor([src, dst], dtype=torch.long)

def _build_flat_edge_index() -> "torch.Tensor":
    src, dst = [], []
    for tls_id, neighbors in NEIGHBOR_MAP.items():
        i = _TLS_INDEX[tls_id]
        for nb in neighbors:
            j = _TLS_INDEX[nb]
            src.append(i); dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)

def _build_adj_mask(device: torch.device) -> torch.Tensor:
    global _STATIC_ADJ_MASK
    if _STATIC_ADJ_MASK is None:
        n   = len(INTERSECTIONS)
        adj = torch.zeros(n, n)
        for tls_id, neighbors in NEIGHBOR_MAP.items():
            i = _TLS_INDEX[tls_id]
            for nb in neighbors:
                j = _TLS_INDEX[nb]
                adj[i, j] = 1.0
        _STATIC_ADJ_MASK = adj
    return _STATIC_ADJ_MASK.to(device)

def _find_free_port(preferred: int, span: int = 200) -> int:
    """Return `preferred` if free, otherwise scan upward for a free port."""
    import socket
    for p in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            try:
                s.bind(("localhost", p))
                return p
            except OSError:
                continue
    raise RuntimeError(
        f"No free port found in range [{preferred}, {preferred + span})")


def _kill_process_on_port(port: int) -> bool:
    """Best-effort kill of whatever process is holding `port`. Returns True
    if a process was found and a kill was attempted."""
    import subprocess
    import platform

    system = platform.system()
    try:
        if system == "Windows":
            out = subprocess.check_output(
                ["netstat", "-ano", "-p", "TCP"],
                text=True, stderr=subprocess.DEVNULL)
            pids = set()
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
                    pids.add(parts[-1])
            for pid in pids:
                subprocess.run(["taskkill", "/F", "/PID", pid],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            return bool(pids)
        else:
            out = subprocess.check_output(
                ["lsof", "-t", f"-i:{port}"],
                text=True, stderr=subprocess.DEVNULL)
            pids = [p for p in out.split() if p]
            for pid in pids:
                subprocess.run(["kill", "-9", pid],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            return bool(pids)
    except Exception:
        return False


def _start_sumo(sumo_cmd, port=8813):
    import subprocess
    import socket

    def _port_busy(p):
        try:
            with socket.create_connection(("localhost", p), timeout=0.3):
                return True
        except OSError:
            return False

    if _port_busy(port):
        for attempt in range(15):
            if not _port_busy(port):
                break
            print(f"    [SUMO] Port {port} busy (attempt {attempt+1}/15), waiting...",
                  flush=True)
            time.sleep(0.5)
        else:
            # Still stuck after waiting -- try to actively reclaim it
            print(f"    [SUMO] Port {port} still busy; attempting to kill "
                  f"the process holding it...", flush=True)
            killed = _kill_process_on_port(port)
            if killed:
                time.sleep(1.0)
            if _port_busy(port):
                # Give up on this exact port and pick a free one instead,
                # rather than aborting the whole training run.
                new_port = _find_free_port(port + 1)
                print(f"    [SUMO] Port {port} unrecoverable; "
                      f"falling back to port {new_port}.", flush=True)
                port = new_port

    full_cmd = sumo_cmd + ["--remote-port", str(port)]
    print("    [SUMO] Launching:", " ".join(full_cmd), flush=True)

    import tempfile, os as _os
    _stderr_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    proc = subprocess.Popen(full_cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=_stderr_file)
    _stderr_file.close()
    print("    [SUMO] Waiting for SUMO to start...", flush=True)

    deadline = time.time() + 90.0
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            # Read and print SUMO's error output so the root cause is visible
            try:
                with open(_stderr_file.name, "r", errors="replace") as _ef:
                    _sumo_err = _ef.read().strip()
            except Exception:
                _sumo_err = "(could not read SUMO stderr)"
            try:
                _os.unlink(_stderr_file.name)
            except Exception:
                pass
            if _sumo_err:
                print(f"\n    [SUMO ERROR OUTPUT]\n{_sumo_err}\n", flush=True)
            raise RuntimeError(
                f"SUMO exited prematurely (rc={rc}) -- check your .sumocfg and network files.")
        time.sleep(0.5)
        try:
            traci.init(port=port, host="localhost", numRetries=0)
            print("    [SUMO] TraCI connected.", flush=True)
            return proc
        except Exception:
            pass

    _kill_sumo(proc)
    raise RuntimeError(
        f"SUMO did not accept TraCI connection within 90s -- port {port} may be blocked.")

def _kill_sumo(proc):
    if proc is None:
        return
    import platform
    pid = proc.pid
    if platform.system() == "Windows":
        # On Windows, terminate() sends CTRL_BREAK which SUMO may ignore.
        # Use taskkill /F to force-kill the entire process tree instantly.
        try:
            import subprocess as _sp
            _sp.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                timeout=5,
            )
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    else:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
            except Exception:
                pass

def _close_sumo(proc):
    """Close the TraCI connection and shut down the SUMO process.

    IMPORTANT (XML output correctness): traci.close() tells SUMO to finish
    writing its output files (tripinfo.xml / emissions.xml / fcd.xml) and
    exit on its own. That file flush + process exit is not instantaneous,
    especially on Windows. Previously this function force-killed the SUMO
    process (via taskkill /F on Windows) immediately after traci.close()
    returned, with no wait -- which could truncate or entirely prevent the
    XML files from being written, even though TraCI itself had already
    received all its data in-memory. That is why TraCI-side totals could
    look correct while tripinfo.xml was missing or empty.

    Fix: after traci.close(), wait briefly for the SUMO process to exit on
    its own (it should, since traci.close() sent the shutdown command).
    Only force-kill if it is still alive after the grace period -- so a
    normal, cooperative shutdown is never interrupted mid-write.
    """
    try:
        traci.close()
    except Exception:
        pass

    if proc is not None:
        # Give SUMO a chance to finish flushing output files and exit
        # cleanly before we consider force-killing it.
        try:
            proc.wait(timeout=10.0)
        except Exception:
            # Process did not exit on its own within the grace period --
            # fall back to the existing force-kill path.
            _kill_sumo(proc)

    _ALL_SUBSCRIBED_LANES.clear()
    _ALL_SUBSCRIBED_DETECTORS.clear()

@contextmanager
def _null_ctx():
    """No-op context manager -- replaces torch.cuda.amp.autocast on CPU."""
    yield

def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"

# -- module-level state ----------------------------------------------
_STEP_CACHE = _StepCache()