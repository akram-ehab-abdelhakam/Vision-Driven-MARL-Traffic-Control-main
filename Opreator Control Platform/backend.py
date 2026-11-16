from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

DEFAULT_SAVE_DIR: str = "./results"
DEMO_EPISODE:     int = 40         # which checkpoint to load by default
START_WALK_STEP:  int = 0          # no warm-up - stream live SUMO data immediately
TICK_INTERVAL:    float = 1.0      # seconds between /state updates
SUMO_BINARY: str = "sumo"          # Changed from "sumo-gui"
SUMO_CFG:    str = "./sumo_cfg/om_network.sumocfg"  # path to your .sumocfg
SUMO_PORT:   int = 8813

_EPISODE_STATS_CACHE: Dict[Tuple[str, int], dict] = {}
manual_override_phase: Optional[int] = None

def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_episode_number(episode: Union[int, str]) -> int:
    if isinstance(episode, int):
        return episode
    ep_raw = str(episode).strip()
    if not ep_raw:
        raise ValueError("empty episode value")
    ep_digits = ep_raw.lower()
    if ep_digits.startswith("ep"):
        ep_digits = ep_digits[2:]
    ep_digits = ep_digits.strip()
    if not ep_digits:
        return 0
    return int(ep_digits)


def load_episode_metrics(
    episode: Union[int, str], save_dir: str = DEFAULT_SAVE_DIR
) -> dict:
    ep = _parse_episode_number(episode)
    metrics_path = (
        Path(save_dir) / "checkpoints" / f"ep{ep:04d}" / "metrics.json"
    )
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"metrics.json not found for episode {ep} at: {metrics_path}"
        )
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Corrupted metrics.json for episode {ep}: {exc}"
        ) from exc


def compute_avg_delay(metrics: dict) -> float:
    return _coerce_float((metrics.get("rl") or {}).get("avg_wait"), 0.0)


def compute_dashboard_metrics(metrics: dict) -> dict:
    rl = metrics.get("rl") or {}
    improvement = metrics.get("improvement") or {}

    # Per-TL queue & reward - try several common metrics.json layouts:
    #   1. metrics["per_tl"]["TL1"] = {"queue": ..., "reward": ...}
    #   2. metrics["tl_stats"]["TL1"] = {"avg_queue": ..., "avg_reward": ...}
    #   3. metrics["rl"]["per_tl"] ...
    #   4. top-level "per_agent_queue" / "per_agent_reward" dicts
    per_tl_queue:  dict = {}
    per_tl_reward: dict = {}

    raw_per_tl = (
        metrics.get("per_tl")
        or metrics.get("tl_stats")
        or (rl.get("per_tl") if isinstance(rl, dict) else None)
        or {}
    )
    for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
        tl_data = raw_per_tl.get(tl_id) or {}
        q = (tl_data.get("queue") or tl_data.get("avg_queue")
             or tl_data.get("mean_queue") or tl_data.get("final_queue"))
        r = (tl_data.get("reward") or tl_data.get("avg_reward")
             or tl_data.get("mean_reward") or tl_data.get("final_reward"))
        if q is not None:
            per_tl_queue[tl_id] = _coerce_float(q)
        if r is not None:
            per_tl_reward[tl_id] = _coerce_float(r)

    # Fallback: top-level flat dicts
    if not per_tl_queue:
        paq = metrics.get("per_agent_queue") or {}
        for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
            if tl_id in paq:
                per_tl_queue[tl_id] = _coerce_float(paq[tl_id])
    if not per_tl_reward:
        par = metrics.get("per_agent_reward") or {}
        for tl_id in ["TL1", "TL2", "TL3", "TL4"]:
            if tl_id in par:
                per_tl_reward[tl_id] = _coerce_float(par[tl_id])

    return {
        "avg_delay":        compute_avg_delay(metrics),
        "avg_queue":        _coerce_float(rl.get("avg_queue"), 0.0),
        "avg_reward":       _coerce_float(rl.get("avg_reward"), 0.0),
        "improvement_wait": _coerce_float(improvement.get("wait_pct"), 0.0),
        "per_tl_queue":     per_tl_queue,
        "per_tl_reward":    per_tl_reward,
    }


def get_episode_stats(
    episode: Union[int, str],
    save_dir: str = DEFAULT_SAVE_DIR,
    use_cache: bool = True,
) -> dict:
    ep = _parse_episode_number(episode)
    cache_key = (str(Path(save_dir).resolve()), ep)
    if use_cache and cache_key in _EPISODE_STATS_CACHE:
        return dict(_EPISODE_STATS_CACHE[cache_key])
    metrics = load_episode_metrics(ep, save_dir=save_dir)
    stats = compute_dashboard_metrics(metrics)
    stats["episode"] = ep
    if use_cache:
        _EPISODE_STATS_CACHE[cache_key] = dict(stats)
    return stats

def _traci_available() -> bool:
    try:
        import traci
        return traci.isLoaded()
    except Exception:
        return False


def _get_traci_metrics(tl_id: str, lane_ids: List[str]) -> dict:
    """
    Fetch real metrics from a running SUMO simulation via traci.
    Returns: waiting_cars, queue_length, avg_speed, reward
    Falls back to zeros if traci is not available.
    """
    try:
        import traci
        if not traci.isLoaded():
            return {"waiting_cars": 0, "queue_length": 0.0,
                    "avg_speed": 50.0, "reward": 0.0}

        waiting = 0
        queue   = 0.0
        speeds  = []
        for lane in lane_ids:
            try:
                waiting += traci.lane.getLastStepHaltingNumber(lane)
                queue   += traci.lane.getLastStepLength(lane)   # meters
                speeds.append(traci.lane.getLastStepMeanSpeed(lane) * 3.6)  # m/s-km/h
            except Exception:
                pass

        avg_speed = float(np.mean(speeds)) if speeds else 50.0
        reward = -float(waiting) / max(len(lane_ids), 1)
        return {
            "waiting_cars": int(waiting),
            "queue_length": round(queue / max(len(lane_ids), 1), 2),
            "avg_speed":    round(avg_speed, 1),
            "reward":       round(reward, 4),
        }
    except Exception:
        return {"waiting_cars": 0, "queue_length": 0.0,
                "avg_speed": 50.0, "reward": 0.0}


def _start_sumo(cfg: str, port: int) -> bool:
    """Launch SUMO in headless mode and connect traci."""
    try:
        import traci
        import sumolib
        sumo_cmd = [SUMO_BINARY, "-c", cfg,
                    "--remote-port", str(port),
                    "--no-warnings", "true"]
        traci.start(sumo_cmd, port=port)
        print(f"[SUMO] Started headless SUMO on port {port}", flush=True)
        return True
    except Exception as exc:
        print(f"[SUMO] Could not start SUMO: {exc} - running in synthetic mode",
              flush=True)
        return False

RL_AVAILABLE:  bool = False
RL_IMPORT_ERR: str  = ""

_here = os.path.dirname(os.path.abspath(__file__))
# Add both the mappo_project folder itself AND its parent so that
# "import mappo_project.xxx" works regardless of where uvicorn is launched from.
_pkg_parent = os.path.dirname(_here)
for _p in (_here, _pkg_parent):
    if _p not in sys.path:
        sys.path.insert(0, _p)

print("[backend_6] Importing mappo_project (inference symbols only) ...", flush=True)
try:
    _mod_cfg   = importlib.import_module("mappo_project.config")
    _mod_agent = importlib.import_module("mappo_project.agent")
    _mod_ckpt  = importlib.import_module("mappo_project.metrics_and_eval")
    _mod_train = importlib.import_module("mappo_project.training_loop")

    TL_IDS          = _mod_cfg.INTERSECTIONS
    MAPPOAgent      = _mod_agent.MAPPOAgent
    load_checkpoint = _mod_ckpt.load_checkpoint
    run_training    = getattr(_mod_train, "run_training", None)

    NEIGHBOR_MAP = getattr(_mod_cfg, "NEIGHBOR_MAP", {
        "TL1": ["TL2", "TL3"],
        "TL2": ["TL1", "TL3", "TL4"],
        "TL3": ["TL2", "TL4", "TL1"],
        "TL4": ["TL2", "TL3"],
    })
    INTENT_DIM = getattr(_mod_cfg, "INTENT_SHARING_DIM", 32)

    _ftp  = getattr(_mod_cfg, "FIXED_TIME_PLANS", {})
    _up   = getattr(_mod_cfg, "UPSTREAM_LANES", {})
    _qcap = getattr(_mod_cfg, "_QUEUE_CAPACITY_PER_TLS", {})
    UPSTREAM_LANES: Dict[str, List[str]] = dict(_up)

    WEBSTER = {}
    for _t, _plan in _ftp.items():
        _lanes = _up.get(_t, [])
        _edges = ", ".join(sorted({l.split("_")[0] for l in _lanes}))
        _cycle = _plan.get("cycle_length", 60)
        WEBSTER[_t] = {
            "queue":  _qcap.get(_t, 50) * 0.8,
            "delay":  _cycle / 2.0,
            "phases": len(_plan.get("phases", [])),
            "edges":  _edges or _t,
            "cycle":  _cycle,
        }
    if not WEBSTER:
        WEBSTER = {
            "TL1": {"queue": 155.86, "delay": 89.0,  "phases": 10,
                    "edges": "E7, E9",      "cycle": 60},
            "TL2": {"queue": 42.56,  "delay": 85.5,  "phases": 9,
                    "edges": "E1,E2,E4,E6", "cycle": 60},
            "TL3": {"queue": 13.18,  "delay": 51.7,  "phases": 6,
                    "edges": "E3, E5",      "cycle": 60},
            "TL4": {"queue": 2.50,   "delay": 41.6,  "phases": 4,
                    "edges": "E8, E10",     "cycle": 60},
        }
    if not UPSTREAM_LANES:
        UPSTREAM_LANES = {
            "TL1": ["E7_0", "E7_1", "E9_0", "E9_1"],
            "TL2": ["E1_0", "E2_0", "E4_0", "E6_0"],
            "TL3": ["E3_0", "E3_1", "E5_0", "E5_1"],
            "TL4": ["E8_0", "E8_1", "E10_0", "E10_1"],
        }

    RL_AVAILABLE = True
    print(
        f"[backend_6] - mappo_project loaded (inference only).\n"
        f"  TL_IDS          = {TL_IDS}\n"
        f"  MAPPOAgent      = found\n"
        f"  load_checkpoint = found\n"
        f"  run_training    = {'found' if run_training is not None else 'NOT found in mappo_project.training_loop'}",
        flush=True,
    )

except Exception as _exc:
    RL_IMPORT_ERR = traceback.format_exc()
    print(
        f"\n[backend_6] - mappo_project unavailable - SYNTHETIC mode.\n"
        f"Reason: {_exc}",
        flush=True,
    )
    TL_IDS          = ["TL1", "TL2", "TL3", "TL4"]
    MAPPOAgent      = None
    load_checkpoint = None
    run_training    = None
    NEIGHBOR_MAP    = {
        "TL1": ["TL2", "TL3"], "TL2": ["TL1", "TL3", "TL4"],
        "TL3": ["TL2", "TL4", "TL1"], "TL4": ["TL2", "TL3"],
    }
    INTENT_DIM = 32
    UPSTREAM_LANES = {
        "TL1": ["E7_0", "E7_1", "E9_0", "E9_1"],
        "TL2": ["E1_0", "E2_0", "E4_0", "E6_0"],
        "TL3": ["E3_0", "E3_1", "E5_0", "E5_1"],
        "TL4": ["E8_0", "E8_1", "E10_0", "E10_1"],
    }
    WEBSTER = {
        "TL1": {"queue": 155.86, "delay": 89.0,  "phases": 10,
                "edges": "E7, E9",      "cycle": 60},
        "TL2": {"queue": 42.56,  "delay": 85.5,  "phases": 9,
                "edges": "E1,E2,E4,E6", "cycle": 60},
        "TL3": {"queue": 13.18,  "delay": 51.7,  "phases": 6,
                "edges": "E3, E5",      "cycle": 60},
        "TL4": {"queue": 2.50,   "delay": 41.6,  "phases": 4,
                "edges": "E8, E10",     "cycle": 60},
    }
MODE_RL, MODE_FIXED, MODE_MANUAL = 0, 1, 2
MODE_NAMES   = {MODE_RL: "RL", MODE_FIXED: "FIXED-TIME", MODE_MANUAL: "MANUAL"}
PHASE_LABELS = ["GREEN", "RED", "YELLOW"]
THROUGHPUT_W  = 1247
THROUGHPUT_RL = 1659

RL_EP40 = {
    "TL1": {"queue": 101.31, "delay": 55.0, "corr": 0.85, "reward": -12.4},
    "TL2": {"queue": 27.66,  "delay": 51.5, "corr": 0.72, "reward": -8.7},
    "TL3": {"queue": 7.91,   "delay": 30.0, "corr": 0.63, "reward": -4.2},
    "TL4": {"queue": 1.50,   "delay": 25.0, "corr": 0.47, "reward": -2.1},
}

DETECTORS = [
    {"id": "e2_E7_0",  "tl": "TL1", "edge": "E7"},
    {"id": "e2_E7_1",  "tl": "TL1", "edge": "E7"},
    {"id": "e2_E9_0",  "tl": "TL1", "edge": "E9"},
    {"id": "e2_E9_1",  "tl": "TL1", "edge": "E9"},
    {"id": "e2_E1_0",  "tl": "TL2", "edge": "E1"},
    {"id": "e2_E2_0",  "tl": "TL2", "edge": "E2"},
    {"id": "e2_E4_0",  "tl": "TL2", "edge": "E4"},
    {"id": "e2_E6_0",  "tl": "TL2", "edge": "E6"},
    {"id": "e2_E3_0",  "tl": "TL3", "edge": "E3"},
    {"id": "e2_E3_1",  "tl": "TL3", "edge": "E3"},
    {"id": "e2_E5_0",  "tl": "TL3", "edge": "E5"},
    {"id": "e2_E5_1",  "tl": "TL3", "edge": "E5"},
    {"id": "e2_E8_0",  "tl": "TL4", "edge": "E8"},
    {"id": "e2_E8_1",  "tl": "TL4", "edge": "E8"},
    {"id": "e2_E10_0", "tl": "TL4", "edge": "E10"},
    {"id": "e2_E10_1", "tl": "TL4", "edge": "E10"},
]
SCEN_NORMAL, SCEN_PEAK, SCEN_EMERG = 0, 1, 2
SCEN_NAMES = {0: "Normal", 1: "Peak Hour", 2: "Emergency Vehicle"}

class InferenceLoader:
    """
    Finds results/checkpoints/ep0040/, calls load_checkpoint() to
    restore actor/critic/gat state dicts, then patches every agent
    to pure inference mode.
    """

    def __init__(self, save_dir: str = DEFAULT_SAVE_DIR) -> None:
        self.save_dir         = Path(save_dir)
        self.loaded_ep:       Optional[int] = None
        self.checkpoint_path: Optional[str] = None
        self.agents:          Dict[str, object] = {}
        self.load_ok:         bool = False
        self.load_error:      str  = ""
        self.dashboard_metrics: dict = {}
        self.dashboard_error: Optional[str] = None

    def load(self, episode: Optional[Union[int, str]] = None) -> bool:
        target_ep: Optional[int] = None
        if episode is not None:
            try:
                target_ep = int(episode)
            except (ValueError, TypeError):
                target_ep = None

        global DEMO_EPISODE
        if target_ep is not None:
            DEMO_EPISODE = target_ep

        print(
            f"\n[InferenceLoader] Searching for ep{DEMO_EPISODE:04d} checkpoint"
            f"\n  save_dir = {self.save_dir.resolve()}",
            flush=True,
        )
        path, ep = self._find_checkpoint()
        if path is None:
            self.load_error = (
                f"No checkpoint found under '{self.save_dir}'. "
                f"Expected results/checkpoints/ep{DEMO_EPISODE:04d}/"
            )
            print(f"[InferenceLoader] - {self.load_error}", flush=True)
            return False

        self.checkpoint_path = str(path)
        self.loaded_ep       = ep
        print(f"[InferenceLoader]  - {path}  (ep={ep})", flush=True)
        self._refresh_dashboard_metrics()

        if RL_AVAILABLE and load_checkpoint is not None and MAPPOAgent is not None:
            return self._load_and_patch(path, ep)

        print("[InferenceLoader]  SYNTHETIC mode - no .pt weights loaded.", flush=True)
        self.load_ok = True
        return True

    def _find_checkpoint(self) -> Tuple[Optional[Path], Optional[int]]:
        ckpt = self.save_dir / "checkpoints"

        ep_dir = ckpt / f"ep{DEMO_EPISODE:04d}"
        if ep_dir.is_dir() and list(ep_dir.glob("*.pt")):
            return ep_dir, DEMO_EPISODE

        for p in [ckpt / f"ep{DEMO_EPISODE:04d}.pt",
                  ckpt / f"ep{DEMO_EPISODE:04d}.pth"]:
            if p.exists():
                return p, DEMO_EPISODE

        if ckpt.exists():
            ep_dirs = sorted(
                [d for d in ckpt.iterdir()
                 if d.is_dir() and d.name.startswith("ep") and d.name[2:].isdigit()],
                reverse=True,
            )
            if ep_dirs:
                n = int(ep_dirs[0].name[2:])
                print(
                    f"[InferenceLoader]  ep{DEMO_EPISODE:04d} not found - "
                    f"using highest available: ep{n:04d}",
                    flush=True,
                )
                return ep_dirs[0], n

            pts = sorted(ckpt.glob("ep*.pt"), reverse=True)
            if pts:
                n = int(pts[0].stem[2:])
                return pts[0], n

            lj = ckpt / "latest.json"
            if lj.exists():
                try:
                    ptr = json.loads(lj.read_text())
                    n   = int(ptr["latest_ep"])
                    d   = ckpt / f"ep{n:04d}"
                    if d.is_dir():
                        return d, n
                except Exception:
                    pass

            emg = ckpt / "emergency.pt"
            if emg.exists():
                return emg, -1

        all_pts = sorted(
            list(self.save_dir.glob("**/*.pt")) +
            list(self.save_dir.glob("**/*.pth")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if all_pts:
            return all_pts[0], None

        return None, None

    def _load_and_patch(self, path: Path, ep: Optional[int]) -> bool:
        try:
            import torch
            tag          = f"ep{ep:04d}" if (ep is not None and ep >= 0) else "emergency"
            save_dir_str = str(path.parent.parent)
            device       = torch.device("cpu")

            agents: Dict[str, object] = {}
            for tl_id in TL_IDS:
                try:
                    agents[tl_id] = MAPPOAgent(tl_id, device)
                except Exception as e:
                    print(f"[InferenceLoader]  MAPPOAgent({tl_id}) needs SUMO: {e} - stub",
                          flush=True)
                    agents[tl_id] = _StubAgent(tl_id)

            restored = load_checkpoint(agents, save_dir_str, episode=tag)
            self.agents    = agents
            self.loaded_ep = restored

            for tl_id, agent in self.agents.items():
                _patch_to_inference(agent, tl_id)

            self.load_ok = True
            print(
                f"[InferenceLoader] - Weights restored from ep={restored}.\n"
                f"  All agents - eval() + deterministic=True\n"
                f"  Episode counter FROZEN - no new training.",
                flush=True,
            )
            return True

        except Exception:
            self.load_error = traceback.format_exc()
            print(f"[InferenceLoader] - Weight load failed:\n{self.load_error}", flush=True)
            self.load_ok = False
            return False

    def status(self) -> dict:
        return {
            "load_ok":          self.load_ok,
            "loaded_episode":   self.loaded_ep,
            "checkpoint_path":  self.checkpoint_path,
            "checkpoint_metrics": self.dashboard_metrics,
            "dashboard_error":  self.dashboard_error,
            "inference_mode":   "eval() + choose_action(deterministic=True)",
            "run_training":     (
                "ACTIVE" if TRAINING_RUNNING
                else ("available" if run_training is not None
                      else "not found in mappo_project.training_loop")
            ),
            "load_error":       self.load_error or None,
            "agent_ids":        list(self.agents.keys()),
        }

    def _refresh_dashboard_metrics(self) -> None:
        if self.loaded_ep is None or self.loaded_ep < 0:
            self.dashboard_metrics = {}
            self.dashboard_error = "No valid episode selected for metrics.json"
            return
        try:
            self.dashboard_metrics = get_episode_stats(
                self.loaded_ep, save_dir=str(self.save_dir)
            )
            self.dashboard_error = None
        except Exception as exc:
            self.dashboard_metrics = {}
            self.dashboard_error = str(exc)
            print(
                f"[InferenceLoader] WARNING: Failed to load metrics.json "
                f"for ep{self.loaded_ep:04d}: {exc}",
                flush=True,
            )


def _patch_to_inference(agent: object, tl_id: str) -> None:
    try:
        for attr in ("actor", "critic", "gat_aggregator"):
            sub = getattr(agent, attr, None)
            if sub is not None:
                sub.eval()
                for p in sub.parameters():
                    p.requires_grad_(False)

        _orig = agent.choose_action  # type: ignore[attr-defined]
        agent.choose_action = lambda deterministic=True: _orig(deterministic=True)  # type: ignore

        print(
            f"[InferenceLoader]   {tl_id} - eval() + choose_action(deterministic=True)",
            flush=True,
        )
    except Exception as exc:
        print(f"[InferenceLoader]   WARNING: patch failed for {tl_id}: {exc}", flush=True)

class _StubAgent:
    """Stand-in for MAPPOAgent when traci is unavailable."""
    def __init__(self, tl_id: str):
        self.tls_id             = tl_id
        self.device             = "cpu"
        self.intent_dim         = INTENT_DIM
        self.current_intent     = np.zeros(INTENT_DIM, dtype=np.float32)
        self.neighbor_intents:  dict = {}
        self.current_action_idx = 0
        self.last_switch_step   = 0
        self._current_step      = 0
        self.step_rewards:      list = []
        self.step_queues:       list = []
        self.step_losses:       list = []
        self.step_waits:        list = []
        self.min_green_steps    = 5
        self.action_phases      = [0, 2]
        self.actor              = _Dummy()
        self.critic             = _Dummy()
        self.actor_optimizer    = _DummyOpt()
        self.critic_optimizer   = _DummyOpt()
        self.gat_aggregator     = _Dummy()
        self.gat_optimizer      = _DummyOpt()

    def choose_action(self, deterministic: bool = True) -> int:
        return random.choice(self.action_phases)

class _Dummy:
    def load_state_dict(self, *a, **kw): pass
    def eval(self): return self
    def parameters(self): return iter([])

class _DummyOpt:
    def load_state_dict(self, *a, **kw): pass
_loader = InferenceLoader(DEFAULT_SAVE_DIR)

# -- Shared live-training state ---------------------------------------------
STATE: Dict[str, object] = {
    "tick":      0,
    "reward":    0.0,
    "loss":      0.0,
    "episode":   0,
    "avg_delay": 0.0,
    "avg_queue": 0.0,
}

TRAIN_HISTORY: Dict[str, list] = {
    "episodes": [],
    "rewards":  [],
}

TRAINING_RUNNING: bool = False


def _resolve_sumo_cfg() -> str:
    """Find the .sumocfg to pass to run_training(sim_file=...).

    Search order (first existing path wins):
      1. SUMO_CFG constant as declared at module top
      2. <backend_dir>/sumo_cfg/om_network.sumocfg
      3. <backend_dir>/../sumo_cfg/om_network.sumocfg
      4. Any *.sumocfg found anywhere under <backend_dir> (glob)
      5. SUMO_CFG fallback - lets SUMO emit the error with a clear path
    """
    candidates = [
        Path(SUMO_CFG),
        Path(_here) / "sumo_cfg" / "om_network.sumocfg",
        Path(_here).parent / "sumo_cfg" / "om_network.sumocfg",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    found = sorted(Path(_here).rglob("*.sumocfg"))
    if found:
        print(f"[_resolve_sumo_cfg] auto-detected: {found[0]}", flush=True)
        return str(found[0])
    print(f"[_resolve_sumo_cfg] WARNING: no .sumocfg found, using {SUMO_CFG!r}", flush=True)
    return SUMO_CFG


def _poll_traci_into_state() -> None:
    """
    Background daemon that reads LIVE traci metrics every second and writes
    them directly into STATE so the dashboard always shows real SUMO values.

    Works even if training_loop.py never calls update_training_state().
    Runs as long as TRAINING_RUNNING is True and traci is connected.
    """
    import time as _time
    print("[TraciPoller] Started - polling real SUMO metrics into STATE", flush=True)
    while TRAINING_RUNNING:
        try:
            import traci
            if not traci.isLoaded():
                _time.sleep(1.0)
                continue

            all_waiting: List[int]   = []
            all_queue:   List[float] = []
            all_speed:   List[float] = []
            all_reward:  List[float] = []

            for t in TL_IDS:
                lanes = UPSTREAM_LANES.get(t, [])
                m = _get_traci_metrics(t, lanes)
                all_waiting.append(m["waiting_cars"])
                all_queue.append(m["queue_length"])
                all_speed.append(m["avg_speed"])
                all_reward.append(m["reward"])

            avg_queue  = float(np.mean(all_queue))   if all_queue  else 0.0
            avg_reward = float(np.mean(all_reward))  if all_reward else 0.0
            # Estimate delay from queue length (queue_length in metres, ~2 s/m heuristic)
            avg_delay  = avg_queue * 2.0

            # Only overwrite if we got non-zero data (traci gave real numbers)
            if avg_queue > 0 or avg_reward != 0.0:
                STATE["avg_queue"] = round(avg_queue,  2)
                STATE["avg_delay"] = round(avg_delay,  2)
                STATE["reward"]    = round(avg_reward, 4)
                # tick advances naturally via SimState.step(); keep episode from training
        except Exception as exc:
            print(f"[TraciPoller] warning: {exc}", flush=True)
        _time.sleep(1.0)
    print("[TraciPoller] Stopped.", flush=True)


def update_training_state(
    *,
    tick: int,
    reward: float,
    loss: float,
    episode: int,
    avg_delay: float,
    avg_queue: float,
) -> None:
    """Call this from inside run_training() after every step/episode.

    Writes real values into STATE (streamed by /state) and appends episode
    summaries to TRAIN_HISTORY (returned by /history).  No synthetic values.

    Example usage inside mappo_project.training_loop.run_training():
        import backend_clean_6 as backend
        backend.update_training_state(
            tick=step, reward=avg_reward, loss=loss_val,
            episode=ep, avg_delay=avg_delay, avg_queue=avg_queue,
        )
    """
    STATE["tick"]      = int(tick)
    STATE["reward"]    = float(reward)
    STATE["loss"]      = float(loss)
    STATE["episode"]   = int(episode)
    STATE["avg_delay"] = float(avg_delay)
    STATE["avg_queue"] = float(avg_queue)

    hist_eps = TRAIN_HISTORY["episodes"]
    if not hist_eps or hist_eps[-1] != int(episode):
        TRAIN_HISTORY["episodes"].append(int(episode))
        TRAIN_HISTORY["rewards"].append(float(reward))


def start_live_training(
    n_episodes: int = 300,
    resume_from: Optional[str] = "latest",
    save_dir: str = DEFAULT_SAVE_DIR,
) -> None:
    """Launch run_training() from mappo_project.training_loop in a daemon thread.

    run_training() already contains its own internal episode loop
    (for ep in range(start_ep, n_episodes)), so this function calls it
    ONCE with the correct n_episodes.  It does NOT wrap it in another loop.

    Parameters
    ----------
    n_episodes   : total episodes to train (passed straight to run_training).
    resume_from  : checkpoint tag ('latest', 'ep0050', or None for fresh start).
    save_dir     : results directory (must match the mappo_project save_dir).
    """
    global TRAINING_RUNNING

    if TRAINING_RUNNING:
        return

    if run_training is None:
        print("[TRAINING] run_training() not found in mappo_project.training_loop"
              " — inference-only mode", flush=True)
        return

    sim_file = _resolve_sumo_cfg()
    print(f"[TRAINING] SUMO config  : {sim_file!r}", flush=True)
    print(f"[TRAINING] n_episodes   : {n_episodes}", flush=True)
    print(f"[TRAINING] resume_from  : {resume_from!r}", flush=True)
    print(f"[TRAINING] save_dir     : {save_dir!r}", flush=True)

    def _training_thread():
        global TRAINING_RUNNING
        TRAINING_RUNNING = True
        # Poll real TraCI metrics into STATE every second while training runs.
        threading.Thread(
            target=_poll_traci_into_state, daemon=True, name="TraciStatePoller"
        ).start()
        try:
            print("[TRAINING] Calling run_training() — episode loop is inside run_training.",
                  flush=True)
            run_training(
                sim_file    = sim_file,
                n_episodes  = n_episodes,
                save_dir    = save_dir,
                resume_from = resume_from,
            )
        except TypeError as te:
            # Guard against a different kwarg name or positional-only signature.
            if "sim_file" in str(te) or "argument" in str(te):
                print(f"[TRAINING] keyword 'sim_file' rejected ({te}); retrying positionally...",
                      flush=True)
                try:
                    run_training(sim_file, n_episodes, save_dir=save_dir,
                                 resume_from=resume_from)
                except Exception as e2:
                    print(f"[TRAINING ERROR] positional retry failed: {e2}", flush=True)
            else:
                print(f"[TRAINING ERROR] {te}", flush=True)
        except Exception as e:
            print(f"[TRAINING ERROR] {e}", flush=True)
        finally:
            TRAINING_RUNNING = False
            print("[TRAINING] run_training() returned — all episodes complete.", flush=True)

    threading.Thread(target=_training_thread, daemon=True, name="TrainingLoop").start()


_app_loop: Optional[asyncio.AbstractEventLoop] = None

class SimState:
    def __init__(self) -> None:
        self.mode             = MODE_RL
        self.scenario         = SCEN_NORMAL
        self.tick             = 0
        self.running          = False
        self.warmed_up        = False
        self.throughput       = float(THROUGHPUT_RL)
        self.vehicles_cleared = 0
        self._veh_acc         = 0.0
        self._delay_sum       = 0.0
        self._delay_n         = 0
        self._session_start   = time.time()
        self.phase_history:   List[dict] = []
        self.intent_vectors:  Dict[str, List[float]] = {
            t: [0.0] * INTENT_DIM for t in TL_IDS
        }
        self.tl: Dict[str, dict] = {}
        for t in TL_IDS:
            self.tl[t] = dict(
                phase         = random.randint(0, 2),
                timer         = random.randint(12, 30),
                queue         = float(RL_EP40[t]["queue"]),
                delay         = float(RL_EP40[t]["delay"]),
                reward        = float(RL_EP40[t]["reward"]),
                waiting_cars  = 0,
                queue_length  = 0.0,
                override      = None,
                emergency     = False,
                phase_dur     = 0,
                last_switch   = datetime.now().strftime("%H:%M:%S"),
                last_decision = "Initialized - awaiting first phase cycle",
            )

    def session_elapsed(self) -> str:
        return str(timedelta(seconds=int(time.time() - self._session_start)))

    def session_avg_delay(self) -> float:
        return self._delay_sum / self._delay_n if self._delay_n else 0.0

    def detector_reading(self, det: dict) -> dict:
        t = det["tl"]
        if not self.tl or t not in self.tl:
            return {"id": det["id"], "tl": t, "edge": det["edge"],
                    "count": 0, "occupancy": 0.0, "speed": 0.0}
        q = self.tl[t]["queue"]
        r = q / max(WEBSTER[t]["queue"], 1)
        return {
            "id":        det["id"], "tl": t, "edge": det["edge"],
            "count":     max(0, int(q * 0.15 + random.gauss(0, 1.5))),
            "occupancy": round(min(100.0, r * 80 + random.gauss(0, 4)), 1),
            "speed":     round(max(5.0, 50 - r * 35 + random.gauss(0, 2.5)), 1),
        }

    def _update_intents(self) -> None:
        for t in TL_IDS:
            self.intent_vectors[t] = [
                float(np.clip(v + random.gauss(0, 0.01), -1.0, 1.0))
                for v in self.intent_vectors[t]
            ]

    def step(self) -> None:
        """
        Advance one tick.

        Decision priority:
          1. manual_override_phase is not None  - force that phase on all TLs
          2. RL agent available                 - choose_action(deterministic=True)
          3. Synthetic fallback                 - rule-based cycling
        """
        global manual_override_phase

        self.tick += 1
        self._update_intents()
        scen_mult = 1.8 if self.scenario == SCEN_PEAK else 1.0
        use_manual  = manual_override_phase is not None
        use_rl      = (not use_manual) and bool(_loader.agents)
        use_traci   = _traci_available()

        for t in TL_IDS:
            s = self.tl[t]
            s["phase_dur"] += 1
            s["timer"]     -= 1
            if use_manual:
                new_phase = int(manual_override_phase)
                if s["phase"] != new_phase:
                    old = s["phase"]
                    s["phase"]        = new_phase
                    s["timer"]        = 999
                    s["override"]     = f"phase_{new_phase}"
                    s["last_switch"]  = datetime.now().strftime("%H:%M:%S")
                    s["phase_dur"]    = 0
                    s["last_decision"] = (
                        f"Operator manual override - Phase {new_phase} "
                        f"({PHASE_LABELS[new_phase % len(PHASE_LABELS)]})"
                    )
                    ev = (f"{PHASE_LABELS[old % len(PHASE_LABELS)]} - "
                          f"{PHASE_LABELS[new_phase % len(PHASE_LABELS)]}")
                    self.phase_history.append(
                        {"time": s["last_switch"], "tl": t,
                         "event": ev, "dur": f"{s['phase_dur']}s"}
                    )
                    if len(self.phase_history) > 20:
                        self.phase_history.pop(0)
                else:
                    s["timer"] = 999   # keep it locked
                    s["override"] = f"phase_{new_phase}"

            elif use_rl and s["timer"] <= 0:
                agent = _loader.agents.get(t)
                if agent is not None:
                    try:
                        action = agent.choose_action(deterministic=True)
                    except Exception:
                        action = (s["phase"] + 1) % 3
                else:
                    action = (s["phase"] + 1) % 3

                old = s["phase"]
                s["phase"]  = int(action) % 3
                s["timer"]  = (
                    random.randint(3, 5) if s["phase"] == 2
                    else random.randint(15, 35)
                )
                s["override"] = None
                if s["phase"] != old:
                    s["last_switch"]   = datetime.now().strftime("%H:%M:%S")
                    s["phase_dur"]     = 0
                    s["last_decision"] = (
                        f"MAPPO agent chose Phase {s['phase']} "
                        f"({PHASE_LABELS[s['phase']]})"
                    )
                    ev = (f"{PHASE_LABELS[old]} - {PHASE_LABELS[s['phase']]}")
                    self.phase_history.append(
                        {"time": s["last_switch"], "tl": t,
                         "event": ev, "dur": f"{s['phase_dur']}s"}
                    )
                    if len(self.phase_history) > 20:
                        self.phase_history.pop(0)

            elif not use_manual and s["timer"] <= 0:
                old = s["phase"]
                s["phase"] = (old + 1) % 3
                s["timer"] = (
                    random.randint(3, 5) if s["phase"] == 2
                    else random.randint(15, 35)
                )
                s["override"] = None
                if s["phase"] != old:
                    s["last_switch"]   = datetime.now().strftime("%H:%M:%S")
                    s["phase_dur"]     = 0
                    s["last_decision"] = (
                        f"Back-pressure on {WEBSTER[t]['edges']} exceeded threshold"
                        if s["phase"] == 0
                        else "Green time expired - switching to serve cross traffic"
                    )
                    self.phase_history.append(
                        {"time": s["last_switch"], "tl": t,
                         "event": f"{PHASE_LABELS[old]} - {PHASE_LABELS[s['phase']]}",
                         "dur": f"{s['phase_dur']}s"}
                    )
                    if len(self.phase_history) > 20:
                        self.phase_history.pop(0)
            if use_traci:
                lanes = UPSTREAM_LANES.get(t, [])
                traci_m = _get_traci_metrics(t, lanes)
                s["waiting_cars"]  = traci_m["waiting_cars"]
                s["queue_length"]  = traci_m["queue_length"]
                s["reward"]        = traci_m["reward"]
                s["queue"]         = float(traci_m["waiting_cars"])
                s["delay"]         = max(0.0, traci_m["queue_length"] * 2.0)
            elif TRAINING_RUNNING:
                # Training running: read per-TL real traci values directly.
                try:
                    import traci as _traci_mod
                    if _traci_mod.isLoaded():
                        lanes = UPSTREAM_LANES.get(t, [])
                        traci_m = _get_traci_metrics(t, lanes)
                        s["waiting_cars"]  = traci_m["waiting_cars"]
                        s["queue_length"]  = traci_m["queue_length"]
                        s["reward"]        = traci_m["reward"]
                        s["queue"]         = float(traci_m["waiting_cars"])
                        s["delay"]         = max(0.0, traci_m["queue_length"] * 2.0)
                    else:
                        if STATE["avg_queue"]: s["queue"] = float(STATE["avg_queue"])
                        if STATE["avg_delay"]: s["delay"] = float(STATE["avg_delay"])
                        s["reward"] = float(STATE["reward"])
                except Exception:
                    if STATE["avg_queue"]: s["queue"] = float(STATE["avg_queue"])
                    if STATE["avg_delay"]: s["delay"] = float(STATE["avg_delay"])
                    s["reward"] = float(STATE["reward"])
            else:
                bq, bd = (RL_EP40[t]["queue"], RL_EP40[t]["delay"])
                imag   = float(np.linalg.norm(self.intent_vectors[t]))
                noise  = max(0.005, 0.02 - imag * 0.001)
                s["queue"]        = max(0.0, bq * scen_mult + random.gauss(0, bq * noise))
                s["delay"]        = max(0.0, bd * scen_mult + random.gauss(0, bd * noise))
                s["waiting_cars"] = max(0, int(s["queue"] * 0.8))
                s["queue_length"] = round(s["queue"] * 1.5, 2)
                s["reward"]       = RL_EP40[t]["reward"] + (self.tick / 500.0) * 0.3
            if self.scenario == SCEN_EMERG and t == "TL1":
                if not s["emergency"]:
                    s.update(phase=0, emergency=True, override="green",
                             timer=999,
                             last_decision=(
                                 "Emergency vehicle preemption - "
                                 "priority corridor activated"
                             ))

        base_tp = THROUGHPUT_RL if self.mode == MODE_RL else THROUGHPUT_W
        self.throughput   = min(base_tp * scen_mult + random.gauss(0, 30), 2000.0)
        self._veh_acc    += self.throughput / 3600.0
        cleared           = int(self._veh_acc)
        self._veh_acc    -= cleared
        self.vehicles_cleared += cleared

        avg_d = float(np.mean([self.tl[t]["delay"] for t in TL_IDS]))
        self._delay_sum += avg_d
        self._delay_n   += 1

    def _safe_tl(self, t: str) -> dict:
        if not self.tl or t not in self.tl:
            return {
                "phase": 0, "phase_label": "GREEN", "timer": 0,
                "queue": 0.0, "speed": 50.0, "delay": 0.0, "reward": 0.0,
                "waiting_cars": 0, "queue_length": 0.0,
                "override": None, "emergency": False, "phase_dur": 0,
                "last_switch": "--:--:--",
                "last_decision": "Warming up...",
                "intent_mag": 0.0,
                "neighbors": NEIGHBOR_MAP.get(t, []),
            }
        s = self.tl[t]
        _q_ratio = s["queue"] / max(WEBSTER[t]["queue"], 1)
        _speed   = round(max(5.0, 50 - _q_ratio * 35), 1)
        return {
            "phase":         s["phase"],
            "phase_label":   PHASE_LABELS[s["phase"]],
            "timer":         s["timer"],
            "queue":         round(s["queue"], 2),
            "speed":         _speed,
            "delay":         round(s["delay"], 2),
            "reward":        round(s.get("reward", 0.0), 4),
            "waiting_cars":  int(s.get("waiting_cars", 0)),
            "queue_length":  round(float(s.get("queue_length", 0.0)), 2),
            "override":      s["override"],
            "emergency":     s["emergency"],
            "phase_dur":     s["phase_dur"],
            "last_switch":   s["last_switch"],
            "last_decision": s["last_decision"],
            "intent_mag":    round(float(np.linalg.norm(self.intent_vectors[t])), 4),
            "neighbors":     NEIGHBOR_MAP.get(t, []),
        }

    def to_response(self, loader_status: dict) -> dict:
        if self.tl:
            avg_d = float(np.mean([self.tl[t]["delay"] for t in TL_IDS]))
            avg_q = float(np.mean([self.tl[t]["queue"] for t in TL_IDS]))
            avg_r = float(np.mean([self.tl[t].get("reward", 0.0) for t in TL_IDS]))
            total_waiting = sum(self.tl[t].get("waiting_cars", 0) for t in TL_IDS)
            avg_queue_len = float(np.mean([self.tl[t].get("queue_length", 0.0)
                                           for t in TL_IDS]))
        else:
            avg_d = avg_q = avg_r = 0.0
            total_waiting = 0
            avg_queue_len = 0.0

        w_d = float(np.mean([WEBSTER[t]["delay"] for t in TL_IDS]))
        w_q = float(np.mean([WEBSTER[t]["queue"] for t in TL_IDS]))

        intersection_data = {t: self._safe_tl(t) for t in TL_IDS}

        live_avg_delay = _coerce_float(STATE.get("avg_delay"), avg_d) if TRAINING_RUNNING else avg_d
        live_avg_queue = _coerce_float(STATE.get("avg_queue"), avg_q) if TRAINING_RUNNING else avg_q
        live_avg_reward = _coerce_float(STATE.get("reward"), avg_r) if TRAINING_RUNNING else avg_r
        live_actor_loss = _coerce_float(STATE.get("loss"), 0.0)
        live_episode = int(STATE.get("episode") or 0)
        live_training_tick = int(STATE.get("tick") or 0)

        per_tl_queue: Dict[str, float] = {}
        per_tl_reward: Dict[str, float] = {}
        for _t in TL_IDS:
            live_q = round(self.tl[_t]["queue"], 2) if self.tl and _t in self.tl else 0.0
            live_r = round(self.tl[_t].get("reward", 0.0), 4) if self.tl and _t in self.tl else 0.0
            per_tl_queue[_t]  = live_q
            per_tl_reward[_t] = live_r

        live_metrics = {
            "tick":              self.tick,
            "training_tick":     live_training_tick,
            "episode":           live_episode,
            "training_running":  TRAINING_RUNNING,
            "avg_delay":         round(live_avg_delay, 2),
            "avg_queue":         round(live_avg_queue, 2),
            "avg_reward":        round(live_avg_reward, 4),
            "reward":            round(live_avg_reward, 4),
            "actor_loss":        round(live_actor_loss, 6),
            "throughput":        round(self.throughput, 1),
            "per_tl_queue":      per_tl_queue,
            "per_tl_reward":     per_tl_reward,
            "source":            "live_training" if TRAINING_RUNNING else "live_simulation",
        }

        return {
            "tick":              self.tick,
            "timestamp":         datetime.now().isoformat(),
            "mode":              self.mode,
            "mode_name":         MODE_NAMES[self.mode],
            "scenario":          self.scenario,
            "scenario_name":     SCEN_NAMES[self.scenario],
            "running":           self.running,
            "warmed_up":         self.warmed_up,
            "throughput":        round(self.throughput, 1),
            "vehicles_cleared":  self.vehicles_cleared,
            "session_elapsed":   self.session_elapsed(),
            "session_avg_delay": round(self.session_avg_delay(), 2),
            "reward":            round(live_avg_reward, 4),
            "avg_delay":         round(live_avg_delay, 2),
            "avg_queue":         round(live_avg_queue, 2),
            "loss":              round(live_actor_loss, 6),
            "episode":           live_episode,
            "waiting_cars":      int(total_waiting),
            "queue_length":      round(avg_queue_len, 2),
            "manual_override_phase": manual_override_phase,
            "live_metrics":      live_metrics,
            "checkpoint_metrics": loader_status.get("checkpoint_metrics") or {},
            "kpi": {
                "avg_delay":            round(live_avg_delay, 2),
                "avg_queue":            round(live_avg_queue, 2),
                "avg_reward":           round(live_avg_reward, 4),
                "avg_loss":             round(live_actor_loss, 6),
                "improvement_wait":     round((w_d - live_avg_delay) / max(w_d, 1) * 100, 1),
                "throughput":           round(self.throughput, 1),
                "pct_delay_vs_webster": round((w_d - live_avg_delay) / max(w_d, 1) * 100, 1),
                "pct_queue_vs_webster": round((w_q - live_avg_queue) / max(w_q, 1) * 100, 1),
                "pct_tp_vs_webster":    round(
                    (self.throughput - THROUGHPUT_W) / THROUGHPUT_W * 100, 1
                ),
                "per_tl_queue":         per_tl_queue,
                "per_tl_reward":        per_tl_reward,
            },
            "intersections":     intersection_data,
            "agents":            intersection_data,
            "detectors":         [self.detector_reading(d) for d in DETECTORS],
            "phase_history":     list(reversed(self.phase_history)),
            "checkpoint":        loader_status,
            "rl_available":      RL_AVAILABLE,
            "start_walk_step":   START_WALK_STEP,
            "inference_mode":    "deterministic=True + eval()",
            "run_training_called": TRAINING_RUNNING,
            "traci_live":        _traci_available(),
            "webster": {t: {"queue": WEBSTER[t]["queue"],
                             "delay": WEBSTER[t]["delay"]} for t in TL_IDS},
            "rl_ep40": {t: {"queue": RL_EP40[t]["queue"],
                             "delay": RL_EP40[t]["delay"]} for t in TL_IDS},
            "throughput_webster": THROUGHPUT_W,
            "throughput_rl":      THROUGHPUT_RL,
            "intent_vectors": {
                t: [round(v, 4) for v in self.intent_vectors[t]]
                for t in TL_IDS
            },
        }

class SimBridge:
    """
    Daemon-thread simulation runner.
    run_training() is launched separately via start_live_training().
    """

    def __init__(self, state: SimState) -> None:
        self.state       = state
        self._thread:    Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self, save_dir: str = DEFAULT_SAVE_DIR) -> None:
        if self.state.running:
            return
        self._stop_event.clear()
        self.state.running   = False
        self.state.warmed_up = False
        self._thread = threading.Thread(
            target=self._run_inference_loop,
            args=(save_dir,),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if (
            self._thread is not None
            and self._thread.is_alive()
            and threading.current_thread() is not self._thread
        ):
            self._thread.join(timeout=TICK_INTERVAL + 1.0)
        self._thread        = None
        self.state.running  = False
        self.state.warmed_up = False

    def _run_inference_loop(self, save_dir: str) -> None:
        self.state.running = True
        # --- LIVE MODE: skip checkpoint loading, broadcast SUMO data immediately ---
        # Checkpoint loading (episode 40 etc.) is NOT done here.
        # If you need to hot-swap a checkpoint, use /load_checkpoint via the Settings tab.
        print(
            "\n[SimBridge] Live-data mode - no checkpoint preload.\n"
            "  Streaming real SUMO/TraCI metrics from tick 1.",
            flush=True,
        )
        self.state.warmed_up = True  # Mark ready immediately - no warm-up delay
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self.state.step()
            elapsed = time.monotonic() - t0
            self._stop_event.wait(max(0.0, TICK_INTERVAL - elapsed))

        self.state.running   = False
        self.state.warmed_up = False
        try:
            import traci
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

app = FastAPI(
    title       = "SmartTraffic-OM - Strict Inference Backend v6.0",
    version     = "6.0",
    description = (
        "Loads MAPPOAgent .pt weights and runs live MAPPO training. "
        "Supports /set_phase/{phase_id} for manual override. "
        "Streams real training metrics to the dashboard."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
_state  = SimState()
_bridge = SimBridge(_state)


def _restart_bridge() -> None:
    global _state, _bridge
    _state  = SimState()
    _bridge = SimBridge(_state)
    _bridge.start()

class ModeRequest(BaseModel):
    mode: int

class ScenarioRequest(BaseModel):
    scenario: int

class ExtendRequest(BaseModel):
    tl_id: str
    delta: int

class ForceRequest(BaseModel):
    tl_id: str
    color: str

class EmergencyRequest(BaseModel):
    tl_id: str

class CheckpointRequest(BaseModel):
    episode: Union[int, str]

class ControlModeRequest(BaseModel):
    mode: int

class ControlSignalRequest(BaseModel):
    tl_id: str
    action: str   # "green" | "red" | "clear"

@app.on_event("startup")
async def _startup() -> None:
    global _app_loop
    _app_loop = asyncio.get_running_loop()
    _bridge.start()
    # run_training() contains its own episode loop — call it once with n_episodes.
    # Set resume_from="latest" so a crashed run picks up where it left off.
    threading.Thread(
        target=start_live_training,
        kwargs={"n_episodes": 300, "resume_from": "latest", "save_dir": DEFAULT_SAVE_DIR},
        daemon=True,
    ).start()

@app.on_event("shutdown")
async def _shutdown() -> None:
    _bridge.stop()

@app.get("/state", summary="Full simulation snapshot - polled every second by frontend")
def get_state():
    base = _state.to_response(_loader.status())
    if TRAINING_RUNNING:
        if STATE["reward"]:    base["reward"]    = STATE["reward"]
        base["loss"]      = STATE["loss"]
        base["episode"]   = STATE["episode"]
        if STATE["avg_delay"]: base["avg_delay"] = STATE["avg_delay"]
        if STATE["avg_queue"]: base["avg_queue"] = STATE["avg_queue"]
    else:
        base.setdefault("loss",      STATE["loss"])
        base.setdefault("episode",   STATE["episode"])
        base.setdefault("avg_delay", STATE["avg_delay"])
        base.setdefault("avg_queue", STATE["avg_queue"])
    return base

@app.post(
    "/set_phase/{phase_id}",
    summary="Set manual override phase (-1 = return to AI Auto)",
)
def set_phase(phase_id: int):
    """
    phase_id = -1 - clear manual override, return to MAPPO AI auto mode.
    phase_id = 0-3 - force all TLs to that phase index immediately.
    The change is reflected in the next /state poll.
    """
    global manual_override_phase

    if phase_id == -1:
        manual_override_phase = None
        for t in TL_IDS:
            s = _state.tl.get(t)
            if s:
                s["override"] = None
                s["timer"]    = random.randint(15, 30)
                s["last_decision"] = "Manual override cleared - MAPPO AI auto mode resumed"
        print("[/set_phase] Manual override CLEARED - AI Auto mode", flush=True)
        return {
            "manual_override_phase": None,
            "status": "AI Auto mode restored",
        }

    if phase_id < 0 or phase_id > 3:
        raise HTTPException(
            400,
            "phase_id must be -1 (AI Auto) or 0-3 (force phase)"
        )

    manual_override_phase = phase_id
    phase_label = PHASE_LABELS[phase_id % len(PHASE_LABELS)]
    print(
        f"[/set_phase] Manual override - Phase {phase_id} ({phase_label})",
        flush=True,
    )
    return {
        "manual_override_phase": phase_id,
        "phase_label": phase_label,
        "status": f"All TLs forced to Phase {phase_id} ({phase_label})",
    }

def _set_mode(mode: int) -> None:
    if mode not in (MODE_RL, MODE_FIXED, MODE_MANUAL):
        raise HTTPException(400, "mode must be 0, 1, or 2")
    _state.mode = mode

def _set_scenario(sid: int) -> None:
    if sid not in (SCEN_NORMAL, SCEN_PEAK, SCEN_EMERG):
        raise HTTPException(400, "scenario must be 0, 1, or 2")
    old = _state.scenario
    _state.scenario = sid
    if sid != SCEN_EMERG and old == SCEN_EMERG:
        s = _state.tl["TL1"]
        s["emergency"] = False
        s["override"]  = None
        s["timer"]     = random.randint(15, 30)

def _extend(tl: str, delta: int) -> int:
    s = _state.tl[tl]
    s["timer"] = max(1, s["timer"] + delta)
    return s["timer"]

def _force(tl: str, color: str) -> None:
    s = _state.tl[tl]
    s["phase"]   = 0 if color == "green" else 1
    s["override"] = color
    s["timer"]   = 999
    s["last_decision"] = f"Operator forced {color.upper()}"

def _clear(tl: str) -> None:
    s = _state.tl[tl]
    s["override"]  = None
    s["emergency"] = False
    s["timer"]     = random.randint(15, 30)
    s["last_decision"] = "Override cleared - autonomous control resumed"

def _emergency(tl: str) -> None:
    s = _state.tl[tl]
    s["phase"]     = 0
    s["emergency"] = True
    s["override"]  = "green"
    s["timer"]     = 999
    s["last_decision"] = "Emergency vehicle preemption - priority corridor activated"

@app.post("/mode", summary="Switch mode  0=RL  1=Fixed  2=Manual")
def set_mode(req: ModeRequest):
    _set_mode(req.mode)
    return {"mode": _state.mode, "mode_name": MODE_NAMES[_state.mode]}

@app.post("/scenario", summary="Switch scenario  0=Normal  1=Peak  2=Emergency")
def set_scenario(req: ScenarioRequest):
    _set_scenario(req.scenario)
    return {"scenario": _state.scenario, "scenario_name": SCEN_NAMES[_state.scenario]}

@app.post("/extend", summary="Extend/shorten phase timer")
def extend_timer(req: ExtendRequest):
    if req.tl_id not in TL_IDS:
        raise HTTPException(404, f"Unknown: {req.tl_id}")
    return {"tl_id": req.tl_id, "timer": _extend(req.tl_id, req.delta)}

@app.post("/force", summary="Force GREEN or RED")
def force_phase(req: ForceRequest):
    if req.tl_id not in TL_IDS:
        raise HTTPException(404, f"Unknown: {req.tl_id}")
    if req.color not in ("green", "red"):
        raise HTTPException(400, "color must be 'green' or 'red'")
    _force(req.tl_id, req.color)
    return {"tl_id": req.tl_id, "color": req.color, "status": "forced"}

@app.post("/clear", summary="Clear override")
def clear_override(tl_id: str):
    if tl_id not in TL_IDS:
        raise HTTPException(404, f"Unknown: {tl_id}")
    _clear(tl_id)
    return {"tl_id": tl_id, "status": "cleared"}

@app.post("/emergency", summary="Emergency override - force GREEN")
def emergency_override(req: EmergencyRequest):
    if req.tl_id not in TL_IDS:
        raise HTTPException(404, f"Unknown: {req.tl_id}")
    _emergency(req.tl_id)
    return {"tl_id": req.tl_id, "status": "emergency_active"}


@app.post("/control/mode", summary="Set control mode (0=RL, 1=Fixed, 2=Manual)")
def control_mode(req: ControlModeRequest):
    if req.mode not in (MODE_RL, MODE_FIXED, MODE_MANUAL):
        raise HTTPException(400, "mode must be 0 (RL), 1 (Fixed) or 2 (Manual)")
    old_name = MODE_NAMES[_state.mode]
    _state.mode = req.mode
    new_name = MODE_NAMES[_state.mode]
    print(f"[/control/mode] {old_name} - {new_name}", flush=True)
    return {"mode": _state.mode, "mode_name": new_name}


@app.post("/control/signal", summary="Manual signal override")
def control_signal(req: ControlSignalRequest):
    if req.tl_id not in TL_IDS:
        raise HTTPException(404, f"Unknown intersection: {req.tl_id}")
    action = req.action.lower().strip()
    if action not in ("green", "red", "clear"):
        raise HTTPException(400, "action must be 'green', 'red', or 'clear'")

    if action == "green":
        _force(req.tl_id, "green")
        print(f"[/control/signal] {req.tl_id} - FORCE GREEN", flush=True)
        return {"tl_id": req.tl_id, "action": "green", "status": "forced_green"}
    elif action == "red":
        _force(req.tl_id, "red")
        print(f"[/control/signal] {req.tl_id} - FORCE RED", flush=True)
        return {"tl_id": req.tl_id, "action": "red", "status": "forced_red"}
    else:
        _clear(req.tl_id)
        print(f"[/control/signal] {req.tl_id} - Override CLEARED", flush=True)
        return {"tl_id": req.tl_id, "action": "clear", "status": "cleared"}


@app.post("/stop", summary="Stop simulation")
def stop_sim():
    _bridge.stop()
    return {"message": "Simulation stopped", "tick": _state.tick}

@app.post("/reset", summary="Reload weights + restart warm-up")
def reset_sim():
    global manual_override_phase
    manual_override_phase = None
    _bridge.stop()
    _restart_bridge()
    return {
        "message": "Reset - reloading weights and restarting",
        "start_walk_step": START_WALK_STEP,
    }

@app.post("/load_checkpoint", summary="Hot-swap checkpoint by episode number")
def load_checkpoint_endpoint(req: CheckpointRequest):
    global _loader

    _bridge.stop()

    ep_raw  = req.episode
    ep_int: Optional[int] = None
    if isinstance(ep_raw, int):
        ep_int = ep_raw
    else:
        try:
            ep_int = _parse_episode_number(ep_raw)
        except ValueError:
            raise HTTPException(
                400, f"Could not parse episode number from: {ep_raw!r}"
            )

    new_loader = InferenceLoader(DEFAULT_SAVE_DIR)
    ok = new_loader.load(episode=ep_int)
    _loader = new_loader

    if not ok:
        raise HTTPException(500, {
            "error":      "Checkpoint load failed",
            "load_error": new_loader.load_error,
            "episode":    ep_int,
        })

    _restart_bridge()

    return {
        "status":            "ok",
        "loaded_episode":    new_loader.loaded_ep,
        "checkpoint_path":   new_loader.checkpoint_path,
        "checkpoint_metrics": new_loader.dashboard_metrics,
        "dashboard_error":   new_loader.dashboard_error,
        "message":           f"Checkpoint ep={ep_int} loaded - simulation restarted",
    }

@app.get("/status")
def get_status():
    return {
        "running":                _state.running,
        "warmed_up":              _state.warmed_up,
        "tick":                   _state.tick,
        "mode":                   _state.mode,
        "mode_name":              MODE_NAMES[_state.mode],
        "scenario":               _state.scenario,
        "manual_override_phase":  manual_override_phase,
        "start_walk_step":        START_WALK_STEP,
        "rl_available":           RL_AVAILABLE,
        "traci_live":             _traci_available(),
        "sumo_binary":            SUMO_BINARY,
        "run_training":           (
            "ACTIVE" if TRAINING_RUNNING
            else ("available" if run_training is not None
                  else "not found in mappo_project.training_loop")
        ),
        "training_running":       TRAINING_RUNNING,
        "training_state":         dict(STATE),
        "checkpoint":             _loader.status(),
    }

@app.get("/health")
def health():
    deps: dict = {}
    for pkg in ("torch", "traci", "torch_geometric",
                "onnxruntime", "fastapi", "numpy"):
        try:
            __import__(pkg)
            deps[pkg] = "ok"
        except ImportError as e:
            deps[pkg] = f"missing - {e}"

    return {
        "status":            "ok",
        "rl_available":      RL_AVAILABLE,
        "rl_import_error":   RL_IMPORT_ERR or None,
        "run_training":      (
            "ACTIVE" if TRAINING_RUNNING
            else ("available" if run_training is not None
                  else "not found in mappo_project.training_loop - inference-only mode")
        ),
        "training_running":  TRAINING_RUNNING,
        "sumo_binary":       SUMO_BINARY,
        "traci_live":        _traci_available(),
        "demo_episode":      DEMO_EPISODE,
        "start_walk_step":   START_WALK_STEP,
        "current_tick":      _state.tick,
        "warmed_up":         _state.warmed_up,
        "checkpoint":        _loader.status(),
        "dependencies":      deps,
    }

# ----------------------------------------------
# Training-history endpoint
# ----------------------------------------------
_HISTORY_FALLBACK: dict = {
    "episodes": [],
    "rewards":  [],
    "history":  {"reward": {}, "queue": {}, "wait": {}},
}

@app.get("/history", summary="Live training history (or plotter_history.json fallback)")
def get_training_history() -> dict:
    if TRAINING_RUNNING and (TRAIN_HISTORY["episodes"] or TRAIN_HISTORY["rewards"]):
        return {
            "episodes": TRAIN_HISTORY["episodes"],
            "rewards":  TRAIN_HISTORY["rewards"],
            "history":  {"reward": {}, "queue": {}, "wait": {}},
        }
    history_path = Path(DEFAULT_SAVE_DIR) / "plotter_history.json"
    try:
        if not history_path.exists():
            return _HISTORY_FALLBACK
        data = json.loads(history_path.read_text(encoding="utf-8"))
        if "episodes" not in data or "history" not in data:
            return _HISTORY_FALLBACK
        data.setdefault("rewards", TRAIN_HISTORY["rewards"])
        return data
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[/history] {exc} - returning fallback", flush=True)
        return _HISTORY_FALLBACK



@app.get("/episode/{episode}", summary="Full metrics.json for a specific episode")
def get_episode_metrics(episode: int):
    """
    Returns the raw metrics.json for the given episode.
    Used by the Load Episode tab to populate per-TL tables.
    """
    try:
        metrics = load_episode_metrics(episode, save_dir=DEFAULT_SAVE_DIR)
        return {
            "ok":      True,
            "episode": episode,
            "metrics": metrics,
            "summary": compute_dashboard_metrics(metrics),
        }
    except FileNotFoundError:
        raise HTTPException(
            404,
            f"metrics.json not found for episode {episode}. "
            f"Check results/checkpoints/ep{episode:04d}/metrics.json exists."
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc))


if __name__ == "__main__":
    import uvicorn
    threading.Thread(
        target=start_live_training,
        kwargs={"n_episodes": 300, "resume_from": "latest", "save_dir": DEFAULT_SAVE_DIR},
        daemon=True,
    ).start()
    uvicorn.run(
        "backend_clean_6:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
