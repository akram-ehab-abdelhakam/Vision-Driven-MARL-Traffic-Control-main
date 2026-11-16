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
from .sumo_env import *
from .networks import *
from .agent import *
from .visualization import *
from .metrics_and_eval import *
from .rewards import BackPressureReward, WaitingTimeReward, QueueReductionReward, MaxPressureReward
from .intent_aggregators import GATIntentAggregator, FlatGATAggregator, MeanIntentAggregator, NoSharingIntentAggregator
from .ev_priority import DefaultEVPriority, NoEVPriority

from .training_loop_ippo import run_training_ippo
from .training_loop_vanilla_ppo import run_training_vanilla_ppo

__all__ = ['_build_global_intent_vec_worker', '_WorkerAgent', '_broadcast_worker_intents', '_run_episode_worker', '_train_agents_on_collected', 'run_training', '_run_training_sequential', '_run_training_parallel', 'main']


def _build_global_intent_vec_worker(wagents: dict, intent_dim: int) -> np.ndarray:
    n   = len(INTERSECTIONS)
    vec = np.zeros(n * intent_dim, dtype=np.float32)
    for tls_id, wa in wagents.items():
        idx = _TLS_INDEX.get(tls_id)
        if idx is not None:
            vec[idx * intent_dim: (idx + 1) * intent_dim] = wa.current_intent
    return vec

class _WorkerAgent:
    _lane_length_cache: dict[str, float] = {}

    def __init__(self, tls_id: str, state_dicts: dict, spec: dict):
        self.tls_id          = tls_id
        self.action_phases   = spec["action_phases"]
        self.phase_links     = spec["phase_links"]
        self.incoming_lanes  = spec["incoming_lanes"]
        self.outgoing_lanes  = spec["outgoing_lanes"]
        self.action_dim      = spec["action_dim"]
        self.state_dim       = spec["state_dim"]
        self.intent_dim      = spec["intent_dim"]

        self.current_action_idx = 0
        self.last_switch_step   = -MIN_GREEN_STEPS
        self._current_step      = 0

        self.actor  = ActorNetwork(self.state_dim, self.action_dim, self.intent_dim)
        self.critic = CriticNetwork(self.state_dim, self.intent_dim)
        self.actor.load_state_dict(state_dicts["actor"])
        self.critic.load_state_dict(state_dicts["critic"])
        self.actor.eval()
        self.critic.eval()

        self.gat_aggregator = GATIntentAggregator(self.intent_dim)
        gat_sd = state_dicts.get("gat")
        if gat_sd is not None:
            self.gat_aggregator.load_state_dict(gat_sd)
        self.gat_aggregator.eval()

        self.state           = None
        self.current_intent  = np.zeros(self.intent_dim, dtype=np.float32)
        self.neighbor_intents: Dict[str, np.ndarray] = {}
        self.transitions:    list = []
        self.step_rewards:   list[float] = []
        self.step_queues:    list[float] = []
        self.step_waits:     list[float] = []

    @classmethod
    def _get_lane_length(cls, lane_id: str) -> float:
        v = cls._lane_length_cache.get(lane_id)
        if v is None:
            v = max(traci.lane.getLength(lane_id), 1.0)
            cls._lane_length_cache[lane_id] = v
        return v

    def _get_state(self) -> np.ndarray:
        ph = np.zeros(self.action_dim, dtype=np.float32)
        ph[self.current_action_idx] = 1.0
        vals = list(ph)

        for lane_id in self.incoming_lanes:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])
        for lane_id in self.outgoing_lanes:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])

        dets = TLS_DETECTOR_MAP.get(self.tls_id, [])
        for det_id in dets:
            vals.append(_STEP_CACHE.detector_halting(det_id))

        for lane_id in self.incoming_lanes:
            vals.append(_lane_queue(lane_id))

        steps_since_switch = float(self._current_step - self.last_switch_step)
        vals.append(min(steps_since_switch / 100.0, 1.0))

        total_queue = _get_tls_queue_from_detectors(self.tls_id)
        qcap = _QUEUE_CAPACITY_PER_TLS.get(self.tls_id, QUEUE_CAPACITY)
        vals.append(min(total_queue / max(qcap, 1.0), 1.0))

        # -- Shock-wave features (Rankine-Hugoniot) -- must match MAPPOAgent._get_state --
        in_list   = list(self.incoming_lanes)
        out_list  = list(self.outgoing_lanes)
        shock_sum = 0.0
        shock_min = 0.0
        n_pairs   = max(len(in_list), 1)
        for k, in_l in enumerate(in_list):
            out_l = out_list[k % len(out_list)] if out_list else None
            if out_l:
                vs = _shock_wave_speed(in_l, out_l)
                shock_sum += vs
                shock_min  = min(shock_min, vs)
        vals.append(float(shock_sum / n_pairs) / max(V_FREE, 1.0))
        vals.append(float(shock_min)            / max(V_FREE, 1.0))

        return np.array(vals, dtype=np.float32)

    def _build_intent_matrix(self) -> torch.Tensor:
        n      = len(INTERSECTIONS)
        mat    = torch.zeros(n, self.intent_dim, dtype=torch.float32)
        my_idx = _TLS_INDEX[self.tls_id]
        mat[my_idx] = torch.from_numpy(self.current_intent)
        for nb_id, nb_intent in self.neighbor_intents.items():
            if nb_id in _TLS_INDEX:
                mat[_TLS_INDEX[nb_id]] = torch.from_numpy(nb_intent)
        return mat

    def aggregate_neighbor_intents(self) -> np.ndarray:
        if not self.neighbor_intents:
            return np.zeros(self.intent_dim, dtype=np.float32)
        mat = self._build_intent_matrix()
        with torch.no_grad():
            out = self.gat_aggregator(mat)
        my_idx = _TLS_INDEX[self.tls_id]
        return out[my_idx].numpy().astype(np.float32)

    def choose_action(self) -> int:
        agg_intent = self.aggregate_neighbor_intents()
        combined_intent = (self.current_intent + agg_intent) / 2.0

        with torch.no_grad():
            s = torch.from_numpy(self.state).unsqueeze(0)
            i = torch.from_numpy(combined_intent).unsqueeze(0)
            logits, new_intent = self.actor(s, i)
            self.current_intent = new_intent[0].numpy()
            probs  = torch.softmax(logits, dim=-1)
            dist   = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
        return action

    def apply_action(self, action_idx: int, current_step: int) -> None:
        self._current_step = current_step
        chosen = self.action_phases[action_idx]

        try:
            current       = traci.trafficlight.getPhase(self.tls_id)
            phase_state   = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            in_transition = any(c in ('y', 'Y', 'u') for c in phase_state)
        except Exception:
            current       = chosen
            in_transition = False

        if in_transition:
            return

        if chosen != current:
            if (current_step - self.last_switch_step) >= MIN_GREEN_STEPS:
                traci.trafficlight.setPhase(self.tls_id, chosen)
                self.last_switch_step = current_step

    def get_reward(self) -> float:
        """Back-Pressure value under the fixed schedule.

        Uses per-TLS capacity normalisation (same as MAPPOAgent / BackPressureReward)
        so that the 'FT BP-Score' column in the UI and Excel export is directly
        comparable to the RL reward on a consistent scale.
        """
        qcap = float(_QUEUE_CAPACITY_PER_TLS.get(self.tls_id, UNIFIED_BP_CAPACITY))
        out_lanes = list(self.outgoing_lanes)
        n_pairs   = max(len(self.incoming_lanes), 1)

        pressure = 0.0
        for k, in_lane in enumerate(self.incoming_lanes):
            q_in     = _lane_queue(in_lane) / qcap
            out_lane = out_lanes[k % len(out_lanes)] if out_lanes else None
            q_out    = (_lane_queue(out_lane) / qcap) if out_lane else 0.0
            pressure += (q_in - q_out)

        raw = -(pressure / n_pairs)
        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))

    def _estimate_flow(self, lane_id: str) -> float:
        ll = self._get_lane_length(lane_id)
        n  = _STEP_CACHE.lane_vehicles(lane_id)
        k  = n / ll
        return max(V_FREE * k - _V_FREE_OVER_KJAM * k * k, 0.0)

def _broadcast_worker_intents(wagents: dict):
    for tls_id, wa in wagents.items():
        neighbors = NEIGHBOR_MAP.get(tls_id, [])
        for neighbor_id in neighbors:
            neighbor_wa = wagents.get(neighbor_id)
            if neighbor_wa is not None:
                neighbor_wa.neighbor_intents[tls_id] = wa.current_intent.copy()

def _run_episode_worker(args: tuple) -> dict:
    port, sumo_cmd, sim_max_steps, agent_specs, state_dicts, _epsilons = args

    collected: dict = {
        tls_id: {
            "transitions":  [],
            "step_rewards": [],
            "step_queues":  [],
            "step_waits":   [],
        }
        for tls_id in agent_specs
    }

    proc = None
    try:
        proc = _start_sumo(sumo_cmd, port)

        wagents: dict[str, _WorkerAgent] = {}
        for tls_id, spec in agent_specs.items():
            wagents[tls_id] = _WorkerAgent(tls_id, state_dicts[tls_id], spec)

        all_lanes: set[str] = set()
        for wa in wagents.values():
            all_lanes.update(wa.incoming_lanes)
            all_lanes.update(wa.outgoing_lanes)
        for tid in INTERSECTIONS:
            all_lanes.update(UPSTREAM_LANES.get(tid, []))
        _subscribe_lanes(list(all_lanes))

        all_dets: set[str] = {d for dets in TLS_DETECTOR_MAP.values() for d in dets}
        _subscribe_detectors(list(all_dets))

        for wa in wagents.values():
            wa.state = wa._get_state()

        for step in range(1, sim_max_steps + 1):
            if step > 200 and traci.simulation.getMinExpectedNumber() == 0:
                break

            # Subscribe newly-departed vehicles every step (cheap with getDepartedIDList)
            _subscribe_vehicles_bulk()

            _STEP_CACHE.clear()

            _broadcast_worker_intents(wagents)

            actions: dict[str, int] = {}
            for tls_id, wa in wagents.items():
                a = wa.choose_action()
                wa.current_action_idx = a
                wa.apply_action(a, step)
                actions[tls_id] = a

            traci.simulationStep()

            _STEP_CACHE.bulk_load_lanes()
            _STEP_CACHE.bulk_load_detectors()
            _STEP_CACHE.bulk_load_vehicles()

            for tls_id, wa in wagents.items():
                ns     = wa._get_state()
                reward = wa.get_reward()
                done   = 1.0 if step >= sim_max_steps else 0.0

                agg_intent = wa.aggregate_neighbor_intents()
                combined_intent = (wa.current_intent + agg_intent) / 2.0
                global_intent_vec = _build_global_intent_vec_worker(
                    wagents, wa.intent_dim)

                collected[tls_id]["transitions"].append(
                    (wa.state.copy(), combined_intent.copy(),
                     actions[tls_id], reward, ns.copy(), done,
                     global_intent_vec.copy()))
                wa.state = ns

                collected[tls_id]["step_rewards"].append(reward)
                collected[tls_id]["step_queues"].append(
                    _get_tls_queue_from_detectors(tls_id))
                collected[tls_id]["step_waits"].append(_get_tls_mean_wait(tls_id))

    except Exception as exc:
        import traceback as tb
        collected["_error"] = f"{exc}\n{tb.format_exc()}"

    finally:
        if "_error" not in collected:
            for tls_id in INTERSECTIONS:
                n = len(collected.get(tls_id, {}).get("transitions", []))
                if n == 0:
                    print(f"  [WORKER-WARN] {tls_id}: "
                          f"0 transitions in this episode", flush=True)
        try:
            traci.close()
        except Exception:
            pass
        if proc is not None:
            _kill_sumo(proc)
        _ALL_SUBSCRIBED_LANES.clear()
        _ALL_SUBSCRIBED_DETECTORS.clear()
        _ALL_SUBSCRIBED_LANES.clear()
        _ALL_SUBSCRIBED_DETECTORS.clear()

    return collected

def _train_agents_on_collected(agents: dict, all_results: list[dict],
                                device: torch.device) -> dict[str, float]:
    for tls_id, ag in agents.items():
        total_transitions = sum(
            len(ep.get(tls_id, {}).get("transitions", []))
            for ep in all_results
            if "_error" not in ep
        )
        print(f"  [BUFFER] {tls_id}: {total_transitions} transitions collected", flush=True)
        if total_transitions == 0:
            print(f"  [WARN] {tls_id}: ZERO transitions -- "
                  f"agent will not train this batch!", flush=True)

    for ep_result in all_results:
        if "_error" in ep_result:
            continue
        for tls_id, ag in agents.items():
            data = ep_result.get(tls_id)
            if not data:
                continue
            transitions = data["transitions"]
            if not transitions:
                continue

            all_s      = np.stack([t[0] for t in transitions], axis=0)
            all_intent = np.stack([t[1] for t in transitions], axis=0)
            all_a      = np.array([t[2] for t in transitions], dtype=np.int64)
            all_ns     = np.stack([t[4] for t in transitions], axis=0)
            all_d      = np.array([t[5] for t in transitions], dtype=np.float32)
            all_gi     = np.stack([t[6] for t in transitions], axis=0)

            if all_s.shape[1] != ag.state_dim:
                print(f"  [DIM-ERROR] {tls_id}: worker state_dim={all_s.shape[1]} "
                      f"!= agent state_dim={ag.state_dim} -- skipping batch "
                      f"(fix _WorkerAgent._get_state to match MAPPOAgent._get_state)",
                      flush=True)
                continue

            all_values    = ag.get_values_batch(all_s, all_gi)
            all_log_probs = ag.get_log_probs_batch(all_s, all_intent, all_a)

            all_next_values = np.zeros(len(transitions), dtype=np.float32)
            non_terminal = np.where(all_d == 0.0)[0]
            if len(non_terminal) > 0:
                next_gi = np.stack([
                    transitions[i + 1][6] if i + 1 < len(transitions) else all_gi[i]
                    for i in non_terminal
                ], axis=0)
                nv = ag.get_values_batch(all_ns[non_terminal], next_gi)
                all_next_values[non_terminal] = nv

            for t_idx, (s, intent, a, r, ns, d, gi) in enumerate(transitions):
                ag.rollout_buffer.push(
                    s, intent, a, r,
                    float(all_values[t_idx]),
                    float(all_log_probs[t_idx]),
                    d, intent,
                    float(all_next_values[t_idx]),
                    global_intent=gi)

    mean_losses: dict[str, float] = {}
    for tls_id, ag in agents.items():
        buf_size = ag.rollout_buffer.size
        if buf_size < 1:
            mean_losses[tls_id] = 0.0
            print(f"  [LOSS-WARN] {tls_id}: rollout buffer empty -- skipping train_epoch", flush=True)
            continue
        if buf_size < ag.batch_size:
            # Buffer has data but fewer samples than one batch.
            # Temporarily lower batch_size so train_epoch can still run.
            _orig_bs = ag.batch_size
            ag.batch_size = max(1, buf_size // 2)
            print(f"  [LOSS-WARN] {tls_id}: buffer={buf_size} < batch_size={_orig_bs} "
                  f"-- temporarily using batch_size={ag.batch_size}", flush=True)
        else:
            _orig_bs = None

        advantages, returns = ag.rollout_buffer.compute_gae()
        loss, mean_entropy, mean_kl = ag.train_epoch(advantages, returns)
        ag.rollout_buffer.clear()
        ag._update_gat_from_peers(agents)
        mean_losses[tls_id] = loss

        # KL diagnostic: was previously computed inside train_epoch and
        # discarded every episode. Surfacing it here answers directly
        # whether the existing KLAdaptiveLRController is firing often
        # (constraint genuinely binding) or rarely (instability has a
        # different cause) -- see counts dict for the breakdown.
        # NOTE: counts are cumulative since training start, not per-episode.
        counts = getattr(ag, "last_kl_action_counts", None)
        true_kl = getattr(ag, "last_true_kl", 0.0)
        if counts is not None:
            print(f"  [KL-DIAG] {tls_id}: mean_kl={mean_kl:.5f}  "
                  f"true_kl={true_kl:.5f}  "
                  f"target={KL_TARGET:.4f}  "
                  f"consec_inc={counts.get('consecutive_increases', 0)}  "
                  f"lr_pinned={counts.get('is_lr_pinned', False)}  "
                  f"cumulative(degen={counts.get('degenerate', 0)}  "
                  f"inc={counts['increased']}  "
                  f"dec={counts['decreased']}  "
                  f"unch={counts['unchanged']})", flush=True)

        # Diagnostic entropy + true_kl per episode so we can verify
        # the policy starts with high entropy (~ln(N_actions)) and true_kl
        # moves in a reasonable range (0.001-0.02) instead of freezing at ~0.
        _n_actions = getattr(ag, "action_dim", 8)
        _expected_uniform_entropy = np.log(_n_actions) if _n_actions > 1 else 0.0
        print(f"  [ENT-DIAG] {tls_id}: entropy={mean_entropy:.6f}  "
              f"uniform_expected={_expected_uniform_entropy:.4f}  "
              f"ratio={mean_entropy / max(_expected_uniform_entropy, 1e-8):.3f}  "
              f"true_kl={true_kl:.6f}", flush=True)

        # NOTE: CosineAnnealingLR .step() calls intentionally removed.
        # The KL-adaptive controller (_kl_controller) is the sole authority
        # on actor/critic LRs -- it adjusts param_groups['lr'] in-place
        # during train_epoch().  Stepping the cosine schedulers here would
        # overwrite those KL-driven values every episode.
        ag._check_entropy_collapse(mean_entropy)

        if _orig_bs is not None:
            ag.batch_size = _orig_bs

    return mean_losses

def run_training(
        sim_file,
        n_episodes    = 1,
        sim_max_steps = 3600,
        gui           = "disable",
        save_dir      = "./results",
        resume_from   = None,
        n_workers     = 1,
        save_every: int = 10,
        eval_interval: int = 10,
        # ═══════════════════════════════════════════════════
        # EXPERIMENT CONFIGURATION — Change these for ablation studies
        # ═══════════════════════════════════════════════════
        reward_type:      str  = "backpressure",
        aggregator_type:  str  = "gat",
        ev_priority_type: str  = "default",
        adaptive_lr:      bool = True,
        adaptive_clip:    bool = True,
        seed_enable:      bool = False,
        base_seed:        int  = 42,
        fixed_seed:       bool = False,
        reset_lr:         bool = False,
        cuda_dynamic:     bool = True,
):
    if not TRACI_AVAILABLE:
        raise RuntimeError("traci is not available.")

    # ═══════════════════════════════════════════════════
    # EXPERIMENT CONFIGURATION — Change these for ablation studies
    # ═══════════════════════════════════════════════════
    REWARD_MAP = {
        "backpressure": BackPressureReward(),
        "waiting_time": WaitingTimeReward(),
        "queue":        QueueReductionReward(),
        "max_pressure":  MaxPressureReward(),
    }
    AGGREGATOR_MAP = {
        "gat":      GATIntentAggregator(INTENT_SHARING_DIM),
        "flat_gat": FlatGATAggregator(INTENT_SHARING_DIM),
        "mean":     MeanIntentAggregator(INTENT_SHARING_DIM),
        "none":     NoSharingIntentAggregator(INTENT_SHARING_DIM),
    }
    EV_MAP = {
        "default": DefaultEVPriority(),
        "none":    NoEVPriority(),
    }

    _reward_func      = REWARD_MAP.get(reward_type,      BackPressureReward())
    _intent_aggr      = AGGREGATOR_MAP.get(aggregator_type, GATIntentAggregator(INTENT_SHARING_DIM))
    _ev_module        = EV_MAP.get(ev_priority_type,     DefaultEVPriority())

    # When EV priority is disabled, propagate EV_PRIORITY_ENABLED=False into
    # sumo_env and rewards, which cache the flag at import time via
    # `from .config import EV_PRIORITY_ENABLED`.  Patching sys.modules entries
    # ensures their module-level guards (_get_ev_features, _ev_potential,
    # BackPressureReward shaping) are respected without restarting the process.
    if ev_priority_type == "none":
        import sys as _sys
        _ev_flag_modules = [
            __name__.rsplit(".", 1)[0] + ".sumo_env",
            __name__.rsplit(".", 1)[0] + ".rewards",
        ]
        for _mod_name in _ev_flag_modules:
            if _mod_name in _sys.modules:
                _sys.modules[_mod_name].EV_PRIORITY_ENABLED = False
        print("[INFO] Emergency-vehicle reward shaping: DISABLED", flush=True)
    else:
        print("[INFO] Emergency-vehicle reward shaping: ENABLED", flush=True)

    # Non-deterministic by default for better exploration and GPU throughput.
    # No fixed seed -- runs are fully stochastic.

    import platform as _plat
    _is_windows = _plat.system() == "Windows"

    # Parallel SUMO workers on Windows with Python's spawn start-method
    # frequently deadlock due to traci socket inheritance. Force sequential.
    if _is_windows and n_workers > 1:
        print(f"[WARNING] Parallel mode (n_workers={n_workers}) is not supported "
              f"on Windows -- switching to sequential (n_workers=1).", flush=True)
        n_workers = 1

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")

    # On Windows, SUMO binaries may need the .exe suffix and must be on PATH.
    if _is_windows:
        sumo_binary = "sumo-gui.exe" if gui == "enable" else "sumo.exe"
    else:
        sumo_binary = "sumo-gui" if gui == "enable" else "sumo"
    sumo_cmd    = [
        sumo_binary,
        "-c", sim_file,
        "--step-length",      str(int(STEP_LENGTH)),
        "--no-step-log",
        "--time-to-teleport", "300",
        "--collision.action", "warn",
    ]

    n_workers = max(1, n_workers)

    # -- GPU / CPU device selection --------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if seed_enable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            torch.backends.cudnn.benchmark        = False
            torch.backends.cudnn.deterministic    = True
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif not cuda_dynamic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            torch.backends.cudnn.benchmark        = False
            torch.backends.cudnn.deterministic    = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True
            torch.backends.cudnn.deterministic    = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        n_cpu = os.cpu_count() or 4
        try:
            torch.set_num_threads(max(1, n_cpu - 1))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    parallel_mode = n_workers > 1
    mode_label    = f"PARALLEL x{n_workers}" if parallel_mode else "SEQUENTIAL"

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM            : {vram_gb:.1f} GB")
        try:
            _x = torch.randn(64, 64, device="cuda")
            _r = (_x @ _x).sum().item()
            print(f"  CUDA smoke test : OK  (result={_r:.2f})")
            del _x, _r
        except Exception as _ce:
            print(f"  CUDA smoke test : FAILED -- {_ce}")
    print(f"\n{'='*70}")
    print(f"  OM Network  |  MAPPO-PIS + GAT (Multi-Agent PPO with Graph Attention Intent Sharing)")
    print(f"  Device           : {device}")
    print(f"  Mode             : {mode_label}")
    print(f"  Episodes         : {n_episodes}")
    print(f"  Steps / episode  : {sim_max_steps}")
    print(f"  Reward           : {_reward_func.name}")
    print(f"  Flow model       : Greenshields (Eq. 4.15)")
    print(f"  Intent module    : {_intent_aggr.name} ({'PyG GATConv' if PYG_AVAILABLE else 'Fallback Attn'})" + "  heads=" + str(GAT_HEADS))
    print(f"  EV Priority      : {_ev_module.name}")
    print(f"  gamma={GAMMA}  lr={LEARNING_RATE}  gae_lambda={GAE_LAMBDA}"
          f"  clip={CLIP_RATIO}")
    print(f"{'='*70}\n")

    # ── Seed before agent network initialisation ─────────────────────────
    if seed_enable:
        _seed_msg = f"  Seeding torch/numpy/random with base_seed={base_seed}"
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed)
            torch.cuda.manual_seed_all(base_seed)
        print(_seed_msg, flush=True)

    print("  Initialising agents (one-time SUMO bootstrap) ...")
    # Bootstrap run only reads network topology (TLS IDs, lane lists).
    _bootstrap_cmd = sumo_cmd
    _proc = _start_sumo(_bootstrap_cmd, port=BASE_PORT)
    agents = {}
    try:
        for tls_id in traci.trafficlight.getIDList():
            try:
                agents[tls_id] = MAPPOAgent(
                    tls_id,
                    device,
                    reward_func=_reward_func,
                    intent_aggregator=_intent_aggr,
                    ev_module=_ev_module,
                    use_adaptive_lr=adaptive_lr,
                    use_adaptive_clip=adaptive_clip,
                )
                print(f"    Agent [{tls_id}]"
                      f"  state_dim={agents[tls_id].state_dim}"
                      f"  action_dim={agents[tls_id].action_dim}"
                      f"  intent_dim={agents[tls_id].intent_dim}"
                      f"  phases={agents[tls_id].action_phases}")
            except RuntimeError as e:
                print(f"    [WARN] Skipping {tls_id}: {e}")
    except Exception as exc:
        _close_sumo(_proc)
        raise RuntimeError(f"Agent init failed: {exc}")
    _close_sumo(_proc)

    if not agents:
        raise RuntimeError("No usable TLS agents could be created.")

    if sim_max_steps != MEMORY_CAPACITY:
        print(f"  Resizing rollout buffers: {MEMORY_CAPACITY} -> {sim_max_steps} steps")
        for ag in agents.values():
            ag.rollout_buffer = RolloutBuffer(sim_max_steps, ag.state_dim, ag.intent_dim)

    # -- Resolve resume tag ONCE here so downstream history loading uses the same tag --
    # Resolving "latest" in load_checkpoint and again in the plotter-restore block
    # risks a race if latest.json is updated between calls (e.g. auto-saver fires).
# ═══════════════════════════════════════════════════════════════════════════════
# === 4. Resume block inside run_training() ===
#     Replace from "# -- Resolve resume tag" through "ag.setup_lr_schedulers()"
# ═══════════════════════════════════════════════════════════════════════════════

    # -- Resolve resume tag ONCE -- shared by load_checkpoint and history restore --
    #
    # We resolve "latest" here so that both load_checkpoint() and the plotter-
    # history block below use exactly the same tag.  Resolving separately in each
    # call would create a race window: if EpisodeAutoSaver fires between the two
    # calls, the two blocks could end up loading weights from ep=N and history
    # from ep=N+1 (or vice-versa), causing subtle length mismatches.
    _resume_tag: "str | None" = None

    if resume_from is not None:
        _ckpt_dir_early = os.path.join(save_dir, "checkpoints")

        if resume_from == "latest":
            _ptr_json_early = os.path.join(_ckpt_dir_early, "latest.json")
            _ptr_link_early = os.path.join(_ckpt_dir_early, "latest")

            if os.path.isfile(_ptr_json_early):
                try:
                    with open(_ptr_json_early) as _pf:
                        _ptr_data = json.load(_pf)
                    _resume_tag = f"ep{int(_ptr_data['latest_ep']):04d}"
                    print(f"  [RESUME] 'latest' resolved via latest.json -> {_resume_tag}",
                          flush=True)
                except Exception as _re:
                    print(f"  [RESUME] WARNING: Could not parse latest.json: {_re}",
                          flush=True)

            if _resume_tag is None and os.path.islink(_ptr_link_early):
                _resume_tag = os.path.basename(os.readlink(_ptr_link_early))
                print(f"  [RESUME] 'latest' resolved via symlink -> {_resume_tag}",
                      flush=True)

            if _resume_tag is None:
                print(
                    f"  [RESUME] WARNING: --resume_from=latest but no pointer found "
                    f"in {_ckpt_dir_early}. Starting from ep 0.",
                    flush=True,
                )
        else:
            # Caller supplied an explicit tag (e.g. "ep0050") -- use as-is.
            _resume_tag = resume_from
            print(f"  [RESUME] Explicit tag supplied: {_resume_tag}", flush=True)

    # -- Load checkpoint --------------------------------------------------------
    start_ep = 0
    if _resume_tag is not None:
        # Pass the already-resolved tag so load_checkpoint() skips the
        # "latest" resolution entirely (avoids a second latest.json read).
        _restored_ep = load_checkpoint(agents, save_dir, episode=_resume_tag)
        start_ep     = _restored_ep + 1

        print(
            f"  [RESUME] Checkpoint '{_resume_tag}' loaded "
            f"-> continuing from ep {start_ep} "
            f"({n_episodes - start_ep} episodes remaining)",
            flush=True,
        )

        # Clear rollout buffers explicitly.
        # An emergency-save checkpoint may have been written mid-episode with a
        # partially filled buffer (size > 0).  If left populated the first
        # train_step would train on stale transitions from the interrupted episode.
        for _ag in agents.values():
            _ag.rollout_buffer.clear()
        print(f"  [RESUME] Rollout buffers cleared (size=0).", flush=True)

    # -- LR schedulers ---------------------------------------------------------
    # Always pass the TOTAL episode count as n_episodes so the cosine shape is
    # identical whether training is fresh or resumed.
    #
    # On resume:  load_checkpoint() stashed saved state_dicts on each agent;
    #             setup_lr_schedulers() reads the original T_max from those
    #             state_dicts and calls load_state_dict() to restore last_epoch,
    #             giving a seamless cosine with no LR jump at the resume point.
    #
    # On fresh:   pending state_dicts are None, so a new scheduler is created
    #             with T_max=n_episodes and last_epoch=0.
    # -- LR schedulers ---------------------------------------------------------
    # Always pass n_episodes (the TOTAL count) as T_max so the cosine shape is
    # the same whether training is fresh or resumed.
    # On resume, load_checkpoint() stashed the saved state_dict on each agent;
    # setup_lr_schedulers() reads the original T_max from that state_dict and
    # calls load_state_dict() to restore last_epoch -- giving a seamless cosine.
    # On a fresh start the pending state_dicts are None, so a fresh scheduler
    # starting at last_epoch=0 is created with T_max=n_episodes.
    #
    # pending state_dict after restoring it (so it isn't re-applied). A second
    # call on the same agents would find pending=None and silently overwrite
    # the just-restored scheduler with a brand-new one at last_epoch=0 --
    # causing an LR spike back to its peak value at the resume boundary.
    print(f"  [LR] Setting up schedulers: n_episodes={n_episodes}  start_ep={start_ep}",
          flush=True)
    for ag in agents.values():
        ag.setup_lr_schedulers(n_episodes)

    if reset_lr and resume_from is not None:
        _init_actor_lr = LEARNING_RATE
        _init_critic_lr = LEARNING_RATE
        _init_gat_lr = 1e-4
        for ag in agents.values():
            for pg in ag.actor_optimizer.param_groups:
                pg['lr'] = _init_actor_lr
            for pg in ag.critic_optimizer.param_groups:
                pg['lr'] = _init_critic_lr
            if hasattr(ag, 'gat_optimizer') and ag.gat_optimizer is not None:
                for pg in ag.gat_optimizer.param_groups:
                    pg['lr'] = _init_gat_lr
        print(f"  [LR-RESET] All optimizers reset to initial LR "
              f"(actor/critic={_init_actor_lr}, gat={_init_gat_lr})", flush=True)

    print(f"  {len(agents)} agent(s) ready. Starting training ({mode_label}) ...\n")

    # -- Rich UI ---------------------------------------------------------------
    rich_ui = RichUI(
        n_episodes    = n_episodes,
        sim_max_steps = sim_max_steps,
        device        = str(device),
        mode          = mode_label,
        intersections = INTERSECTIONS,
        reward_name   = _reward_func.name,
    )


    fixed_controllers = {}
    for tl_id in INTERSECTIONS:
        if tl_id in FIXED_TIME_PLANS:
            try:
                fixed_controllers[tl_id] = FixedTimeController(tl_id)
            except ValueError as e:
                print(f"  [FixedTime] {e}")


    # Build a seed-aware command for the FT baseline so that, when
    # fixed_seed=True, the Fixed-Time comparison is also reproducible.
    if fixed_seed and seed_enable:
        _ft_sumo_cmd = sumo_cmd + ["--seed", str(base_seed)]
    else:
        _ft_sumo_cmd = sumo_cmd

    _cached_fx_result: dict = {}
    _fx_cache_path = os.path.join(save_dir, "checkpoints", "fixed_time_baseline.json")
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    if fixed_controllers:
        _fx_steps = max(900, sim_max_steps)

        # Try to load from disk first (always on resume, opportunistically on fresh start)
        _loaded_from_disk = False
        if os.path.isfile(_fx_cache_path):
            try:
                with open(_fx_cache_path) as _fcf:
                    _cached_fx_result = json.load(_fcf)
                _tl_items_disk = {k: v for k, v in _cached_fx_result.items()
                                  if not k.startswith("__")}
                _fx_q = {tl: float(np.mean(d["queues"])) if d["queues"] else 0.0
                         for tl, d in _tl_items_disk.items()}
                _fx_w = {tl: float(np.mean(d["waits"]))  if d["waits"]  else 0.0
                         for tl, d in _tl_items_disk.items()}
                print(f"  [FixedTime] Loaded baseline from disk "
                      f"(avg_queue={np.mean(list(_fx_q.values())):.2f}  "
                      f"avg_wait={np.mean(list(_fx_w.values())):.2f}) -- "
                      f"skipping SUMO re-run.", flush=True)
                _loaded_from_disk = True
            except Exception as _load_e:
                print(f"  [FixedTime] WARNING: Could not load cached baseline "
                      f"({_load_e}) -- will re-run SUMO.", flush=True)
                _cached_fx_result = {}

        if not _loaded_from_disk:
            _ft_seed_label = (f"seed={base_seed} (fixed)"
                              if fixed_seed else "random seed")
            print(f"  [FixedTime] Running baseline ({_ft_seed_label}) …",
                  flush=True)
            _fx_port_run = BASE_PORT + 700


            _seed_results: list[dict] = []
            try:
                _sr = _run_fixed_time_episode(
                    _ft_sumo_cmd, _fx_steps, _fx_port_run, fixed_controllers)
                _seed_results.append(_sr)
                _sr_tls = {k: v for k, v in _sr.items() if not k.startswith("__")}
                _sr_q   = {tl: float(np.mean(d["queues"])) if d["queues"] else 0.0
                           for tl, d in _sr_tls.items()}
                _sr_w   = {tl: float(np.mean(d["waits"]))  if d["waits"]  else 0.0
                           for tl, d in _sr_tls.items()}
                print(f"    avg_queue={np.mean(list(_sr_q.values())):.2f}  "
                      f"avg_wait={np.mean(list(_sr_w.values())):.2f}  "
                      f"cars={_sr.get('__cars_passed', 0)}", flush=True)
            except Exception as _seed_e:
                print(f"  [FixedTime] Baseline run failed: {_seed_e} -- continuing without baseline.",
                      flush=True)

            if not _seed_results:
                print(f"  [FixedTime] All baseline seeds failed -- continuing without baseline.",
                      flush=True)
                _cached_fx_result = {}
            else:
                # -- Average per-TL step-lists across seeds --------------------
                # For queue/wait/reward we concatenate the step-lists from all
                # seeds so that downstream np.mean() sees the full distribution.
                # Scalar metrics (EV, emissions, cars) are averaged numerically.
                _cached_fx_result = {}
                _tl_keys = [k for k in _seed_results[0] if not k.startswith("__")]
                for _tl in _tl_keys:
                    # Concatenate step-level lists across seeds for each metric
                    _all_q = []
                    _all_w = []
                    _all_r = []
                    for _sr in _seed_results:
                        _all_q.extend(_sr[_tl].get("queues",  []))
                        _all_w.extend(_sr[_tl].get("waits",   []))
                        _all_r.extend(_sr[_tl].get("rewards", []))
                    _cached_fx_result[_tl] = {
                        "queues":  _all_q,
                        "waits":   _all_w,
                        "rewards": _all_r,
                    }

                # Scalar meta-metrics: numerical average across seeds
                def _avg_scalar(key, default=0.0):
                    vals = [_sr.get(key, default) for _sr in _seed_results
                            if _sr.get(key) is not None]
                    return float(np.mean(vals)) if vals else default

                def _std_scalar(key, default=0.0):
                    vals = [_sr.get(key, default) for _sr in _seed_results
                            if _sr.get(key) is not None]
                    return float(np.std(vals)) if len(vals) > 1 else 0.0

                _cached_fx_result["__ev_cleared"]          = _avg_scalar("__ev_cleared", 0)
                _cached_fx_result["__cars_passed"]         = _avg_scalar("__cars_passed", 0)
                _cached_fx_result["__co2_total"]           = round(_avg_scalar("__co2_total"),  2)
                _cached_fx_result["__nox_total"]           = round(_avg_scalar("__nox_total"),  2)
                _cached_fx_result["__fuel_total"]          = round(_avg_scalar("__fuel_total"), 2)
                # EV wait: only average over seeds where EVs actually ran
                _ev_wait_vals = [_sr["__ev_avg_wait"] for _sr in _seed_results
                                 if _sr.get("__ev_avg_wait") is not None]
                _cached_fx_result["__ev_avg_wait"] = (
                    round(float(np.mean(_ev_wait_vals)), 3) if _ev_wait_vals else None)
                # Standard deviations stored for downstream uncertainty reporting
                _cached_fx_result["__std_co2_total"]   = round(_std_scalar("__co2_total"),  2)
                _cached_fx_result["__std_nox_total"]   = round(_std_scalar("__nox_total"),  2)
                _cached_fx_result["__std_fuel_total"]  = round(_std_scalar("__fuel_total"), 2)
                _cached_fx_result["__std_cars_passed"] = round(_std_scalar("__cars_passed"), 1)
                # Provenance metadata
                _cached_fx_result["__baseline_type"]   = "multi_seed_monte_carlo"
                # seed list removed
                _cached_fx_result["__n_seeds_ok"]       = len(_seed_results)

                _tl_items = {k: v for k, v in _cached_fx_result.items()
                             if not k.startswith("__")}
                _fx_q = {tl: float(np.mean(d["queues"])) if d["queues"] else 0.0
                         for tl, d in _tl_items.items()}
                _fx_w = {tl: float(np.mean(d["waits"]))  if d["waits"]  else 0.0
                         for tl, d in _tl_items.items()}
                print(f"  [FixedTime] Monte Carlo baseline complete -- "
                      f""
                      f"avg_queue={np.mean(list(_fx_q.values())):.2f}  "
                      f"avg_wait={np.mean(list(_fx_w.values())):.2f}  "
                      f"ev_cleared={_cached_fx_result['__ev_cleared']:.1f}  "
                      f"cars={_cached_fx_result['__cars_passed']:.0f}", flush=True)
                try:
                    with open(_fx_cache_path, "w") as _fcf:
                        json.dump(_cached_fx_result, _fcf, indent=2)
                    print(f"  [FixedTime] Baseline saved -> {_fx_cache_path}", flush=True)
                except Exception as _save_e:
                    print(f"  [FixedTime] WARNING: Could not save baseline: {_save_e}",
                          flush=True)


    # -- TensorBoard comparison logger -------------------------------------
    tb_logger = TBComparisonLogger(save_dir)

    # -- histories for comparison Excel + plots ----------------------------
    rl_history    = {"reward":      {tl: [] for tl in INTERSECTIONS},
                     "queue":       {tl: [] for tl in INTERSECTIONS},
                     "wait":        {tl: [] for tl in INTERSECTIONS},
                     # EV & throughput metrics (flat lists, one entry per episode)
                     "ev_cleared":  [],   # EVs that completed trip
                     "ev_avg_wait": [],   # mean EV accumulated wait (s)
                     "cars_passed": [],   # total vehicles that completed trip
                     # Emission totals per episode (all in mg)
                     "co2_total":   [],
                     "nox_total":   [],
                     "fuel_total":  []}
    fixed_history = {"reward":      {tl: [] for tl in INTERSECTIONS},
                     "queue":       {tl: [] for tl in INTERSECTIONS},
                     "wait":        {tl: [] for tl in INTERSECTIONS},
                     # Fixed-time EV & throughput (single cached run, repeated per episode)
                     "ev_cleared":  [],
                     "ev_avg_wait": [],
                     "cars_passed": [],
                     # Emission totals (from single deterministic FT baseline run)
                     "co2_total":   [],
                     "nox_total":   [],
                     "fuel_total":  []}

    plotter = TrainingPlotter(intersections=INTERSECTIONS, save_dir=plot_dir)

    # _resume_tag is already resolved above (None if not resuming).
    if _resume_tag is not None:
        _ckpt_dir = os.path.join(save_dir, "checkpoints")
        _tag      = _resume_tag   # already resolved; consistent with load_checkpoint call

        # Locate history files -- prefer exact ep-dir, scan backwards if absent.
        # EpisodeAutoSaver:  checkpoints/ep{N:04d}/plotter_history.json
        # CheckpointManager: checkpoints/plotter_history_{tag}.json (legacy flat format)
        _ep_dir_hist = os.path.join(_ckpt_dir, _tag, "plotter_history.json")
        _ep_dir_cmp  = os.path.join(_ckpt_dir, _tag, "comparison_history.json")
        _flat_hist   = os.path.join(_ckpt_dir, f"plotter_history_{_tag}.json")

        _hist_path = (_ep_dir_hist if os.path.exists(_ep_dir_hist) else
                      _flat_hist   if os.path.exists(_flat_hist)   else None)
        _cmp_path  = _ep_dir_cmp  if os.path.exists(_ep_dir_cmp)  else None

        # Fallback: scan backwards up to 50 checkpoints for the most recent save.
        # Handles emergency saves that write .pt but no JSON history alongside.
        if _hist_path is None or _cmp_path is None:
            _ep_num_tag = int(''.join(filter(str.isdigit, _tag)) or '0')
            for _back in range(_ep_num_tag - 1, max(_ep_num_tag - 50, -1), -1):
                _back_dir  = os.path.join(_ckpt_dir, f"ep{_back:04d}")
                _back_hist = os.path.join(_back_dir, "plotter_history.json")
                _back_cmp  = os.path.join(_back_dir, "comparison_history.json")
                if _hist_path is None and os.path.exists(_back_hist):
                    _hist_path = _back_hist
                    print(f"  [RESUME] plotter_history not at {_tag}, "
                          f"using ep{_back:04d} instead.", flush=True)
                if _cmp_path is None and os.path.exists(_back_cmp):
                    _cmp_path = _back_cmp
                    print(f"  [RESUME] comparison_history not at {_tag}, "
                          f"using ep{_back:04d} instead.", flush=True)
                if _hist_path is not None and _cmp_path is not None:
                    break

        # -- Restore plotter history ----------------------------------------
        if _hist_path:
            try:
                with open(_hist_path) as _f:
                    _hd = json.load(_f)

                # Load ALL fields BEFORE deduplication so the dedup block has
                # access to ep_duration_ms and wall_times. Previously these were
                # loaded after the dedup guard, making the guard always False
                # (dead code) and leaving ep_duration_ms length mismatched.
                plotter._episodes            = list(_hd.get("episodes", []))
                plotter._ep_duration_ms      = list(_hd.get("ep_duration_ms", []))
                plotter._global_step_counter = int(_hd.get("global_step_counter", 0))
                plotter._global_steps        = list(_hd.get("global_steps", []))
                plotter._wall_times          = list(_hd.get("wall_times", []))

                for _metric, _tl_data in _hd.get("history", {}).items():
                    for _tl, _vals in _tl_data.items():
                        plotter._history[_metric][_tl] = list(_vals)

                for _metric, _tl_data in _hd.get("step_history", {}).items():
                    for _tl, _pairs in _tl_data.items():
                        plotter._step_history[_metric][_tl] = [
                            tuple(p) for p in _pairs]

                # Deduplication: keep first occurrence of each episode number.
                # Overlapping CSV + JSON saves (e.g. crash mid-write) can produce
                # duplicate entries that break plot x-axes and the Excel export.
                _seen_eps:  set  = set()
                _dedup_idx: list = []
                for _i, _ep_n in enumerate(plotter._episodes):
                    if _ep_n not in _seen_eps:
                        _seen_eps.add(_ep_n)
                        _dedup_idx.append(_i)

                _n_raw = len(plotter._episodes)
                if len(_dedup_idx) < _n_raw:
                    print(f"  [RESUME] Deduplicating plotter history: "
                          f"{_n_raw} -> {len(_dedup_idx)} episodes", flush=True)
                    plotter._episodes = [plotter._episodes[_i] for _i in _dedup_idx]
                    for _metric in list(plotter._history.keys()):
                        for _tl in list(plotter._history[_metric].keys()):
                            _v = plotter._history[_metric][_tl]
                            plotter._history[_metric][_tl] = [
                                _v[_i] for _i in _dedup_idx if _i < len(_v)]
                    plotter._ep_duration_ms = [
                        plotter._ep_duration_ms[_i]
                        for _i in _dedup_idx
                        if _i < len(plotter._ep_duration_ms)]
                    plotter._global_steps = [
                        plotter._global_steps[_i]
                        for _i in _dedup_idx
                        if _i < len(plotter._global_steps)]
                    plotter._wall_times = [
                        plotter._wall_times[_i]
                        for _i in _dedup_idx
                        if _i < len(plotter._wall_times)]

                # Ensure ep_duration_ms has the same length as _episodes (pad with 0).
                _dur_short = len(plotter._episodes) - len(plotter._ep_duration_ms)
                if _dur_short > 0:
                    plotter._ep_duration_ms.extend([0.0] * _dur_short)

                # Reset training_start so wall_times accumulate from last saved value,
                # not from zero -- prevents the wall-time axis jumping on resume.
                if plotter._wall_times:
                    plotter._training_start = time.time() - plotter._wall_times[-1]

                _ep_first = plotter._episodes[0]  if plotter._episodes else "?"
                _ep_last  = plotter._episodes[-1] if plotter._episodes else "?"
                print(f"  [RESUME] Plotter history restored: {len(plotter._episodes)} episodes "
                      f"(ep {_ep_first}-{_ep_last})  "
                      f"global_step={plotter._global_step_counter}", flush=True)

            except Exception as _he:
                print(f"  [RESUME] WARNING: Could not restore plotter history: {_he}",
                      flush=True)
                import traceback as _ht; _ht.print_exc()
        else:
            print(f"  [RESUME] No plotter history found for tag '{_tag}' -- "
                  f"plots start from ep {start_ep}.", flush=True)

        if _cmp_path:
            try:
                with open(_cmp_path) as _f:
                    _cd = json.load(_f)
                _rl_cd = _cd.get("rl_history",    {})
                _fx_cd = _cd.get("fixed_history", {})

                # Per-TL dict metrics
                for _metric in ("reward", "queue", "wait"):
                    for _tl in INTERSECTIONS:
                        rl_history[_metric].setdefault(_tl, [])
                        fixed_history[_metric].setdefault(_tl, [])
                        rl_history[_metric][_tl]    = list(_rl_cd.get(_metric, {}).get(_tl, []))
                        fixed_history[_metric][_tl] = list(_fx_cd.get(_metric, {}).get(_tl, []))

                # Flat list metrics (EV / throughput / emissions)
                for _fk in ("ev_cleared", "ev_avg_wait", "cars_passed",
                            "co2_total", "nox_total", "fuel_total"):
                    rl_history.setdefault(_fk, [])
                    fixed_history.setdefault(_fk, [])
                    rl_history[_fk]    = list(_rl_cd.get(_fk, []))
                    fixed_history[_fk] = list(_fx_cd.get(_fk, []))

                # Length normalisation: pad shorter lists to the canonical episode count.
                # Uses the per-TL reward list length as the reference.
                _n_eps_rl = len(rl_history["reward"].get(INTERSECTIONS[0], []))
                _n_eps_fx = len(fixed_history["reward"].get(INTERSECTIONS[0], []))

                def _pad_to(lst, target_len, default=0.0):
                    if len(lst) < target_len:
                        fill = lst[-1] if lst else default
                        lst.extend([fill] * (target_len - len(lst)))

                for _tl in INTERSECTIONS:
                    for _metric in ("reward", "queue", "wait"):
                        _pad_to(rl_history[_metric][_tl],    _n_eps_rl)
                        _pad_to(fixed_history[_metric][_tl], _n_eps_fx)
                for _fk in ("ev_cleared", "ev_avg_wait", "cars_passed",
                            "co2_total", "nox_total", "fuel_total"):
                    _pad_to(rl_history[_fk],    _n_eps_rl)
                    _pad_to(fixed_history[_fk], _n_eps_fx)

                # Always reset global episode-stat trackers so the first resumed
                # episode starts from a clean state (no cross-episode leakage).
                _reset_ep_stats()

                print(f"  [RESUME] RL/Fixed histories restored "
                      f"(RL={_n_eps_rl} eps, Fixed={_n_eps_fx} eps)", flush=True)

            except Exception as _ce:
                print(f"  [RESUME] WARNING: Could not restore comparison histories: {_ce}",
                      flush=True)
                import traceback as _ct; _ct.print_exc()
                _reset_ep_stats()
        else:
            print(f"  [RESUME] No comparison history found -- "
                  f"RL/Fixed histories start fresh from ep {start_ep}.", flush=True)
            _reset_ep_stats()

        rich_ui.log(
            f"[RESUME] '{_tag}' loaded -- continuing from ep {start_ep} "
            f"({n_episodes - start_ep} episodes remaining)",
            style="bold green")

    reward_writer = ResultsWriter(
        INTERSECTIONS,
        os.path.join(save_dir, "episode_rewards"),
        header={"sim_file": sim_file, "episodes": n_episodes,
                "reward":    "BackPressure_Eq4.92",
                "algorithm": "MAPPO_PIS_GAT",
                "mode":      mode_label,
                # so any episode can be exactly replicated for debugging.
                "seed_scheme": "random (no fixed seed)"},
        extra_fields=[],
    )
    # TL*_peak = worst congestion point in episode; TL*_p95 = 95th-percentile wait time.
    _tl_peak_fields = [f"{tl}_peak" for tl in INTERSECTIONS]
    _tl_p95_fields  = [f"{tl}_p95"  for tl in INTERSECTIONS]
    queue_writer = ResultsWriter(INTERSECTIONS, os.path.join(save_dir, "episode_queues"),
                                 extra_fields=_tl_peak_fields)
    wait_writer  = ResultsWriter(INTERSECTIONS, os.path.join(save_dir, "episode_waits"),
                                 extra_fields=_tl_p95_fields)
    # (parallel mode: all episodes in a batch have identical loss = one update per batch).
    # transparent in the CSV -- was: users couldn't tell the Fixed column never changed.
    loss_writer  = ResultsWriter(INTERSECTIONS, os.path.join(save_dir, "episode_losses"),
                                 extra_fields=["batch_no", "baseline_is_cached", "parallel_batch_loss"])
    # Emission totals -- network-wide scalars (not per-TL), so intersections=[]
    # Columns: episode | ep_duration_ms | co2_rl_mg | nox_rl_mg | fuel_rl_mg
    #          | co2_fx_mg | nox_fx_mg | fuel_fx_mg
    #          | co2_saving_pct | nox_saving_pct | fuel_saving_pct | baseline_is_cached
    emissions_writer = ResultsWriter(
        [],
        os.path.join(save_dir, "episode_emissions"),
        header={"description": "Network-wide emission totals per episode",
                "units":       "mg (milligrams) for all values",
                # FT baseline: single run with random seed
                "note":        "Fixed-Time baseline = single run with random seed; refreshed every 50 RL episodes"},
        extra_fields=["co2_rl_mg", "nox_rl_mg", "fuel_rl_mg",
                      "co2_fx_mg", "nox_fx_mg", "fuel_fx_mg",
                      "co2_saving_pct", "nox_saving_pct", "fuel_saving_pct",
                      "baseline_is_cached"],
    )

    # KL diagnostics -- one row per (episode, tls_id): mean_kl + cumulative
    # KLAdaptiveLRController action counts. Purely diagnostic, does not
    # affect training. See KLDiagnosticsWriter in visualization.py.
    kl_writer = KLDiagnosticsWriter(os.path.join(save_dir, "kl_diagnostics"))

    proc_ref = [None]
    ckpt     = CheckpointManager(save_dir)
    ckpt.register(agents, plotter,
                  [reward_writer, queue_writer, wait_writer, loss_writer, emissions_writer, kl_writer],
                  proc_ref)

    # -- Episode Auto-Saver ------------------------------------------------
    auto_saver = EpisodeAutoSaver(
        save_dir      = save_dir,
        agents        = agents,
        plotter       = plotter,
        writers       = [reward_writer, queue_writer, wait_writer, loss_writer, emissions_writer, kl_writer],
        rl_history    = rl_history,
        fixed_history = fixed_history,
        onnx_export_interval  = 20,
        excel_export_interval = max(1, save_every),
        save_every            = save_every,
        rich_ui       = rich_ui,
    )

    _STEP_INTERVAL = 25   # UI update every 25 steps (was 10) -- reduces overhead

    # -- Start the Rich Live UI now (all print() calls are done above) -----
    rich_ui.start()

    try:
        if parallel_mode:
            _run_training_parallel(
                agents, sumo_cmd, n_episodes, sim_max_steps,
                start_ep, n_workers, device,
                plotter, reward_writer, queue_writer, wait_writer, loss_writer,
                ckpt, proc_ref,
                tb_logger=tb_logger,
                fixed_controllers=fixed_controllers,
                cached_fx_result=_cached_fx_result,
                rl_history=rl_history, fixed_history=fixed_history,
                rich_ui=rich_ui,
                auto_saver=auto_saver,
                emissions_writer=emissions_writer,
                seed_enable=seed_enable, base_seed=base_seed,
                fixed_seed=fixed_seed,
                step_interval=_STEP_INTERVAL,
            )
        else:
            _run_training_sequential(
                agents, sumo_cmd, n_episodes, sim_max_steps,
                start_ep, device, _STEP_INTERVAL,
                plotter, reward_writer, queue_writer, wait_writer, loss_writer,
                ckpt, proc_ref,
                tb_logger=tb_logger,
                fixed_controllers=fixed_controllers,
                cached_fx_result=_cached_fx_result,
                rl_history=rl_history, fixed_history=fixed_history,
                rich_ui=rich_ui,
                auto_saver=auto_saver,
                emissions_writer=emissions_writer,
                seed_enable=seed_enable, base_seed=base_seed,
                fixed_seed=fixed_seed,
                kl_writer=kl_writer,
            )

    except Exception as exc:
        _is_sumo_disconnect = ("Connection closed by SUMO" in str(exc) or
                               "FatalTraCIError" in type(exc).__name__)
        if _is_sumo_disconnect:
            rich_ui.log("[INFO] SUMO connection closed (user quit or simulation ended).",
                        style="bold yellow")
        else:
            print(f"\n[ERROR] Unexpected exception: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        ckpt.save_now(emergency=True)
        if proc_ref[0] is not None:
            _close_sumo(proc_ref[0])
        if not _is_sumo_disconnect:
            raise

    finally:
        rich_ui.set_phase("plot")
        ckpt.save_now()
        reward_writer.close()
        queue_writer.close()
        wait_writer.close()
        loss_writer.close()
        emissions_writer.close()
        tb_logger.close()
        if plotter._episodes:
            rich_ui.log("Generating plots and Excel workbook …", style="bold white")
            try:
                plotter.plot_all()
                plotter.export_excel("training_results.xlsx")
            except Exception as plot_exc:
                rich_ui.log(f"[WARN] Plot/Excel failed: {plot_exc}", style="bold red")
        # -- Final comparison plots + Excel -------------------------------
        # export_comparison_excel() is fully defensive and handles all errors
        # internally (including PermissionError when file is open in Excel).
        if plotter._episodes:
            # Always run comparison export if we have any episodes -- even if
            # fixed_controllers is None, the function will just show FT cols as 0.
            rich_ui.log("Generating RL vs Fixed-Time comparison outputs …",
                        style="bold white")
            try:
                plot_comparison(save_dir, rl_history, fixed_history,
                                plotter._episodes,
                                ep_duration_ms=plotter._ep_duration_ms)
            except Exception as _pe:
                rich_ui.log(f"[WARN] Comparison plots failed: {_pe}", style="bold red")
            # Excel is exported independently so a plot failure never blocks it
            _final_xlsx = export_comparison_excel(
                save_dir,
                rl_history,
                fixed_history,
                plotter._episodes,
                ep_duration_ms=plotter._ep_duration_ms,
                verbose=True,
            )
            if _final_xlsx:
                rich_ui.log(
                    f"[Excel] Final comparison workbook: "
                    f"{os.path.basename(_final_xlsx)}",
                    style="bold green")
            else:
                rich_ui.log(
                    "[WARN] Final Excel export did not complete "
                    "(see console for reason).",
                    style="bold yellow")
        rich_ui.set_phase("done")
        rich_ui.stop()
        print(f"\nAll results saved to: {save_dir}")

def _run_training_sequential(
    agents, sumo_cmd, n_episodes, sim_max_steps,
    start_ep, device, step_interval,
    plotter, reward_writer, queue_writer, wait_writer, loss_writer,
    ckpt, proc_ref,
    tb_logger=None, fixed_controllers=None, cached_fx_result=None,
    rl_history=None, fixed_history=None,
    rich_ui=None, auto_saver=None,
    emissions_writer=None,
    seed_enable=False, base_seed=42,
    fixed_seed=False,
    kl_writer=None,
):
    _ui = rich_ui  # shorthand

    def _log(msg, style="dim white"):
        if _ui and RICH_AVAILABLE:
            # Must use the live's console -- plain print() breaks the Live display
            _ui.log(msg, style=style)
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"{ts}  {msg}", flush=True)

    is_first_ep = True
    for ep in range(start_ep, n_episodes):
        ckpt.update_episode(ep)

        _is_sumo_disconnect  = False
        _sumo_closed_by_user = False

        if _ui:
            _ui.begin_episode(ep)
        _ep_start_ms = time.time() * 1000.0
        _log(f"[Ep {ep}] Resetting logs …")
        for agent in agents.values():
            agent.step_rewards.clear()
            agent.step_queues.clear()
            agent.step_losses.clear()
            agent.step_waits.clear()
            # Reset EV potential cache so shaping starts fresh each episode.
            # across episode boundaries giving a spurious shaping bonus at step 1.
            agent._ev_phi_prev = 0.0
            agent._ev_phi_curr = 0.0

        MAPPOAgent.clear_lane_length_cache()

        ep_port = BASE_PORT + (ep % 500) + 1

        if seed_enable:
            ep_seed_val = base_seed if fixed_seed else base_seed + ep
            random.seed(ep_seed_val)
            np.random.seed(ep_seed_val)
            torch.manual_seed(ep_seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(ep_seed_val)
                torch.cuda.manual_seed_all(ep_seed_val)
            _ep_sumo_cmd = sumo_cmd + ["--seed", str(ep_seed_val)]
            _log(f"[Ep {ep}] Deterministic seed = {ep_seed_val}"
                 + (" (fixed)" if fixed_seed else f" (base+ep={base_seed}+{ep})"))
        else:
            _ep_sumo_cmd = sumo_cmd
            _log(f"[Ep {ep}] Starting SUMO on port {ep_port} (random seed) …")
        _ep_sumo_cmd = list(_ep_sumo_cmd) + _xml_output_args(os.path.join("results", "training_xml"))
        try:
            _proc = _start_sumo(_ep_sumo_cmd, port=ep_port)
            proc_ref[0] = _proc
        except Exception as exc:
            _log(f"[ERROR] Ep {ep}: Cannot start SUMO: {exc}", style="bold red")
            import traceback; traceback.print_exc()
            break

        _setup_all_subscriptions(agents)

        _log(f"[Ep {ep}] SUMO started. Refreshing states …")
        for agent in agents.values():
            if is_first_ep and start_ep > 0:
                # First episode after a resume: keep action_idx, last_switch_step,
                # and neighbor_intents as restored from the checkpoint.
                # Only zero _current_step so the step counter starts fresh for
                # this new SUMO instance (the old value was the last step of the
                # previous run, not meaningful for a new simulation).
                agent._current_step = 0
            else:
                # Fresh episode (no resume) or any subsequent episode:
                # reset all per-episode agent state to safe defaults.
                agent.last_switch_step   = -agent.min_green_steps
                agent.current_action_idx = 0
                agent._current_step      = 0
                agent.neighbor_intents.clear()
            try:
                agent.state = agent._get_state()
            except Exception:
                agent.state = np.zeros(agent.state_dim, dtype=np.float32)
        is_first_ep = False

        _reset_ep_stats()   # zero EV & throughput counters for this episode
        _log(f"[Ep {ep}] Running {sim_max_steps} steps …")
        # Raw transitions: (state, intent, action, reward, next_state, next_intent, done)
        _raw_transitions: dict[str, list] = {tid: [] for tid in agents}
        _min_expected_interval = 50   # check every 50 steps instead of every step
        try:
            for step in range(1, sim_max_steps + 1):
                if step % _min_expected_interval == 0:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        _log(f"[Ep {ep}] Early stop at step {step}")
                        break

                if step % step_interval == 0:
                    if _ui:
                        _ui.update_step(step)
                # Subscribe newly-departed vehicles every step (cheap now)
                _subscribe_vehicles_bulk()

                _STEP_CACHE.clear()
                _broadcast_intents(agents)

                # pre_step: choose & apply action, capture current state+intent
                step_states  = {}
                step_intents = {}
                actions      = {}
                for tid, ag in agents.items():
                    agg = ag.aggregate_neighbor_intents(step)
                    combined = (ag.current_intent + agg) / 2.0
                    step_states[tid]  = ag.state.copy()
                    step_intents[tid] = combined.copy()
                    action = ag._choose_action_with_intent(combined)
                    ag.current_action_idx = action
                    ag.apply_action(action, step)
                    actions[tid] = action

                traci.simulationStep()

                _STEP_CACHE.bulk_load_lanes()
                _STEP_CACHE.bulk_load_detectors()
                _STEP_CACHE.bulk_load_vehicles()

                done = 1.0 if step >= sim_max_steps else 0.0
                global_intent_vec = _build_global_intent_vec(
                    agents, next(iter(agents.values())).intent_dim)
                for tid, ag in agents.items():
                    ns     = ag._get_state()
                    reward = ag.get_reward()

                    agg_next    = ag.aggregate_neighbor_intents(step)
                    next_intent = (ag.current_intent + agg_next) / 2.0

                    _raw_transitions[tid].append((
                        step_states[tid],
                        step_intents[tid],
                        actions[tid],
                        reward,
                        ns.copy(),
                        next_intent.copy(),
                        done,
                        global_intent_vec.copy(),
                    ))
                    ag.state = ns

                    ag.step_rewards.append(reward)
                    ag.step_queues.append(_get_tls_queue_from_detectors(tid))
                    ag.step_waits.append(_get_tls_mean_wait(tid))

        except Exception as step_exc:
            _is_sumo_disconnect  = ("Connection closed by SUMO" in str(step_exc)
                                    or "FatalTraCIError" in type(step_exc).__name__)
            _sumo_closed_by_user = _is_sumo_disconnect
            if _is_sumo_disconnect:
                _log(f"[Ep {ep}] SUMO GUI closed by user at step {step}.", style="bold yellow")
            else:
                _log(f"[Ep {ep}] Step error at step {step}: {step_exc}", style="bold red")
            _log(f"[Ep {ep}] Saving partial episode data …")

        # -- Post-episode: batch compute values & log-probs, fill rollout buffer --
        if _ui: _ui.set_phase("train")
        _log(f"[Ep {ep}] Training on collected rollouts …")
        for tid, ag in agents.items():
            trans = _raw_transitions[tid]
            if not trans:
                continue
            all_s      = np.stack([t[0] for t in trans])
            all_intent = np.stack([t[1] for t in trans])
            all_a      = np.array([t[2] for t in trans], dtype=np.int64)
            all_ns     = np.stack([t[4] for t in trans])
            all_ni     = np.stack([t[5] for t in trans])
            all_d      = np.array([t[6] for t in trans], dtype=np.float32)
            all_gi     = np.stack([t[7] for t in trans])

            if all_s.shape[1] != ag.state_dim:
                _log(f"[DIM-ERROR] {tid}: state_dim={all_s.shape[1]} "
                     f"!= agent.state_dim={ag.state_dim} -- skipping", style="bold red")
                continue

            all_values    = ag.get_values_batch(all_s, all_gi)
            all_log_probs = ag.get_log_probs_batch(all_s, all_intent, all_a)

            non_term = np.where(all_d == 0.0)[0]
            all_next_values = np.zeros(len(trans), dtype=np.float32)
            if len(non_term) > 0:
                next_gi = np.stack([
                    trans[i + 1][7] if i + 1 < len(trans) else all_gi[i]
                    for i in non_term
                ])
                all_next_values[non_term] = ag.get_values_batch(
                    all_ns[non_term], next_gi)

            for i, (s, intent, a, r, ns, ni, d, gi) in enumerate(trans):
                ag.rollout_buffer.push(
                    s, intent, a, r,
                    float(all_values[i]),
                    float(all_log_probs[i]),
                    d, intent,
                    float(all_next_values[i]),
                    global_intent=gi)

            loss = ag.train_step(agents=agents)
            # Always record the loss for this episode, even if zero.
            # This prevents TL2/others from showing 0.0 in CSV when
            # train_step returns a valid value but step_losses was empty.
            if not ag.step_losses:
                ag.step_losses.append(loss)
            elif ag.step_losses[-1] == 0.0 and loss > 0.0:
                ag.step_losses[-1] = loss

        if _ui: _ui.set_phase("checkpoint")
        _log(f"[Ep {ep}] Closing SUMO …")
        _close_sumo(_proc)
        proc_ref[0] = None

        if _sumo_closed_by_user:
            has_data = any(ag.step_rewards for ag in agents.values())
            if not has_data:
                _log(f"[Ep {ep}] No data collected - saving checkpoint only.")
                ckpt.save_now(emergency=True)
                break
            _log(f"[Ep {ep}] Partial data collected - recording and saving.")

        ep_reward, ep_queue, ep_wait, ep_loss = {}, {}, {}, {}
        ep_queue_peak, ep_wait_p95 = {}, {}
        for tl in INTERSECTIONS:
            agent = agents.get(tl)
            if agent and agent.step_rewards:
                # step_rewards/queues/waits are appended once per sim step,
                # so np.mean here gives the true episode mean.
                ep_reward[tl] = float(np.mean(agent.step_rewards))
                ep_queue[tl]  = float(np.mean(agent.step_queues))
                ep_wait[tl]   = float(np.mean(agent.step_waits)) \
                                if agent.step_waits else 0.0
                ep_loss[tl]   = float(np.mean(agent.step_losses)) \
                                if agent.step_losses else 0.0
                ep_queue_peak[tl] = float(np.max(agent.step_queues)) if agent.step_queues else 0.0
                ep_wait_p95[tl]   = float(np.percentile(agent.step_waits, 95)) if agent.step_waits else 0.0
            else:
                ep_reward[tl] = ep_queue[tl] = ep_wait[tl] = ep_loss[tl] = 0.0
                ep_queue_peak[tl] = ep_wait_p95[tl] = 0.0

            # KL diagnostics (MAPPO+HierGAT only): true_kl
            # (agent.last_true_kl) is the full distributional KL from logits
            # that actually drives KLAdaptiveLRController and the clip
            # controller (see agent.py train_epoch()). approx_kl
            # (agent.last_mean_kl) is the sample-based PPO estimator
            # (mean(old_lp − new_lp)) kept for diagnostic comparison only.
            # The old comment claiming the reverse was stale — it predated
            # the switch to batch_true_kl as the controller signal.
            # getattr guards keep this safe for any agent type that lacks
            # these attributes (e.g. IPPO agents, which don't track them).
            if kl_writer is not None and agent is not None:
                _approx_kl = getattr(agent, "last_mean_kl", None)
                if _approx_kl is not None:
                    kl_writer.write_row(
                        episode=ep, tls_id=tl,
                        approx_kl=_approx_kl,
                        true_kl=getattr(agent, "last_true_kl", 0.0),
                        counts=getattr(agent, "last_kl_action_counts", None))

        ep_duration_ms = time.time() * 1000.0 - _ep_start_ms

        # ── FIX-C: Per-TLS reward normalisation diagnostic ────────────────
        # Print raw queue / capacity / normalised reward every 50 episodes
        # so we can verify that per-TLS capacity scaling is working
        # (TL1:90, TL2:70, TL3:20, TL4:20) and rewards are on comparable
        # magnitudes across intersections.
        if ep % 50 == 0 or ep == n_episodes - 1:
            _cap = _QUEUE_CAPACITY_PER_TLS
            for tl in INTERSECTIONS:
                _raw_q = ep_queue.get(tl, 0.0)
                _cap_v = _cap.get(tl, UNIFIED_BP_CAPACITY)
                _raw_r = ep_reward.get(tl, 0.0)
                _log(f"    {tl}: raw_queue={_raw_q:.1f}  capacity={_cap_v:.0f}  reward={_raw_r:+.4f}",
                     style="dim white")

        plotter.record(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                       ep_duration_ms=ep_duration_ms)
        plotter.record_steps(agents)
        reward_writer.write_row({"episode": ep, **ep_reward,
                                 "ep_duration_ms": round(ep_duration_ms, 1),
                                 })
        _q_peak_cols = {f"{tl}_peak": ep_queue_peak.get(tl, 0.0) for tl in INTERSECTIONS}
        _w_p95_cols  = {f"{tl}_p95":  ep_wait_p95.get(tl,   0.0) for tl in INTERSECTIONS}
        queue_writer.write_row({"episode": ep, **ep_queue, "ep_duration_ms": round(ep_duration_ms, 1), **_q_peak_cols})
        wait_writer.write_row({"episode":  ep, **ep_wait,  "ep_duration_ms": round(ep_duration_ms, 1), **_w_p95_cols})
        loss_writer.write_row({"episode": ep, **ep_loss,
                               "ep_duration_ms": round(ep_duration_ms, 1),
                               "batch_no": 0,
                               "baseline_is_cached": True,
                               "parallel_batch_loss": False})
        # Emission CSV -- one row per episode (RL vs cached Fixed-Time)
        if emissions_writer is not None:
            _collect_final_emissions()
            _em_rl   = _snapshot_ep_stats()
            _co2_fx  = cached_fx_result.get("__co2_total",  0.0) if cached_fx_result else 0.0
            _nox_fx  = cached_fx_result.get("__nox_total",  0.0) if cached_fx_result else 0.0
            _fuel_fx = cached_fx_result.get("__fuel_total", 0.0) if cached_fx_result else 0.0
            _co2_rl  = round(_em_rl["co2_sum"],  2)
            _nox_rl  = round(_em_rl["nox_sum"],  2)
            _fuel_rl = round(_em_rl["fuel_sum"], 2)
            _e_pct   = lambda rl_v, fx_v: round(((fx_v - rl_v) / max(fx_v, 1e-6)) * 100, 2) if fx_v else 0.0
            emissions_writer.write_row({
                "episode":          ep,
                "ep_duration_ms":   round(ep_duration_ms, 1),
                "co2_rl_mg":        _co2_rl,
                "nox_rl_mg":        _nox_rl,
                "fuel_rl_mg":       _fuel_rl,
                "co2_fx_mg":        round(_co2_fx,  2),
                "nox_fx_mg":        round(_nox_fx,  2),
                "fuel_fx_mg":       round(_fuel_fx, 2),
                "co2_saving_pct":   _e_pct(_co2_rl,  _co2_fx),
                "nox_saving_pct":   _e_pct(_nox_rl,  _nox_fx),
                "fuel_saving_pct":  _e_pct(_fuel_rl, _fuel_fx),
                "baseline_is_cached": True,
            })

        # -- Log RL metrics to TensorBoard ---------------------------------
        if tb_logger is not None:
            tb_logger.log_episode_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss)


        FT_REFRESH_INTERVAL = 50  # re-run FT every N episodes (0 = first run only)
        if (fixed_controllers and
                (ep % FT_REFRESH_INTERVAL == 0) and
                ep > 0):   # ep==0 already run at startup; skip to avoid duplicate
            _ft_seed_label = (f"seed={base_seed} (fixed)"
                              if fixed_seed else "random seed")
            if fixed_seed and seed_enable:
                _refresh_cmd = sumo_cmd + ["--seed", str(base_seed)]
            else:
                _refresh_cmd = sumo_cmd
            _log(f"[Ep {ep}] Refreshing Fixed-Time baseline ({_ft_seed_label})…")
            try:
                _fx_port_refresh = BASE_PORT + 701
                _run_result_refresh = _run_fixed_time_episode(
                    _refresh_cmd, max(900, sim_max_steps), _fx_port_refresh, fixed_controllers)
                cached_fx_result = {}
                for _tl in [k for k in _run_result_refresh if not k.startswith("__")]:
                    cached_fx_result[_tl] = _run_result_refresh[_tl]
                cached_fx_result["__ev_cleared"]       = _run_result_refresh.get("__ev_cleared", 0)
                cached_fx_result["__ev_avg_wait"]      = _run_result_refresh.get("__ev_avg_wait", None)
                cached_fx_result["__cars_passed"]      = _run_result_refresh.get("__cars_passed", 0)
                cached_fx_result["__co2_total"]        = _run_result_refresh.get("__co2_total",  0.0)
                cached_fx_result["__nox_total"]        = _run_result_refresh.get("__nox_total",  0.0)
                cached_fx_result["__fuel_total"]       = _run_result_refresh.get("__fuel_total", 0.0)
                cached_fx_result["__ft_baseline_ep"]   = ep
                # seed removed
                _log(f"[Ep {ep}] Fixed-Time baseline refreshed -- "
                     f"ev_cleared={cached_fx_result['__ev_cleared']}  "
                     f"cars={cached_fx_result['__cars_passed']}")
            except Exception as _refresh_e:
                _log(f"[Ep {ep}] WARNING: FT refresh failed: {_refresh_e} -- keeping previous cached result.")

        fx_reward = {tl: 0.0 for tl in INTERSECTIONS}
        fx_queue  = {tl: 0.0 for tl in INTERSECTIONS}
        fx_wait   = {tl: 0.0 for tl in INTERSECTIONS}
        if fixed_controllers and cached_fx_result:
            if _ui:
                _ui.begin_fixed(ep)
            for tl, data in cached_fx_result.items():
                if tl.startswith("__"):
                    continue   # skip __ev_cleared / __ev_avg_wait / __cars_passed
                fx_reward[tl] = float(np.mean(data["rewards"])) if data["rewards"] else 0.0
                fx_queue[tl]  = float(np.mean(data["queues"]))  if data["queues"]  else 0.0
                fx_wait[tl]   = float(np.mean(data["waits"]))   if data["waits"]   else 0.0

            if tb_logger is not None:
                tb_logger.log_episode_fixed(ep, fixed_controllers)
                avg_rl_q  = float(np.mean(list(ep_queue.values())))
                avg_fx_q  = float(np.mean(list(fx_queue.values())))
                avg_rl_w  = float(np.mean(list(ep_wait.values())))
                avg_fx_w  = float(np.mean(list(fx_wait.values())))
                tb_logger.log_comparison(ep, avg_rl_q,  avg_fx_q,  "Compare/Avg_Queue")
                tb_logger.log_comparison(ep, avg_rl_w,  avg_fx_w,  "Compare/Avg_Wait")
                tb_logger.log_comparison(ep,
                                         float(np.mean(list(ep_reward.values()))),
                                         float(np.mean(list(fx_reward.values()))),
                                         "Compare/Avg_Reward")

        # -- Accumulate histories for final comparison plots ---------------
        # Capture RL EV & throughput stats collected during this episode's steps
        _rl_ep_stats = _snapshot_ep_stats()
        # "all EVs cleared instantly", which misleads the reader.
        if _rl_ep_stats["ev_cleared"] > 0:
            _rl_ev_avg = round(_rl_ep_stats["ev_wait_sum"] / _rl_ep_stats["ev_cleared"], 3)
        else:
            _rl_ev_avg = None   # sentinel: no EV data this episode
        if rl_history is not None:
            for tl in INTERSECTIONS:
                rl_history["reward"][tl].append(ep_reward.get(tl, 0.0))
                rl_history["queue"][tl].append(ep_queue.get(tl, 0.0))
                rl_history["wait"][tl].append(ep_wait.get(tl, 0.0))
            rl_history["ev_cleared"].append(_rl_ep_stats["ev_cleared"])
            rl_history["ev_avg_wait"].append(_rl_ev_avg)
            rl_history["cars_passed"].append(_rl_ep_stats["cars_passed"])
            # Emission totals for this RL episode (all in mg, including vehicles still in network)
            rl_history["co2_total"].append(round(_rl_ep_stats["co2_sum"],  2))
            rl_history["nox_total"].append(round(_rl_ep_stats["nox_sum"],  2))
            rl_history["fuel_total"].append(round(_rl_ep_stats["fuel_sum"], 2))
        # Fixed EV & emission stats come from the cached deterministic baseline run
        if fixed_history is not None:
            for tl in INTERSECTIONS:
                fixed_history["reward"][tl].append(fx_reward.get(tl, 0.0))
                fixed_history["queue"][tl].append(fx_queue.get(tl, 0.0))
                fixed_history["wait"][tl].append(fx_wait.get(tl, 0.0))
            fixed_history["ev_cleared"].append(
                cached_fx_result.get("__ev_cleared", 0) if cached_fx_result else 0)
            _fx_ev_avg_wait = cached_fx_result.get("__ev_avg_wait", None) if cached_fx_result else None
            fixed_history["ev_avg_wait"].append(_fx_ev_avg_wait)
            fixed_history["cars_passed"].append(
                cached_fx_result.get("__cars_passed", 0) if cached_fx_result else 0)
            # Emission totals from the cached FT baseline (refreshed every FT_REFRESH_INTERVAL episodes)
            fixed_history["co2_total"].append(
                cached_fx_result.get("__co2_total",  0.0) if cached_fx_result else 0.0)
            fixed_history["nox_total"].append(
                cached_fx_result.get("__nox_total",  0.0) if cached_fx_result else 0.0)
            fixed_history["fuel_total"].append(
                cached_fx_result.get("__fuel_total", 0.0) if cached_fx_result else 0.0)
            fixed_history.setdefault("ft_baseline_ep", []).append(
                cached_fx_result.get("__ft_baseline_ep", 0) if cached_fx_result else 0)

        # -- Update Rich UI for this episode -------------------------------
        if _ui:
            _ui.record_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                          ep_stats=_snapshot_ep_stats())
            if fixed_controllers and cached_fx_result:
                _ui.record_fixed(fx_reward, fx_queue, fx_wait,
                                 fx_stats=cached_fx_result)

        # -- Auto-save: checkpoint + plots + Excel every episode -----------
        auto_saver.save_episode(
            ep,
            ep_reward, ep_queue, ep_wait, ep_loss,
            fx_reward, fx_queue, fx_wait,
        )

        if ep % auto_saver.save_every == 0 or ep == n_episodes - 1:
            auto_saver.tag_latest(ep)

        # -- If user closed SUMO GUI, stop gracefully ----------------------
        if _sumo_closed_by_user:
            _log("[INFO] SUMO GUI closed -- stopping training.", style="bold yellow")
            ckpt.save_now(emergency=True)
            break

        _log(f"[Ep {ep}] Done. "
             f"avg_r={float(np.mean(list(ep_reward.values()))):+.4f}  "
             f"avg_q={float(np.mean(list(ep_queue.values()))):.2f}  "
             f"avg_w={float(np.mean(list(ep_wait.values()))):.2f}",
             style="dim cyan")

    # end of episode loop
    _log("Sequential training loop complete.", style="bold green")

def _run_training_parallel(
    agents, sumo_cmd, n_episodes, sim_max_steps,
    start_ep, n_workers, device,
    plotter, reward_writer, queue_writer, wait_writer, loss_writer,
    ckpt, proc_ref,
    tb_logger=None, fixed_controllers=None, cached_fx_result=None,
    rl_history=None, fixed_history=None,
    rich_ui=None, auto_saver=None,
    emissions_writer=None,
    seed_enable=False, base_seed=42,
    fixed_seed=False,
    step_interval=25,
):
    """
    Thin wrapper: run episodes in sequential batches of n_workers,
    then do one shared PPO update per batch.
    On Windows this is forced to n_workers=1 by run_training().
    """
    _ui = rich_ui

    def _log(msg, style="dim white"):
        if _ui and RICH_AVAILABLE:
            _ui.log(msg, style=style)
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"{ts}  {msg}", flush=True)

    batch_no = 0
    for batch_start in range(start_ep, n_episodes, n_workers):
        batch_eps   = list(range(batch_start, min(batch_start + n_workers, n_episodes)))
        batch_no   += 1
        ep_port_base = BASE_PORT + (batch_start % 400) + 1

        _log(f"[Batch {batch_no}] Running episodes {batch_eps[0]}-{batch_eps[-1]} …")

        # -- Each worker runs one sequential episode ------------------------
        batch_rewards     = {tl: [] for tl in INTERSECTIONS}
        batch_queues      = {tl: [] for tl in INTERSECTIONS}
        batch_waits       = {tl: [] for tl in INTERSECTIONS}
        batch_losses      = {tl: [] for tl in INTERSECTIONS}
        batch_queue_peaks = {tl: [] for tl in INTERSECTIONS}
        batch_wait_p95s   = {tl: [] for tl in INTERSECTIONS}
        # compute a real ep_duration_ms (was hardcoded 0.0 for all parallel eps).
        batch_ep_start_ms: list[float] = []
        batch_ep_dur_ms:   list[float] = []
        batch_ep_seeds:    list[int]   = []

        for w_idx, ep in enumerate(batch_eps):
            ckpt.update_episode(ep)
            if _ui:
                _ui.begin_episode(ep)
            ep_port = ep_port_base + w_idx
            _par_ep_start_ms = time.time() * 1000.0

            for agent in agents.values():
                agent.step_rewards.clear()
                agent.step_queues.clear()
                agent.step_losses.clear()
                agent.step_waits.clear()
                agent._ev_phi_prev = 0.0
                agent._ev_phi_curr = 0.0
            MAPPOAgent.clear_lane_length_cache()
            _reset_ep_stats()

            try:
                if seed_enable:
                    ep_seed_val = base_seed if fixed_seed else base_seed + ep
                    random.seed(ep_seed_val)
                    np.random.seed(ep_seed_val)
                    torch.manual_seed(ep_seed_val)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(ep_seed_val)
                        torch.cuda.manual_seed_all(ep_seed_val)
                    _ep_sumo_cmd = sumo_cmd + ["--seed", str(ep_seed_val)]
                    batch_ep_seeds.append(ep_seed_val)
                    _log(f"[Ep {ep}] Deterministic seed = {ep_seed_val}"
                         + (" (fixed)" if fixed_seed else f" (base+ep={base_seed}+{ep})"))
                else:
                    ep_seed_val = None
                    batch_ep_seeds.append(None)
                    _ep_sumo_cmd = sumo_cmd
                _ep_sumo_cmd = list(_ep_sumo_cmd) + _xml_output_args(os.path.join("results", "training_xml"))
                _proc = _start_sumo(_ep_sumo_cmd, port=ep_port)
                proc_ref[0] = _proc
            except Exception as exc:
                _log(f"[ERROR] Ep {ep}: Cannot start SUMO: {exc}", style="bold red")
                batch_ep_seeds.append(None)  # keep list aligned even on failure
                continue

            _setup_all_subscriptions(agents)
            for agent in agents.values():
                agent.last_switch_step   = -agent.min_green_steps
                agent.current_action_idx = 0
                agent._current_step      = 0
                agent.neighbor_intents.clear()
                try:
                    agent.state = agent._get_state()
                except Exception:
                    agent.state = np.zeros(agent.state_dim, dtype=np.float32)

            _raw_transitions = {tid: [] for tid in agents}
            try:
                for step in range(1, sim_max_steps + 1):
                    if step % 50 == 0:
                        if traci.simulation.getMinExpectedNumber() == 0:
                            break
                    if step % step_interval == 0:
                        if _ui:
                            _ui.update_step(step)
                    _subscribe_vehicles_bulk()
                    _STEP_CACHE.clear()
                    _broadcast_intents(agents)

                    step_states  = {}
                    step_intents = {}
                    actions      = {}
                    for tid, ag in agents.items():
                        agg      = ag.aggregate_neighbor_intents(step)
                        combined = (ag.current_intent + agg) / 2.0
                        step_states[tid]  = ag.state.copy()
                        step_intents[tid] = combined.copy()
                        action = ag._choose_action_with_intent(combined)
                        ag.current_action_idx = action
                        ag.apply_action(action, step)
                        actions[tid] = action

                    traci.simulationStep()
                    _STEP_CACHE.bulk_load_lanes()
                    _STEP_CACHE.bulk_load_detectors()
                    _STEP_CACHE.bulk_load_vehicles()

                    done = 1.0 if step >= sim_max_steps else 0.0
                    global_intent_vec = _build_global_intent_vec(
                        agents, next(iter(agents.values())).intent_dim)
                    for tid, ag in agents.items():
                        ns          = ag._get_state()
                        reward      = ag.get_reward()
                        agg_next    = ag.aggregate_neighbor_intents(step)
                        next_intent = (ag.current_intent + agg_next) / 2.0
                        _raw_transitions[tid].append((
                            step_states[tid], step_intents[tid], actions[tid],
                            reward, ns.copy(), next_intent.copy(), done,
                            global_intent_vec.copy(),
                        ))
                        ag.state = ns
                        ag.step_rewards.append(reward)
                        ag.step_queues.append(_get_tls_queue_from_detectors(tid))
                        ag.step_waits.append(_get_tls_mean_wait(tid))

            except Exception as step_exc:
                _log(f"[Ep {ep}] Step error: {step_exc}", style="bold red")

            # Fill rollout buffers
            for tid, ag in agents.items():
                trans = _raw_transitions[tid]
                if not trans:
                    continue
                all_s      = np.stack([t[0] for t in trans])
                all_intent = np.stack([t[1] for t in trans])
                all_a      = np.array([t[2] for t in trans], dtype=np.int64)
                all_ns     = np.stack([t[4] for t in trans])
                all_ni     = np.stack([t[5] for t in trans])
                all_d      = np.array([t[6] for t in trans], dtype=np.float32)
                all_gi     = np.stack([t[7] for t in trans])
                all_values    = ag.get_values_batch(all_s, all_gi)
                all_log_probs = ag.get_log_probs_batch(all_s, all_intent, all_a)
                non_term      = np.where(all_d == 0.0)[0]
                all_nv        = np.zeros(len(trans), dtype=np.float32)
                if len(non_term) > 0:
                    next_gi = np.stack([
                        trans[i + 1][7] if i + 1 < len(trans) else all_gi[i]
                        for i in non_term])
                    all_nv[non_term] = ag.get_values_batch(all_ns[non_term], next_gi)
                for i, (s, intent, a, r, ns, ni, d, gi) in enumerate(trans):
                    ag.rollout_buffer.push(
                        s, intent, a, r,
                        float(all_values[i]), float(all_log_probs[i]),
                        d, intent, float(all_nv[i]), global_intent=gi)

            _collect_final_emissions()
            _close_sumo(_proc)
            proc_ref[0] = None

            _par_ep_dur = time.time() * 1000.0 - _par_ep_start_ms
            batch_ep_dur_ms.append(_par_ep_dur)

            # episode's SUMO run completes.  Previously all episodes in the batch
            # read from batch_rewards[tl][w_idx] which was populated *sequentially*,
            # meaning episode N always got the rewards from the last-run agent state,
            # not the state that existed when episode N actually executed.
            for tl in INTERSECTIONS:
                ag = agents.get(tl)
                if ag and ag.step_rewards:
                    batch_rewards[tl].append(float(np.mean(ag.step_rewards)))
                    batch_queues[tl].append(float(np.mean(ag.step_queues)))
                    batch_waits[tl].append(float(np.mean(ag.step_waits)) if ag.step_waits else 0.0)
                    batch_queue_peaks[tl].append(float(np.max(ag.step_queues)) if ag.step_queues else 0.0)
                    batch_wait_p95s[tl].append(float(np.percentile(ag.step_waits, 95)) if ag.step_waits else 0.0)
                else:
                    batch_rewards[tl].append(0.0)
                    batch_queues[tl].append(0.0)
                    batch_waits[tl].append(0.0)
                    batch_queue_peaks[tl].append(0.0)
                    batch_wait_p95s[tl].append(0.0)

        # -- Shared PPO update (once per batch) ----------------------------
        batch_loss = {tl: 0.0 for tl in INTERSECTIONS}
        for tid, ag in agents.items():
            if ag.rollout_buffer.size > 0:
                loss = ag.train_step(agents=agents)
                batch_loss[tid] = loss
                if not ag.step_losses:
                    ag.step_losses.append(loss)

        # -- Record each episode in the batch ------------------------------
        for w_idx, ep in enumerate(batch_eps):
            ep_reward = {tl: batch_rewards[tl][w_idx] if w_idx < len(batch_rewards[tl]) else 0.0
                         for tl in INTERSECTIONS}
            ep_queue  = {tl: batch_queues[tl][w_idx]  if w_idx < len(batch_queues[tl])  else 0.0
                         for tl in INTERSECTIONS}
            ep_wait   = {tl: batch_waits[tl][w_idx]   if w_idx < len(batch_waits[tl])   else 0.0
                         for tl in INTERSECTIONS}
            ep_loss   = batch_loss
            _par_q_peak = {f"{tl}_peak": batch_queue_peaks[tl][w_idx] if w_idx < len(batch_queue_peaks[tl]) else 0.0
                           for tl in INTERSECTIONS}
            _par_w_p95  = {f"{tl}_p95":  batch_wait_p95s[tl][w_idx]  if w_idx < len(batch_wait_p95s[tl])  else 0.0
                           for tl in INTERSECTIONS}
            _ep_dur  = batch_ep_dur_ms[w_idx]  if w_idx < len(batch_ep_dur_ms)  else 0.0
            _ep_seed = None

            # ── FIX-C: Per-TLS reward normalisation diagnostic (parallel) ──
            if ep % 50 == 0 or ep == n_episodes - 1:
                _cap = _QUEUE_CAPACITY_PER_TLS
                for tl in INTERSECTIONS:
                    _raw_q = ep_queue.get(tl, 0.0)
                    _cap_v = _cap.get(tl, UNIFIED_BP_CAPACITY)
                    _raw_r = ep_reward.get(tl, 0.0)
                    _log(f"    {tl}: raw_queue={_raw_q:.1f}  capacity={_cap_v:.0f}  reward={_raw_r:+.4f}",
                         style="dim white")

            plotter.record(ep, ep_reward, ep_queue, ep_wait, ep_loss, ep_duration_ms=_ep_dur)
            reward_writer.write_row({"episode": ep, **ep_reward,
                                     "ep_duration_ms": round(_ep_dur, 1),
                                     })
            queue_writer.write_row({"episode":  ep, **ep_queue,  "ep_duration_ms": round(_ep_dur, 1), **_par_q_peak})
            wait_writer.write_row({"episode":   ep, **ep_wait,   "ep_duration_ms": round(_ep_dur, 1), **_par_w_p95})
            loss_writer.write_row({"episode":   ep, **ep_loss,
                                   "ep_duration_ms": round(_ep_dur, 1),
                                   "batch_no": batch_no,
                                   "baseline_is_cached": True,
                                   "parallel_batch_loss": True})  # BUG-10 FIX: all eps in batch share one loss value
            # Emission CSV -- parallel mode (ep_duration_ms not tracked per-ep)
            if emissions_writer is not None:
                _em_rl_p   = _snapshot_ep_stats()
                _co2_fx_p  = cached_fx_result.get("__co2_total",  0.0) if cached_fx_result else 0.0
                _nox_fx_p  = cached_fx_result.get("__nox_total",  0.0) if cached_fx_result else 0.0
                _fuel_fx_p = cached_fx_result.get("__fuel_total", 0.0) if cached_fx_result else 0.0
                _co2_rl_p  = round(_em_rl_p["co2_sum"],  2)
                _nox_rl_p  = round(_em_rl_p["nox_sum"],  2)
                _fuel_rl_p = round(_em_rl_p["fuel_sum"], 2)
                _ep_pct = lambda rl_v, fx_v: round(((fx_v - rl_v) / max(fx_v, 1e-6)) * 100, 2) if fx_v else 0.0
                emissions_writer.write_row({
                    "episode":          ep,
                    "ep_duration_ms":   round(_ep_dur, 1),
                    "co2_rl_mg":        _co2_rl_p,
                    "nox_rl_mg":        _nox_rl_p,
                    "fuel_rl_mg":       _fuel_rl_p,
                    "co2_fx_mg":        round(_co2_fx_p,  2),
                    "nox_fx_mg":        round(_nox_fx_p,  2),
                    "fuel_fx_mg":       round(_fuel_fx_p, 2),
                    "co2_saving_pct":   _ep_pct(_co2_rl_p,  _co2_fx_p),
                    "nox_saving_pct":   _ep_pct(_nox_rl_p,  _nox_fx_p),
                    "fuel_saving_pct":  _ep_pct(_fuel_rl_p, _fuel_fx_p),
                    "baseline_is_cached": True,
                })

            if tb_logger is not None:
                tb_logger.log_episode_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss)

            FT_REFRESH_INTERVAL = 50
            if (fixed_controllers and
                    (ep % FT_REFRESH_INTERVAL == 0) and
                    ep > 0):
                _ft_seed_label_p = (f"seed={base_seed} (fixed)"
                                    if fixed_seed else "random seed")
                if fixed_seed and seed_enable:
                    _par_ft_cmd = sumo_cmd + ["--seed", str(base_seed)]
                else:
                    _par_ft_cmd = sumo_cmd
                try:
                    _fx_port_par = BASE_PORT + 702
                    _run_result_par = _run_fixed_time_episode(
                        _par_ft_cmd, max(900, sim_max_steps),
                        _fx_port_par, fixed_controllers)
                    cached_fx_result = {}
                    for _tl in [k for k in _run_result_par if not k.startswith("__")]:
                        cached_fx_result[_tl] = _run_result_par[_tl]
                    cached_fx_result["__ev_cleared"]     = _run_result_par.get("__ev_cleared", 0)
                    cached_fx_result["__ev_avg_wait"]    = _run_result_par.get("__ev_avg_wait", None)
                    cached_fx_result["__cars_passed"]    = _run_result_par.get("__cars_passed", 0)
                    cached_fx_result["__co2_total"]      = _run_result_par.get("__co2_total",  0.0)
                    cached_fx_result["__nox_total"]      = _run_result_par.get("__nox_total",  0.0)
                    cached_fx_result["__fuel_total"]     = _run_result_par.get("__fuel_total", 0.0)
                    cached_fx_result["__ft_baseline_ep"]   = ep
                    _log(f"[Ep {ep}] Fixed-Time baseline refreshed ({_ft_seed_label_p})")
                except Exception as _ref_e:
                    _log(f"[Ep {ep}] WARNING: FT refresh failed (parallel): {_ref_e}")

            fx_reward = {tl: 0.0 for tl in INTERSECTIONS}
            fx_queue  = {tl: 0.0 for tl in INTERSECTIONS}
            fx_wait   = {tl: 0.0 for tl in INTERSECTIONS}
            if fixed_controllers and cached_fx_result:
                for tl, data in cached_fx_result.items():
                    if tl.startswith("__"):
                        continue
                    fx_reward[tl] = float(np.mean(data["rewards"])) if data["rewards"] else 0.0
                    fx_queue[tl]  = float(np.mean(data["queues"]))  if data["queues"]  else 0.0
                    fx_wait[tl]   = float(np.mean(data["waits"]))   if data["waits"]   else 0.0

            if tb_logger is not None:
                tb_logger.log_episode_fixed(ep, fixed_controllers)

            _rl_ep_stats = _snapshot_ep_stats()
            if _rl_ep_stats["ev_cleared"] > 0:
                _rl_ev_avg_par = round(_rl_ep_stats["ev_wait_sum"] / _rl_ep_stats["ev_cleared"], 3)
            else:
                _rl_ev_avg_par = None
            if rl_history is not None:
                for tl in INTERSECTIONS:
                    rl_history["reward"][tl].append(ep_reward.get(tl, 0.0))
                    rl_history["queue"][tl].append(ep_queue.get(tl, 0.0))
                    rl_history["wait"][tl].append(ep_wait.get(tl, 0.0))
                rl_history["ev_cleared"].append(_rl_ep_stats["ev_cleared"])
                rl_history["ev_avg_wait"].append(_rl_ev_avg_par)
                rl_history["cars_passed"].append(_rl_ep_stats["cars_passed"])
                # Emission totals for this RL episode
                rl_history["co2_total"].append(round(_rl_ep_stats["co2_sum"],  2))
                rl_history["nox_total"].append(round(_rl_ep_stats["nox_sum"],  2))
                rl_history["fuel_total"].append(round(_rl_ep_stats["fuel_sum"], 2))
            if fixed_history is not None:
                for tl in INTERSECTIONS:
                    fixed_history["reward"][tl].append(fx_reward.get(tl, 0.0))
                    fixed_history["queue"][tl].append(fx_queue.get(tl, 0.0))
                    fixed_history["wait"][tl].append(fx_wait.get(tl, 0.0))
                fixed_history["ev_cleared"].append(
                    cached_fx_result.get("__ev_cleared", 0) if cached_fx_result else 0)
                _fx_ev_avg_par = cached_fx_result.get("__ev_avg_wait", None) if cached_fx_result else None
                fixed_history["ev_avg_wait"].append(_fx_ev_avg_par)
                fixed_history["cars_passed"].append(
                    cached_fx_result.get("__cars_passed", 0) if cached_fx_result else 0)
                # Emission totals from cached FT baseline
                fixed_history["co2_total"].append(
                    cached_fx_result.get("__co2_total",  0.0) if cached_fx_result else 0.0)
                fixed_history["nox_total"].append(
                    cached_fx_result.get("__nox_total",  0.0) if cached_fx_result else 0.0)
                fixed_history["fuel_total"].append(
                    cached_fx_result.get("__fuel_total", 0.0) if cached_fx_result else 0.0)
                fixed_history.setdefault("ft_baseline_ep", []).append(
                    cached_fx_result.get("__ft_baseline_ep", 0) if cached_fx_result else 0)

            if _ui:
                _ui.record_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                              ep_stats=_snapshot_ep_stats())
                if fixed_controllers and cached_fx_result:
                    _ui.record_fixed(fx_reward, fx_queue, fx_wait,
                                     fx_stats=cached_fx_result)

            auto_saver.save_episode(
                ep,
                ep_reward, ep_queue, ep_wait, ep_loss,
                fx_reward, fx_queue, fx_wait,
            )
            if ep % auto_saver.save_every == 0 or ep == n_episodes - 1:
                auto_saver.tag_latest(ep)

        _log(f"[Batch {batch_no}] Done. "
             f"avg_r={float(np.mean([v for vals in batch_rewards.values() for v in vals])):+.4f}",
             style="dim cyan")

    _log("Parallel training loop complete.", style="bold green")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SmartTraffic-OM — Traffic Signal Control Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Shared arguments (identical for both algorithms) ─────────────────────────
    parser.add_argument(
        "--sim_file", required=True,
        help="Path to the SUMO .sumocfg simulation config file",
    )
    parser.add_argument(
        "--n_episodes", type=int, default=300,
        help="Total number of training episodes",
    )
    parser.add_argument(
        "--max_steps", type=int, default=3600,
        help="Maximum simulation steps per episode",
    )
    parser.add_argument(
        "--gui", choices=["enable", "disable"], default="disable",
        help="Enable or disable SUMO GUI (sumo-gui vs sumo)",
    )
    parser.add_argument(
        "--save_dir", default="./results",
        help="Directory to save checkpoints, plots, and results",
    )
    parser.add_argument(
        "--resume_from", default=None,
        help="Checkpoint tag to resume from, e.g. 'ep0050' or 'latest'",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of parallel SUMO workers (>1 uses parallel mode; forced to 1 on Windows)",
    )
    parser.add_argument(
        "--save_every", type=int, default=10,
        help="Save checkpoint, plots, and Excel every N episodes",
    )
    parser.add_argument(
        "--reward",
        choices=["backpressure", "waiting_time", "queue", "max_pressure"],
        default="backpressure",
        help="Reward function: backpressure (default), waiting_time, queue, or max_pressure (MaxPressure per active phase)",
    )
    parser.add_argument(
        "--ev_priority",
        choices=["default", "none"],
        default="default",
        help="EV priority module: default (full EV logic) or none (disabled)",
    )

    # ── MAPPO+GAT specific ───────────────────────────────────────────
    parser.add_argument(
        "--aggregator",
        choices=["gat", "flat_gat", "mean", "none"],
        default="gat",
        help=(
            "Intent aggregator (mappo_gat only, ignored by flat_gat and ippo): "
            "gat (default) — hierarchical GAT with local clusters + inter-cluster attention; "
            "flat_gat — single-level GATConv over all agents via NEIGHBOR_MAP (no hierarchy); "
            "mean — non-parametric mean of neighbour intents; "
            "none — independent agents, no intent sharing"
        ),
    )
    parser.add_argument(
        "--adaptive_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable KL-adaptive learning rate controller (default: on). "
            "When enabled, actor/critic/GAT learning rates are scaled up or down "
            "after every mini-batch based on the observed KL divergence so the "
            "policy never drifts too far or too little per update. "
            "Use --no-adaptive_lr for fixed LRs throughout training."
        ),
    )
    parser.add_argument(
        "--adaptive_clip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable KL-adaptive clip ratio controller (default: on). "
            "When enabled, the PPO clip ratio is scaled up or down "
            "after every mini-batch based on observed KL divergence. "
            "High KL -> tighter clipping; low KL -> looser clipping. "
            "Use --no-adaptive_clip for fixed clip ratio throughout training."
        ),
    )

    # ── Algorithm selector ─────────────────────────────────────────────
    parser.add_argument(
        "--algorithm",
        choices=["mappo_gat", "flat_gat", "ippo", "vanilla_ppo"],
        default="mappo_gat",
        help=(
            "Training algorithm: "
            "mappo_gat (default) — full MAPPO+HierGAT intent sharing; "
            "flat_gat — MAPPO+GAT without the hierarchical architecture (single-level GATConv over all agents); "
            "ippo — independent PPO without intent sharing (baseline); "
            "vanilla_ppo — absolute simple single-agent vanilla PPO (simplest baseline)"
        ),
    )

    # ── Reproducibility ──────────────────────────────────────────────
    parser.add_argument(
        "--seed_enable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable deterministic per-episode seeding for reproducibility (default: disabled)",
    )
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base seed for deterministic seeding (used when --seed_enable is set) (default: 42)",
    )
    parser.add_argument(
        "--fixed_seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the same seed for every episode (default: disabled). "
            "When enabled, --base_seed is used directly as the seed for all "
            "episodes and SUMO, ensuring fully reproducible runs. "
            "When disabled with --seed_enable, the seed increments per episode "
            "(base_seed + ep). Ignored when --seed_enable is not set."
        ),
    )

    # ── CUDA / performance ────────────────────────────────────
    parser.add_argument(
        "--cuda_dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable CUDA dynamic optimisations (cudnn.benchmark, TF32). "
             "Disable for stable/debug runs. (default: enabled)",
    )

    # ── Resume behaviour ──────────────────────────────────────
    parser.add_argument(
        "--reset_lr",
        action="store_true",
        default=False,
        help="Reset learning rate to initial value when resuming from a checkpoint",
    )

    # ── IPPO specific ─────────────────────────────────────────
    parser.add_argument(
        "--ippo_critic",
        choices=["local", "centralised"],
        default="local",
        help=(
            "Critic architecture for ippo: "
            "local (per-agent, fast) or centralised (shared over all agents). "
            "Ignored when --algorithm=mappo_gat or flat_gat."
        ),
    )

    args = parser.parse_args()

    # --fixed_seed implies --seed_enable: a user requesting a fixed seed
    # clearly wants reproducible seeding, so enable it automatically.
    if args.fixed_seed and not args.seed_enable:
        args.seed_enable = True
        print("[INFO] --fixed_seed implies --seed_enable; "
              "enabling deterministic seeding.", flush=True)

    # ── Fix PYTHONHASHSEED before any hash-based data structures ───
    if args.seed_enable:
        os.environ.setdefault("PYTHONHASHSEED", str(args.base_seed))

    # ── Dispatch ────────────────────────────────────────────────────────
    if args.algorithm == "ippo":
        # save_dir gets a _ippo suffix so results never collide with MAPPO+GAT
        # runs that use the same base --save_dir. If the user already chose a
        # save_dir that identifies itself as vanilla-PPO-specific (e.g.
        # "./results_ippo", "./results_ippo_central"), we leave it untouched
        # rather than producing an ugly double suffix.
        import re as _re
        _base_name = os.path.basename(args.save_dir.rstrip("/\\"))
        _already_ippo = bool(_re.search(r"(?:^|_)ippo(?:_|$)", _base_name))
        _ippo_save_dir = (args.save_dir if _already_ippo
                          else args.save_dir.rstrip("/\\") + "_ippo")
        print(f"[INFO] Algorithm: IPPO  (critic={args.ippo_critic}, "
              f"save_dir={_ippo_save_dir})", flush=True)
        run_training_ippo(
            sim_file          = args.sim_file,
            n_episodes        = args.n_episodes,
            sim_max_steps     = args.max_steps,
            gui               = args.gui,
            save_dir          = _ippo_save_dir,
            resume_from       = args.resume_from,
            n_workers         = args.n_workers,
            save_every        = args.save_every,
            reward_type       = args.reward,
            ev_priority_type  = args.ev_priority,
            critic_mode       = args.ippo_critic,
            adaptive_clip     = args.adaptive_clip,
            seed_enable       = args.seed_enable,
            base_seed         = args.base_seed,
            fixed_seed        = args.fixed_seed,
            reset_lr          = args.reset_lr,
            cuda_dynamic      = args.cuda_dynamic,
        )
    elif args.algorithm == "vanilla_ppo":
        import re as _re
        _base_name = os.path.basename(args.save_dir.rstrip("/\\"))
        _already_vpp = bool(_re.search(r"(?:^|_)vanilla_ppo(?:_|$)", _base_name))
        _vpp_save_dir = (args.save_dir if _already_vpp
                         else args.save_dir.rstrip("/\\") + "_vanilla_ppo")
        print(f"[INFO] Algorithm: Vanilla PPO  (save_dir={_vpp_save_dir})",
              flush=True)
        run_training_vanilla_ppo(
            sim_file          = args.sim_file,
            n_episodes        = args.n_episodes,
            sim_max_steps     = args.max_steps,
            gui               = args.gui,
            save_dir          = _vpp_save_dir,
            resume_from       = args.resume_from,
            n_workers         = args.n_workers,
            save_every        = args.save_every,
            reward_type       = args.reward,
            ev_priority_type  = args.ev_priority,
            seed_enable       = args.seed_enable,
            base_seed         = args.base_seed,
            fixed_seed        = args.fixed_seed,
            reset_lr          = args.reset_lr,
            cuda_dynamic      = args.cuda_dynamic,
        )
    elif args.algorithm == "flat_gat":
        # [FLAT-GAT] Full MAPPO+GAT pipeline with a single-level GATConv
        # aggregator instead of the hierarchical one.  Everything else
        # (actor/critic, PIS, EV priority, checkpointing) is unchanged.
        # save_dir gets a _flatgat suffix so results never collide with
        # HierGAT runs that share the same base directory.
        import re as _re
        _base_name    = os.path.basename(args.save_dir.rstrip("/\\"))
        _already_flat = bool(_re.search(r"(?:^|_)flatgat(?:_|$)", _base_name))
        _flat_save_dir = (args.save_dir if _already_flat
                          else args.save_dir.rstrip("/\\") + "_flatgat")
        print(f"[INFO] Algorithm: Flat-GAT  (save_dir={_flat_save_dir})",
              flush=True)
        run_training(
            sim_file         = args.sim_file,
            n_episodes       = args.n_episodes,
            sim_max_steps    = args.max_steps,
            gui              = args.gui,
            save_dir         = _flat_save_dir,
            resume_from      = args.resume_from,
            n_workers        = args.n_workers,
            save_every       = args.save_every,
            reward_type      = args.reward,
            aggregator_type  = "flat_gat",   # always flat; ignores --aggregator
            ev_priority_type = args.ev_priority,
            adaptive_lr      = args.adaptive_lr,
            adaptive_clip    = args.adaptive_clip,
            seed_enable      = args.seed_enable,
            base_seed        = args.base_seed,
            fixed_seed       = args.fixed_seed,
            reset_lr         = args.reset_lr,
            cuda_dynamic     = args.cuda_dynamic,
        )
    else:
        # [UNCHANGED] MAPPO+HierGAT — identical to the original main()
        print(f"[INFO] Algorithm: MAPPO+GAT  (aggregator={args.aggregator}, "
              f"save_dir={args.save_dir})", flush=True)
        run_training(
            sim_file             = args.sim_file,
            n_episodes           = args.n_episodes,
            sim_max_steps        = args.max_steps,
            gui                  = args.gui,
            save_dir             = args.save_dir,
            resume_from          = args.resume_from,
            n_workers            = args.n_workers,
            save_every           = args.save_every,
            reward_type          = args.reward,
            aggregator_type      = args.aggregator,
            ev_priority_type     = args.ev_priority,
            adaptive_lr          = args.adaptive_lr,
            adaptive_clip        = args.adaptive_clip,
            seed_enable          = args.seed_enable,
            base_seed            = args.base_seed,
            fixed_seed           = args.fixed_seed,
            reset_lr             = args.reset_lr,
            cuda_dynamic         = args.cuda_dynamic,
        )

if __name__ == "__main__":
    main()