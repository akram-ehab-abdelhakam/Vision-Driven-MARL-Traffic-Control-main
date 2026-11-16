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
from .rewards import BaseRewardFunction, BackPressureReward
from .intent_aggregators import BaseIntentAggregator, GATIntentAggregator as _GATAggregatorCls
from .ev_priority import BaseEVPriorityModule, DefaultEVPriority
from .kl_adaptive_lr import (
    KLAdaptiveLRController,
    KLAdaptiveClipController,
    build_kl_adaptive_optimizer,
    compute_kl_divergence,
    compute_kl_from_logits,
    KL_TARGET_DEFAULT,
    LR_MIN, LR_MAX,
    LR_DECREASE_FACTOR, LR_INCREASE_FACTOR,
)

__all__ = ['RolloutBuffer', 'MAPPOAgent', '_broadcast_intents', '_build_global_intent_vec',
           'KLAdaptiveLRController', 'KLAdaptiveClipController', 'compute_kl_divergence',
           'compute_kl_from_logits', 'build_kl_adaptive_optimizer']

# Module-level flag: once we discover torch.compile/Triton is broken on this
# machine (e.g. WSL2 GPU-virtualization + Triton autotune failures), we stop
# retrying compilation for every subsequent agent in this process. Without
# this, N intersections == N slow failed compile attempts on every run.
_COMPILE_BROKEN_ON_THIS_MACHINE = False


def _try_compile_with_fallback(actor, critic, state_dim, intent_dim, device):
    """
    Wrap actor/critic with torch.compile and *actually* verify it works by
    running one real forward pass through each (compilation + Triton
    autotuning is lazy and only happens on first call — a bare
    `torch.compile(model)` never raises, so wrapping it in try/except like
    the old code did was not a real safety net).

    If the warm-up forward pass throws (e.g. `cudaErrorUnknown` from Triton
    autotuning on WSL2), we discard the compiled wrapper and fall back to
    plain eager-mode modules so training can proceed at full n_workers
    throughput instead of crashing mid-run.
    """
    global _COMPILE_BROKEN_ON_THIS_MACHINE
    if _COMPILE_BROKEN_ON_THIS_MACHINE:
        return actor, critic

    n_agents = len(INTERSECTIONS)
    compiled_actor  = torch.compile(actor,  mode="default")
    compiled_critic = torch.compile(critic, mode="default")

    try:
        with torch.no_grad():
            dummy_state  = torch.zeros(2, state_dim, device=device)
            dummy_intent = torch.zeros(2, intent_dim, device=device)
            dummy_global = torch.zeros(2, intent_dim * n_agents, device=device)
            compiled_actor(dummy_state, dummy_intent)
            compiled_critic(dummy_state, dummy_global)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return compiled_actor, compiled_critic
    except Exception as exc:
        _COMPILE_BROKEN_ON_THIS_MACHINE = True
        print(f"[torch.compile] Warm-up failed ({exc.__class__.__name__}: {exc}). "
              f"Falling back to eager mode for this and all subsequent agents.",
              flush=True)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return actor, critic


class RolloutBuffer:
    def __init__(self, capacity: int, state_dim: int, intent_dim: int):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.intent_dim = intent_dim
        n_agents        = len(INTERSECTIONS)

        self.states         = np.zeros((capacity, state_dim),              dtype=np.float32)
        self.intents        = np.zeros((capacity, intent_dim),             dtype=np.float32)
        self.global_intents = np.zeros((capacity, intent_dim * n_agents),  dtype=np.float32)
        self.actions        = np.zeros(capacity,                            dtype=np.int64)
        self.rewards        = np.zeros(capacity,                            dtype=np.float32)
        self.values         = np.zeros(capacity,                            dtype=np.float32)
        self.log_probs      = np.zeros(capacity,                            dtype=np.float32)
        self.dones          = np.zeros(capacity,                            dtype=np.float32)
        self.next_values    = np.zeros(capacity,                            dtype=np.float32)
        self.next_intents   = np.zeros((capacity, intent_dim),             dtype=np.float32)
        self.size = 0

    def push(self, state, intent, action, reward, value, log_prob, done, next_intent,
             next_value: float = 0.0, global_intent: "np.ndarray | None" = None):
        idx = self.size % self.capacity
        self.states[idx]       = state
        self.intents[idx]      = intent
        self.actions[idx]      = action
        self.rewards[idx]      = reward
        self.values[idx]       = value
        self.log_probs[idx]    = log_prob
        self.dones[idx]        = done
        self.next_intents[idx] = next_intent
        self.next_values[idx]  = next_value
        if global_intent is not None:
            self.global_intents[idx] = global_intent
        else:
            self.global_intents[idx] = np.tile(intent, len(INTERSECTIONS))
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, gamma: float = GAMMA, lam: float = GAE_LAMBDA):
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            next_value = self.next_values[t] * (1 - self.dones[t])
            delta = self.rewards[t] + gamma * next_value - self.values[t]
            if self.dones[t]:
                gae = delta
            else:
                gae = delta + gamma * lam * gae
            advantages[t] = gae
        returns = advantages + self.values[:self.size]
        # FIX-B (audit): Advantage normalisation is strictly PER-AGENT.
        # Each MAPPOAgent owns its own RolloutBuffer, which is populated
        # exclusively with that agent's transitions.  No pooling or sharing
        # of advantages across agents occurs — mean/std are computed over
        # this single agent's rollout only.  This is the correct MAPPO
        # design: each agent normalises its own advantage signal so that
        # gradient magnitudes are comparable across agents with different
        # reward scales (important after FIX-A switched to per-TLS capacity).
        std = advantages.std()
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        return advantages, returns

    def sample(self, batch_size: int):
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        return (
            self.states[indices],
            self.intents[indices],
            self.global_intents[indices],
            self.actions[indices],
            self.rewards[indices],
            self.values[indices],
            self.log_probs[indices],
            self.dones[indices],
            indices,
        )

    def clear(self):
        self.size = 0

class MAPPOAgent:
    _lane_length_cache: dict[str, float] = {}

    @property
    def clip_ratio(self) -> float:
        return self._clip_holder[0]

    def __init__(
        self,
        tls_id: str,
        device: torch.device,
        reward_func: "BaseRewardFunction | None" = None,
        intent_aggregator: "BaseIntentAggregator | None" = None,
        ev_module: "BaseEVPriorityModule | None" = None,
        use_adaptive_lr: bool = True,
        use_adaptive_clip: bool = True,
    ):
        self.tls_id    = tls_id
        self.device    = device
        self._use_cuda = device.type == "cuda"

        self.learning_rate   = LEARNING_RATE
        self.gamma           = GAMMA
        self.gae_lambda      = GAE_LAMBDA
        self._clip_holder    = [CLIP_RATIO]  # mutable container for adaptive clip control
        self.entropy_coeff   = ENTROPY_COEFF
        self.value_loss_coeff = VALUE_LOSS_COEFF
        self.n_epochs        = N_EPOCHS
        self.batch_size      = BATCH_SIZE
        self.min_green_steps = MIN_GREEN_STEPS

        print(f"  [{tls_id}] lr={self.learning_rate}  gamma={self.gamma}"
              f"  gae_lambda={self.gae_lambda}  clip={self.clip_ratio}"
              f"  entropy={self.entropy_coeff}")

        self.incoming_lanes, self.outgoing_lanes = _get_lanes_for_tls(tls_id)
        self.action_phases, self.phase_links     = _get_green_phases_and_links(tls_id)

        if not self.action_phases:
            raise RuntimeError(f"No valid green phases for TLS '{tls_id}'.")

        self.action_dim = len(self.action_phases)

        n_in  = len(self.incoming_lanes)
        n_out = len(self.outgoing_lanes)
        n_dets = len(TLS_DETECTOR_MAP.get(tls_id, []))
        # EV feature block size: 2·n_in (count + priority per lane) + 2 (total + closest)
        self._ev_feat_dim = (2 * n_in + 2) if EV_PRIORITY_ENABLED else 0
        self.state_dim = (
            self.action_dim          +
            2 * (n_in + n_out)       +
            n_dets                   +
            n_in                     +
            1                        +  # time-since-switch
            1                        +  # normalised total queue
            2                        +  # shock features: mean_v_shock + min_v_shock
            self._ev_feat_dim           # EV priority features (0 when disabled)
        )
        self.intent_dim = INTENT_SHARING_DIM
        # EV potential cache (updated in _get_state, consumed in get_reward)
        self._ev_phi_prev: float = 0.0
        self._ev_phi_curr: float = 0.0

        # -- Injectable experiment modules ------------------------------------
        # Defaults preserve identical behaviour to the original hardcoded code.
        self.reward_func:       BaseRewardFunction    = reward_func       or BackPressureReward()
        self.ev_module:         BaseEVPriorityModule  = ev_module         or DefaultEVPriority()
        self._use_adaptive_lr:   bool = use_adaptive_lr
        self._use_adaptive_clip: bool = use_adaptive_clip
        # intent_aggregator is stored after the GAT optimiser is set up below.

        self.actor  = ActorNetwork(self.state_dim, self.action_dim, self.intent_dim).to(device)
        self.critic = CriticNetwork(self.state_dim, self.intent_dim).to(device)

        # -- GPU optimisations ----------------------------------------------
        # torch.compile requires Triton which is not available on Windows.
        # Only attempt compilation on non-Windows platforms.
        import platform as _plat_agent
        if self._use_cuda and _plat_agent.system() != "Windows":
            self.actor, self.critic = _try_compile_with_fallback(
                self.actor, self.critic, self.state_dim, self.intent_dim, device)
        if self._use_cuda:
            self._scaler = torch.amp.GradScaler(device="cuda")
        else:
            self._scaler = None

        # -- ONNX inference session (populated lazily after first export) ---
        self._onnx_actor_session  = None
        self._onnx_export_done    = False
        # BUGFIX (policy-collapse @ export cadence): export_onnx() snapshots
        # the actor and, once done, choose_action() would silently switch to
        # that frozen snapshot for every future action -- even though
        # self.actor keeps training live in the background. Every re-export
        # (every `onnx_export_interval` episodes) then dumped N episodes of
        # accumulated gradient drift into behaviour in one discontinuous
        # jump, which looked like periodic policy collapse.
        # `use_onnx_for_training` gates that swap: it must stay False for
        # the whole training run so choose_action() always uses the live
        # PyTorch actor. Only a genuine post-training inference/deployment
        # path should ever set this True.
        self.use_onnx_for_training = False

        # -- KL-Adaptive per-group optimizers --------------------------------
        # Three distinct learning rates:
        #   actor_lr  = KL_ACTOR_LR  (3e-4) — policy head (medium LR)
        #   critic_lr = KL_CRITIC_LR (1e-3) — centralized critic (higher LR)
        #   gat_lr    = KL_GAT_LR    (1e-4) — intent aggregator (lower LR)
        # These are stored on the optimizer param_groups so the KL controller
        # can scale them in-place after every mini-batch update.
        self.actor_optimizer  = optim.Adam(
            self.actor.parameters(), lr=KL_ACTOR_LR)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=KL_CRITIC_LR)

        self._actor_scheduler:  "optim.lr_scheduler.CosineAnnealingLR | None" = None
        self._critic_scheduler: "optim.lr_scheduler.CosineAnnealingLR | None" = None

        # -- KL-Adaptive LR controller (wired AFTER gat_optimizer is built) --
        # Initialised with a placeholder here; replaced in _setup_kl_controller()
        # once gat_optimizer exists.  The None guard in train_epoch / update_gat
        # prevents any adjustment before the controller is ready.
        self._kl_controller: "KLAdaptiveLRController | None" = None
        # -- KL-Adaptive Clip controller (wired alongside LR controller) --
        self._clip_controller: "KLAdaptiveClipController | None" = None
        # Running EMA of per-batch KL for diagnostics / TensorBoard logging
        self._kl_ema: float = 0.0
        self._kl_ema_alpha: float = 0.1   # smoothing factor
        # Per-episode mean_kl / KL-controller action counts, set by train_step()
        # after each train_epoch() call. Defaults let readers (e.g. a CSV
        # logger in training_loop.py) safely check them before the first
        # training step has happened (buffer not full yet, etc.).
        self.last_mean_kl: float = 0.0
        self.last_kl_action_counts: "dict | None" = None
        # Diagnostic-only distributional KL (see compute_kl_from_logits usage
        # in train_epoch). Never read by any controller / optimizer logic.
        self.last_true_kl: float = 0.0

        self._entropy_history: list[float] = []
        self._entropy_boost_remaining: int  = 0      # episodes left in boost window
        self._base_entropy_coeff: float     = self.entropy_coeff  # saved for restore

        # -- Intent aggregator (injectable) ----------------------------------
        # Default: GATIntentAggregator (original behaviour).
        # Accepts any BaseIntentAggregator subclass for ablation studies.
        if intent_aggregator is not None:
            self.intent_aggregator: BaseIntentAggregator = intent_aggregator
        else:
            self.intent_aggregator = _GATAggregatorCls(self.intent_dim)
        self.intent_aggregator.to(device)

        # Keep gat_aggregator as an alias so checkpointing code and any
        # external callers that reference agent.gat_aggregator still work.
        self.gat_aggregator = self.intent_aggregator

        # Build the GAT optimiser over whatever parameters the aggregator exposes.
        # Non-parametric aggregators (Mean, NoSharing) return an empty iterator
        # so Adam gets a dummy param list -- that is safe: it simply does nothing.
        # KL-Adaptive: GAT uses KL_GAT_LR (1e-4), lower than actor/critic.
        _agg_params = list(self.intent_aggregator.parameters())
        self.gat_optimizer = optim.Adam(
            _agg_params if _agg_params else [torch.zeros(1, requires_grad=True)],
            lr=KL_GAT_LR,
        )

        # -- Finalise KL-Adaptive controllers now that all three optimizers exist --
        self._setup_kl_controller()
        self._setup_kl_clip_controller()

        self.rollout_buffer = RolloutBuffer(MEMORY_CAPACITY, self.state_dim, self.intent_dim)

        self.current_action_idx  = 0
        self.last_switch_step    = -self.min_green_steps
        self._current_step       = 0
        self.state               = self._get_state()
        self.current_intent      = np.zeros(self.intent_dim, dtype=np.float32)
        self.neighbor_intents: Dict[str, np.ndarray] = {}

        self.step_rewards: list[float] = []
        self.step_queues:  list[float] = []
        self.step_losses:  list[float] = []
        self.step_waits:   list[float] = []
        self._cached_agg_intent: np.ndarray = np.zeros(self.intent_dim, dtype=np.float32)
        self._agg_intent_step: int = -GAT_UPDATE_INTERVAL  # force compute on first step

        # Pinned-memory tensors for zero-copy H->D transfers on CUDA.
        # On Windows, pin_memory() can raise if the CUDA driver is not fully
        # initialised yet -- fall back to regular tensors in that case.
        if self._use_cuda:
            try:
                self._state_tensor  = torch.zeros(
                    1, self.state_dim,  dtype=torch.float32).pin_memory()
                self._intent_tensor = torch.zeros(
                    1, self.intent_dim, dtype=torch.float32).pin_memory()
                self._pinned = True
            except Exception:
                self._state_tensor  = torch.zeros(1, self.state_dim,  dtype=torch.float32)
                self._intent_tensor = torch.zeros(1, self.intent_dim, dtype=torch.float32)
                self._pinned = False
            self._state_gpu  = torch.zeros(
                1, self.state_dim,  dtype=torch.float32, device=device)
            self._intent_gpu = torch.zeros(
                1, self.intent_dim, dtype=torch.float32, device=device)
        else:
            self._pinned        = False
            self._state_tensor  = torch.zeros(1, self.state_dim,  dtype=torch.float32, device=device)
            self._intent_tensor = torch.zeros(1, self.intent_dim, dtype=torch.float32, device=device)
            self._state_gpu     = self._state_tensor
            self._intent_gpu    = self._intent_tensor

    @classmethod
    def _get_lane_length(cls, lane_id: str) -> float:
        v = cls._lane_length_cache.get(lane_id)
        if v is None:
            v = max(traci.lane.getLength(lane_id), 1.0)
            cls._lane_length_cache[lane_id] = v
        return v

    @classmethod
    def clear_lane_length_cache(cls):
        cls._lane_length_cache.clear()

    def _get_state(self) -> np.ndarray:
        phase_onehot = np.zeros(self.action_dim, dtype=np.float32)
        phase_onehot[self.current_action_idx] = 1.0
        values = list(phase_onehot)

        for lane_id in self.incoming_lanes:
            m, s = _moving_stopped_counts(lane_id)
            values.extend([float(m), float(s)])
        for lane_id in self.outgoing_lanes:
            m, s = _moving_stopped_counts(lane_id)
            values.extend([float(m), float(s)])

        dets = TLS_DETECTOR_MAP.get(self.tls_id, [])
        for det_id in dets:
            values.append(_STEP_CACHE.detector_halting(det_id))

        for lane_id in self.incoming_lanes:
            values.append(_lane_queue(lane_id))

        steps_since_switch = float(self._current_step - self.last_switch_step)
        values.append(min(steps_since_switch / 100.0, 1.0))

        total_queue = _get_tls_queue_from_detectors(self.tls_id)
        qcap = _QUEUE_CAPACITY_PER_TLS.get(self.tls_id, QUEUE_CAPACITY)
        values.append(min(total_queue / max(qcap, 1.0), 1.0))

        # -- Shock-wave features (Rankine-Hugoniot) ----------------------------
        # For each adjacent incoming-outgoing lane pair, compute the shock speed.
        # Negative v_shock = upstream-propagating queue (most dangerous for TL1).
        # Normalise by V_FREE so features stay in [-1, 1].
        in_list  = list(self.incoming_lanes)
        out_list = list(self.outgoing_lanes)
        shock_sum  = 0.0
        shock_min  = 0.0   # most negative = worst upstream propagation
        n_pairs    = max(len(in_list), 1)
        for k, in_l in enumerate(in_list):
            out_l = out_list[k % len(out_list)] if out_list else None
            if out_l:
                vs = _shock_wave_speed(in_l, out_l)
                shock_sum += vs
                shock_min  = min(shock_min, vs)
        values.append(float(shock_sum / n_pairs) / max(V_FREE, 1.0))   # mean shock speed
        values.append(float(shock_min)            / max(V_FREE, 1.0))   # worst upstream shock

        # -- EV priority features -------------------------------------------------
        # _get_ev_features returns zeros when EV_PRIORITY_ENABLED=False so the
        # state vector length stays consistent with self._ev_feat_dim.
        if self._ev_feat_dim > 0:
            ev_feats = self.ev_module.get_ev_features(self.tls_id, list(self.incoming_lanes))
            values.extend(ev_feats.tolist())
            # Refresh potential cache so get_reward() can apply shaping:
            #   _ev_phi_prev = Φ(s_t)   (set from the previous _get_state call)
            #   _ev_phi_curr = Φ(s_t+1) (just computed from current SUMO state)
            self._ev_phi_prev = self._ev_phi_curr
            self._ev_phi_curr = self.ev_module.get_ev_potential(
                list(self.incoming_lanes),
                self.current_action_idx,
                self.action_phases,
            )

        return np.array(values, dtype=np.float32)

    def _build_intent_matrix(self) -> torch.Tensor:
        n      = len(INTERSECTIONS)
        mat    = torch.zeros(n, self.intent_dim, dtype=torch.float32)
        my_idx = _TLS_INDEX[self.tls_id]
        mat[my_idx] = torch.from_numpy(self.current_intent)
        for nb_id, nb_intent in self.neighbor_intents.items():
            if nb_id in _TLS_INDEX:
                mat[_TLS_INDEX[nb_id]] = torch.from_numpy(nb_intent)
        return mat

    def aggregate_neighbor_intents(self, current_step: int = -1) -> np.ndarray:
        if not self.neighbor_intents:
            return np.zeros(self.intent_dim, dtype=np.float32)
        # Only recompute GAT every GAT_UPDATE_INTERVAL steps
        if current_step >= 0 and (current_step - self._agg_intent_step) < GAT_UPDATE_INTERVAL:
            return self._cached_agg_intent
        mat     = self._build_intent_matrix().to(self.device)
        with torch.no_grad():
            out = self.intent_aggregator(mat)
        my_idx = _TLS_INDEX[self.tls_id]
        result = out[my_idx].cpu().numpy().astype(np.float32)
        self._cached_agg_intent = result
        self._agg_intent_step   = current_step
        return result

    def update_gat(self, intent_matrix: torch.Tensor,
                   target_matrix: torch.Tensor) -> float:
        # Check if the aggregator actually has trainable parameters.
        # Non-parametric aggregators (Mean, NoSharing) should not call .backward().
        agg_params = list(self.intent_aggregator.parameters())
        if not agg_params:
            with torch.no_grad():
                pred = self.intent_aggregator(intent_matrix.to(self.device))
                loss = nn.functional.mse_loss(pred, target_matrix.to(self.device))
            return float(loss.item())

        self.gat_optimizer.zero_grad(set_to_none=True)
        pred = self.intent_aggregator(intent_matrix.to(self.device))
        loss = nn.functional.mse_loss(pred, target_matrix.to(self.device))
        loss.backward()
        nn.utils.clip_grad_norm_(agg_params, 0.5)
        self.gat_optimizer.step()
        return float(loss.item())


    def _setup_kl_controller(self) -> None:
        """Wire the KLAdaptiveLRController over all three optimizers.

        Called once from __init__ after actor_optimizer, critic_optimizer,
        and gat_optimizer are all created.  Safe to call again on resume
        (re-creates the controller without resetting optimizer states).

        When ``use_adaptive_lr=False`` the controller is left as ``None``
        and the KL block inside ``train_epoch`` is skipped entirely —
        learning rates stay fixed at their initial values for the whole run.
        """
        if not self._use_adaptive_lr:
            self._kl_controller = None
            return
        self._kl_controller = KLAdaptiveLRController(
            optimizers=[
                (self.actor_optimizer,  [0]),   # actor LR
                (self.critic_optimizer, [0]),   # critic LR
                (self.gat_optimizer,    [0]),   # GAT LR
            ],
            kl_target       = KL_TARGET,
            lr_min          = KL_LR_MIN,
            lr_max          = KL_LR_MAX,
            decrease_factor = KL_LR_DOWN,
            increase_factor = KL_LR_UP,
            verbose         = False,
        )

    def _setup_kl_clip_controller(self) -> None:
        """Wire the KLAdaptiveClipController to adjust clip_ratio based on KL.

        Called once from __init__ after ``_setup_kl_controller()``.
        When ``use_adaptive_clip=False`` the controller is left as ``None``
        and the clip ratio stays fixed at its initial value for the whole run.
        """
        if not self._use_adaptive_clip:
            self._clip_controller = None
            return
        self._clip_controller = KLAdaptiveClipController(
            clip_holder = self._clip_holder,
            kl_target   = KL_TARGET,
            clip_min    = CLIP_MIN,
            clip_max    = CLIP_MAX,
            high_mult   = KL_HIGH_MULT,
            low_mult    = KL_LOW_MULT,
            clip_down   = CLIP_DOWN,
            clip_up     = CLIP_UP,
            verbose     = False,
        )

    def export_onnx(self, save_dir: str) -> str:
        """
        Export the actor to ONNX and load an OrtSession for fast CPU/CUDA inference.
        Called once after the first training step.  Returns the .onnx path.
        Uses a temp file strategy to avoid Windows path issues with .onnx.data files.
        """
        if not ONNX_AVAILABLE:
            return ""
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"actor_{self.tls_id}.onnx")

        dummy_s = torch.zeros(1, self.state_dim,  dtype=torch.float32)
        dummy_i = torch.zeros(1, self.intent_dim, dtype=torch.float32)

        # unwrap compiled model for export
        actor_raw = (self.actor._orig_mod
                     if hasattr(self.actor, "_orig_mod") else self.actor)
        actor_raw.eval()

        # Export to a temp dir first to avoid Windows Invalid Argument on .onnx.data
        import tempfile, shutil
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, f"actor_{self.tls_id}.onnx")
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    actor_raw.cpu(),
                    (dummy_s, dummy_i),
                    tmp_path,
                    input_names  = ["state", "intent"],
                    output_names = ["logits", "new_intent"],
                    dynamic_axes = {
                        "state":      {0: "batch"},
                        "intent":     {0: "batch"},
                        "logits":     {0: "batch"},
                        "new_intent": {0: "batch"},
                    },
                    opset_version = 17,
                    do_constant_folding = True,
                )
            # Move exported files to target dir (handles .onnx.data side files)
            for fname in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, fname)
                # remap name so .data file matches target .onnx name
                dst_name = fname.replace(f"actor_{self.tls_id}", f"actor_{self.tls_id}")
                dst = os.path.join(save_dir, dst_name)
                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
                shutil.move(src, dst)
        except Exception as export_err:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise export_err
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # restore model to training device
        actor_raw.to(self.device).train()

        # build OrtSession with GPU EP when available
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._use_cuda else ["CPUExecutionProvider"])
        try:
            self._onnx_actor_session = ort.InferenceSession(
                path, sess_options=sess_opts, providers=providers)
            self._onnx_export_done = True
            if not self.use_onnx_for_training:
                print(f"  [ONNX] {self.tls_id}: snapshot exported to {path} "
                      f"(artifact only -- choose_action still uses the live "
                      f"PyTorch actor since use_onnx_for_training=False)",
                      flush=True)
        except Exception as e:
            print(f"[ONNX] Session creation failed for {self.tls_id}: {e}", flush=True)
            self._onnx_actor_session = None
        return path

    def _choose_action_onnx(self, combined_intent: np.ndarray) -> tuple[int, np.ndarray]:
        """Fast inference path via ONNX Runtime (no Python overhead)."""
        feeds = {
            "state":  self.state.reshape(1, -1),
            "intent": combined_intent.reshape(1, -1),
        }
        logits, new_intent = self._onnx_actor_session.run(
            ["logits", "new_intent"], feeds)
        logits = logits[0]  # shape (action_dim,)
        # softmax + sample
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()
        action  = int(np.random.choice(len(probs), p=probs))
        return action, new_intent[0]

    def _choose_action_with_intent(self, combined_intent: np.ndarray,
                                    deterministic: bool = False) -> int:
        """Choose action given a pre-computed combined intent (avoids recomputing GAT).

        """
        # ONNX fast path -- ONLY for genuine post-training inference. Gated
        # behind use_onnx_for_training (default False) so a mid-training
        # export never silently swaps the acting policy away from the live,
        # currently-training PyTorch actor (see BUGFIX note in __init__).
        if self.use_onnx_for_training and self._onnx_export_done \
                and self._onnx_actor_session is not None \
                and not deterministic:
            action, new_intent = self._choose_action_onnx(combined_intent)
            self.current_intent = new_intent.astype(np.float32)
            return action
        # PyTorch path
        with torch.no_grad():
            s_gpu = self._state_gpu
            i_gpu = self._intent_gpu
            s_gpu.copy_(torch.from_numpy(self.state).unsqueeze(0), non_blocking=True)
            i_gpu.copy_(torch.from_numpy(combined_intent).unsqueeze(0), non_blocking=True)
            action_logits, new_intent = self.actor(s_gpu, i_gpu)
            self.current_intent = new_intent[0].cpu().numpy()
            if deterministic:
                action = int(action_logits.argmax().item())
            else:
                probs  = torch.softmax(action_logits, dim=-1)
                action = int(torch.distributions.Categorical(probs).sample().item())
        return action

    def choose_action(self, deterministic: bool = False) -> int:
        agg_intent = self.aggregate_neighbor_intents()
        combined_intent = (self.current_intent + agg_intent) / 2.0

        # -- ONNX fast path: post-training inference only, see above -------
        if self.use_onnx_for_training and self._onnx_export_done \
                and self._onnx_actor_session is not None \
                and not deterministic:
            action, new_intent = self._choose_action_onnx(combined_intent)
            self.current_intent = new_intent.astype(np.float32)
            return action

        # -- PyTorch path (fallback / deterministic) -----------------------
        with torch.no_grad():
            if self._use_cuda:
                self._state_tensor.copy_(
                    torch.from_numpy(self.state).unsqueeze(0))
                self._intent_tensor.copy_(
                    torch.from_numpy(combined_intent).unsqueeze(0))
                self._state_gpu.copy_(self._state_tensor, non_blocking=True)
                self._intent_gpu.copy_(self._intent_tensor, non_blocking=True)
                s_gpu = self._state_gpu
                i_gpu = self._intent_gpu
            else:
                s_gpu = self._state_gpu
                i_gpu = self._intent_gpu
                s_gpu.copy_(torch.from_numpy(self.state).unsqueeze(0), non_blocking=True)
                i_gpu.copy_(torch.from_numpy(combined_intent).unsqueeze(0), non_blocking=True)

            action_logits, new_intent = self.actor(s_gpu, i_gpu)
            self.current_intent = new_intent[0].cpu().numpy()
            if deterministic:
                action = int(action_logits.argmax().item())
            else:
                probs  = torch.softmax(action_logits, dim=-1)
                dist   = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())
        return action

    def apply_action(self, action_idx: int, current_step: int) -> None:
        self._current_step = current_step
        chosen_phase = self.action_phases[action_idx]

        try:
            active_phase  = traci.trafficlight.getPhase(self.tls_id)
            phase_state   = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            in_transition = any(c in ('y', 'Y', 'u') for c in phase_state)
        except Exception:
            active_phase  = chosen_phase   # safe fallback: pretend no change needed
            in_transition = False

        if in_transition:
            return


        if chosen_phase != active_phase:
            if (current_step - self.last_switch_step) >= self.min_green_steps:
                traci.trafficlight.setPhase(self.tls_id, chosen_phase)
                self.last_switch_step = current_step

    def get_reward(self) -> float:
        """Delegate to the injected reward function.

        Formula (default: BackPressureReward)
        -------
        r_i = -mean_{phi} [ (q_in_phi - q_out_phi) / capacity(tls_id) ]

        Per-TLS capacity
        ----------------
        Each agent now divides by its own ``_QUEUE_CAPACITY_PER_TLS`` value
        (TL1->90, TL2->70, TL3/TL4->20) which reflects the actual lane count
        and saturation flow.  This prevents TL1 (16 lanes) from producing
        reward magnitudes ~4.5x larger than TL3/TL4 (8-10 lanes), which was
        the primary cause of TL1 reward instability and gradient dominance.
        """
        # -- EV potential-based reward shaping (Ng et al. 1999) ------------------
        # r'(s,a,s') = r_BP(s,a) + α·(γ·Φ(s') − Φ(s))
        # _ev_phi_curr = Φ(s') computed in _get_state() AFTER simulationStep()
        # _ev_phi_prev = Φ(s)  cached from the previous _get_state() call
        # This formulation preserves the optimal policy (PPO-safe) and adds
        # a positive bonus when EVs hold green or a negative penalty when a
        # green EV lane is about to turn red.
        return self.reward_func.get_reward(
            self.tls_id,
            self.incoming_lanes,
            self.outgoing_lanes,
            ev_phi_prev=self._ev_phi_prev,
            ev_phi_curr=self._ev_phi_curr,
            action_idx=self.current_action_idx,
            action_phases=self.action_phases,
            phase_links=self.phase_links,
        )

    def get_value(self, state: np.ndarray, global_intent: np.ndarray) -> float:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            g = torch.from_numpy(global_intent).unsqueeze(0).to(self.device)
            v = self.critic(s, g)
        return float(v[0].item())

    def get_action_log_prob(self, state: np.ndarray, intent: np.ndarray, action: int) -> float:
        with torch.no_grad():
            s = torch.from_numpy(state).unsqueeze(0).to(self.device)
            i = torch.from_numpy(intent).unsqueeze(0).to(self.device)
            logits, _ = self.actor(s, i)
            log_probs = torch.log_softmax(logits, dim=-1)
        return float(log_probs[0, action].item())

    def get_values_batch(self, states: np.ndarray, global_intents: np.ndarray) -> np.ndarray:
        """Batch inference for centralized critic."""
        kw = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device, **kw)
            g = torch.from_numpy(global_intents).to(self.device, **kw)
            with torch.amp.autocast(device_type="cuda") if use_amp else _null_ctx():
                v = self.critic(s, g).squeeze(1)
        return v.float().cpu().numpy()

    def get_log_probs_batch(self, states: np.ndarray, intents: np.ndarray,
                            actions: np.ndarray) -> np.ndarray:
        """Batch inference for actor log-probs -- replaces per-transition calls."""
        kw = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device, **kw)
            i = torch.from_numpy(intents).to(self.device, **kw)
            acts_t = torch.from_numpy(actions).long().to(self.device, **kw)
            with torch.amp.autocast(device_type="cuda") if use_amp else _null_ctx():
                logits, _ = self.actor(s, i)
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                lp        = log_probs.gather(1, acts_t.unsqueeze(1)).squeeze(1)
        return lp.float().cpu().numpy()

    def train_epoch(self, advantages: np.ndarray, returns: np.ndarray) -> tuple[float, float, float]:
        total_loss    = 0.0
        total_entropy = 0.0
        total_kl      = 0.0   # approx_kl (sample-based) accumulator — diagnostic only
        total_true_kl = 0.0   # distributional KL accumulator — diagnostic only
        n_batches     = 0
        use_amp       = self._use_cuda and self._scaler is not None

        # -- Store old values for clipped value loss (PPO-style critic) --------
        with torch.no_grad():
            old_values = self.get_values_batch(
                self.rollout_buffer.states[:self.rollout_buffer.size],
                self.rollout_buffer.global_intents[:self.rollout_buffer.size],
            )  # shape: (size,)

        for _epoch in range(self.n_epochs):
            for _ in range(max(1, self.rollout_buffer.size // self.batch_size)):
                states, intents, global_intents, actions, _, _, old_log_probs, _, indices = \
                    self.rollout_buffer.sample(self.batch_size)
                batch_size = len(states)
                if batch_size == 0:
                    continue

                kw = {"non_blocking": True} if self._use_cuda else {}
                states_t        = torch.from_numpy(states).to(self.device, **kw)
                intents_t       = torch.from_numpy(intents).to(self.device, **kw)
                global_intents_t= torch.from_numpy(global_intents).to(self.device, **kw)
                actions_t    = torch.from_numpy(actions).unsqueeze(1).to(self.device, **kw)
                old_lp_t     = torch.from_numpy(old_log_probs).to(self.device, **kw)
                advantages_t = torch.from_numpy(advantages[indices]).to(self.device, **kw)
                returns_t    = torch.from_numpy(returns[indices]).to(self.device, **kw)
                old_vals_t   = torch.from_numpy(old_values[indices]).to(self.device, **kw)

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda") if use_amp else _null_ctx():
                    # -- Actor loss (PPO-clip) ----------------------------------
                    logits, new_intent = self.actor(states_t, intents_t)
                    log_probs = torch.log_softmax(logits, dim=-1)
                    log_probs = log_probs.gather(1, actions_t).squeeze(1)

                    ratio  = torch.exp(log_probs - old_lp_t)
                    surr1  = ratio * advantages_t
                    surr2  = torch.clamp(ratio,
                                         1 - self.clip_ratio,
                                         1 + self.clip_ratio) * advantages_t

                    log_pi   = torch.log_softmax(logits, dim=-1)
                    entropy  = -(torch.softmax(logits, dim=-1) * log_pi).sum(dim=1).mean()
                    actor_loss = (-torch.min(surr1, surr2).mean()
                                  - self.entropy_coeff * entropy)

                    # -- Centralized Critic loss -------------------------------
                    values      = self.critic(states_t, global_intents_t).squeeze(1)
                    v_clipped   = old_vals_t + torch.clamp(
                                      values - old_vals_t,
                                      -self.clip_ratio, self.clip_ratio)
                    critic_loss = torch.max(
                        nn.functional.mse_loss(values,    returns_t),
                        nn.functional.mse_loss(v_clipped, returns_t),
                    )

                    # -- INTENT-CONSISTENCY (auxiliary) --------------------------
                    # Direct learning signal for the actor's intent_head: does
                    # THIS agent's freshly-produced intent (new_intent, still
                    # attached to the actor graph) help the *centralized*
                    # critic explain the joint return better?
                    #
                    # We splice new_intent into this agent's slot of the global
                    # intent vector and re-query the critic with FROZEN critic
                    # weights (via torch.func.functional_call), so gradient
                    # flows only into new_intent -> intent_head -> actor, never
                    # into the critic itself (that stays critic_loss's job,
                    # using the untouched global_intents_t from the buffer).
                    my_idx = _TLS_INDEX[self.tls_id]
                    global_intents_spliced = global_intents_t.clone()
                    start = my_idx * self.intent_dim
                    end   = start + self.intent_dim
                    global_intents_spliced[:, start:end] = new_intent

                    frozen_params = {
                        k: v.detach() for k, v in self.critic.named_parameters()
                    }
                    frozen_buffers = {
                        k: v.detach() for k, v in self.critic.named_buffers()
                    }
                    values_from_new_intent = torch.func.functional_call(
                        self.critic,
                        (frozen_params, frozen_buffers),
                        (states_t, global_intents_spliced),
                    ).squeeze(1)

                    intent_consistency_loss = nn.functional.mse_loss(
                        values_from_new_intent, returns_t.detach())

                    loss = (actor_loss
                            + self.value_loss_coeff * critic_loss
                            + INTENT_CONSISTENCY_COEFF * intent_consistency_loss)

                if use_amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.actor_optimizer)
                    self._scaler.unscale_(self.critic_optimizer)
                    nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self._scaler.step(self.actor_optimizer)
                    self._scaler.step(self.critic_optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),  0.5)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

                total_loss    += float(loss.item())
                total_entropy += float(entropy.item())

                # ── KL-ADAPTIVE LR ──────────────────────────────────────────
                # Step 1: Compute KL divergence between old and new policy
                # AFTER optimizer.step() so log_probs reflects updated weights.
                # We use the scalar estimator: KL ≈ mean(old_lp − new_lp).
                # log_probs is computed BEFORE the backward pass (inside the
                # autocast block above) and reflects the current (pre-step)
                # parameters, so we re-evaluate it post-step here.
                # NOTE: re-evaluating after step is the correct approach for
                # measuring how much the update moved the policy.
                if self._kl_controller is not None:
                    with torch.no_grad():
                        post_logits, _ = self.actor(states_t, intents_t)
                        post_log_probs = torch.log_softmax(
                            post_logits.float(), dim=-1
                        ).gather(1, actions_t).squeeze(1)
                        # approx_kl: the sample-based PPO estimator (same one
                        # OpenAI's PPO implementations use for early-stopping /
                        # LR control). It is computed only on the SELECTED
                        # actions' log-probs, not the full distribution, so it
                        # has no non-negativity guarantee and can go negative
                        # (e.g. when the update happens to raise the
                        # probability of the sampled actions on average).
                        # DIAGNOSTIC ONLY — the controller, clip adapter,
                        # early stopping, and all optimizer logic are driven
                        # by batch_true_kl (see below).
                        batch_kl = compute_kl_divergence(
                            old_lp_t.float(), post_log_probs)

                        # true_kl: full distributional KL(old || new) computed
                        # from the complete logit vectors (`logits` = pre-step,
                        # `post_logits` = post-step -- both already available,
                        # so this adds no extra forward pass). Mathematically
                        # guaranteed >= 0 (up to floating-point precision) by
                        # Gibbs' inequality.  This IS the signal fed into
                        # _kl_controller and _clip_controller.
                        batch_true_kl = compute_kl_from_logits(
                            logits.float(), post_logits.float())

                    # Step 2: Adjust LRs based on observed KL
                    # Previously used batch_kl (approx_kl),
                    # a high-variance sample-based estimator computed on selected
                    # actions only, re-evaluated on the same mini-batch right
                    # after its own gradient step.  This gives a noisy, possibly
                    # negative signal that does not reflect true distributional
                    # drift.  Switch to batch_true_kl — the full distributional
                    # KL(old||new) computed from complete logit vectors, which is
                    # mathematically >= 0 (Gibbs' inequality) and far more stable.
                    self._kl_controller.step(batch_true_kl, agent_id=self.tls_id)

                    # Step 3: Adjust clip ratio based on same KL
                    if self._clip_controller is not None:
                        self._clip_controller.step(batch_true_kl, agent_id=self.tls_id)

                    # Diagnostic: print old vs new KL so we can monitor the
                    # difference between the noisy approx_kl and the stable
                    # true_kl that now drives the controller.
                    if n_batches <= 2:
                        print(
                            f"  [KL-COMPARE] {self.tls_id}: "
                            f"approx_kl={batch_kl:.6f}  "
                            f"true_kl={batch_true_kl:.6f}",
                            flush=True,
                        )

                    # Update EMA for diagnostics (now tracks true_kl)
                    self._kl_ema = (
                        self._kl_ema_alpha * batch_true_kl
                        + (1.0 - self._kl_ema_alpha) * self._kl_ema
                    )
                    total_kl      += batch_kl        # approx_kl (sample-based) for diagnostics
                    total_true_kl += batch_true_kl    # distributional KL for diagnostics
                # ── END KL-ADAPTIVE LR ──────────────────────────────────────

                n_batches  += 1

        mean_loss    = total_loss    / max(n_batches, 1)
        mean_entropy = total_entropy / max(n_batches, 1)
        mean_kl      = total_kl      / max(n_batches, 1)
        mean_true_kl = total_true_kl / max(n_batches, 1)
        # Diagnostics were already tracked inside KLAdaptiveLRController
        # (n_increased / n_decreased / n_unchanged) but never surfaced --
        # expose them here as instance attributes so training_loop.py can
        # log per-episode without changing the controller's own logic.
        if self._kl_controller is not None:
            self.last_kl_action_counts = {
                "increased":            self._kl_controller.n_increased,
                "decreased":            self._kl_controller.n_decreased,
                "unchanged":            self._kl_controller.n_unchanged,
                "degenerate":           self._kl_controller.n_degenerate,
                "consecutive_increases": self._kl_controller.consecutive_increases,
                "is_lr_pinned":         self._kl_controller.is_lr_pinned,
            }
        else:
            self.last_kl_action_counts = None
        # last_true_kl: epoch-mean of the distributional KL that drives the
        # controller, surfaced as an instance attribute so training_loop.py
        # can log it to CSV without changing train_epoch()'s return signature.
        # The *instance attribute* is diagnostic-only (controllers read
        # batch_true_kl during the training loop, not this attribute).
        self.last_true_kl = mean_true_kl
        return mean_loss, mean_entropy, mean_kl



    def pre_step(self, current_step: int) -> int:
        action = self.choose_action(deterministic=False)
        self.current_action_idx = action
        self.apply_action(action, current_step)
        return action

    def post_step(self, action: int, current_step: int, next_value: float = 0.0,
                  sim_max_steps: int = TOTAL_STEPS,
                  combined_intent: "np.ndarray | None" = None) -> float:
        next_state = self._get_state()
        reward = self.get_reward()
        done = 1.0 if current_step >= sim_max_steps else 0.0

        if combined_intent is None:
            agg_intent = self.aggregate_neighbor_intents()
            combined_intent = (self.current_intent + agg_intent) / 2.0

        value = self.get_value(self.state, combined_intent)
        log_prob = self.get_action_log_prob(self.state, combined_intent, action)

        self.rollout_buffer.push(
            self.state, combined_intent, action, reward, value, log_prob, done,
            combined_intent.copy(), next_value)
        self.state = next_state

        incoming_queue = _get_tls_queue_from_detectors(self.tls_id)
        mean_wait = _get_tls_mean_wait(self.tls_id)
        self.step_rewards.append(reward)
        self.step_queues.append(incoming_queue)
        self.step_waits.append(mean_wait)
        return reward

    def train_step(self, agents: dict = None) -> float:
        if self.rollout_buffer.size < self.batch_size:
            print(f"  [TRAIN-SKIP] {self.tls_id}: buffer size "
                  f"{self.rollout_buffer.size} < batch_size {self.batch_size} -- skipping",
                  flush=True)
            return 0.0
        advantages, returns = self.rollout_buffer.compute_gae()
        loss, mean_entropy, _mean_kl = self.train_epoch(advantages, returns)
        # KL diagnostic: expose the per-episode mean_kl (previously discarded
        # here) as an instance attribute so training_loop.py can log it to a
        # CSV without changing this method's return signature or behavior.
        self.last_mean_kl = _mean_kl
        self.rollout_buffer.clear()
        if agents is not None:
            self._update_gat_from_peers(agents)
        self.step_losses.append(loss)

        # NOTE: CosineAnnealingLR .step() calls intentionally removed.
        # The KL-adaptive controller (_kl_controller) is the sole authority
        # on actor/critic LRs -- it adjusts param_groups['lr'] in-place
        # during train_epoch().  Stepping the cosine schedulers here would
        # overwrite those KL-driven values every episode.

        self._check_entropy_collapse(mean_entropy)
        return loss

    def _update_gat_from_peers(self, agents: dict) -> None:
        tls_ids = list(agents.keys())
        if len(tls_ids) < 2:
            return

        # Compute per-agent mean reward over the last episode.
        # GAT target will weight neighbors by relative performance:
        # a neighbor with better (less-negative) BP reward contributes more
        # to the target intent, so the GAT learns to attend to informative nodes.
        agent_mean_reward: dict[str, float] = {}
        for tls_id, ag in agents.items():
            rewards = ag.step_rewards
            agent_mean_reward[tls_id] = float(np.mean(rewards)) if rewards else 0.0

        # Build intent matrix from current agent intents.
        mat = torch.zeros(len(INTERSECTIONS), self.intent_dim, dtype=torch.float32)
        for tls_id, ag in agents.items():
            if tls_id in _TLS_INDEX:
                mat[_TLS_INDEX[tls_id]] = torch.from_numpy(ag.current_intent)

        # Reward-weighted target: softmax over neighbor rewards (temperature=50)
        # amplifies small reward differences so attention is meaningful even when
        # all rewards cluster near -0.05.  Fallback to self-intent when isolated.
        _TEMP = 5
        target = torch.zeros_like(mat)
        for tls_id in INTERSECTIONS:
            i         = _TLS_INDEX[tls_id]
            neighbors = [nb for nb in NEIGHBOR_MAP.get(tls_id, [])
                         if nb in _TLS_INDEX and nb in agents]
            if neighbors:
                nb_rewards = torch.tensor(
                    [agent_mean_reward[nb] for nb in neighbors],
                    dtype=torch.float32,
                )
                # Shift to non-negative then apply temperature-scaled softmax.
                nb_rewards = (nb_rewards - nb_rewards.min()) * _TEMP + 1e-6
                weights    = torch.softmax(nb_rewards, dim=0)          # (n_nb,)
                nb_intents = torch.stack([mat[_TLS_INDEX[nb]] for nb in neighbors])
                target[i]  = (weights.unsqueeze(1) * nb_intents).sum(dim=0)
            else:
                target[i] = mat[i]

        mat_in = mat.clone().detach().to(self.device)
        target  = target.clone().detach().to(self.device)
        self.update_gat(mat_in, target)


# ═══════════════════════════════════════════════════════════════════════════════
# === 2. Updated setup_lr_schedulers() -- inside MAPPOAgent ===
# ═══════════════════════════════════════════════════════════════════════════════

    def setup_lr_schedulers(self, n_episodes: int) -> None:
        """Create CosineAnnealingLR schedulers for actor and critic.

        Called once from ``run_training()`` with the *total* episode count so
        the LR decays from ``LEARNING_RATE`` down to ``eta_min=1e-5`` over the
        full training run.

        Resume behaviour
        ----------------
        ``load_checkpoint()`` stashes the raw scheduler ``state_dict``s on the
        agent via ``_pending_actor_scheduler_sd`` / ``_pending_critic_scheduler_sd``
        before this method is called.

        When those are present we:

        1. Read the **original** ``T_max`` from *inside* the state_dict.
           PyTorch stores ``T_max`` there verbatim -- so it always reflects the
           value used when the scheduler was first created, regardless of what
           episode we are resuming from.

        2. Instantiate a new scheduler with that original ``T_max``.

        3. Call ``load_state_dict()`` to restore ``last_epoch`` (and any other
           internal counters), placing the scheduler exactly where it was when
           the checkpoint was saved.

        This gives a perfectly seamless cosine curve:

            LR(t) = η_min + ½(η_max − η_min)(1 + cos(π · last_epoch / T_max_orig))

        Using ``T_max = remaining_episodes`` instead would restart the cosine
        from its peak -- an LR jump at every resume boundary.

        On a fresh start (no pending state_dicts) a new scheduler is created
        with ``T_max = n_episodes`` and ``last_epoch = 0``.
        """
        ETA_MIN = 1e-5

        def _build(optimizer, pending_sd, role: str):
            """Return a scheduler, restored from *pending_sd* if available."""
            if pending_sd is not None:
                # Step 1 -- recover the original T_max from the checkpoint.
                orig_tmax = pending_sd.get("T_max", n_episodes)
                if orig_tmax is None or orig_tmax < 1:
                    orig_tmax = n_episodes
                    print(f"  [LR-RESUME] {self.tls_id}/{role}: T_max missing in SD, "
                          f"using n_episodes={n_episodes}", flush=True)

                # Step 2 -- create the scheduler with the original shape.
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=orig_tmax, eta_min=ETA_MIN)

                # Step 3 -- restore internal state (last_epoch, _last_lr, …).
                try:
                    sched.load_state_dict(pending_sd)
                    # get_last_lr() raises if step() has never been called on a
                    # fresh instance before load_state_dict; safe after load.
                    try:
                        _lr_str = f"{sched.get_last_lr()[0]:.3e}"
                    except Exception:
                        _lr_str = "N/A"
                    print(
                        f"  [LR-RESUME] {self.tls_id}/{role}: restored "
                        f"T_max={orig_tmax}  last_epoch={sched.last_epoch}"
                        f"  lr={_lr_str}",
                        flush=True,
                    )
                    return sched
                except Exception as _e:
                    print(
                        f"  [LR-RESUME] {self.tls_id}/{role}: load_state_dict failed "
                        f"({_e}) -- falling back to fresh scheduler "
                        f"(T_max={n_episodes})",
                        flush=True,
                    )
                    # Fall through to fresh creation below.

            # Fresh start (or fallback after failed restore).
            sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(n_episodes, 1), eta_min=ETA_MIN)
            print(f"  [LR-FRESH]  {self.tls_id}/{role}: "
                  f"CosineAnnealingLR T_max={n_episodes} eta_min={ETA_MIN}",
                  flush=True)
            return sched

        _actor_sd  = getattr(self, "_pending_actor_scheduler_sd",  None)
        _critic_sd = getattr(self, "_pending_critic_scheduler_sd", None)

        self._actor_scheduler  = _build(self.actor_optimizer,  _actor_sd,  "actor")
        self._critic_scheduler = _build(self.critic_optimizer, _critic_sd, "critic")

        # Clear pending state -- must not be re-applied on subsequent calls.
        self._pending_actor_scheduler_sd  = None
        self._pending_critic_scheduler_sd = None

    def _check_entropy_collapse(self, current_entropy: float) -> None:
        """Monitor entropy per episode. If entropy < 0.01 for 3 consecutive
        episodes -> warn and temporarily 2x entropy_coeff for next 5 episodes, then restore.
        Prevents the policy from permanently collapsing to a deterministic distribution."""
        self._entropy_history.append(current_entropy)

        # Decrement boost counter and restore base coeff when boost window expires
        if self._entropy_boost_remaining > 0:
            self._entropy_boost_remaining -= 1
            if self._entropy_boost_remaining == 0:
                self.entropy_coeff = self._base_entropy_coeff
                print(f"  [ENTROPY] {self.tls_id}: entropy boost expired -- "
                      f"entropy_coeff restored to {self.entropy_coeff:.4f}", flush=True)

        # Check last 3 episodes for collapse (only when not already in a boost window)
        if self._entropy_boost_remaining == 0 and len(self._entropy_history) >= 3:
            last3 = self._entropy_history[-3:]
            if all(e < 0.01 for e in last3):
                self._entropy_boost_remaining = 5
                self.entropy_coeff = self._base_entropy_coeff * 2.0
                print(f"  [ENTROPY-WARN] {self.tls_id}: entropy collapsed! "
                      f"(last 3 eps: {[f'{e:.5f}' for e in last3]}) "
                      f"-- boosting entropy_coeff to {self.entropy_coeff:.4f} "
                      f"for next 5 episodes.", flush=True)

    def get_spec(self) -> dict:
        return {
            "state_dim":      self.state_dim,
            "action_dim":     self.action_dim,
            "intent_dim":     self.intent_dim,
            "action_phases":  self.action_phases,
            "phase_links":    self.phase_links,
            "incoming_lanes": self.incoming_lanes,
            "outgoing_lanes": self.outgoing_lanes,
        }

    def get_cpu_state_dict(self) -> dict:
        return {
            "actor":  {k: v.cpu() for k, v in self.actor.state_dict().items()},
            "critic": {k: v.cpu() for k, v in self.critic.state_dict().items()},
            "gat":    {k: v.cpu() for k, v in self.intent_aggregator.state_dict().items()},
        }

def _broadcast_intents(agents: dict):
    for tls_id, agent in agents.items():
        neighbors = NEIGHBOR_MAP.get(tls_id, [])
        for neighbor_id in neighbors:
            neighbor_agent = agents.get(neighbor_id)
            if neighbor_agent is not None:
                neighbor_agent.neighbor_intents[tls_id] = agent.current_intent.copy()

def _build_global_intent_vec(agents: dict, intent_dim: int) -> np.ndarray:
    n   = len(INTERSECTIONS)
    vec = np.zeros(n * intent_dim, dtype=np.float32)
    for tls_id, ag in agents.items():
        idx = _TLS_INDEX.get(tls_id)
        if idx is not None:
            vec[idx * intent_dim: (idx + 1) * intent_dim] = ag.current_intent
    return vec