"""
ippo.py
==============
IPPO baseline for the SmartTraffic-OM project.

Design contract
---------------
* No GAT, no intent sharing, no neighbour broadcasts.
* Identical state representation to MAPPOAgent._get_state() (same feature
  engineering, same dimension formula, same shock-wave / EV blocks).
* Independent per-intersection actor + local critic (decentralised execution,
  decentralised training).  A centralized-critic variant is available via
  ``critic_mode="centralised"`` (shares state from all agents).
* Uses every hyperparameter defined in config.py:
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_RATIO, BATCH_SIZE, N_EPOCHS, HIDDEN_DIM, MEMORY_CAPACITY.
* Supports all three reward functions (BackPressureReward, WaitingTimeReward,
  QueueReductionReward) and the EV priority module.
* Compatible with:
    – CheckpointManager / EpisodeAutoSaver (save / resume)
    – RichUI, TrainingPlotter, ResultsWriter, TBComparisonLogger
    – export_comparison_excel / plot_comparison
    – ONNX export (actor only, no intent input — different signature to
      ActorNetwork but same export/ORT approach)
* Switching: pass ``--algorithm ippo`` to main(); default is mappo_gat.

Class hierarchy
---------------
IPPOBuffer          — rollout buffer (no intent columns)
IPPOActorNet        — state -> logits  (no intent head)
IPPOCriticNet       — state -> value   (local) or concat_states -> value
IPPOAgent           — per-intersection PPO agent
IPPOCentralCritic   — optional shared critic object (centralised mode)
"""

from __future__ import annotations

import os
import sys
import time
import csv
import json
import random
import signal
import warnings
import platform
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo

# ── project imports ────────────────────────────────────────────────────────────
# These are all guaranteed to exist by the outer project.
from .config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_RATIO, BATCH_SIZE, N_EPOCHS, HIDDEN_DIM, MEMORY_CAPACITY,
    TOTAL_STEPS, INTERSECTIONS, TLS_DETECTOR_MAP, UPSTREAM_LANES,
    DOWNSTREAM_LANES, NEIGHBOR_MAP, QUEUE_CAPACITY, UNIFIED_BP_CAPACITY,
    _QUEUE_CAPACITY_PER_TLS, EV_PRIORITY_ENABLED, EV_BETA, EV_ALPHA,
    MIN_GREEN_STEPS, MIN_GREEN_TIME, STEP_LENGTH, REWARD_CLIP, V_FREE,
    BASE_PORT,
    CLIP_MIN, CLIP_MAX, CLIP_DOWN, CLIP_UP,
    KL_TARGET, KL_HIGH_MULT, KL_LOW_MULT,
)
from .kl_adaptive_lr import (
    KLAdaptiveClipController,
    compute_kl_divergence,
)
from .sumo_env import (
    _get_lanes_for_tls, _get_green_phases_and_links, _moving_stopped_counts,
    _lane_queue, _get_tls_queue_from_detectors, _get_tls_mean_wait,
    _shock_wave_speed, _STEP_CACHE, _null_ctx,
    TRACI_AVAILABLE,
)
from .rewards import BaseRewardFunction, BackPressureReward
from .ev_priority import BaseEVPriorityModule, DefaultEVPriority

try:
    import traci
except ImportError:
    traci = None  # type: ignore

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

__all__ = [
    "IPPOBuffer",
    "IPPOActorNet",
    "IPPOCriticNet",
    "IPPOAgent",
    "IPPOCentralCritic",
]

# Shared with agent.py's flag via the module-level import below is not
# possible without a circular import, so IPPO keeps its own flag. Once
# torch.compile/Triton is found to be broken on this machine, subsequent
# IPPOAgent instances skip straight to eager mode instead of re-attempting
# a doomed compile + warm-up on every intersection.
_IPPO_COMPILE_BROKEN_ON_THIS_MACHINE = False


def _ippo_try_compile_with_fallback(actor, critic, state_dim, device):
    """
    Same rationale as agent.py's _try_compile_with_fallback: torch.compile()
    itself never raises (compilation is lazy), so the only way to actually
    detect a broken Triton/WSL setup is to run a real forward pass through
    the compiled module and catch the failure there, then fall back to
    eager mode instead of letting it crash mid-training.
    """
    global _IPPO_COMPILE_BROKEN_ON_THIS_MACHINE
    if _IPPO_COMPILE_BROKEN_ON_THIS_MACHINE:
        return actor, critic

    compiled_actor = torch.compile(actor, mode="default")
    compiled_critic = torch.compile(critic, mode="default") if critic is not None else None

    try:
        with torch.no_grad():
            dummy_state = torch.zeros(2, state_dim, device=device)
            compiled_actor(dummy_state)
            if compiled_critic is not None:
                compiled_critic(dummy_state)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return compiled_actor, compiled_critic
    except Exception as exc:
        _IPPO_COMPILE_BROKEN_ON_THIS_MACHINE = True
        print(f"[torch.compile] IPPO warm-up failed ({exc.__class__.__name__}: {exc}). "
              f"Falling back to eager mode for this and all subsequent agents.",
              flush=True)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return actor, critic


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Rollout Buffer (no intent columns)
# ══════════════════════════════════════════════════════════════════════════════

class IPPOBuffer:
    """
    Fixed-capacity ring buffer for one intersection's rollout data.

    Fields stored per transition
    ----------------------------
    states      (capacity, state_dim)   float32
    actions     (capacity,)             int64
    rewards     (capacity,)             float32
    values      (capacity,)             float32   V(s_t)
    log_probs   (capacity,)             float32   log π(a|s) at collection time
    dones       (capacity,)             float32   1.0 at episode end
    next_values (capacity,)             float32   V(s_{t+1})
    """

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity  = capacity
        self.state_dim = state_dim  # use passed-in dim, not hardcoded 77

        self.states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions     = np.zeros(capacity,              dtype=np.int64)
        self.rewards     = np.zeros(capacity,              dtype=np.float32)
        self.values      = np.zeros(capacity,              dtype=np.float32)
        self.log_probs   = np.zeros(capacity,              dtype=np.float32)
        self.dones       = np.zeros(capacity,              dtype=np.float32)
        self.next_values = np.zeros(capacity,              dtype=np.float32)
        self.size = 0

    # ------------------------------------------------------------------
    def push(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        value:      float,
        log_prob:   float,
        done:       float,
        next_value: float = 0.0,
    ) -> None:
        idx = self.size % self.capacity
        self.states[idx]      = state
        self.actions[idx]     = action
        self.rewards[idx]     = reward
        self.values[idx]      = value
        self.log_probs[idx]   = log_prob
        self.dones[idx]       = done
        self.next_values[idx] = next_value
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------
    def compute_gae(
        self,
        gamma: float = GAMMA,
        lam:   float = GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalised Advantage Estimation.
        Returns (advantages, returns), both shape (size,).
        Advantages are normalised to zero mean / unit std.
        """
        advantages = np.zeros(self.size, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(self.size)):
            nv    = self.next_values[t] * (1.0 - self.dones[t])
            delta = self.rewards[t] + gamma * nv - self.values[t]
            gae   = delta + gamma * lam * gae * (1.0 - self.dones[t])
            advantages[t] = gae
        returns = advantages + self.values[: self.size]
        std = advantages.std()
        if std > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)
        return advantages, returns

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> Tuple:
        idx = np.random.choice(
            self.size, min(batch_size, self.size), replace=False)
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.values[idx],
            idx,
        )

    def clear(self) -> None:
        self.size = 0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Neural Networks
# ══════════════════════════════════════════════════════════════════════════════

class IPPOActorNet(nn.Module):
    """
    Actor: state → action logits.
    No intent head (distinguishes from ActorNetwork which shares intent).
    Architecture mirrors ActorNetwork's feature_net depth / width.
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(HIDDEN_DIM, action_dim)

        # Orthogonal init (recommended for PPO actors)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_head(self.net(state))


class IPPOCriticNet(nn.Module):
    """
    Local critic: state → scalar value.
    Used in independent (decentralised) mode.
    """

    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.net(state))  # (batch, 1)


class IPPOCentralCritic(nn.Module):
    """
    Optional centralised critic: concatenated states of all agents → value.
    Constructed once and shared across all IPPOAgent instances.

    Input dimension = n_agents * state_dim  (states concatenated in
    INTERSECTIONS order).
    """

    def __init__(self, n_agents: int, state_dim: int) -> None:
        super().__init__()
        in_dim = n_agents * state_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.net(global_state))  # (batch, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  IPPOAgent
# ══════════════════════════════════════════════════════════════════════════════

# Ordered index lookup (same as _TLS_INDEX in networks.py)
_IPPO_TLS_INDEX: Dict[str, int] = {
    tls: i for i, tls in enumerate(INTERSECTIONS)
}


class IPPOAgent:
    """
    Per-intersection IPPO agent.

    Identical state extraction to MAPPOAgent._get_state() — same feature
    engineering, same dimension.  No intent sharing / GAT.

    Parameters
    ----------
    tls_id : str
        Intersection identifier, e.g. "TL1".
    device : torch.device
    critic_mode : {"local", "centralised"}
        "local"       — each agent maintains its own CriticNet (default).
        "centralised" — agents share one IPPOCentralCritic passed in
                         via ``shared_critic``; the agent stores no local
                         critic network.
    shared_critic : IPPOCentralCritic | None
        Must be provided when critic_mode == "centralised".
    reward_func : BaseRewardFunction | None
        Defaults to BackPressureReward (matches MAPPO baseline).
    ev_module : BaseEVPriorityModule | None
        Defaults to DefaultEVPriority.
    """

    _lane_length_cache: Dict[str, float] = {}

    @property
    def clip_ratio(self) -> float:
        return self._clip_holder[0]

    def __init__(
        self,
        tls_id:              str,
        device:              torch.device,
        critic_mode:         str                          = "local",
        shared_critic:       "IPPOCentralCritic | None" = None,
        reward_func:         "BaseRewardFunction | None"  = None,
        ev_module:           "BaseEVPriorityModule | None" = None,
        use_adaptive_clip:   bool                         = True,
    ) -> None:
        assert critic_mode in ("local", "centralised"), (
            f"critic_mode must be 'local' or 'centralised', got '{critic_mode}'")
        if critic_mode == "centralised" and shared_critic is None:
            raise ValueError(
                "critic_mode='centralised' requires shared_critic to be provided.")

        self.tls_id      = tls_id
        self.device      = device
        self._use_cuda   = device.type == "cuda"
        self.critic_mode = critic_mode

        # ── Hyperparameters (from config) ──────────────────────────────────
        self.gamma           = GAMMA
        self.gae_lambda      = GAE_LAMBDA
        self._clip_holder    = [CLIP_RATIO]  # mutable container for adaptive clip control
        self._use_adaptive_clip = use_adaptive_clip
        self._clip_controller: "KLAdaptiveClipController | None" = None
        self.entropy_coeff   = ENTROPY_COEFF
        self.value_loss_coeff = VALUE_LOSS_COEFF
        self.n_epochs        = N_EPOCHS
        self.batch_size      = BATCH_SIZE
        self.min_green_steps = MIN_GREEN_STEPS
        self.learning_rate   = LEARNING_RATE

        print(f"  [IPPO/{tls_id}] "
              f"lr={self.learning_rate}  γ={self.gamma}  "
              f"λ={self.gae_lambda}  clip={self.clip_ratio}  "
              f"entropy={self.entropy_coeff}  critic={critic_mode}")

        # ── Lane / phase topology ─────────────────────────────────────────
        self.incoming_lanes, self.outgoing_lanes = _get_lanes_for_tls(tls_id)
        self.action_phases, self.phase_links     = _get_green_phases_and_links(tls_id)

        if not self.action_phases:
            raise RuntimeError(
                f"No valid green phases for TLS '{tls_id}'.")

        self.action_dim = len(self.action_phases)

        n_in   = len(self.incoming_lanes)
        n_out  = len(self.outgoing_lanes)
        n_dets = len(TLS_DETECTOR_MAP.get(tls_id, []))
        self._ev_feat_dim = (2 * n_in + 2) if EV_PRIORITY_ENABLED else 0

        # State dim MUST mirror MAPPOAgent._get_state() exactly.
        self.state_dim = (
            self.action_dim      +   # phase one-hot
            2 * (n_in + n_out)   +   # moving/stopped per lane
            n_dets               +   # detector halting counts
            n_in                 +   # lane queue occupancy
            1                    +   # time-since-switch
            1                    +   # normalised total queue
            2                    +   # shock-wave (mean, min)
            self._ev_feat_dim        # EV features (0 if disabled)
        )

        # ── Injectable modules ────────────────────────────────────────────
        self.reward_func: BaseRewardFunction   = reward_func or BackPressureReward()
        self.ev_module:   BaseEVPriorityModule = ev_module   or DefaultEVPriority()

        # EV potential cache for reward shaping
        self._ev_phi_prev: float = 0.0
        self._ev_phi_curr: float = 0.0

        # ── Networks ──────────────────────────────────────────────────────
        self.actor = IPPOActorNet(self.state_dim, self.action_dim).to(device)

        if critic_mode == "local":
            self.critic:  "IPPOCriticNet | None" = (
                IPPOCriticNet(self.state_dim).to(device))
            self._shared_critic = None
        else:  # centralised
            self.critic = None
            self._shared_critic: "IPPOCentralCritic | None" = shared_critic

        # torch.compile (Linux/Mac only — skipped on Windows)
        if self._use_cuda and platform.system() != "Windows":
            self.actor, self.critic = _ippo_try_compile_with_fallback(
                self.actor, self.critic, self.state_dim, device)

        # Mixed-precision scaler
        self._scaler = (
            torch.amp.GradScaler(device="cuda") if self._use_cuda else None)

        # ── Optimisers ───────────────────────────────────────────────────
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.learning_rate)

        if critic_mode == "local":
            assert self.critic is not None
            self.critic_optimizer: "optim.Adam | None" = optim.Adam(
                self.critic.parameters(), lr=self.learning_rate)
        else:
            # Centralised critic is optimised via IPPOCentralCritic.optimizer
            self.critic_optimizer = None

        # ── LR schedulers (populated by setup_lr_schedulers) ─────────────
        self._actor_scheduler:  "optim.lr_scheduler.CosineAnnealingLR | None" = None
        self._critic_scheduler: "optim.lr_scheduler.CosineAnnealingLR | None" = None
        # Pending state_dicts stashed by load_checkpoint()
        self._pending_actor_scheduler_sd  = None
        self._pending_critic_scheduler_sd = None

        # ── ONNX inference session ────────────────────────────────────────
        self._onnx_actor_session = None
        self._onnx_export_done   = False

        # ── Rollout buffer ────────────────────────────────────────────────
        self.rollout_buffer = IPPOBuffer(MEMORY_CAPACITY, self.state_dim)

        # ── Per-episode state ─────────────────────────────────────────────
        self.current_action_idx  = 0
        self.last_switch_step    = -self.min_green_steps
        self._current_step       = 0
        self.state               = self._get_state()

        # IPPO has no intent vector (no GAT / intent sharing).
        # Zero placeholder so shared visualization/checkpoint code that
        # accesses current_intent does not raise AttributeError.
        self.current_intent: np.ndarray = np.zeros(1, dtype=np.float32)

        # Metrics accumulators (cleared each episode by training_loop)
        self.step_rewards: List[float] = []
        self.step_queues:  List[float] = []
        self.step_losses:  List[float] = []
        self.step_waits:   List[float] = []

        # Entropy collapse tracking
        self._entropy_history:           List[float] = []
        self._entropy_boost_remaining:   int         = 0
        self._base_entropy_coeff:        float       = self.entropy_coeff

        # Pinned tensors for fast CPU→GPU copies
        if self._use_cuda:
            try:
                self._state_buf = torch.zeros(
                    1, self.state_dim, dtype=torch.float32).pin_memory()
                self._pinned = True
            except Exception:
                self._state_buf = torch.zeros(
                    1, self.state_dim, dtype=torch.float32)
                self._pinned = False
            self._state_gpu = torch.zeros(
                1, self.state_dim, dtype=torch.float32, device=device)
        else:
            self._pinned    = False
            self._state_buf = torch.zeros(
                1, self.state_dim, dtype=torch.float32, device=device)
            self._state_gpu = self._state_buf

        # -- KL-Adaptive Clip controller --
        self._setup_kl_clip_controller()

    # ──────────────────────────────────────────────────────────────────────
    # State extraction  (MUST match MAPPOAgent._get_state() exactly)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def _get_lane_length(cls, lane_id: str) -> float:
        v = cls._lane_length_cache.get(lane_id)
        if v is None:
            v = max(traci.lane.getLength(lane_id), 1.0)
            cls._lane_length_cache[lane_id] = v
        return v

    @classmethod
    def clear_lane_length_cache(cls) -> None:
        cls._lane_length_cache.clear()

    def _get_state(self) -> np.ndarray:
        # 1. Phase one-hot
        ph = np.zeros(self.action_dim, dtype=np.float32)
        ph[self.current_action_idx] = 1.0
        vals: List[float] = list(ph)

        # 2. Moving / stopped vehicle counts per lane
        for lane_id in self.incoming_lanes:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])
        for lane_id in self.outgoing_lanes:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])

        # 3. Detector halting counts
        for det_id in TLS_DETECTOR_MAP.get(self.tls_id, []):
            vals.append(_STEP_CACHE.detector_halting(det_id))

        # 4. Per-incoming-lane queue occupancy
        for lane_id in self.incoming_lanes:
            vals.append(_lane_queue(lane_id))

        # 5. Time since last phase switch (normalised)
        steps_since = float(self._current_step - self.last_switch_step)
        vals.append(min(steps_since / 100.0, 1.0))

        # 6. Normalised total queue at this intersection
        total_q = _get_tls_queue_from_detectors(self.tls_id)
        qcap    = _QUEUE_CAPACITY_PER_TLS.get(self.tls_id, QUEUE_CAPACITY)
        vals.append(min(total_q / max(qcap, 1.0), 1.0))

        # 7. Shock-wave features (Rankine-Hugoniot)
        in_list  = list(self.incoming_lanes)
        out_list = list(self.outgoing_lanes)
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

        # 8. EV priority features
        if self._ev_feat_dim > 0:
            ev_feats = self.ev_module.get_ev_features(
                self.tls_id, list(self.incoming_lanes))
            vals.extend(ev_feats.tolist())
            self._ev_phi_prev = self._ev_phi_curr
            self._ev_phi_curr = self.ev_module.get_ev_potential(
                list(self.incoming_lanes),
                self.current_action_idx,
                self.action_phases,
            )

        return np.array(vals, dtype=np.float32)

    # ------------------------------------------------------------------
    def _setup_kl_clip_controller(self) -> None:
        """Wire the KLAdaptiveClipController to adjust clip_ratio based on KL.

        Called once from ``__init__``.  When ``use_adaptive_clip=False``
        the controller is left as ``None``.
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

    # ──────────────────────────────────────────────────────────────────────
    # Action selection
    # ──────────────────────────────────────────────────────────────────────

    def _to_device(self, state: np.ndarray) -> torch.Tensor:
        if self._use_cuda:
            self._state_buf.copy_(
                torch.from_numpy(state).unsqueeze(0))
            self._state_gpu.copy_(self._state_buf, non_blocking=True)
            return self._state_gpu
        else:
            self._state_gpu.copy_(
                torch.from_numpy(state).unsqueeze(0), non_blocking=True)
            return self._state_gpu

    def choose_action(self, deterministic: bool = False) -> int:
        with torch.no_grad():
            s_gpu   = self._to_device(self.state)
            logits  = self.actor(s_gpu)
            if deterministic:
                action = int(logits.argmax().item())
            else:
                probs  = torch.softmax(logits, dim=-1)
                action = int(torch.distributions.Categorical(probs).sample().item())
        return action

    def apply_action(self, action_idx: int, current_step: int) -> None:
        """Mirror MAPPOAgent.apply_action: respects min-green and yellow guard."""
        self._current_step = current_step
        chosen_phase       = self.action_phases[action_idx]
        try:
            active_phase  = traci.trafficlight.getPhase(self.tls_id)
            phase_state   = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            in_transition = any(c in ('y', 'Y', 'u') for c in phase_state)
        except Exception:
            active_phase  = chosen_phase
            in_transition = False

        if in_transition:
            return

        if chosen_phase != active_phase:
            if (current_step - self.last_switch_step) >= self.min_green_steps:
                traci.trafficlight.setPhase(self.tls_id, chosen_phase)
                self.last_switch_step = current_step

    def get_reward(self) -> float:
        return self.reward_func.get_reward(
            self.tls_id,
            self.incoming_lanes,
            self.outgoing_lanes,
            ev_phi_prev=self._ev_phi_prev,
            ev_phi_curr=self._ev_phi_curr,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Value / log-prob inference
    # ──────────────────────────────────────────────────────────────────────

    def get_value(self, state: np.ndarray,
                  global_state: "np.ndarray | None" = None) -> float:
        """
        Compute V(s).
        For centralised critic mode, global_state must be the concatenation
        of all agents' states (built by build_global_state_vec).
        """
        with torch.no_grad():
            if self.critic_mode == "local":
                assert self.critic is not None
                s = torch.from_numpy(state).unsqueeze(0).to(self.device)
                v = self.critic(s)
            else:
                assert self._shared_critic is not None
                assert global_state is not None
                g = torch.from_numpy(global_state).unsqueeze(0).to(self.device)
                v = self._shared_critic(g)
        return float(v[0].item())

    def get_values_batch(
        self,
        states:       np.ndarray,
        global_states: "np.ndarray | None" = None,
    ) -> np.ndarray:
        kw     = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        with torch.no_grad():
            if self.critic_mode == "local":
                assert self.critic is not None
                s = torch.from_numpy(states).to(self.device, **kw)
                with (torch.amp.autocast(device_type="cuda")
                      if use_amp else _null_ctx()):
                    v = self.critic(s).squeeze(1)
            else:
                assert self._shared_critic is not None
                assert global_states is not None
                g = torch.from_numpy(global_states).to(self.device, **kw)
                with (torch.amp.autocast(device_type="cuda")
                      if use_amp else _null_ctx()):
                    v = self._shared_critic(g).squeeze(1)
        return v.float().cpu().numpy()

    def get_log_prob(self, state: np.ndarray, action: int) -> float:
        with torch.no_grad():
            s      = torch.from_numpy(state).unsqueeze(0).to(self.device)
            logits = self.actor(s)
            lp     = torch.log_softmax(logits, dim=-1)
        return float(lp[0, action].item())

    def get_log_probs_batch(
        self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        kw      = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        with torch.no_grad():
            s  = torch.from_numpy(states).to(self.device, **kw)
            a  = torch.from_numpy(actions).long().to(self.device, **kw)
            with (torch.amp.autocast(device_type="cuda")
                  if use_amp else _null_ctx()):
                logits = self.actor(s)
                lp     = torch.log_softmax(logits.float(), dim=-1)
                lp_a   = lp.gather(1, a.unsqueeze(1)).squeeze(1)
        return lp_a.float().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────────────────

    def train_epoch(
        self,
        advantages: np.ndarray,
        returns:    np.ndarray,
        global_states: "np.ndarray | None" = None,
    ) -> Tuple[float, float]:
        """
        Run N_EPOCHS PPO mini-batch updates.

        Returns
        -------
        (mean_loss, mean_entropy)
        """
        total_loss    = 0.0
        total_entropy = 0.0
        n_batches     = 0
        use_amp       = self._use_cuda and self._scaler is not None
        buf           = self.rollout_buffer

        # Compute old values once (for clipped value loss, PPO-style)
        with torch.no_grad():
            old_values = self.get_values_batch(
                buf.states[: buf.size],
                global_states,
            )

        for _epoch in range(self.n_epochs):
            for _ in range(max(1, buf.size // self.batch_size)):
                states_b, actions_b, old_lp_b, _, idx = buf.sample(self.batch_size)
                if len(idx) == 0:
                    continue

                kw = {"non_blocking": True} if self._use_cuda else {}
                s_t      = torch.from_numpy(states_b).to(self.device, **kw)
                a_t      = torch.from_numpy(actions_b).unsqueeze(1).to(self.device, **kw)
                olp_t    = torch.from_numpy(old_lp_b).to(self.device, **kw)
                adv_t    = torch.from_numpy(advantages[idx]).to(self.device, **kw)
                ret_t    = torch.from_numpy(returns[idx]).to(self.device, **kw)
                ov_t     = torch.from_numpy(old_values[idx]).to(self.device, **kw)

                # Optional: global states for centralised critic
                g_t = None
                if global_states is not None:
                    g_t = torch.from_numpy(global_states[idx]).to(self.device, **kw)

                self.actor_optimizer.zero_grad(set_to_none=True)
                if self.critic_optimizer is not None:
                    self.critic_optimizer.zero_grad(set_to_none=True)

                with (torch.amp.autocast(device_type="cuda")
                      if use_amp else _null_ctx()):
                    # ── Actor ──────────────────────────────────────────
                    logits  = self.actor(s_t)
                    log_pi  = torch.log_softmax(logits, dim=-1)
                    new_lp  = log_pi.gather(1, a_t).squeeze(1)

                    ratio   = torch.exp(new_lp - olp_t)
                    surr1   = ratio * adv_t
                    surr2   = torch.clamp(
                        ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_t

                    entropy    = -(torch.softmax(logits, dim=-1) * log_pi).sum(1).mean()
                    actor_loss = (-torch.min(surr1, surr2).mean()
                                  - self.entropy_coeff * entropy)

                    # ── Critic ─────────────────────────────────────────
                    if self.critic_mode == "local":
                        assert self.critic is not None
                        vals     = self.critic(s_t).squeeze(1)
                    else:
                        assert self._shared_critic is not None and g_t is not None
                        vals     = self._shared_critic(g_t).squeeze(1)

                    v_clipped = ov_t + torch.clamp(
                        vals - ov_t, -self.clip_ratio, self.clip_ratio)
                    critic_loss = torch.max(
                        nn.functional.mse_loss(vals,      ret_t),
                        nn.functional.mse_loss(v_clipped, ret_t),
                    )

                    loss = actor_loss + self.value_loss_coeff * critic_loss

                if use_amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.actor_optimizer)
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    if self.critic_optimizer is not None:
                        self._scaler.unscale_(self.critic_optimizer)
                        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self._scaler.step(self.actor_optimizer)
                    if self.critic_optimizer is not None:
                        self._scaler.step(self.critic_optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    if self.critic_optimizer is not None:
                        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                    self.actor_optimizer.step()
                    if self.critic_optimizer is not None:
                        self.critic_optimizer.step()

                # ── KL-ADAPTIVE CLIP ────────────────────────────────────
                if self._clip_controller is not None:
                    with torch.no_grad():
                        post_logits  = self.actor(s_t)
                        post_log_pi  = torch.log_softmax(
                            post_logits.float(), dim=-1)
                        post_new_lp  = post_log_pi.gather(1, a_t).squeeze(1)
                        batch_kl = compute_kl_divergence(
                            olp_t.float(), post_new_lp)
                    self._clip_controller.step(batch_kl, agent_id=self.tls_id)
                # ── END KL-ADAPTIVE CLIP ────────────────────────────────

                total_loss    += float(loss.item())
                total_entropy += float(entropy.item())
                n_batches     += 1

        mean_loss    = total_loss    / max(n_batches, 1)
        mean_entropy = total_entropy / max(n_batches, 1)
        return mean_loss, mean_entropy

    def train_step(self, global_states: "np.ndarray | None" = None) -> float:
        """
        Compute GAE, run train_epoch, clear buffer.
        Returns mean_loss for logging.
        """
        if self.rollout_buffer.size < self.batch_size:
            print(f"  [IPPO-SKIP] {self.tls_id}: buffer "
                  f"{self.rollout_buffer.size} < batch_size {self.batch_size}",
                  flush=True)
            return 0.0

        advantages, returns = self.rollout_buffer.compute_gae()
        loss, mean_entropy  = self.train_epoch(advantages, returns, global_states)
        self.rollout_buffer.clear()
        self.step_losses.append(loss)

        # LR schedulers
        if self._actor_scheduler is not None:
            self._actor_scheduler.step()
        if self._critic_scheduler is not None:
            self._critic_scheduler.step()

        # Entropy collapse detection (same logic as MAPPOAgent)
        self._check_entropy_collapse(mean_entropy)
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # LR Schedulers
    # ──────────────────────────────────────────────────────────────────────

    def setup_lr_schedulers(self, n_episodes: int) -> None:
        """
        Create / restore CosineAnnealingLR schedulers for actor (and local
        critic if applicable).  Mirror of MAPPOAgent.setup_lr_schedulers().
        """
        ETA_MIN = 1e-5

        def _build(optimizer, pending_sd, role: str):
            if pending_sd is not None:
                orig_tmax = pending_sd.get("T_max", n_episodes) or n_episodes
                sched = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=orig_tmax, eta_min=ETA_MIN)
                try:
                    sched.load_state_dict(pending_sd)
                    try:
                        lr_str = f"{sched.get_last_lr()[0]:.3e}"
                    except Exception:
                        lr_str = "N/A"
                    print(f"  [IPPO-LR-RESUME] {self.tls_id}/{role}: "
                          f"T_max={orig_tmax}  last_epoch={sched.last_epoch}"
                          f"  lr={lr_str}", flush=True)
                    return sched
                except Exception as e:
                    print(f"  [IPPO-LR-RESUME] {self.tls_id}/{role}: "
                          f"load_state_dict failed ({e}) -- fresh scheduler",
                          flush=True)
            sched = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(n_episodes, 1), eta_min=ETA_MIN)
            print(f"  [IPPO-LR-FRESH] {self.tls_id}/{role}: "
                  f"T_max={n_episodes}  eta_min={ETA_MIN}", flush=True)
            return sched

        self._actor_scheduler = _build(
            self.actor_optimizer,
            getattr(self, "_pending_actor_scheduler_sd", None),
            "actor",
        )
        if self.critic_optimizer is not None:
            self._critic_scheduler = _build(
                self.critic_optimizer,
                getattr(self, "_pending_critic_scheduler_sd", None),
                "critic",
            )
        self._pending_actor_scheduler_sd  = None
        self._pending_critic_scheduler_sd = None

    # ──────────────────────────────────────────────────────────────────────
    # Entropy collapse detection (same as MAPPOAgent)
    # ──────────────────────────────────────────────────────────────────────

    def _check_entropy_collapse(self, current_entropy: float) -> None:
        self._entropy_history.append(current_entropy)
        if self._entropy_boost_remaining > 0:
            self._entropy_boost_remaining -= 1
            if self._entropy_boost_remaining == 0:
                self.entropy_coeff = self._base_entropy_coeff
                print(f"  [IPPO-ENTROPY] {self.tls_id}: boost expired -- "
                      f"coeff restored to {self.entropy_coeff:.4f}", flush=True)
        if (self._entropy_boost_remaining == 0
                and len(self._entropy_history) >= 3):
            last3 = self._entropy_history[-3:]
            if all(e < 0.01 for e in last3):
                self._entropy_boost_remaining = 5
                self.entropy_coeff = self._base_entropy_coeff * 2.0
                print(f"  [IPPO-ENTROPY-WARN] {self.tls_id}: collapse detected "
                      f"(last 3: {[f'{e:.5f}' for e in last3]}) -- "
                      f"boosting to {self.entropy_coeff:.4f} for 5 episodes",
                      flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # ONNX export (actor only)
    # ──────────────────────────────────────────────────────────────────────

    def export_onnx(self, save_dir: str) -> str:
        """
        Export the actor to ONNX for fast inference.
        Single input: state.  Single output: logits.
        """
        if not ONNX_AVAILABLE:
            return ""
        import tempfile, shutil
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"ippo_actor_{self.tls_id}.onnx")

        dummy_s  = torch.zeros(1, self.state_dim, dtype=torch.float32)
        actor_raw = (self.actor._orig_mod
                     if hasattr(self.actor, "_orig_mod") else self.actor)
        actor_raw.eval()

        tmp_dir  = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, f"ippo_actor_{self.tls_id}.onnx")
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    actor_raw.cpu(),
                    (dummy_s,),
                    tmp_path,
                    input_names  = ["state"],
                    output_names = ["logits"],
                    dynamic_axes = {"state": {0: "batch"}, "logits": {0: "batch"}},
                    opset_version       = 17,
                    do_constant_folding = True,
                )
            for fname in os.listdir(tmp_dir):
                dst = os.path.join(save_dir, fname)
                if os.path.exists(dst):
                    try:
                        os.remove(dst)
                    except Exception:
                        pass
                shutil.move(os.path.join(tmp_dir, fname), dst)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[IPPO-ONNX] Export failed for {self.tls_id}: {e}", flush=True)
            return ""
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        actor_raw.to(self.device).train()

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        sess_opts.intra_op_num_threads = 1
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if self._use_cuda else ["CPUExecutionProvider"])
        try:
            self._onnx_actor_session = ort.InferenceSession(
                path, sess_options=sess_opts, providers=providers)
            self._onnx_export_done = True
        except Exception as e:
            print(f"[IPPO-ONNX] Session failed for {self.tls_id}: {e}", flush=True)
        return path

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_spec(self) -> dict:
        return {
            "state_dim":      self.state_dim,
            "action_dim":     self.action_dim,
            "action_phases":  self.action_phases,
            "phase_links":    self.phase_links,
            "incoming_lanes": self.incoming_lanes,
            "outgoing_lanes": self.outgoing_lanes,
            "critic_mode":    self.critic_mode,
        }

    def get_cpu_state_dict(self) -> dict:
        sd = {
            "actor":  {k: v.cpu() for k, v in self.actor.state_dict().items()},
        }
        if self.critic is not None:
            sd["critic"] = {
                k: v.cpu() for k, v in self.critic.state_dict().items()}
        return sd


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Checkpoint helpers (save / load)
# ══════════════════════════════════════════════════════════════════════════════

def save_ippo_checkpoint(
    agents:    Dict[str, IPPOAgent],
    save_dir:  str,
    episode:   int,
    tag:       str,
    plotter    = None,
    writers:   list = (),
    shared_critic: "IPPOCentralCritic | None" = None,
    shared_critic_optimizer: "optim.Adam | None" = None,
    shared_critic_scheduler = None,
    rl_history:    "dict | None" = None,
    fixed_history: "dict | None" = None,
) -> None:
    """
    Persist all IPPO state to disk under
    ``save_dir/checkpoints/<tag>/``.

    Compatible with CheckpointManager's recovery flow: the caller can point
    ``load_ippo_checkpoint`` at the same directory to resume.
    """
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    ep_dir   = ckpt_dir  # flat layout: files live directly under checkpoints/
    os.makedirs(ep_dir, exist_ok=True)

    # ── Per-agent weights ─────────────────────────────────────────────────
    # NOTE: filename convention is "{tls_id}_{tag}.pt" (e.g. "TL1_ep0000.pt",
    # "TL1_emergency.pt") to match the existing checkpoints already on disk.
    for tls_id, ag in agents.items():
        path = os.path.join(ep_dir, f"{tls_id}_{tag}.pt")
        payload = {
            "episode":             episode,
            "algorithm":           "ippo",
            "critic_mode":         ag.critic_mode,
            "actor_state":         ag.actor.state_dict(),
            "actor_optimizer":     ag.actor_optimizer.state_dict(),
            "actor_scheduler":     (ag._actor_scheduler.state_dict()
                                    if ag._actor_scheduler else None),
            "current_action_idx":  ag.current_action_idx,
            "last_switch_step":    ag.last_switch_step,
            "_current_step":       ag._current_step,
            "entropy_history":     list(ag._entropy_history),
            "entropy_boost_remaining": ag._entropy_boost_remaining,
            "entropy_coeff":       ag.entropy_coeff,
            "base_entropy_coeff":  ag._base_entropy_coeff,
        }
        if ag.critic is not None:
            payload["critic_state"]     = ag.critic.state_dict()
            payload["critic_optimizer"] = ag.critic_optimizer.state_dict()
            payload["critic_scheduler"] = (ag._critic_scheduler.state_dict()
                                           if ag._critic_scheduler else None)
        torch.save(payload, path)

    # ── Centralised critic (shared) ───────────────────────────────────────
    if shared_critic is not None:
        sc_path = os.path.join(ep_dir, f"shared_critic_{tag}.pt")
        sc_payload: dict = {"critic_state": shared_critic.state_dict()}
        if shared_critic_optimizer is not None:
            sc_payload["optimizer"] = shared_critic_optimizer.state_dict()
        if shared_critic_scheduler is not None:
            sc_payload["scheduler"] = shared_critic_scheduler.state_dict()
        torch.save(sc_payload, sc_path)

    # ── Plotter history ───────────────────────────────────────────────────
    if plotter is not None:
        hist_path = os.path.join(ep_dir, f"plotter_history_{tag}.json")
        with open(hist_path, "w") as f:
            json.dump({
                "episodes":            plotter._episodes,
                "history":             {k: dict(v) for k, v in plotter._history.items()},
                "ep_duration_ms":      plotter._ep_duration_ms,
                "global_step_counter": plotter._global_step_counter,
                "global_steps":        plotter._global_steps,
                "wall_times":          plotter._wall_times,
                "step_history":        {k: {tl: pairs
                                            for tl, pairs in tl_data.items()}
                                        for k, tl_data in
                                        plotter._step_history.items()},
            }, f, indent=2)

    # ── RL / Fixed-Time comparison history ───────────────────────────────
    if rl_history is not None or fixed_history is not None:
        cmp_path = os.path.join(ep_dir, f"comparison_history_{tag}.json")
        with open(cmp_path, "w") as f:
            json.dump({
                "rl_history":    rl_history    or {},
                "fixed_history": fixed_history or {},
            }, f, indent=2)

    # ── Flush writers ─────────────────────────────────────────────────────
    for w in writers:
        try:
            w.f.flush()
        except Exception:
            pass

    # ── Update latest.json pointer ────────────────────────────────────────
    try:
        ptr = os.path.join(ckpt_dir, "latest.json")
        with open(ptr, "w") as pf:
            json.dump({"latest_ep": episode, "tag": tag,
                       "algorithm": "ippo",
                       "path": ep_dir}, pf)
    except OSError:
        pass

    print(f"  [IPPO-CKPT] Saved checkpoint '{tag}' -> {ep_dir}", flush=True)


def load_ippo_checkpoint(
    agents:   Dict[str, IPPOAgent],
    save_dir: str,
    episode:  str = "latest",
    shared_critic: "IPPOCentralCritic | None" = None,
    shared_critic_optimizer: "optim.Adam | None" = None,
) -> int:
    """
    Load a IPPO checkpoint.  Returns the restored episode number.
    """
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    # Resolve tag
    if episode == "latest":
        ptr = os.path.join(ckpt_dir, "latest.json")
        if os.path.isfile(ptr):
            with open(ptr) as pf:
                data = json.load(pf)
            tag = data.get("tag", "emergency")
        else:
            tag = "emergency"
        print(f"  [IPPO-RESUME] 'latest' resolved -> {tag}", flush=True)
    else:
        tag = episode

    restored_ep = 0
    missing = []

    for tls_id, ag in agents.items():
        # Preferred (current) flat layout: checkpoints/{tls_id}_{tag}.pt
        candidates = [
            os.path.join(ckpt_dir, f"{tls_id}_{tag}.pt"),
            # Back-compat with older/alternate layouts, just in case:
            os.path.join(ckpt_dir, tag, f"ippo_{tls_id}.pt"),
            os.path.join(ckpt_dir, tag, f"{tls_id}.pt"),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            missing.append(tls_id)
            print(f"  [IPPO-RESUME] WARNING: no checkpoint found for "
                  f"{tls_id} (tag={tag!r}); tried: {candidates}",
                  flush=True)
            continue
        ckpt = torch.load(path, map_location=ag.device, weights_only=False)
        restored_ep = max(restored_ep, int(ckpt.get("episode", 0)))

        # Actor
        ag.actor.load_state_dict(ckpt["actor_state"])
        ag.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        ag._pending_actor_scheduler_sd = ckpt.get("actor_scheduler")

        # Local critic (if present in checkpoint and agent)
        if ag.critic is not None and "critic_state" in ckpt:
            ag.critic.load_state_dict(ckpt["critic_state"])
            if ag.critic_optimizer is not None and "critic_optimizer" in ckpt:
                ag.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
            ag._pending_critic_scheduler_sd = ckpt.get("critic_scheduler")

        # Per-episode state
        ag.current_action_idx        = int(ckpt.get("current_action_idx", 0))
        ag.last_switch_step          = int(ckpt.get("last_switch_step",
                                                     -ag.min_green_steps))
        ag._current_step             = int(ckpt.get("_current_step", 0))
        ag._entropy_history          = list(ckpt.get("entropy_history", []))
        ag._entropy_boost_remaining  = int(ckpt.get("entropy_boost_remaining", 0))
        ag.entropy_coeff             = float(ckpt.get("entropy_coeff",
                                                       ENTROPY_COEFF))
        ag._base_entropy_coeff       = float(ckpt.get("base_entropy_coeff",
                                                        ENTROPY_COEFF))
        print(f"  [IPPO-RESUME] {tls_id} loaded (ep={restored_ep})", flush=True)

    # Centralised critic
    sc_candidates = [
        os.path.join(ckpt_dir, f"shared_critic_{tag}.pt"),
        os.path.join(ckpt_dir, tag, "ippo_shared_critic.pt"),
    ]
    sc_path = next((p for p in sc_candidates if os.path.isfile(p)), None)
    if shared_critic is not None and sc_path is not None:
        sc_ckpt = torch.load(sc_path, map_location="cpu", weights_only=False)
        shared_critic.load_state_dict(sc_ckpt["critic_state"])
        if shared_critic_optimizer is not None and "optimizer" in sc_ckpt:
            shared_critic_optimizer.load_state_dict(sc_ckpt["optimizer"])
        print(f"  [IPPO-RESUME] Shared critic loaded.", flush=True)

    if missing:
        raise FileNotFoundError(
            f"[IPPO-RESUME] Refusing to silently continue: no checkpoint "
            f"found for agent(s) {missing} under tag {tag!r} in {ckpt_dir}. "
            f"Fix --resume_from or the checkpoint directory, don't train "
            f"from scratch by accident."
        )

    return restored_ep


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Global state vector helper (centralised critic)
# ══════════════════════════════════════════════════════════════════════════════

def build_global_state_vec(
    agents:    Dict[str, IPPOAgent],
    state_dim: int,  # kept for API compat; actual dims come from each agent
) -> np.ndarray:
    """
    Concatenate current agent states in INTERSECTIONS order.
    Each agent may have a different state_dim, so states are concatenated
    sequentially (variable-width) rather than into fixed-width slots.
    Shape: (sum of each agent's state_dim,)
    """
    parts = []
    for tls_id in INTERSECTIONS:
        ag = agents.get(tls_id)
        if ag is not None:
            parts.append(ag.state.astype(np.float32))
        else:
            # agent missing: fill zeros using a representative dim
            fallback = next(iter(agents.values())).state_dim if agents else state_dim
            parts.append(np.zeros(fallback, dtype=np.float32))
    return np.concatenate(parts)