"""
vanilla_ppo.py
==============
Absolute single-agent Vanilla PPO for SmartTraffic-OM.

ONE agent controls the ENTIRE traffic network.
ONE actor (shared backbone + per-intersection heads), ONE critic, ONE buffer.

Design
------
* Global state = concat of all intersections' local states.
* Joint action = tuple of per-intersection phase choices.
* Joint log-prob = sum of per-intersection log-probs (factored heads).
* Single PPO update over the joint trajectory.
* No GAT, no intent sharing, no KL-adaptive LR/clip.
* Compatible with CheckpointManager, EpisodeAutoSaver, RichUI,
  TrainingPlotter, ResultsWriter, TBComparisonLogger.

Usage
-----
  python main.py --sim_file sumo/OM.sumocfg --algorithm vanilla_ppo
"""

from __future__ import annotations

import os
import json
import warnings
import platform
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo

from .config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_RATIO, BATCH_SIZE, N_EPOCHS, HIDDEN_DIM, MEMORY_CAPACITY,
    TOTAL_STEPS, INTERSECTIONS, TLS_DETECTOR_MAP, UPSTREAM_LANES,
    DOWNSTREAM_LANES, QUEUE_CAPACITY,
    _QUEUE_CAPACITY_PER_TLS, EV_PRIORITY_ENABLED,
    MIN_GREEN_STEPS, STEP_LENGTH, REWARD_CLIP, V_FREE,
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
    traci = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

__all__ = [
    "VanillaPPONetwork",
    "VanillaPPOBuffer",
    "VanillaPPOAgent",
]

_VPP_COMPILE_BROKEN = False


def _vpp_try_compile(actor, critic, state_dim, device):
    global _VPP_COMPILE_BROKEN
    if _VPP_COMPILE_BROKEN:
        return actor, critic
    ca = torch.compile(actor, mode="default")
    cc = torch.compile(critic, mode="default")
    try:
        with torch.no_grad():
            d = torch.zeros(2, state_dim, device=device)
            ca(d)
            cc(d)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return ca, cc
    except Exception as exc:
        _VPP_COMPILE_BROKEN = True
        print(f"[torch.compile] VPP warm-up failed ({exc.__class__.__name__}: "
              f"{exc}). Falling back to eager.", flush=True)
        try:
            torch._dynamo.reset()
        except Exception:
            pass
        return actor, critic


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Rollout Buffer  (stores global states + joint actions)
# ══════════════════════════════════════════════════════════════════════════════

class VanillaPPOBuffer:
    """Fixed-capacity ring buffer for the single centralized agent."""

    def __init__(self, capacity: int, global_state_dim: int,
                 n_intersections: int) -> None:
        self.capacity = capacity
        self.global_state_dim = global_state_dim
        self.n_int = n_intersections

        self.states      = np.zeros((capacity, global_state_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, n_intersections), dtype=np.int64)
        self.rewards     = np.zeros(capacity, dtype=np.float32)
        self.values      = np.zeros(capacity, dtype=np.float32)
        self.log_probs   = np.zeros(capacity, dtype=np.float32)
        self.dones       = np.zeros(capacity, dtype=np.float32)
        self.next_values = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def push(
        self,
        global_state: np.ndarray,
        actions:      np.ndarray,
        reward:       float,
        value:        float,
        log_prob:     float,
        done:         float,
        next_value:   float = 0.0,
    ) -> None:
        idx = self.size % self.capacity
        self.states[idx]      = global_state
        self.actions[idx]     = actions
        self.rewards[idx]     = reward
        self.values[idx]      = value
        self.log_probs[idx]   = log_prob
        self.dones[idx]       = done
        self.next_values[idx] = next_value
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(
        self,
        gamma: float = GAMMA,
        lam:   float = GAE_LAMBDA,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def sample(self, batch_size: int):
        idx = np.random.choice(
            self.size, min(batch_size, self.size), replace=False)
        return self.states[idx], self.actions[idx], idx

    def clear(self) -> None:
        self.size = 0


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Centralized Networks
# ══════════════════════════════════════════════════════════════════════════════

class VanillaPPONetwork(nn.Module):
    """
    Single centralized actor + critic for the entire network.

    Actor
    -----
    Shared 2-layer backbone -> per-intersection action heads.
    Each head outputs logits for that intersection's green phases.

    Critic
    ------
    Shared 2-layer backbone -> scalar value.
    """

    def __init__(
        self,
        global_state_dim: int,
        action_dims: Dict[str, int],
    ) -> None:
        super().__init__()
        self.action_dims = action_dims
        self.tls_ids = list(action_dims.keys())
        n_heads = len(self.tls_ids)

        # ── Shared actor backbone ─────────────────────────────────────────
        self.actor_backbone = nn.Sequential(
            nn.Linear(global_state_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        # Per-intersection action heads
        self.actor_heads = nn.ModuleDict({
            tls: nn.Linear(HIDDEN_DIM, adim)
            for tls, adim in action_dims.items()
        })

        # ── Critic ────────────────────────────────────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

        # ── Orthogonal init ──────────────────────────────────────────────
        for m in self.actor_backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        for head in self.actor_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def get_logits(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return per-intersection logits from the shared backbone."""
        h = self.actor_backbone(global_state)
        return {tls: self.actor_heads[tls](h) for tls in self.tls_ids}

    def get_value(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.critic(global_state)

    def forward(
        self, global_state: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return self.get_logits(global_state), self.get_value(global_state)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Single Centralized Agent
# ══════════════════════════════════════════════════════════════════════════════

class VanillaPPOAgent:
    """
    ONE agent controls ALL intersections.

    Maintains per-intersection state extraction for compatibility with the
    existing training infrastructure, but uses a single centralized network.
    """

    _lane_length_cache: Dict[str, float] = {}

    @property
    def clip_ratio(self) -> float:
        return self._clip_holder[0]

    def __init__(
        self,
        device:      torch.device,
        reward_func: "BaseRewardFunction | None"  = None,
        ev_module:   "BaseEVPriorityModule | None" = None,
    ) -> None:
        self.device    = device
        self._use_cuda = device.type == "cuda"

        # ── Hyperparameters ──────────────────────────────────────────────
        self.gamma            = GAMMA
        self.gae_lambda       = GAE_LAMBDA
        self._clip_holder     = [CLIP_RATIO]
        self.entropy_coeff    = ENTROPY_COEFF
        self.value_loss_coeff = VALUE_LOSS_COEFF
        self.n_epochs         = N_EPOCHS
        self.batch_size       = BATCH_SIZE
        self.min_green_steps  = MIN_GREEN_STEPS
        self.learning_rate    = LEARNING_RATE

        # ── Per-intersection topology ────────────────────────────────────
        self.tls_ids = list(INTERSECTIONS)
        self._tls_info: Dict[str, dict] = {}
        action_dims: Dict[str, int] = {}
        total_state_dim = 0

        for tls_id in self.tls_ids:
            inc, out = _get_lanes_for_tls(tls_id)
            phases, links = _get_green_phases_and_links(tls_id)
            if not phases:
                raise RuntimeError(f"No valid green phases for TLS '{tls_id}'.")
            n_in   = len(inc)
            n_out  = len(out)
            n_dets = len(TLS_DETECTOR_MAP.get(tls_id, []))
            ev_dim = (2 * n_in + 2) if EV_PRIORITY_ENABLED else 0
            sd = (len(phases) + 2*(n_in+n_out) + n_dets + n_in
                  + 1 + 1 + 2 + ev_dim)

            self._tls_info[tls_id] = {
                "incoming_lanes": inc,
                "outgoing_lanes": out,
                "action_phases":  phases,
                "phase_links":    links,
                "state_dim":      sd,
                "ev_feat_dim":    ev_dim,
                "action_dim":     len(phases),
                "last_switch_step": -self.min_green_steps,
                "current_action_idx": 0,
                "ev_phi_prev": 0.0,
                "ev_phi_curr": 0.0,
            }
            action_dims[tls_id] = len(phases)
            total_state_dim += sd

        self.global_state_dim = total_state_dim
        self.n_intersections  = len(self.tls_ids)

        print(f"  [VanillaPPO] Single centralized agent | "
              f"global_state_dim={total_state_dim} | "
              f"intersections={self.n_intersections} | "
              f"action_dims={action_dims}")
        print(f"  [VanillaPPO] lr={self.learning_rate} gamma={self.gamma} "
              f"lambda={self.gae_lambda} clip={self.clip_ratio} "
              f"entropy={self.entropy_coeff}")

        # ── Injectable modules ────────────────────────────────────────────
        self.reward_func: BaseRewardFunction   = reward_func or BackPressureReward()
        self.ev_module:   BaseEVPriorityModule = ev_module   or DefaultEVPriority()

        # ── Network ──────────────────────────────────────────────────────
        self.network = VanillaPPONetwork(total_state_dim, action_dims).to(device)

        if self._use_cuda and platform.system() != "Windows":
            self.network, _ = _vpp_try_compile(
                self.network, None, total_state_dim, device)

        self._scaler = (
            torch.amp.GradScaler(device="cuda") if self._use_cuda else None)

        # ── Single optimizer ─────────────────────────────────────────────
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=self.learning_rate)

        # ── LR scheduler ─────────────────────────────────────────────────
        self._scheduler: "optim.lr_scheduler.CosineAnnealingLR | None" = None
        self._pending_scheduler_sd = None

        # ── Rollout buffer ────────────────────────────────────────────────
        self.rollout_buffer = VanillaPPOBuffer(
            MEMORY_CAPACITY, total_state_dim, self.n_intersections)

        # ── Per-episode metrics (per TLS, for compatibility) ─────────────
        self.step_rewards: Dict[str, List[float]] = {t: [] for t in self.tls_ids}
        self.step_queues:  Dict[str, List[float]] = {t: [] for t in self.tls_ids}
        self.step_losses:  List[float] = []
        self.step_waits:   Dict[str, List[float]] = {t: [] for t in self.tls_ids}

        # Entropy collapse tracking
        self._entropy_history:         List[float] = []
        self._entropy_boost_remaining: int         = 0
        self._base_entropy_coeff:      float       = self.entropy_coeff

        # Pinned tensors
        if self._use_cuda:
            try:
                self._state_buf = torch.zeros(
                    1, total_state_dim, dtype=torch.float32).pin_memory()
                self._pinned = True
            except Exception:
                self._state_buf = torch.zeros(
                    1, total_state_dim, dtype=torch.float32)
                self._pinned = False
            self._state_gpu = torch.zeros(
                1, total_state_dim, dtype=torch.float32, device=device)
        else:
            self._pinned    = False
            self._state_buf = torch.zeros(
                1, total_state_dim, dtype=torch.float32, device=device)
            self._state_gpu = self._state_buf

        # Current global state & joint action
        self.global_state = np.zeros(total_state_dim, dtype=np.float32)
        self.joint_action = np.zeros(self.n_intersections, dtype=np.int64)
        self._current_step = 0

        # Placeholder for shared visualization code
        self.current_intent: np.ndarray = np.zeros(1, dtype=np.float32)

    # ── Duck-typing compatibility for CheckpointManager.save_now() ──────────
    #    CheckpointManager expects .actor, .critic, .actor_optimizer,
    #    .critic_optimizer, ._actor_scheduler, ._critic_scheduler,
    #    .current_action_idx, .last_switch_step

    @property
    def actor(self):
        """Map actor -> the whole network (has .state_dict())."""
        return self.network

    @property
    def critic(self):
        return None

    @property
    def actor_optimizer(self):
        return self.optimizer

    @property
    def critic_optimizer(self):
        return None

    @property
    def _actor_scheduler(self):
        return self._scheduler

    @property
    def _critic_scheduler(self):
        return None

    @property
    def current_action_idx(self):
        first_tls = self.tls_ids[0] if self.tls_ids else None
        if first_tls and first_tls in self._tls_info:
            return self._tls_info[first_tls]["current_action_idx"]
        return 0

    @property
    def last_switch_step(self):
        first_tls = self.tls_ids[0] if self.tls_ids else None
        if first_tls and first_tls in self._tls_info:
            return self._tls_info[first_tls]["last_switch_step"]
        return -self.min_green_steps

    @property
    def incoming_lanes(self):
        lanes = set()
        for info in self._tls_info.values():
            lanes.update(info["incoming_lanes"])
        return lanes

    @property
    def outgoing_lanes(self):
        lanes = set()
        for info in self._tls_info.values():
            lanes.update(info["outgoing_lanes"])
        return lanes

    # ──────────────────────────────────────────────────────────────────────
    # Per-intersection state extraction (same features as MAPPOAgent)
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def clear_lane_length_cache(cls) -> None:
        cls._lane_length_cache.clear()

    def _get_tls_state(self, tls_id: str) -> np.ndarray:
        info = self._tls_info[tls_id]
        action_dim    = info["action_dim"]
        incoming      = info["incoming_lanes"]
        outgoing      = info["outgoing_lanes"]
        phases        = info["action_phases"]
        cur_act       = info["current_action_idx"]
        ev_feat_dim   = info["ev_feat_dim"]

        ph = np.zeros(action_dim, dtype=np.float32)
        ph[cur_act] = 1.0
        vals: List[float] = list(ph)

        for lane_id in incoming:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])
        for lane_id in outgoing:
            m, s = _moving_stopped_counts(lane_id)
            vals.extend([float(m), float(s)])

        for det_id in TLS_DETECTOR_MAP.get(tls_id, []):
            vals.append(_STEP_CACHE.detector_halting(det_id))

        for lane_id in incoming:
            vals.append(_lane_queue(lane_id))

        steps_since = float(self._current_step - info["last_switch_step"])
        vals.append(min(steps_since / 100.0, 1.0))

        total_q = _get_tls_queue_from_detectors(tls_id)
        qcap    = _QUEUE_CAPACITY_PER_TLS.get(tls_id, QUEUE_CAPACITY)
        vals.append(min(total_q / max(qcap, 1.0), 1.0))

        in_list  = list(incoming)
        out_list = list(outgoing)
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

        if ev_feat_dim > 0:
            ev_feats = self.ev_module.get_ev_features(tls_id, list(incoming))
            vals.extend(ev_feats.tolist())
            info["ev_phi_prev"] = info["ev_phi_curr"]
            info["ev_phi_curr"] = self.ev_module.get_ev_potential(
                list(incoming), cur_act, phases)

        return np.array(vals, dtype=np.float32)

    def _build_global_state(self) -> np.ndarray:
        parts = [self._get_tls_state(t) for t in self.tls_ids]
        self.global_state = np.concatenate(parts)
        return self.global_state

    # ──────────────────────────────────────────────────────────────────────
    # Action selection (joint)
    # ──────────────────────────────────────────────────────────────────────

    def _to_device(self, state: np.ndarray) -> torch.Tensor:
        if self._use_cuda:
            self._state_buf.copy_(torch.from_numpy(state).unsqueeze(0))
            self._state_gpu.copy_(self._state_buf, non_blocking=True)
            return self._state_gpu
        self._state_gpu.copy_(
            torch.from_numpy(state).unsqueeze(0), non_blocking=True)
        return self._state_gpu

    def choose_action(self, deterministic: bool = False) -> Dict[str, int]:
        """Return per-intersection action dict."""
        with torch.no_grad():
            s_gpu = self._to_device(self.global_state)
            logits_dict = self.network.get_logits(s_gpu)

            actions = {}
            total_lp = 0.0
            for tls_id in self.tls_ids:
                logits = logits_dict[tls_id]
                if deterministic:
                    a = int(logits.argmax().item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    a = int(torch.distributions.Categorical(probs).sample().item())
                actions[tls_id] = a
                if not deterministic:
                    lp = torch.log_softmax(logits, dim=-1)
                    total_lp += float(lp[0, a].item())
                else:
                    total_lp = 0.0

            self.joint_action = np.array(
                [actions[t] for t in self.tls_ids], dtype=np.int64)
        return actions

    def apply_actions(
        self, actions: Dict[str, int], current_step: int,
    ) -> None:
        """Apply per-intersection actions to SUMO."""
        self._current_step = current_step
        for tls_id, action_idx in actions.items():
            info = self._tls_info[tls_id]
            info["current_action_idx"] = action_idx
            chosen_phase = info["action_phases"][action_idx]
            try:
                active_phase = traci.trafficlight.getPhase(tls_id)
                phase_state  = traci.trafficlight.getRedYellowGreenState(tls_id)
                in_transition = any(c in ('y', 'Y', 'u') for c in phase_state)
            except Exception:
                active_phase  = chosen_phase
                in_transition = False
            if in_transition:
                continue
            if chosen_phase != active_phase:
                if (current_step - info["last_switch_step"]) >= self.min_green_steps:
                    traci.trafficlight.setPhase(tls_id, chosen_phase)
                    info["last_switch_step"] = current_step

    def get_reward(self) -> float:
        """Sum of per-intersection rewards."""
        total = 0.0
        for tls_id in self.tls_ids:
            info = self._tls_info[tls_id]
            total += self.reward_func.get_reward(
                tls_id,
                info["incoming_lanes"],
                info["outgoing_lanes"],
                ev_phi_prev=info["ev_phi_prev"],
                ev_phi_curr=info["ev_phi_curr"],
            )
        return total

    # ──────────────────────────────────────────────────────────────────────
    # Value / log-prob inference (batch)
    # ──────────────────────────────────────────────────────────────────────

    def get_values_batch(self, states: np.ndarray) -> np.ndarray:
        kw     = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device, **kw)
            with (torch.amp.autocast(device_type="cuda")
                  if use_amp else _null_ctx()):
                v = self.network.get_value(s).squeeze(1)
        return v.float().cpu().numpy()

    def get_joint_log_probs_batch(
        self, states: np.ndarray, actions: np.ndarray,
    ) -> np.ndarray:
        """Compute joint log-prob for batch of (state, joint_action) pairs."""
        kw      = {"non_blocking": True} if self._use_cuda else {}
        use_amp = self._use_cuda and self._scaler is not None
        n_samples = len(states)
        with torch.no_grad():
            s = torch.from_numpy(states).to(self.device, **kw)
            with (torch.amp.autocast(device_type="cuda")
                  if use_amp else _null_ctx()):
                logits_dict = self.network.get_logits(s)
                total_lp = torch.zeros(n_samples, device=self.device)
                for i, tls_id in enumerate(self.tls_ids):
                    logits = logits_dict[tls_id]
                    a_i = torch.from_numpy(
                        actions[:, i]).long().to(self.device)
                    lp_i = torch.log_softmax(logits.float(), dim=-1)
                    total_lp += lp_i.gather(1, a_i.unsqueeze(1)).squeeze(1)
        return total_lp.float().cpu().numpy()

    # ──────────────────────────────────────────────────────────────────────
    # PPO update (vanilla)
    # ──────────────────────────────────────────────────────────────────────

    def train_epoch(
        self, advantages: np.ndarray, returns: np.ndarray,
    ) -> Tuple[float, float]:
        total_loss    = 0.0
        total_entropy = 0.0
        n_batches     = 0
        use_amp       = self._use_cuda and self._scaler is not None
        buf           = self.rollout_buffer

        with torch.no_grad():
            old_values = self.get_values_batch(buf.states[: buf.size])
            old_lp_all = self.get_joint_log_probs_batch(
                buf.states[: buf.size], buf.actions[: buf.size])

        for _epoch in range(self.n_epochs):
            for _ in range(max(1, buf.size // self.batch_size)):
                states_b, actions_b, idx = buf.sample(self.batch_size)
                if len(idx) == 0:
                    continue

                kw      = {"non_blocking": True} if self._use_cuda else {}
                s_t     = torch.from_numpy(states_b).to(self.device, **kw)
                a_t     = torch.from_numpy(actions_b).to(self.device, **kw)
                old_lp  = torch.from_numpy(old_lp_all[idx]).to(self.device, **kw)
                adv_t   = torch.from_numpy(advantages[idx]).to(self.device, **kw)
                ret_t   = torch.from_numpy(returns[idx]).to(self.device, **kw)
                old_v   = torch.from_numpy(old_values[idx]).to(self.device, **kw)

                self.optimizer.zero_grad(set_to_none=True)

                with (torch.amp.autocast(device_type="cuda")
                      if use_amp else _null_ctx()):
                    # ── Joint log-prob ───────────────────────────────────
                    logits_dict = self.network.get_logits(s_t)
                    new_lp = torch.zeros(len(idx), device=self.device)
                    entropy = torch.zeros(len(idx), device=self.device)
                    for i, tls_id in enumerate(self.tls_ids):
                        logits = logits_dict[tls_id]
                        lp_i   = torch.log_softmax(logits, dim=-1)
                        a_i    = a_t[:, i].unsqueeze(1)
                        new_lp += lp_i.gather(1, a_i).squeeze(1)
                        ent_i  = -(torch.softmax(logits, dim=-1) * lp_i).sum(1)
                        entropy += ent_i

                    # ── PPO-clip actor loss ──────────────────────────────
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(
                        ratio, 1 - self.clip_ratio,
                        1 + self.clip_ratio) * adv_t
                    actor_loss = (-torch.min(surr1, surr2).mean()
                                  - self.entropy_coeff * entropy.mean())

                    # ── Clipped critic loss ──────────────────────────────
                    vals     = self.network.get_value(s_t).squeeze(1)
                    v_clipped = old_v + torch.clamp(
                        vals - old_v, -self.clip_ratio, self.clip_ratio)
                    critic_loss = torch.max(
                        nn.functional.mse_loss(vals,    ret_t),
                        nn.functional.mse_loss(v_clipped, ret_t),
                    )
                    loss = actor_loss + self.value_loss_coeff * critic_loss

                if use_amp:
                    self._scaler.scale(loss).backward()
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), 0.5)
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.network.parameters(), 0.5)
                    self.optimizer.step()

                total_loss    += float(loss.item())
                total_entropy += float(entropy.mean().item())
                n_batches     += 1

        return (total_loss / max(n_batches, 1),
                total_entropy / max(n_batches, 1))

    def train_step(self) -> float:
        if self.rollout_buffer.size < self.batch_size:
            print(f"  [VPP-SKIP] buffer {self.rollout_buffer.size} "
                  f"< batch_size {self.batch_size}", flush=True)
            return 0.0

        advantages, returns = self.rollout_buffer.compute_gae()
        loss, mean_entropy  = self.train_epoch(advantages, returns)
        self.rollout_buffer.clear()
        self.step_losses.append(loss)

        if self._scheduler is not None:
            self._scheduler.step()

        self._check_entropy_collapse(mean_entropy)
        return loss

    # ──────────────────────────────────────────────────────────────────────
    # LR Scheduler
    # ──────────────────────────────────────────────────────────────────────

    def setup_lr_scheduler(self, n_episodes: int) -> None:
        ETA_MIN = 1e-5
        if self._pending_scheduler_sd is not None:
            sd = self._pending_scheduler_sd
            orig_tmax = sd.get("T_max", n_episodes) or n_episodes
            sched = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=orig_tmax, eta_min=ETA_MIN)
            try:
                sched.load_state_dict(sd)
                print(f"  [VPP-LR-RESUME] T_max={orig_tmax} "
                      f"lr={sched.get_last_lr()[0]:.3e}", flush=True)
                self._scheduler = sched
                self._pending_scheduler_sd = None
                return
            except Exception as e:
                print(f"  [VPP-LR-RESUME] load failed ({e}) -- fresh",
                      flush=True)
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(n_episodes, 1), eta_min=ETA_MIN)
        print(f"  [VPP-LR-FRESH] T_max={n_episodes}", flush=True)
        self._pending_scheduler_sd = None

    # ──────────────────────────────────────────────────────────────────────
    # Entropy collapse detection
    # ──────────────────────────────────────────────────────────────────────

    def _check_entropy_collapse(self, current_entropy: float) -> None:
        self._entropy_history.append(current_entropy)
        if self._entropy_boost_remaining > 0:
            self._entropy_boost_remaining -= 1
            if self._entropy_boost_remaining == 0:
                self.entropy_coeff = self._base_entropy_coeff
                print(f"  [VPP-ENTROPY] boost expired -- "
                      f"coeff={self.entropy_coeff:.4f}", flush=True)
        if (self._entropy_boost_remaining == 0
                and len(self._entropy_history) >= 3):
            last3 = self._entropy_history[-3:]
            if all(e < 0.01 for e in last3):
                self._entropy_boost_remaining = 5
                self.entropy_coeff = self._base_entropy_coeff * 2.0
                print(f"  [VPP-ENTROPY-WARN] collapse "
                      f"{[f'{e:.5f}' for e in last3]} -- "
                      f"boost to {self.entropy_coeff:.4f}", flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # ONNX export (single actor)
    # ──────────────────────────────────────────────────────────────────────

    def export_onnx(self, save_dir: str) -> str:
        if not ONNX_AVAILABLE:
            return ""
        import tempfile, shutil
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, "vanilla_ppo_actor.onnx")
        dummy = torch.zeros(1, self.global_state_dim, dtype=torch.float32)
        raw = (self.network._orig_mod
               if hasattr(self.network, "_orig_mod") else self.network)
        raw.eval()
        tmp_dir  = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "vanilla_ppo_actor.onnx")
        try:
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    raw.cpu(), (dummy,), tmp_path,
                    input_names=["global_state"],
                    output_names=["critic_value"],
                    dynamic_axes={"global_state": {0: "batch"}},
                    opset_version=17,
                    do_constant_folding=True,
                )
            for f in os.listdir(tmp_dir):
                dst = os.path.join(save_dir, f)
                if os.path.exists(dst):
                    try: os.remove(dst)
                    except Exception: pass
                shutil.move(os.path.join(tmp_dir, f), dst)
        except Exception as e:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"[VPP-ONNX] Export failed: {e}", flush=True)
            return ""
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        raw.to(self.device).train()
        return path

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_spec(self) -> dict:
        return {
            "global_state_dim": self.global_state_dim,
            "n_intersections":  self.n_intersections,
            "tls_ids":          self.tls_ids,
            "action_dims":      {t: self._tls_info[t]["action_dim"]
                                 for t in self.tls_ids},
        }

    def get_cpu_state_dict(self) -> dict:
        return {
            "network": {k: v.cpu()
                        for k, v in self.network.state_dict().items()},
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_vanilla_ppo_checkpoint(
    agent:     VanillaPPOAgent,
    save_dir:  str,
    episode:   int,
    tag:       str,
    plotter    = None,
    writers:   list = (),
    rl_history:    "dict | None" = None,
    fixed_history: "dict | None" = None,
) -> None:
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    path = os.path.join(ckpt_dir, f"vanilla_ppo_{tag}.pt")
    payload = {
        "episode":             episode,
        "algorithm":           "vanilla_ppo",
        "network_state":       agent.network.state_dict(),
        "optimizer":           agent.optimizer.state_dict(),
        "scheduler":           (agent._scheduler.state_dict()
                                if agent._scheduler else None),
        "tls_info":            {
            t: {
                "current_action_idx": info["current_action_idx"],
                "last_switch_step":   info["last_switch_step"],
            }
            for t, info in agent._tls_info.items()
        },
        "_current_step":           agent._current_step,
        "entropy_history":         list(agent._entropy_history),
        "entropy_boost_remaining": agent._entropy_boost_remaining,
        "entropy_coeff":           agent.entropy_coeff,
        "base_entropy_coeff":      agent._base_entropy_coeff,
    }
    torch.save(payload, path)

    if plotter is not None:
        hp = os.path.join(ckpt_dir, f"plotter_history_{tag}.json")
        with open(hp, "w") as f:
            json.dump({
                "episodes":       plotter._episodes,
                "history":        {k: dict(v) for k, v in plotter._history.items()},
                "ep_duration_ms": plotter._ep_duration_ms,
            }, f, indent=2)

    if rl_history is not None or fixed_history is not None:
        cp = os.path.join(ckpt_dir, f"comparison_history_{tag}.json")
        with open(cp, "w") as f:
            json.dump({"rl_history": rl_history or {},
                       "fixed_history": fixed_history or {}}, f, indent=2)

    for w in writers:
        try: w.f.flush()
        except Exception: pass

    try:
        with open(os.path.join(ckpt_dir, "latest.json"), "w") as pf:
            json.dump({"latest_ep": episode, "tag": tag,
                       "algorithm": "vanilla_ppo", "path": ckpt_dir}, pf)
    except OSError:
        pass

    print(f"  [VPP-CKPT] Saved '{tag}' -> {ckpt_dir}", flush=True)


def load_vanilla_ppo_checkpoint(
    agent:    VanillaPPOAgent,
    save_dir: str,
    episode:  str = "latest",
) -> int:
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    if episode == "latest":
        ptr = os.path.join(ckpt_dir, "latest.json")
        if os.path.isfile(ptr):
            with open(ptr) as pf:
                tag = json.load(pf).get("tag", "emergency")
        else:
            tag = "emergency"
        print(f"  [VPP-RESUME] 'latest' -> {tag}", flush=True)
    else:
        tag = episode

    ep_num = tag.replace("ep", "").lstrip("0") or "0"
    candidates = [
        os.path.join(ckpt_dir, f"vanilla_ppo_{tag}.pt"),
        os.path.join(ckpt_dir, tag, "__vpp__.pt"),
        os.path.join(ckpt_dir, f"__vpp___{tag}.pt"),
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            f"[VPP-RESUME] No checkpoint for tag {tag!r} in {ckpt_dir}.")

    ckpt = torch.load(path, map_location=agent.device, weights_only=False)
    restored_ep = int(ckpt.get("episode", 0))

    if "network_state" in ckpt:
        agent.network.load_state_dict(ckpt["network_state"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        agent._pending_scheduler_sd = ckpt.get("scheduler")
        tls_state = ckpt.get("tls_info", {})
        for tls_id, info in agent._tls_info.items():
            if tls_id in tls_state:
                ts = tls_state[tls_id]
                info["current_action_idx"] = int(ts.get("current_action_idx", 0))
                info["last_switch_step"]   = int(ts.get("last_switch_step",
                                                        -agent.min_green_steps))
    elif "actor_state" in ckpt:
        agent.network.load_state_dict(ckpt["actor_state"])
        if ckpt.get("actor_optimizer") is not None:
            agent.optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent._pending_scheduler_sd = ckpt.get("actor_scheduler")
        ca = ckpt.get("current_action_idx", 0)
        lss = ckpt.get("last_switch_step", -agent.min_green_steps)
        for tls_id, info in agent._tls_info.items():
            info["current_action_idx"] = int(ca)
            info["last_switch_step"]   = int(lss)

    agent._current_step = int(ckpt.get("_current_step", 0))
    agent._entropy_history         = list(ckpt.get("entropy_history", []))
    agent._entropy_boost_remaining = int(ckpt.get("entropy_boost_remaining", 0))
    agent.entropy_coeff            = float(ckpt.get("entropy_coeff",
                                                    ENTROPY_COEFF))
    agent._base_entropy_coeff      = float(ckpt.get("base_entropy_coeff",
                                                     ENTROPY_COEFF))
    print(f"  [VPP-RESUME] Loaded (ep={restored_ep})", flush=True)
    return restored_ep
