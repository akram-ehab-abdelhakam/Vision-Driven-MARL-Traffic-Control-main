from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from .config import (
    UNIFIED_BP_CAPACITY, QUEUE_CAPACITY, _QUEUE_CAPACITY_PER_TLS,
    EV_PRIORITY_ENABLED, EV_ALPHA, GAMMA, REWARD_CLIP,
    UPSTREAM_LANES,
)
from .sumo_env import _lane_queue, _get_tls_queue_from_detectors, _get_tls_mean_wait, _STEP_CACHE

__all__ = ['BaseRewardFunction', 'BackPressureReward', 'WaitingTimeReward', 'QueueReductionReward', 'MaxPressureReward']


class BaseRewardFunction(ABC):
    """Abstract interface for per-agent reward functions.

    All reward functions must implement ``get_reward()``.
    The method receives the TLS id, incoming and outgoing lane lists,
    and any extra per-agent context passed as keyword arguments.
    """

    @property
    def name(self) -> str:
        """Human-readable name shown in the UI and logs."""
        return self.__class__.__name__

    @abstractmethod
    def get_reward(
        self,
        tls_id: str,
        incoming_lanes,
        outgoing_lanes,
        **kwargs,
    ) -> float:
        """Compute and return the scalar reward for one agent at the current step.

        Parameters
        ----------
        tls_id : str
            Traffic light system identifier.
        incoming_lanes : sequence of str
            Ordered list of incoming lane IDs controlled by this TLS.
        outgoing_lanes : sequence of str
            Ordered list of outgoing lane IDs from this TLS.
        **kwargs :
            Optional extra context (e.g. ``ev_phi_prev``, ``ev_phi_curr``
            for ``BackPressureReward``).
        """


class BackPressureReward(BaseRewardFunction):
    """Back-Pressure reward with per-TLS capacity normalisation + optional EV potential shaping.

    Formula
    -------
    r_i = -mean_{k} [ (q_in_k - q_out_k) / capacity(tls_id) ]
          + EV_ALPHA * (GAMMA * Phi(s') - Phi(s))   [when EV_PRIORITY_ENABLED]

    Per-TLS capacity
    ----------------
    Each intersection now divides by its own ``_QUEUE_CAPACITY_PER_TLS``
    value (TL1->90, TL2->70, TL3/TL4->20), which reflects the actual
    lane count and saturation flow of that intersection.  This prevents
    TL1 (the largest intersection with 16 lanes) from producing reward
    magnitudes ~4.5x larger than TL3/TL4 (8-10 lanes), which was the
    primary cause of TL1 reward instability and gradient dominance.

    The fallback is ``UNIFIED_BP_CAPACITY`` (50.0) for any unknown TLS.
    """

    @property
    def name(self) -> str:
        return "BackPressure"

    def get_reward(
        self,
        tls_id: str,
        incoming_lanes,
        outgoing_lanes,
        ev_phi_prev: float = 0.0,
        ev_phi_curr: float = 0.0,
        **kwargs,
    ) -> float:
        qcap    = float(_QUEUE_CAPACITY_PER_TLS.get(tls_id, UNIFIED_BP_CAPACITY))
        out_lanes = list(outgoing_lanes)
        n_pairs   = max(len(incoming_lanes), 1)

        pressure = 0.0
        for k, in_lane in enumerate(incoming_lanes):
            q_in     = _lane_queue(in_lane) / qcap
            out_lane = out_lanes[k % len(out_lanes)] if out_lanes else None
            q_out    = (_lane_queue(out_lane) / qcap) if out_lane else 0.0
            pressure += (q_in - q_out)

        raw = -(pressure / n_pairs)

        # -- EV potential-based reward shaping (Ng et al. 1999) ------------------
        # r'(s,a,s') = r_BP(s,a) + α·(γ·Φ(s') − Φ(s))
        # _ev_phi_curr = Φ(s') computed in _get_state() AFTER simulationStep()
        # _ev_phi_prev = Φ(s)  cached from the previous _get_state() call
        # This formulation preserves the optimal policy (PPO-safe) and adds
        # a positive bonus when EVs hold green or a negative penalty when a
        # green EV lane is about to turn red.
        if EV_PRIORITY_ENABLED and (ev_phi_prev != 0.0 or ev_phi_curr != 0.0):
            shaping = EV_ALPHA * (GAMMA * ev_phi_curr - ev_phi_prev)
            raw = raw + shaping

        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))


class WaitingTimeReward(BaseRewardFunction):
    """Reward = -mean accumulated waiting time across all upstream vehicles.

    Encourages the agent to minimise overall vehicle delay at the intersection.
    """

    @property
    def name(self) -> str:
        return "WaitingTime"

    def get_reward(
        self,
        tls_id: str,
        incoming_lanes,
        outgoing_lanes,
        **kwargs,
    ) -> float:
        mean_wait = _get_tls_mean_wait(tls_id)
        raw = -mean_wait
        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))


class QueueReductionReward(BaseRewardFunction):
    """Reward = -total_queue / QUEUE_CAPACITY.

    Penalises accumulation of halting vehicles relative to the network-wide
    capacity constant so the signal is on the same scale as BackPressureReward.
    """

    @property
    def name(self) -> str:
        return "QueueReduction"

    def get_reward(
        self,
        tls_id: str,
        incoming_lanes,
        outgoing_lanes,
        **kwargs,
    ) -> float:
        total_queue = _get_tls_queue_from_detectors(tls_id)
        capacity    = float(UNIFIED_BP_CAPACITY)
        raw = -(total_queue / max(capacity, 1.0))
        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))


class MaxPressureReward(BaseRewardFunction):
    """MaxPressure (Varaiya 2013) reward using actual phase movements.

    Formula
    -------
    r_i = Σ_{m ∈ active_phase} (q_upstream_m - q_downstream_m) / capacity(tls_id)
          + EV_ALPHA * (GAMMA * Phi(s') - Phi(s))   [when EV_PRIORITY_ENABLED]

    Unlike BackPressureReward which pairs incoming/outgoing lanes in round-robin
    order regardless of the active signal phase, this reward only considers
    movements that are actually served by the chosen phase.  The agent is
    rewarded for selecting phases whose movements carry high pressure
    (upstream queue > downstream queue), matching the standard MaxPressure
    control objective.

    The action index and phase-to-lane mapping are passed via ``**kwargs``
    (``action_idx``, ``action_phases``, ``phase_links``) so this class works
    with the existing ``MAPPOAgent.get_reward()`` signature.  Other reward
    functions simply ignore these extra arguments.

    Per-TLS capacity: same as BackPressureReward — uses
    ``_QUEUE_CAPACITY_PER_TLS`` with fallback to ``UNIFIED_BP_CAPACITY``.
    """

    @property
    def name(self) -> str:
        return "MaxPressure"

    def get_reward(
        self,
        tls_id: str,
        incoming_lanes,
        outgoing_lanes,
        ev_phi_prev: float = 0.0,
        ev_phi_curr: float = 0.0,
        **kwargs,
    ) -> float:
        qcap = float(_QUEUE_CAPACITY_PER_TLS.get(tls_id, UNIFIED_BP_CAPACITY))

        action_idx    = kwargs.get("action_idx", 0)
        phase_links   = kwargs.get("phase_links", {})
        action_phases = kwargs.get("action_phases", [])

        if not phase_links or not action_phases:
            return 0.0

        current_phase = action_phases[action_idx]
        movements     = phase_links.get(current_phase, [])

        pressure = 0.0
        for in_lane, out_lane in movements:
            q_in  = _lane_queue(in_lane) / qcap
            q_out = _lane_queue(out_lane) / qcap
            pressure += (q_in - q_out)

        raw = -pressure

        if EV_PRIORITY_ENABLED and (ev_phi_prev != 0.0 or ev_phi_curr != 0.0):
            shaping = EV_ALPHA * (GAMMA * ev_phi_curr - ev_phi_prev)
            raw = raw + shaping

        return float(np.clip(raw, -REWARD_CLIP, REWARD_CLIP))
