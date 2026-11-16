from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from .config import EV_PRIORITY_ENABLED, EV_BETA, EV_CLASSES, EV_TYPE_KEYWORDS
from .sumo_env import _STEP_CACHE

__all__ = ['BaseEVPriorityModule', 'DefaultEVPriority', 'NoEVPriority']

# ---------------------------------------------------------------------------
# Lazy traci import (mirrors pattern used throughout the project)
# ---------------------------------------------------------------------------
try:
    import traci
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


class BaseEVPriorityModule(ABC):
    """Abstract interface for Emergency Vehicle (EV) priority logic.

    Implementors provide two methods:

    * ``get_ev_features`` -- builds the EV sub-vector appended to the agent
      state.  Must return an ``np.ndarray`` of shape ``(2*n_in + 2,)``.

    * ``get_ev_potential`` -- computes the potential Φ(s) used for reward
      shaping (Ng et al., 1999).  Must return a scalar ``float``.

    Both methods must return zero-valued results when EV priority is
    conceptually disabled so that the state/reward dimensions stay constant.
    """

    @property
    def name(self) -> str:
        """Human-readable name shown in logs."""
        return self.__class__.__name__

    @abstractmethod
    def get_ev_features(self, tls_id: str, incoming_lanes) -> np.ndarray:
        """Return EV feature vector for this agent.

        Parameters
        ----------
        tls_id : str
            Traffic light system identifier.
        incoming_lanes : sequence of str
            Ordered list of incoming lane IDs for this TLS.

        Returns
        -------
        np.ndarray, shape (2*n_in + 2,), dtype float32
            ``[ev_count_norm[0..n_in-1], ev_priority[0..n_in-1],
               ev_total_priority, ev_closest_dist]``.
            All values in [0, 1].
        """

    @abstractmethod
    def get_ev_potential(
        self,
        incoming_lanes,
        action_idx: int,
        action_phases: list,
    ) -> float:
        """Compute potential Φ(s) for reward shaping.

        Parameters
        ----------
        incoming_lanes : sequence of str
            Ordered list of incoming lane IDs.
        action_idx : int
            Index of the currently active action (phase).
        action_phases : list
            Mapping from action index to SUMO phase index.

        Returns
        -------
        float
            Φ(s) ≥ 0.  High when EVs already hold a green phase.
        """


# ---------------------------------------------------------------------------
# Helpers (moved from sumo_env.py to keep EV logic co-located)
# ---------------------------------------------------------------------------

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


class DefaultEVPriority(BaseEVPriorityModule):
    """Full EV priority module — mirrors the original project logic exactly.

    ``get_ev_features`` scans incoming lanes for emergency vehicles and
    builds a proximity-weighted feature vector.

    ``get_ev_potential`` computes Φ(s) = Σ_l max_v[w_prox(v,l)] × is_green(l).
    """

    @property
    def name(self) -> str:
        return "DefaultEV"

    def get_ev_features(self, tls_id: str, incoming_lanes) -> np.ndarray:
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

    def get_ev_potential(
        self,
        incoming_lanes,
        action_idx: int,
        action_phases: list,
    ) -> float:
        """
        Potential Φ(s) = Σ_l  max_v[w_prox(v,l)] × is_green(l)
        High when EVs already have a green phase -> encourages holding green for EVs.
        Returns 0.0 when EV_PRIORITY_ENABLED=False.
        """
        if not EV_PRIORITY_ENABLED:
            return 0.0
        # lanes that are currently green
        try:
            green_lanes = set(action_phases[action_idx]) if action_phases else set()
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


class NoEVPriority(BaseEVPriorityModule):
    """Ablation module — disables all EV priority logic.

    Both methods return zero-valued results of the correct shape/type.
    The state vector length produced by the agent is unchanged because
    ``_ev_feat_dim`` is computed from ``EV_PRIORITY_ENABLED`` in config, and
    this module is selected explicitly via ``--ev_priority none``.

    When ``EV_PRIORITY_ENABLED=True`` in config but ``NoEVPriority`` is
    injected, ``get_ev_features`` returns a zero vector of length
    ``2*n_in + 2`` so the state dimension is consistent.
    """

    @property
    def name(self) -> str:
        return "NoEV"

    def get_ev_features(self, tls_id: str, incoming_lanes) -> np.ndarray:
        n_in = len(incoming_lanes)
        return np.zeros(2 * n_in + 2, dtype=np.float32)

    def get_ev_potential(
        self,
        incoming_lanes,
        action_idx: int,
        action_phases: list,
    ) -> float:
        return 0.0
