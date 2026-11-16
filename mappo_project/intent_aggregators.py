from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from .config import (
    INTERSECTIONS, NEIGHBOR_MAP, GAT_HEADS, GAT_DROPOUT,
    LOCAL_CLUSTERS, GATEWAY_NODE,
)
from .networks import HierarchicalGATAggregator, _TLS_INDEX, PYG_AVAILABLE

if PYG_AVAILABLE:
    from torch_geometric.nn import GATConv

__all__ = [
    'BaseIntentAggregator',
    'GATIntentAggregator',
    'FlatGATAggregator',
    'MeanIntentAggregator',
    'NoSharingIntentAggregator',
]


class BaseIntentAggregator(ABC):
    """Abstract interface for neighbor intent aggregation.

    Implementors receive the full ``(n_agents, intent_dim)`` intent matrix
    and return an aggregated matrix of the same shape.  Each agent's own
    row in the output is the aggregated (context-enriched) representation
    it will use during action selection and PPO updates.
    """

    @property
    def name(self) -> str:
        """Human-readable name shown in logs."""
        return self.__class__.__name__

    @abstractmethod
    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor intents.

        Parameters
        ----------
        intent_matrix : torch.Tensor, shape (n_agents, intent_dim)
            Current intent vectors for all agents, ordered by ``INTERSECTIONS``.

        Returns
        -------
        torch.Tensor, shape (n_agents, intent_dim)
            Aggregated intent matrix.  Each row ``[i]`` contains the
            neighbourhood-aggregated context for agent ``i``.
        """

    # Convenience: make instances callable like nn.Module
    def __call__(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward(intent_matrix)

    def parameters(self):
        """Yield trainable parameters (empty by default for non-parametric aggregators)."""
        return iter([])

    def state_dict(self) -> dict:
        """Return serialisable state (empty by default)."""
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore from a serialised state (no-op by default)."""
        pass

    def to(self, device) -> "BaseIntentAggregator":
        """Move to device (no-op by default for non-parametric aggregators)."""
        return self

    def eval(self) -> "BaseIntentAggregator":
        """Set evaluation mode (no-op by default)."""
        return self

    def train(self, mode: bool = True) -> "BaseIntentAggregator":
        """Set training mode (no-op by default)."""
        return self


class GATIntentAggregator(BaseIntentAggregator):
    """Graph Attention Network aggregator (wraps ``HierarchicalGATAggregator``).

    This is the default aggregator and preserves the original MAPPO-PIS+GAT
    behaviour exactly.  It delegates all computation to the existing
    ``HierarchicalGATAggregator`` from ``networks.py``.
    """

    def __init__(self, intent_dim: int):
        self.intent_dim = intent_dim
        self._module    = HierarchicalGATAggregator(intent_dim)

    @property
    def name(self) -> str:
        return "GAT"

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        return self._module(intent_matrix)

    def parameters(self):
        return self._module.parameters()

    def state_dict(self) -> dict:
        return self._module.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._module.load_state_dict(state_dict)

    def to(self, device) -> "GATIntentAggregator":
        self._module = self._module.to(device)
        return self

    def eval(self) -> "GATIntentAggregator":
        self._module.eval()
        return self

    def train(self, mode: bool = True) -> "GATIntentAggregator":
        self._module.train(mode)
        return self


class FlatGATAggregator(BaseIntentAggregator):
    """Flat (single-level) GAT aggregator — full MAPPO+GAT system, no HierGAT.

    Runs **one** GATConv layer directly over *all* agents using the edges
    defined by ``NEIGHBOR_MAP``.  Everything else in the MAPPO+GAT pipeline
    (actor/critic, intent sharing, PIS, EV priority, checkpointing …) is
    identical to the default ``mappo_gat`` run.  Only the aggregator itself
    is simplified: no local-cluster pass, no inter-cluster pooling, no
    dynamic context gate.

    Architecture
    ------------
    With torch-geometric (preferred)::

        h = GATConv(intent_dim → intent_dim*heads, concat=True)(x, edge_index)
        h = Linear(intent_dim*heads → intent_dim)(h)
        h = ReLU(h)
        out = LayerNorm(h + x)          # residual

    Fallback (no PyG) — masked scaled dot-product attention::

        scores[i,j] = (Wq·x[i]) · (Wk·x[j])^T / sqrt(d),  masked to NEIGHBOR_MAP
        h = softmax(scores) · Wv·x
        out = LayerNorm(ReLU(h) + x)

    The ``save_dir`` for flat-GAT runs gets a ``_flatgat`` suffix so results
    never collide with HierGAT runs that share the same base directory.

    Parameters
    ----------
    intent_dim : int   Dimensionality of each agent's intent vector.
    heads      : int   GAT attention heads  (default: ``GAT_HEADS``).
    dropout    : float Attention dropout    (default: ``GAT_DROPOUT``).
    """

    def __init__(
        self,
        intent_dim: int,
        heads:      int   = GAT_HEADS,
        dropout:    float = GAT_DROPOUT,
    ):
        self.intent_dim = intent_dim
        self.heads      = heads
        self.dropout    = dropout

        # ── static edge index (directed, from NEIGHBOR_MAP) ──────────────────
        # Built once on CPU; moved to the right device lazily in forward().
        src_l: list[int] = []
        dst_l: list[int] = []
        for tls_id in INTERSECTIONS:
            i = _TLS_INDEX[tls_id]
            for nb in NEIGHBOR_MAP.get(tls_id, []):
                if nb in _TLS_INDEX:
                    src_l.append(i)
                    dst_l.append(_TLS_INDEX[nb])
        self._ei_cpu: torch.Tensor = torch.tensor([src_l, dst_l],
                                                   dtype=torch.long)
        self._ei_cached: "torch.Tensor | None" = None   # device-local copy

        # ── adjacency mask for dot-product fallback ──────────────────────────
        n = len(INTERSECTIONS)
        mask = torch.eye(n, dtype=torch.bool)            # self-loops included
        for tls_id in INTERSECTIONS:
            i = _TLS_INDEX[tls_id]
            for nb in NEIGHBOR_MAP.get(tls_id, []):
                if nb in _TLS_INDEX:
                    mask[i, _TLS_INDEX[nb]] = True
        self._mask_cpu: torch.Tensor = mask
        self._mask_cached: "torch.Tensor | None" = None

        # ── learnable layers ─────────────────────────────────────────────────
        if PYG_AVAILABLE:
            self._conv = GATConv(
                intent_dim, intent_dim,
                heads=heads, dropout=dropout, concat=True,
            )
            self._proj = nn.Linear(intent_dim * heads, intent_dim)
        else:
            self._Wq = nn.Linear(intent_dim, intent_dim, bias=False)
            self._Wk = nn.Linear(intent_dim, intent_dim, bias=False)
            self._Wv = nn.Linear(intent_dim, intent_dim, bias=False)

        self._norm = nn.LayerNorm(intent_dim)

    # ── BaseIntentAggregator interface ───────────────────────────────────────

    @property
    def name(self) -> str:
        return "FlatGAT"

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        device = intent_matrix.device

        if PYG_AVAILABLE:
            if self._ei_cached is None or self._ei_cached.device != device:
                self._ei_cached = self._ei_cpu.to(device)
            h = torch.relu(self._proj(self._conv(intent_matrix,
                                                  self._ei_cached)))
        else:
            if self._mask_cached is None or self._mask_cached.device != device:
                self._mask_cached = self._mask_cpu.to(device)
            scale  = self.intent_dim ** 0.5
            Q = self._Wq(intent_matrix)
            K = self._Wk(intent_matrix)
            V = self._Wv(intent_matrix)
            scores = (Q @ K.t()) / scale
            scores = scores.masked_fill(~self._mask_cached, float("-inf"))
            h      = torch.relu(torch.softmax(scores, dim=-1) @ V)

        return self._norm(h + intent_matrix)   # residual + LayerNorm

    def parameters(self):
        if PYG_AVAILABLE:
            return iter(
                list(self._conv.parameters())
                + list(self._proj.parameters())
                + list(self._norm.parameters())
            )
        return iter(
            list(self._Wq.parameters())
            + list(self._Wk.parameters())
            + list(self._Wv.parameters())
            + list(self._norm.parameters())
        )

    def state_dict(self) -> dict:
        sd: dict = {"norm": self._norm.state_dict()}
        if PYG_AVAILABLE:
            sd["conv"] = self._conv.state_dict()
            sd["proj"] = self._proj.state_dict()
        else:
            sd["Wq"] = self._Wq.state_dict()
            sd["Wk"] = self._Wk.state_dict()
            sd["Wv"] = self._Wv.state_dict()
        return sd

    def load_state_dict(self, state_dict: dict) -> None:
        self._norm.load_state_dict(state_dict["norm"])
        if PYG_AVAILABLE:
            self._conv.load_state_dict(state_dict["conv"])
            self._proj.load_state_dict(state_dict["proj"])
        else:
            self._Wq.load_state_dict(state_dict["Wq"])
            self._Wk.load_state_dict(state_dict["Wk"])
            self._Wv.load_state_dict(state_dict["Wv"])

    def to(self, device) -> "FlatGATAggregator":
        # Invalidate cached device-tensors; they will be re-created on demand.
        self._ei_cached   = None
        self._mask_cached = None
        if PYG_AVAILABLE:
            self._conv = self._conv.to(device)
            self._proj = self._proj.to(device)
        else:
            self._Wq = self._Wq.to(device)
            self._Wk = self._Wk.to(device)
            self._Wv = self._Wv.to(device)
        self._norm = self._norm.to(device)
        return self

    def eval(self) -> "FlatGATAggregator":
        if PYG_AVAILABLE:
            self._conv.eval(); self._proj.eval()
        else:
            self._Wq.eval(); self._Wk.eval(); self._Wv.eval()
        self._norm.eval()
        return self

    def train(self, mode: bool = True) -> "FlatGATAggregator":
        if PYG_AVAILABLE:
            self._conv.train(mode); self._proj.train(mode)
        else:
            self._Wq.train(mode); self._Wk.train(mode); self._Wv.train(mode)
        self._norm.train(mode)
        return self


class MeanIntentAggregator(BaseIntentAggregator):
    """Non-parametric aggregator: each agent receives the *mean* of its neighbours.

    For agents without neighbours the row is kept as-is (self-intent).
    No learnable parameters -- purely topology-driven pooling.

    This provides a simple baseline that removes GAT's learned attention
    weights while keeping the communication topology from ``NEIGHBOR_MAP``.
    """

    def __init__(self, intent_dim: int):
        self.intent_dim = intent_dim

    @property
    def name(self) -> str:
        return "MeanAggregator"

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        out = intent_matrix.clone()
        for tls_id in INTERSECTIONS:
            i         = _TLS_INDEX[tls_id]
            neighbors = [nb for nb in NEIGHBOR_MAP.get(tls_id, []) if nb in _TLS_INDEX]
            if neighbors:
                nb_idx   = [_TLS_INDEX[nb] for nb in neighbors]
                nb_stack = intent_matrix[nb_idx]   # (n_nb, intent_dim)
                out[i]   = nb_stack.mean(dim=0)
        return out


class NoSharingIntentAggregator(BaseIntentAggregator):
    """Ablation aggregator: returns an all-zero intent matrix (no communication).

    Each agent acts as if it has no neighbours and contributes/receives no
    intent information.  The actor still receives an intent vector of the
    correct dimensionality (all zeros) so ONNX export compatibility and
    the Actor input signature are preserved.

    This is the independent-agents baseline for ablation studies.
    """

    def __init__(self, intent_dim: int):
        self.intent_dim = intent_dim

    @property
    def name(self) -> str:
        return "NoSharing"

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        # Return zeros with the same shape and device as the input.
        # The actor input signature (state + intent_dim) is unchanged.
        return torch.zeros_like(intent_matrix)
