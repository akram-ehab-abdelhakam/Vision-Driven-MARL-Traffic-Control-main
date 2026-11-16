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

__all__ = ['ActorNetwork', 'CriticNetwork', 'HierarchicalGATAggregator', 'FlatGATAggregator', '_TLS_INDEX', 'GATIntentAggregator', '_STATIC_ADJ_MASK']


# ---------------------------------------------------------------------------
# Weight initialization helpers
# ---------------------------------------------------------------------------
# PyTorch defaults use kaiming_uniform_ (designed for ReLU), which produces
# logits with large initial magnitude — causing the Categorical policy to
# start near-deterministic from episode 0.  Orthogonal init with small gain
# on the action head ensures ~uniform initial distribution (high entropy).

def _init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal init for Linear layers; zero bias. Applied via model.apply()."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, intent_dim: int = INTENT_SHARING_DIM):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.intent_dim = intent_dim

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + intent_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(HIDDEN_DIM, action_dim)
        self.intent_head = nn.Linear(HIDDEN_DIM, intent_dim)

        # Orthogonal init — feature_net with gain=sqrt(2) (ReLU),
        # action_head with gain=0.01 (near-uniform initial policy),
        # intent_head with gain=1.0 (followed by tanh, no need for small gain).
        _relu_gain = nn.init.calculate_gain('relu')
        self.feature_net.apply(lambda m: _init_weights(m, gain=_relu_gain))
        self.action_head.apply(lambda m: _init_weights(m, gain=0.01))
        self.intent_head.apply(lambda m: _init_weights(m, gain=1.0))

    def forward(self, state: torch.Tensor, intent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined  = torch.cat([state, intent], dim=-1)
        features  = self.feature_net(combined)
        action_logits = self.action_head(features)
        new_intent    = torch.tanh(self.intent_head(features))
        return action_logits, new_intent

class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, intent_dim: int = INTENT_SHARING_DIM):
        super().__init__()
        self.state_dim   = state_dim
        self.intent_dim  = intent_dim
        n_agents         = len(INTERSECTIONS)
        # Centralized critic: local state + full intent matrix from all agents
        global_input_dim = state_dim + intent_dim * n_agents

        self.feature_net = nn.Sequential(
            nn.Linear(global_input_dim, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.LayerNorm(HIDDEN_DIM),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

        # Orthogonal init — feature_net with gain=sqrt(2) (ReLU),
        # value_head with gain=1.0 (linear value function, no saturation).
        _relu_gain = nn.init.calculate_gain('relu')
        self.feature_net.apply(lambda m: _init_weights(m, gain=_relu_gain))
        self.value_head.apply(lambda m: _init_weights(m, gain=1.0))

    def forward(self, state: torch.Tensor, global_intent: torch.Tensor) -> torch.Tensor:
        # state:         (batch, state_dim)
        # global_intent: (batch, n_agents * intent_dim)  -- flattened intent matrix
        combined = torch.cat([state, global_intent], dim=-1)
        features = self.feature_net(combined)
        return self.value_head(features)

class HierarchicalGATAggregator(nn.Module):
    def __init__(self, intent_dim: int, heads: int = GAT_HEADS,
                 dropout: float = GAT_DROPOUT):
        super().__init__()
        self.intent_dim  = intent_dim
        self.n_clusters  = len(LOCAL_CLUSTERS)
        # FIX-D: gateway_idx is stored for potential future use (e.g.
        # priority aggregation for the gateway intersection) but is NOT
        # currently referenced in forward().  The forward() path uses
        # data_index to slice intents into cluster/local batches; the
        # cluster-level global GAT operates over cluster centroids, and
        # per-agent priority weighting is not yet implemented.
        self.gateway_idx = _TLS_INDEX[GATEWAY_NODE]

        if PYG_AVAILABLE:
            # GAT v2: Use concat=True for richer features, then project back
            self.local_convs = nn.ModuleList([
                GATConv(intent_dim, intent_dim, heads=heads,
                        dropout=dropout, concat=True)
                for _ in LOCAL_CLUSTERS
            ])
            self.local_projs = nn.ModuleList([
                nn.Linear(intent_dim * heads, intent_dim)
                for _ in LOCAL_CLUSTERS
            ])
            self.global_conv = GATConv(intent_dim, intent_dim, heads=heads,
                                       dropout=dropout, concat=True)
            self.global_proj = nn.Linear(intent_dim * heads, intent_dim)
        else:
            self.local_Wq = nn.ModuleList([
                nn.Linear(intent_dim, intent_dim, bias=False)
                for _ in LOCAL_CLUSTERS])
            self.local_Wk = nn.ModuleList([
                nn.Linear(intent_dim, intent_dim, bias=False)
                for _ in LOCAL_CLUSTERS])
            self.local_Wv = nn.ModuleList([
                nn.Linear(intent_dim, intent_dim, bias=False)
                for _ in LOCAL_CLUSTERS])
            self.global_Wq = nn.Linear(intent_dim, intent_dim, bias=False)
            self.global_Wk = nn.Linear(intent_dim, intent_dim, bias=False)
            self.global_Wv = nn.Linear(intent_dim, intent_dim, bias=False)

            # Orthogonal init for fallback Q/K/V attention layers
            for ml in (self.local_Wq, self.local_Wk, self.local_Wv):
                ml.apply(lambda m: _init_weights(m, gain=nn.init.calculate_gain('relu')))
            for w in (self.global_Wq, self.global_Wk, self.global_Wv):
                _init_weights(w, gain=nn.init.calculate_gain('relu'))

        # GAT v2: Attention-based cluster pooling
        self.cluster_poolers = nn.ModuleList([
            nn.Linear(intent_dim, 1) for _ in LOCAL_CLUSTERS
        ])

        # GAT v2: Dynamic Context Gate (replaces static alpha)
        # Input: [local_h, global_ctx], Output: gate in [0, 1]
        self.context_gate = nn.Sequential(
            nn.Linear(intent_dim * 2, intent_dim // 2),
            nn.ReLU(),
            nn.Linear(intent_dim // 2, 1),
            nn.Sigmoid()
        )

        self.local_norm  = nn.LayerNorm(intent_dim)
        self.global_norm = nn.LayerNorm(intent_dim)
        
        # GAT v2: Store last gates for visualization
        self.last_gates = None

        self._local_ei  = [None] * len(LOCAL_CLUSTERS)
        self._global_ei = None

    def _get_local_ei(self, ci: int, device: torch.device) -> "torch.Tensor":
        if self._local_ei[ci] is None or self._local_ei[ci].device != device:
            self._local_ei[ci] = _build_local_edge_index(
                LOCAL_CLUSTERS[ci]).to(device)
        return self._local_ei[ci]

    def _get_global_ei(self, device: torch.device) -> "torch.Tensor":
        if self._global_ei is None or self._global_ei.device != device:
            self._global_ei = _build_global_edge_index().to(device)
        return self._global_ei

    def _local_attn_fallback(self, ci: int,
                              x: torch.Tensor) -> torch.Tensor:
        scale  = self.intent_dim ** 0.5
        Q = self.local_Wq[ci](x)
        K = self.local_Wk[ci](x)
        V = self.local_Wv[ci](x)
        attn = torch.softmax((Q @ K.t()) / scale, dim=-1)
        return attn @ V

    def _global_attn_fallback(self, x: torch.Tensor) -> torch.Tensor:
        scale  = self.intent_dim ** 0.5
        Q = self.global_Wq(x)
        K = self.global_Wk(x)
        V = self.global_Wv(x)
        attn = torch.softmax((Q @ K.t()) / scale, dim=-1)
        return attn @ V

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        device = intent_matrix.device
        n      = intent_matrix.shape[0]

        # ── local cluster attention ───────────────────────────────────────────
        # Collect each node's local-context vector without in-place writes.
        # local_h[tls_idx] = h_local[j] for that node's cluster output.
        local_h: list["torch.Tensor | None"] = [None] * n
        local_out_list: list[torch.Tensor] = []

        for ci, cluster in enumerate(LOCAL_CLUSTERS):
            cidx    = [_TLS_INDEX[t] for t in cluster]
            x_local = intent_matrix[cidx]
            if PYG_AVAILABLE:
                ei      = self._get_local_ei(ci, device)
                h_loc   = torch.relu(self.local_projs[ci](self.local_convs[ci](x_local, ei)))
            else:
                h_loc   = torch.relu(self._local_attn_fallback(ci, x_local))
            h_loc = self.local_norm(h_loc + x_local)
            for j, tls_idx in enumerate(cidx):
                local_h[tls_idx] = h_loc[j]
            local_out_list.append(h_loc)

        # ── global inter-cluster attention ────────────────────────────────────
        # GAT v2: Attention-based pooling for cluster representatives
        cluster_reps_list = []
        for ci, h_loc in enumerate(local_out_list):
            attn_scores = self.cluster_poolers[ci](h_loc)  # (cluster_size, 1)
            attn_weights = torch.softmax(attn_scores, dim=0)
            rep = (h_loc * attn_weights).sum(dim=0)
            cluster_reps_list.append(rep)
        
        cluster_reps = torch.stack(cluster_reps_list)
        
        if PYG_AVAILABLE:
            g_ei       = self._get_global_ei(device)
            global_ctx = torch.relu(self.global_proj(self.global_conv(cluster_reps, g_ei)))
        else:
            global_ctx = torch.relu(self._global_attn_fallback(cluster_reps))
        global_ctx = self.global_norm(global_ctx + cluster_reps)

        # ── per-node dynamic gating (local ↔ global) ──────────────────────────
        # node_cluster[tls_idx] = which cluster index that node belongs to
        node_cluster = [CLUSTER_OF[INTERSECTIONS[i]] for i in range(n)]

        out_rows:  list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []
        for tls_idx in range(n):
            l_vec  = local_h[tls_idx]
            g_vec  = global_ctx[node_cluster[tls_idx]]
            
            # GAT v2: Dynamic gate based on local and global context
            combined_gate_input = torch.cat([l_vec, g_vec], dim=-1)
            gate = self.context_gate(combined_gate_input) # scalar in [0, 1]
            gate_list.append(gate)
            
            out_rows.append((1.0 - gate) * l_vec + gate * g_vec)

        # GAT v2: Store gates for visualization
        self.last_gates = torch.stack(gate_list).detach().cpu().numpy().flatten() # (n,)
        return torch.stack(out_rows, dim=0)  # (n, intent_dim)

    def _get_local_ei_batched(self, ci: int, batch_size: int,
                               cluster_n: int, device: torch.device) -> "torch.Tensor":
        """Block-diagonal edge index for `batch_size` copies of a local cluster.

        Replaces looping in Python over the batch dimension: PyG's GATConv
        already supports arbitrary disjoint graphs in one call as long as the
        edge index offsets each copy by `b * cluster_n`. We build this once
        per (ci, batch_size) combination and cache it, exactly mirroring the
        single-graph edge-index caching already done for the unbatched path.
        """
        cache_attr = f"_local_ei_batched_{ci}_{batch_size}"
        cached = getattr(self, cache_attr, None)
        if cached is not None and cached.device == device:
            return cached
        base = self._get_local_ei(ci, device)  # (2, E) for one cluster copy
        offsets = (torch.arange(batch_size, device=device) * cluster_n)
        offsets = offsets.view(batch_size, 1, 1)              # (B, 1, 1)
        ei = base.unsqueeze(0) + offsets                       # (B, 2, E)
        ei = ei.permute(1, 0, 2).reshape(2, -1)                # (2, B*E)
        setattr(self, cache_attr, ei)
        return ei

    def _get_global_ei_batched(self, batch_size: int,
                                n_clusters: int, device: torch.device) -> "torch.Tensor":
        cache_attr = f"_global_ei_batched_{batch_size}"
        cached = getattr(self, cache_attr, None)
        if cached is not None and cached.device == device:
            return cached
        base = self._get_global_ei(device)
        offsets = (torch.arange(batch_size, device=device) * n_clusters)
        offsets = offsets.view(batch_size, 1, 1)
        ei = base.unsqueeze(0) + offsets
        ei = ei.permute(1, 0, 2).reshape(2, -1)
        setattr(self, cache_attr, ei)
        return ei

    def forward_batched(self, intent_matrix_batch: torch.Tensor) -> torch.Tensor:
        """Batched HierGAT forward: (B, n_agents, intent_dim) -> (B, n_agents, intent_dim).

        Numerically identical to calling `forward()` once per batch element,
        but replaces the implicit Python-level loop over B with a single PyG
        call per cluster/global step, using block-diagonal edge indices so
        each batch element's graph stays disjoint from the others (no
        cross-batch attention leakage). Falls back to the plain
        Python-loop-over-`forward()` when PyG is unavailable, since the
        dot-product fallback path has no batched-graph primitive to exploit.

        Use this instead of a manual `for b in range(B): agg(mat[b])` loop
        whenever you need to run the aggregator over several intent-matrix
        snapshots at once (e.g. a future `update_gat` that trains on a batch
        of buffered snapshots rather than one at a time).
        """
        if not PYG_AVAILABLE:
            return torch.stack(
                [self.forward(intent_matrix_batch[b])
                 for b in range(intent_matrix_batch.shape[0])],
                dim=0,
            )

        device = intent_matrix_batch.device
        B, n, _ = intent_matrix_batch.shape

        # ── local cluster attention (batched) ─────────────────────────────
        local_h_batched: list["torch.Tensor | None"] = [None] * n  # each (B, intent_dim)
        local_out_list: list[torch.Tensor] = []  # each (B, cluster_n, intent_dim)

        for ci, cluster in enumerate(LOCAL_CLUSTERS):
            cidx      = [_TLS_INDEX[t] for t in cluster]
            cluster_n = len(cidx)
            x_local   = intent_matrix_batch[:, cidx, :]                 # (B, cluster_n, D)
            x_flat    = x_local.reshape(B * cluster_n, self.intent_dim)  # block-diagonal nodes
            ei        = self._get_local_ei_batched(ci, B, cluster_n, device)
            h_flat    = torch.relu(self.local_projs[ci](self.local_convs[ci](x_flat, ei)))
            h_loc     = h_flat.view(B, cluster_n, self.intent_dim)
            h_loc     = self.local_norm(h_loc + x_local)
            for j, tls_idx in enumerate(cidx):
                local_h_batched[tls_idx] = h_loc[:, j, :]
            local_out_list.append(h_loc)

        # ── global inter-cluster attention (batched) ──────────────────────
        cluster_reps_list = []
        for ci, h_loc in enumerate(local_out_list):
            attn_scores  = self.cluster_poolers[ci](h_loc)          # (B, cluster_n, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            rep = (h_loc * attn_weights).sum(dim=1)                 # (B, D)
            cluster_reps_list.append(rep)

        cluster_reps = torch.stack(cluster_reps_list, dim=1)        # (B, n_clusters, D)
        n_clusters   = cluster_reps.shape[1]
        reps_flat    = cluster_reps.reshape(B * n_clusters, self.intent_dim)
        g_ei         = self._get_global_ei_batched(B, n_clusters, device)
        ctx_flat     = torch.relu(self.global_proj(self.global_conv(reps_flat, g_ei)))
        global_ctx   = ctx_flat.view(B, n_clusters, self.intent_dim)
        global_ctx   = self.global_norm(global_ctx + cluster_reps)

        # ── per-node dynamic gating (batched) ──────────────────────────────
        node_cluster = [CLUSTER_OF[INTERSECTIONS[i]] for i in range(n)]

        out_rows:  list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []
        for tls_idx in range(n):
            l_vec = local_h_batched[tls_idx]                 # (B, D)
            g_vec = global_ctx[:, node_cluster[tls_idx], :]  # (B, D)
            combined_gate_input = torch.cat([l_vec, g_vec], dim=-1)
            gate = self.context_gate(combined_gate_input)     # (B, 1)
            gate_list.append(gate)
            out_rows.append((1.0 - gate) * l_vec + gate * g_vec)

        self.last_gates = torch.stack(gate_list, dim=1).detach().cpu().numpy()  # (B, n)
        return torch.stack(out_rows, dim=1)  # (B, n, intent_dim)

class FlatGATAggregator(nn.Module):
    """Single-level GAT aggregator — no hierarchical clustering.

    Applies one GAT layer (or fallback scaled dot-product attention) to the
    full set of intersection nodes using the neighbour topology from
    ``NEIGHBOR_MAP``.  No local clusters, no inter-cluster gating, no
    dynamic context gate — just a flat graph attention step.
    """

    def __init__(self, intent_dim: int, heads: int = GAT_HEADS,
                 dropout: float = GAT_DROPOUT):
        super().__init__()
        self.intent_dim = intent_dim

        if PYG_AVAILABLE:
            self.conv = GATConv(intent_dim, intent_dim, heads=heads,
                                dropout=dropout, concat=True)
            self.proj = nn.Linear(intent_dim * heads, intent_dim)
        else:
            self.Wq = nn.Linear(intent_dim, intent_dim, bias=False)
            self.Wk = nn.Linear(intent_dim, intent_dim, bias=False)
            self.Wv = nn.Linear(intent_dim, intent_dim, bias=False)

            # Orthogonal init for fallback Q/K/V attention layers
            for w in (self.Wq, self.Wk, self.Wv):
                _init_weights(w, gain=nn.init.calculate_gain('relu'))

        self.norm = nn.LayerNorm(intent_dim)
        self._ei  = None

    def _get_edge_index(self, device: torch.device) -> "torch.Tensor":
        if self._ei is None or self._ei.device != device:
            n      = len(INTERSECTIONS)
            src, dst = [], []
            for tls_id in INTERSECTIONS:
                i = _TLS_INDEX[tls_id]
                for nb in NEIGHBOR_MAP.get(tls_id, []):
                    j = _TLS_INDEX[nb]
                    src.append(i); dst.append(j)
            self._ei = torch.tensor([src, dst], dtype=torch.long, device=device)
        return self._ei

    def _attn_fallback(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.intent_dim ** 0.5
        Q = self.Wq(x); K = self.Wk(x); V = self.Wv(x)
        attn = torch.softmax((Q @ K.t()) / scale, dim=-1)
        return attn @ V

    def forward(self, intent_matrix: torch.Tensor) -> torch.Tensor:
        device = intent_matrix.device
        if PYG_AVAILABLE:
            ei = self._get_edge_index(device)
            h  = torch.relu(self.proj(self.conv(intent_matrix, ei)))
        else:
            h  = torch.relu(self._attn_fallback(intent_matrix))
        h = self.norm(h + intent_matrix)
        return h  # (n_agents, intent_dim)

# -- module-level state ----------------------------------------------
_TLS_INDEX = {tls: i for i, tls in enumerate(INTERSECTIONS)}
GATIntentAggregator = HierarchicalGATAggregator
_STATIC_ADJ_MASK: "torch.Tensor | None" = None
