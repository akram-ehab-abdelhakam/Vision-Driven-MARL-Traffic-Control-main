from __future__ import annotations
# config.py  -- module-level globals from V1.py
import sys, os, random, warnings, time, signal, json
import numpy as np
import torch
import torch._dynamo
import multiprocessing as mp
from collections import defaultdict, deque
from typing import Tuple, Dict, List
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
    tc = None
try:
    from torch.utils.tensorboard import SummaryWriter as TBWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    TBWriter = None
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
    try:
        from rich.progress import MofNCompleteColumn, TaskProgressColumn
    except ImportError:
        pass
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
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

import sys
from collections import defaultdict, deque
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch_geometric
import os
_cuda_available_early = False
try:
    import torch as _torch_early
    _cuda_available_early = _torch_early.cuda.is_available()
except Exception:
    pass
if not _cuda_available_early:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import random
import sys
import time
import csv
import json
import signal
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
import torch._dynamo
try:
    from torch.utils.tensorboard import SummaryWriter as TBWriter
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("[WARNING] TensorBoard not found. Run: pip install tensorboard")
    TBWriter = None
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
    # These columns were added in later rich versions -- import safely
    try:
        from rich.progress import MofNCompleteColumn
    except ImportError:
        class MofNCompleteColumn:          # type: ignore
            def __init__(self, *a, **kw): pass
    try:
        from rich.progress import TaskProgressColumn
    except ImportError:
        class TaskProgressColumn:          # type: ignore
            def __init__(self, *a, **kw): pass
    try:
        from rich.layout import Layout
        from rich.columns import Columns
        from rich.style import Style
        from rich.progress import ProgressColumn
    except ImportError:
        Layout = Columns = Style = ProgressColumn = None  # type: ignore
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # stubs so references don't crash
    Console = Live = Table = Panel = Text = Rule = box = None  # type: ignore
    Progress = BarColumn = TextColumn = None                   # type: ignore
    TimeElapsedColumn = TimeRemainingColumn = SpinnerColumn = None  # type: ignore
    MofNCompleteColumn = TaskProgressColumn = None             # type: ignore
    print("[WARNING] rich not found. Run: pip install rich")
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
GAT_HEADS    = 4
GAT_DROPOUT  = 0.1
GAT_LR       = 1e-4   # KL-adaptive: GAT layers use lower base LR (was 3e-4)

# ---------------------------------------------------------------------------
# KL-Adaptive Learning Rate hyperparameters
# ---------------------------------------------------------------------------
# Per-group base learning rates
KL_ACTOR_LR  = 3e-4   # policy head (medium LR)
KL_CRITIC_LR = 1e-3   # centralized critic (higher LR)
KL_GAT_LR    = 1e-4   # GAT / intent-aggregator layers (lower LR)
# KL target and LR clamp bounds
KL_TARGET    = 0.01   # lowered from 0.02 — true_kl (full distributional) is typically smaller than approx_kl
KL_LR_MIN    = 1e-5   # absolute lower bound on any LR
KL_LR_MAX    = 5e-4   # absolute upper bound on any LR
# Adjustment thresholds and multipliers
KL_HIGH_MULT = 1.5    # KL > KL_TARGET * this  → reduce LR
KL_LOW_MULT  = 0.5    # KL < KL_TARGET * this  → increase LR
KL_LR_DOWN   = 0.7    # multiply LR by this when KL is too high
KL_LR_UP     = 1.3    # multiply LR by this when KL is too low
# KL-adaptive clip-ratio thresholds and multipliers
CLIP_MIN     = 0.05   # absolute lower bound on clip ratio
CLIP_MAX     = 0.4    # absolute upper bound on clip ratio
CLIP_DOWN    = 0.8    # multiply clip by this when KL is too high
CLIP_UP      = 1.2    # multiply clip by this when KL is too low
sys.path.append(os.path.join(os.environ.get("SUMO_HOME", "/usr/share/sumo"), "tools"))
warnings.filterwarnings("ignore", message=".*getAllProgramLogics.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*dynamic_axes.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*batch.*will not be used.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*isinstance.*treespec.*", category=FutureWarning)
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    print("[WARNING] traci not found.")
UPSTREAM_LANES = {
    "TL1": ["E7_0", "E7_1", "E7_2", "E9_0", "E9_1", "E9_2"],
    "TL2": ["E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E2_3", "E4_0", "E4_1", "E6_0", "E6_1"],
    "TL3": ["E16_0", "E23_0", "E24_0", "E24_1", "E24_2"],
    "TL4": ["E14_1", "E14_2", "E15_0", "E15_1"],
}
DOWNSTREAM_LANES = {
    "TL1": ["E19_0", "E19_1", "E19_2", "E2_0", "E2_1", "E2_2", "E2_3", "E8_0", "E8_1", "E8_2"],
    "TL2": ["E0_0", "E0_1", "E3_0", "E3_1", "E5_0", "E5_1"],
    "TL3": ["E15_0", "E15_1", "E15_2", "E17_0", "E17_1"],
    "TL4": ["E11_0", "E11_1", "E18_0", "E18_1", "E18_2"],
}
INTERSECTIONS = ["TL1", "TL2", "TL3", "TL4"]
TLS_DETECTOR_MAP = {
    "TL1": ["e2_15", "e2_16", "e2_17", "e2_0", "e2_2", "e2_1"],
    "TL2": ["e2_3", "e2_4", "e2_5", "e2_6"],
    "TL3": ["e2_10", "e2_11", "e2_12", "e2_9"],
    "TL4": ["e2_8", "e2_7", "e2_13", "e2_14"],
}
LANE_DETECTOR_MAP = {
    "E9_0": "e2_15", "E9_1": "e2_16", "E9_2": "e2_17",
    "E7_0": "e2_0",  "E7_1": "e2_2",  "E7_2": "e2_1",
    "E6_0": "e2_3",  "E6_1": "e2_4",
    "E4_0": "e2_5",  "E4_1": "e2_6",
    "E24_0": "e2_10", "E24_1": "e2_11", "E24_2": "e2_12",
    "E16_0": "e2_9",
    "E15_0": "e2_8",  "E15_1": "e2_7",
    "E14_1": "e2_13", "E14_2": "e2_14",
}
NEIGHBOR_MAP = {
    "TL1": ["TL2", "TL3"],
    "TL2": ["TL1", "TL3", "TL4"],
    "TL3": ["TL2", "TL4", "TL1"],
    "TL4": ["TL2", "TL3"],
}
LOCAL_CLUSTERS = [
    ["TL1", "TL2"],
    ["TL3", "TL4"],
]
GATEWAY_NODE = "TL1"
CLUSTER_OF   = {tls: ci for ci, cl in enumerate(LOCAL_CLUSTERS) for tls in cl}
TOTAL_STEPS      = 1800
GAMMA            = 0.97
GAE_LAMBDA       = 0.95
LEARNING_RATE    = 3e-4
ENTROPY_COEFF    = 0.01
VALUE_LOSS_COEFF = 0.5

# INTENT-CONSISTENCY: auxiliary loss giving the actor's intent_head a direct
# gradient signal from the centralized critic (see train_epoch in agent.py).
# Without this term, new_intent is only shaped indirectly through the
# aggregator/neighbor path -- nothing in the loss ever asks "was THIS
# agent's intent useful for the joint value estimate?".
INTENT_CONSISTENCY_COEFF = 0.1
CLIP_RATIO       = 0.2
BATCH_SIZE       = 64
MEMORY_CAPACITY  = 1800
N_EPOCHS         = 3
HIDDEN_DIM       = 256
LAMBDA           = 0.5
STEP_LENGTH      = 1.0
MIN_GREEN_TIME   = 10.0
MIN_GREEN_STEPS  = int(MIN_GREEN_TIME / STEP_LENGTH)
REWARD_CLIP      = 5.0
EV_PRIORITY_ENABLED  = True
EV_BETA              = 3.0
EV_ALPHA             = 0.25
EV_CLASSES           = {"emergency"}
EV_TYPE_KEYWORDS     = {"ambulance", "fire_truck", "police", "emergency"}
_EP_STATS: dict = {
    "ev_cleared":  0,
    "ev_wait_sum": 0.0,
    "cars_passed": 0,
    "co2_sum":     0.0,
    "nox_sum":     0.0,
    "fuel_sum":    0.0,
}
_emission_tracker: dict = {}
_active_evs:      set  = set()
_ev_wait_tracker: dict = {}
# ── Unit-conversion helpers (internal storage is mg; convert at display/export) ──
_MG_TO_KG = 1_000_000.0
_MG_TO_G  = 1_000.0
def _mg_to_kg(v: float) -> float:
    return v / _MG_TO_KG
def _mg_to_g(v: float) -> float:
    return v / _MG_TO_G
def _kg_to_mg(v: float) -> float:
    return v * _MG_TO_KG
def _g_to_mg(v: float) -> float:
    return v * _MG_TO_G

_QUEUE_CAPACITY_PER_TLS = {
    "TL1": 90,
    "TL2": 70,
    "TL3": 20,
    "TL4": 20,
}
QUEUE_CAPACITY = 50
UNIFIED_BP_CAPACITY: float = 50.0
INTENT_SHARING_DIM = 32
INTENT_COMM_RANGE  = 2
GAT_UPDATE_INTERVAL = 2
_TLS_LANE_COUNTS = {
    tls_id: len(UPSTREAM_LANES.get(tls_id, [])) + len(DOWNSTREAM_LANES.get(tls_id, []))
    for tls_id in ["TL1", "TL2", "TL3", "TL4"]
}
BASE_PORT = 8813
_ALL_SUBSCRIBED_LANES:     set[str] = set()
_ALL_SUBSCRIBED_DETECTORS: set[str] = set()
_SUBSCRIPTION_VARS = [
    getattr(tc, 'LAST_STEP_VEHICLE_ID_LIST',  0x12),
    getattr(tc, 'LAST_STEP_HALTING_NUMBER',   0x14),
    getattr(tc, 'LAST_STEP_VEHICLE_NUMBER',   0x10),
]
_DETECTOR_SUB_VARS = [
    getattr(tc, 'LAST_STEP_VEHICLE_HALTING_NUMBER', 0x14),
]
STOPPED_SPEED_THRESHOLD = 0.1
V_FREE = 13.89
K_JAM  = 0.2
_V_FREE_OVER_KJAM = V_FREE / K_JAM
matplotlib.use("Agg")
plt.rcParams.update({
    "figure.facecolor":     "#0D1117",
    "figure.dpi":           150,
    "axes.facecolor":       "#161B22",
    "axes.edgecolor":       "#30363D",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.labelcolor":      "#E6EDF3",
    "axes.labelsize":       11,
    "axes.labelweight":     "bold",
    "axes.titlecolor":      "#E6EDF3",
    "axes.titlesize":       12,
    "axes.titleweight":     "bold",
    "axes.titlelocation":   "left",
    "axes.grid":            True,
    "grid.color":           "#21262D",
    "grid.linestyle":       "-",
    "grid.linewidth":       0.6,
    "xtick.color":          "#8B949E",
    "ytick.color":          "#8B949E",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.facecolor":     "#21262D",
    "legend.edgecolor":     "#30363D",
    "legend.fontsize":      9,
    "lines.linewidth":      2.2,
    "font.family":          "DejaVu Sans",
    "font.size":            10,
    "figure.titlesize":     15,
    "figure.titleweight":   "bold",
    "text.color":           "#E6EDF3",
})
_TL_COLORS = {"TL1": "#58A6FF", "TL2": "#3FB950",
              "TL3": "#F78166", "TL4": "#D2A8FF"}
_TL_FILL   = {"TL1": "#1F6FEB", "TL2": "#238636",
              "TL3": "#DA3633", "TL4": "#8957E5"}
_TL_GLOW   = {"TL1": "#0D419D", "TL2": "#196C2E",
              "TL3": "#B91C1C", "TL4": "#6E40C9"}
FIXED_TIME_PLANS = {
    "TL1": {
        "cycle_length": 60,
        "offset": 0,
        "phases": [
            (29, "rrrrGGGGGG"),
            ( 3, "rrrryyyyyy"),
            (25, "GGGGrrrrrr"),
            ( 3, "yyyyrrrrrr"),
        ],
    },
    "TL2": {
        "cycle_length": 60,
        "offset": 0,
        "phases": [
            (30, "rrrrrGGGrrrr"),
            ( 3, "rrrrryyyrrrr"),
            (14, "GGGGGrrrrrrr"),
            ( 3, "GGyyyrrrrrrr"),
            (10, "rrrrrrrrGGGG"),
            ( 3, "rrrrrrrryyyy"),
        ],
    },
    "TL3": {
        "cycle_length": 60,
        "offset": 0,
        "phases": [
            (18, "rGGGG"),
            ( 3, "ryyGy"),
            (36, "Grrrr"),
            ( 3, "yrrrr"),
        ],
    },
    "TL4": {
        "cycle_length": 60,
        "offset": 10,
        "phases": [
            (38, "GGrr"),
            ( 3, "yyrr"),
            (16, "rrGG"),
            ( 3, "rryy"),
        ],
    },
}
from contextlib import contextmanager

# ---------------------------------------------------------------------------
# XML output / offline-validation configuration
# ---------------------------------------------------------------------------
# When enabled, SUMO episodes also write tripinfo.xml / emissions.xml /
# fcd.xml (via extra sumo_cmd flags -- see sumo_env._xml_output_args) so that
# TraCI-collected metrics can be cross-validated offline via xml_metrics.py.
# Off by default: writing FCD every step is expensive and unnecessary for
# ordinary RL training runs.
# ENABLE_XML_OUTPUT      = False
ENABLE_XML_VALIDATION  = True   # requires ENABLE_XML_OUTPUT=True to have any effect
XML_OUTPUT_DIR         = "results"     # directory SUMO writes tripinfo/emissions/fcd into
XML_TRIPINFO_FILE      = "tripinfo.xml"
XML_EMISSIONS_FILE     = "emissions.xml"
XML_FCD_FILE           = "fcd.xml"
XML_VALIDATION_REL_TOL = 0.05    # 5% relative tolerance for TraCI-vs-XML checks
XML_VALIDATION_ABS_TOL = 1.0     # absolute floor tolerance (mg / count units)

__all__ = ['_EP_STATS', '_emission_tracker', '_active_evs', '_ev_wait_tracker', '_ALL_SUBSCRIBED_LANES', '_ALL_SUBSCRIBED_DETECTORS', '_SUBSCRIPTION_VARS', '_DETECTOR_SUB_VARS', '_V_FREE_OVER_KJAM', '_TLS_LANE_COUNTS', '_QUEUE_CAPACITY_PER_TLS', '_TL_COLORS', '_TL_FILL', '_TL_GLOW', '_cuda_available_early', 'GAT_HEADS', 'GAT_DROPOUT', 'GAT_LR', 'UPSTREAM_LANES', 'DOWNSTREAM_LANES', 'INTERSECTIONS', 'TLS_DETECTOR_MAP', 'LANE_DETECTOR_MAP', 'NEIGHBOR_MAP', 'LOCAL_CLUSTERS', 'GATEWAY_NODE', 'CLUSTER_OF', 'TOTAL_STEPS', 'GAMMA', 'GAE_LAMBDA', 'LEARNING_RATE', 'ENTROPY_COEFF', 'VALUE_LOSS_COEFF', 'CLIP_RATIO', 'BATCH_SIZE', 'MEMORY_CAPACITY', 'N_EPOCHS', 'HIDDEN_DIM', 'LAMBDA', 'STEP_LENGTH', 'MIN_GREEN_TIME', 'MIN_GREEN_STEPS', 'REWARD_CLIP', 'EV_PRIORITY_ENABLED', 'EV_BETA', 'EV_ALPHA', 'EV_CLASSES', 'EV_TYPE_KEYWORDS', 'QUEUE_CAPACITY', 'UNIFIED_BP_CAPACITY', 'INTENT_SHARING_DIM', 'INTENT_COMM_RANGE', 'GAT_UPDATE_INTERVAL', 'INTENT_CONSISTENCY_COEFF', 'BASE_PORT', 'STOPPED_SPEED_THRESHOLD', 'V_FREE', 'K_JAM', 'FIXED_TIME_PLANS', 'KL_ACTOR_LR', 'KL_CRITIC_LR', 'KL_GAT_LR', 'KL_TARGET', 'KL_LR_MIN', 'KL_LR_MAX', 'KL_HIGH_MULT', 'KL_LOW_MULT', 'KL_LR_DOWN', 'KL_LR_UP', 'CLIP_MIN', 'CLIP_MAX', 'CLIP_DOWN', 'CLIP_UP', 'ENABLE_XML_VALIDATION', 'XML_OUTPUT_DIR', 'XML_TRIPINFO_FILE', 'XML_EMISSIONS_FILE', 'XML_FCD_FILE', 'XML_VALIDATION_REL_TOL', 'XML_VALIDATION_ABS_TOL']
