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

# Display-unit conversion (internal storage is mg; convert at display/export)
_MG_TO_KG = 1_000_000.0
_MG_TO_G  = 1_000.0

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
from .agent import *

__all__ = ['_smooth', '_stats', '_add_glow_line', '_safe_fill', '_fix_single_ep_axis', 'ResultsWriter', 'KLDiagnosticsWriter', 'TrainingPlotter', 'TBComparisonLogger', 'export_comparison_excel', 'plot_comparison', 'safe_improvement_pct', 'RichUI']


def _smooth(values, window=5):
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result

def _stats(values):
    a = np.array(values, dtype=float)
    n = len(a)
    w = max(5, n // 10)
    smooth = np.convolve(a, np.ones(w) / w, mode="valid")
    if len(smooth) >= 2:
        _improvement = float(smooth[-1] - smooth[0])
    else:
        _improvement = float(a[-1] - a[0])
    _improvement_pct = float(_improvement / (abs(smooth[0]) + 1e-9) * 100) if len(smooth) >= 1 else 0.0
    return {
        "min":              float(np.min(a)),
        "max":              float(np.max(a)),
        "mean":             float(np.mean(a)),
        "std":              float(np.std(a)),
        "median":           float(np.median(a)),
        "final":            float(a[-1]),
        "first":            float(a[0]),
        "improvement":      _improvement,
        "improvement_pct":  _improvement_pct,
        "best_ep":          int(np.argmax(a)),
        "worst_ep":         int(np.argmin(a)),
        "q25":              float(np.percentile(a, 25)),
        "q75":              float(np.percentile(a, 75)),
        "iqr":              float(np.percentile(a, 75) - np.percentile(a, 25)),
        "pct_positive":     float(np.mean(a > 0) * 100),
        "skewness":         float(((a - np.mean(a)) ** 3).mean() / (np.std(a) ** 3 + 1e-10)),
    }

def _add_glow_line(ax, x, y, color, glow_color, lw=2.2, alpha_glow=0.3, n_layers=3):
    if len(x) == 1:
        ax.scatter(x, y, color=color, s=120, zorder=5, edgecolors=glow_color, linewidths=2.5)
        return
    for i in range(n_layers, 0, -1):
        ax.plot(x, y, color=glow_color, linewidth=lw + i * 2.5,
                alpha=alpha_glow / i, zorder=2, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=lw, zorder=3, solid_capstyle="round")

def _safe_fill(ax, eps, values, **kwargs):
    if len(eps) == 1:
        ax.bar(eps[0], values[0], width=0.6,
               **{k: v for k, v in kwargs.items() if k in ("color", "alpha", "zorder")})
    else:
        ax.fill_between(eps, values, **kwargs)

def _fix_single_ep_axis(ax, eps, pad=0.5):
    if len(eps) <= 1:
        ep0 = eps[0] if eps else 0
        ax.set_xlim(ep0 - pad, ep0 + pad)
        ax.set_xticks([ep0])
    elif len(eps) <= 3:
        ax.set_xlim(min(eps) - pad, max(eps) + pad)
        ax.set_xticks(list(eps))

class ResultsWriter:
    _SAFE_FALLBACK = os.path.join(os.path.expanduser("~"), "sumo_logs")

    @staticmethod
    def _is_writable(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            tp = os.path.join(directory, ".write_test")
            with open(tp, "w") as f:
                f.write("test")
            os.remove(tp)
            return True
        except OSError:
            return False

    def __init__(self, intersections, filepath, header=None,
                 extra_fields: "list[str] | None" = None):
        filepath = os.path.abspath(filepath)
        dir_path = os.path.dirname(filepath)
        if not self._is_writable(dir_path):
            os.makedirs(self._SAFE_FALLBACK, exist_ok=True)
            filepath = os.path.join(self._SAFE_FALLBACK, os.path.basename(filepath))
        self.filepath      = filepath + ".csv"
        self.intersections = list(intersections)
        _extra             = list(extra_fields) if extra_fields else []
        # as trailing columns so existing consumers that only read episode + TL
        # columns are not broken (backward compatible).
        fieldnames         = ["episode"] + self.intersections + ["ep_duration_ms"] + _extra
        file_exists        = os.path.isfile(self.filepath)
        self.f             = open(self.filepath, "a", newline="")
        self.writer        = csv.DictWriter(self.f, fieldnames=fieldnames)
        if not file_exists:
            if header:
                for k, v in header.items():
                    self.f.write(f"# {k}: {v}\n")
            self.writer.writeheader()
        # Track last written episode to prevent duplicate rows on resume.
        # partial writes from a crash mid-flush don't leave undetected duplicates.
        self._last_written_ep: int = -1
        if file_exists:
            try:
                with open(self.filepath, "r") as _rf:
                    _lines = [l for l in _rf if not l.startswith("#") and l.strip()]
                for _line in _lines[1:]:  # skip header row
                    try:
                        _ep_val = int(float(_line.strip().split(",")[0]))
                        if _ep_val > self._last_written_ep:
                            self._last_written_ep = _ep_val
                    except (ValueError, IndexError):
                        pass
            except Exception:
                pass

    def write_row(self, row):
        ep = int(row.get("episode", -1))
        if ep <= self._last_written_ep:
            return  # skip duplicate episode on resume
        self._last_written_ep = ep
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


class KLDiagnosticsWriter:
    """CSV logger for per-episode / per-agent KL diagnostics.

    Unlike ResultsWriter (one row per episode, one column per intersection),
    this writes one row per (episode, tls_id) pair -- i.e. multiple rows can
    share the same episode number, which ResultsWriter's dedup logic does not
    support. Surfaces two KL signals per agent per episode, both computed
    inside agent.train_epoch():

      * approx_kl -- the sample-based PPO estimator
        (mean(old_log_prob - new_log_prob)), computed only on the actions
        that were actually sampled.  DIAGNOSTIC ONLY -- kept for comparison;
        not fed back into any controller, clip adaptation, or optimizer logic.
        Can be negative (no non-negativity guarantee).
      * true_kl   -- the full distributional KL(old || new) computed from
        the complete logit vectors (compute_kl_from_logits).  Mathematically
        guaranteed >= 0 (Gibbs' inequality).  This IS the signal fed into
        KLAdaptiveLRController and KLAdaptiveClipController (see agent.py
        train_epoch()).

    Purely additive logging -- does not touch training behavior.

    Columns: episode, tls_id, approx_kl, true_kl, kl_increased, kl_decreased, kl_unchanged, kl_degenerate, consec_increases, lr_pinned
    """

    _SAFE_FALLBACK = os.path.join(os.path.expanduser("~"), "sumo_logs")
    _FIELDNAMES = ["episode", "tls_id", "approx_kl", "true_kl",
                   "kl_increased", "kl_decreased", "kl_unchanged", "kl_degenerate",
                   "consec_increases", "lr_pinned"]

    @staticmethod
    def _is_writable(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            tp = os.path.join(directory, ".write_test")
            with open(tp, "w") as f:
                f.write("test")
            os.remove(tp)
            return True
        except OSError:
            return False

    def __init__(self, filepath="kl_diagnostics"):
        filepath = os.path.abspath(filepath)
        dir_path = os.path.dirname(filepath)
        if not self._is_writable(dir_path):
            os.makedirs(self._SAFE_FALLBACK, exist_ok=True)
            filepath = os.path.join(self._SAFE_FALLBACK, os.path.basename(filepath))
        self.filepath = filepath + ".csv"
        file_exists   = os.path.isfile(self.filepath)
        self.f        = open(self.filepath, "a", newline="")
        self.writer   = csv.DictWriter(self.f, fieldnames=self._FIELDNAMES)
        if not file_exists:
            self.writer.writeheader()

        # Track (episode, tls_id) pairs already written so resuming training
        # (which replays episodes from a checkpoint) doesn't duplicate rows.
        self._written_keys: set = set()
        if file_exists:
            try:
                with open(self.filepath, "r") as _rf:
                    _reader = csv.DictReader(_rf)
                    for _row in _reader:
                        try:
                            self._written_keys.add(
                                (int(float(_row["episode"])), _row["tls_id"]))
                        except (KeyError, ValueError):
                            pass
            except Exception:
                pass

    def write_row(self, episode: int, tls_id: str, approx_kl: float,
                  true_kl: float = 0.0, counts: "dict | None" = None):
        key = (int(episode), tls_id)
        if key in self._written_keys:
            return  # skip duplicate (episode, tls_id) row on resume
        self._written_keys.add(key)
        counts = counts or {}
        self.writer.writerow({
            "episode":            int(episode),
            "tls_id":             tls_id,
            "approx_kl":          round(float(approx_kl), 6),
            "true_kl":            round(float(true_kl), 6),
            "kl_increased":       counts.get("increased", 0),
            "kl_decreased":       counts.get("decreased", 0),
            "kl_unchanged":       counts.get("unchanged", 0),
            "kl_degenerate":      counts.get("degenerate", 0),
            "consec_increases":   counts.get("consecutive_increases", 0),
            "lr_pinned":          counts.get("is_lr_pinned", False),
        })
        self.f.flush()

    def close(self):
        self.f.close()


class TrainingPlotter:

    def __init__(self, intersections, save_dir="./plots"):
        self.intersections = intersections
        self.save_dir      = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self._history = {
            "reward":        defaultdict(list),
            "queue":         defaultdict(list),
            "wait_time":     defaultdict(list),
            "loss":          defaultdict(list),
        }
        self._ep_duration_ms: list = []
        self._episodes  = []
        self._step_data: dict = {}

        self._global_steps: list = []
        self._wall_times:   list = []
        self._step_history: dict = {
            "reward":    defaultdict(list),
            "queue":     defaultdict(list),
            "wait_time": defaultdict(list),
        }
        self._training_start: float = time.time()
        self._global_step_counter: int = 0

        # Intent tracking: per-agent mean intent vector at the end of each episode.
        # Shape per TLS: list of np.ndarray(intent_dim,), one entry per episode.
        self._intent_history: dict = defaultdict(list)
        # GAT alpha (local/global mix ratio) per episode: list of np.ndarray(n_agents,).
        self._alpha_history:  list = []

    def record(self, episode, reward_dict, queue_dict=None,
               wt_dict=None, loss_dict=None, ep_duration_ms: float = 0.0):
        self._episodes.append(episode)
        self._ep_duration_ms.append(float(ep_duration_ms))
        for tl in self.intersections:
            self._history["reward"][tl].append(float(reward_dict.get(tl, 0)))
            if queue_dict:
                self._history["queue"][tl].append(float(queue_dict.get(tl, 0)))
            if wt_dict:
                self._history["wait_time"][tl].append(float(wt_dict.get(tl, 0)))
            if loss_dict:
                self._history["loss"][tl].append(float(loss_dict.get(tl, 0)))

    def record_steps(self, agents: dict):
        for tl in self.intersections:
            ag = agents.get(tl)
            if ag:
                self._step_data[tl] = {
                    "reward": list(ag.step_rewards),
                    "queue":  list(ag.step_queues),
                    "wait":   list(ag.step_waits),
                    "loss":   list(ag.step_losses),
                }
                n = len(ag.step_rewards)
                for i, (r, q, w) in enumerate(zip(ag.step_rewards,
                                                   ag.step_queues,
                                                   ag.step_waits)):
                    gs = self._global_step_counter + i
                    self._step_history["reward"][tl].append((gs, float(r)))
                    self._step_history["queue"][tl].append((gs, float(q)))
                    self._step_history["wait_time"][tl].append((gs, float(w)))
                    elapsed = time.time() - self._training_start
                    if tl == self.intersections[0]:
                        self._global_steps.append(gs)
                        self._wall_times.append(elapsed)

                # Snapshot end-of-episode intent vector for this agent.
                # IPPOAgent has no intent vector — use a zero placeholder.
                if hasattr(ag, "current_intent"):
                    self._intent_history[tl].append(ag.current_intent.copy())
                else:
                    # Keep history length consistent with episode count
                    prev = (self._intent_history[tl][-1]
                            if self._intent_history[tl] else np.zeros(1, dtype=np.float32))
                    self._intent_history[tl].append(prev.copy())

        # Record GAT alpha (local/global mix) from one representative agent.
        for tl in self.intersections:
            ag = agents.get(tl)
            if ag is None:
                continue
            try:
                # GAT v2: Dynamic alpha (last_gates) per intersection
                # We record the snapshot from the last step of the episode.
                gates = ag.gat_aggregator._module.last_gates
                if gates is not None:
                    self._alpha_history.append(gates.copy())
            except Exception:
                pass
            break

        if agents:
            first_tl = self.intersections[0]
            ag = agents.get(first_tl)
            if ag:
                self._global_step_counter += len(ag.step_rewards)

    def plot_all(self):
        n         = len(self._episodes)
        use_steps = (n <= 2) and bool(self._step_data)
        # Reduced set: only dashboard + correlation heatmap.
        # (reward/queue/wait/loss curves, violins, per-step, intent viz disabled)
        self._plot_dashboard(use_steps=use_steps)
        self._plot_correlation_heatmap()
        print(f"[TrainingPlotter] Plots saved to '{self.save_dir}' (dashboard + correlation only)")

    def _plot_steps_reward(self):
        self._plot_steps_metric("reward", "Reward (Back Pressure)",
                                "reward_curves.png",
                                "Step Reward - Eq. 4.92 Back Pressure (within episode)")

    def _plot_steps_queue(self):
        self._plot_steps_metric("queue", "Vehicles Halting",
                                "queue_curves.png",
                                "Queue Length per Step (within episode)")

    def _plot_steps_wait(self):
        self._plot_steps_metric("wait", "Waiting Time (s / vehicle)",
                                "wait_time_curves.png",
                                "Mean Waiting Time per Step (within episode)")

    def _plot_steps_loss(self):
        self._plot_steps_metric("loss", "PPO Loss",
                                "loss_curves.png",
                                "MAPPO-PIS Training Loss per Step",
                                skip_zeros=True)

    def _plot_steps_metric(self, key, ylabel, fname, title, skip_zeros=False):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle(title, fontsize=14, color="#E6EDF3",
                     fontweight="bold", x=0.5, y=1.01, ha="center")

        for ax, tl in zip(axes.flat, self.intersections):
            ax.set_facecolor("#161B22")
            for spine in ax.spines.values():
                spine.set_color("#30363D")

            raw_all = self._step_data.get(tl, {}).get(key, [])
            if not raw_all:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#8B949E", fontsize=12)
                ax.set_title(f"  {tl}", color=_TL_COLORS[tl],
                             fontweight="bold", fontsize=13, loc="left")
                continue

            color  = _TL_COLORS[tl]
            fill   = _TL_FILL[tl]
            glow   = _TL_GLOW[tl]

            if skip_zeros:
                zero_xs      = [i for i, v in enumerate(raw_all) if v == 0]
                nonzero_pairs = [(i, v) for i, v in enumerate(raw_all) if v > 0]
                if zero_xs:
                    ax.scatter(zero_xs, [0] * len(zero_xs),
                               color=color, alpha=0.20, s=12,
                               marker='x', zorder=2, linewidths=1.0)
                if not nonzero_pairs:
                    ax.set_title(f"  {tl} (all zeros)", fontweight="bold", fontsize=13,
                                 color=color, loc="left", pad=8)
                    ax.text(0.5, 0.5, "⚠ all loss values = 0\n(buffer empty every step)",
                            transform=ax.transAxes, ha="center", va="center",
                            color="#8B949E", fontsize=10)
                    ax.set_xlabel("Simulation Step", fontsize=9, color="#8B949E")
                    ax.set_ylabel(ylabel, fontsize=9, color="#8B949E")
                    continue
                xs       = [p[0] for p in nonzero_pairs]
                plot_raw = [p[1] for p in nonzero_pairs]
            else:
                plot_raw = raw_all
                xs       = list(range(len(plot_raw)))
            w      = max(5, len(plot_raw) // 50)
            smooth = _smooth(plot_raw, window=w)
            mean_v = float(np.mean(plot_raw))
            st     = _stats(plot_raw)

            ax.fill_between(xs, plot_raw, alpha=0.10, color=fill, zorder=1)
            ax.fill_between(xs, smooth,   alpha=0.18, color=fill, zorder=1)
            ax.plot(xs, plot_raw, color=color, alpha=0.18, linewidth=0.6, zorder=2)
            _add_glow_line(ax, xs, smooth, color, glow, lw=2.2)

            if len(plot_raw) > 20:
                ax.axhspan(st["q25"], st["q75"], alpha=0.06, color=color, zorder=0)

            ax.axhline(mean_v, color="#8B949E", linestyle=":", linewidth=1.1, zorder=4)
            ax.text(xs[-1], mean_v, f"  mu={mean_v:.4f}",
                    va="center", ha="left", fontsize=8, color="#8B949E",
                    fontweight="bold", clip_on=False)

            ax.set_title(f"  {tl}", fontweight="bold", fontsize=13,
                         color=color, loc="left", pad=8)
            ax.set_xlabel("Simulation Step", fontsize=9, color="#8B949E")
            ax.set_ylabel(ylabel, fontsize=9, color="#8B949E")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            ax.tick_params(colors="#8B949E")

            txt = (f"min  {st['min']:.4f}\n"
                   f"max  {st['max']:.4f}\n"
                   f"std  {st['std']:.4f}\n"
                   f"IQR  {st['iqr']:.4f}")
            ax.text(0.02, 0.04, txt, transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="left", family="monospace",
                    color="#8B949E",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                              edgecolor="#30363D", alpha=0.95))

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save(fig, fname)

    def _plot_reward_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Episode Reward per Intersection (Eq. 4.92 Back Pressure)",
                     fontsize=15, color="#E6EDF3", fontweight="bold",
                     x=0.5, y=1.01, ha="center")

        eps = self._episodes
        for ax, tl in zip(axes.flat, self.intersections):
            raw    = self._history["reward"][tl]
            smooth = _smooth(raw, window=max(3, len(raw) // 20 + 1))
            color  = _TL_COLORS[tl]
            fill   = _TL_FILL[tl]
            glow   = _TL_GLOW[tl]
            mean_v = float(np.mean(raw))
            st     = _stats(raw)

            ax.set_facecolor("#161B22")
            for spine in ax.spines.values():
                spine.set_color("#30363D")

            ax.fill_between(eps, raw,    alpha=0.12, color=fill, zorder=1)
            ax.fill_between(eps, smooth, alpha=0.20, color=fill, zorder=1)
            ax.plot(eps, raw, color=color, alpha=0.18, linewidth=0.8, zorder=2)
            _add_glow_line(ax, eps, smooth, color, glow, lw=2.5)

            if len(raw) > 10:
                ax.axhspan(st["q25"], st["q75"], alpha=0.07, color=color, zorder=0)
            ax.axhline(mean_v, color="#8B949E", linestyle=":", linewidth=1.1, zorder=4)
            ax.text(eps[-1], mean_v, f"  mu={mean_v:.4f}",
                    va="center", ha="left", fontsize=8, color="#8B949E",
                    fontweight="bold", clip_on=False)

            best_idx = st["best_ep"]
            if best_idx < len(eps):
                ax.scatter([eps[best_idx]], [raw[best_idx]], color="#FFD700",
                           s=90, zorder=6, marker="*",
                           label=f"Best ep {eps[best_idx]}")

            ax.set_title(f"  {tl}", fontweight="bold", fontsize=13,
                         color=color, loc="left", pad=8)
            ax.set_ylabel("Reward (Back Pressure)", fontsize=10, color="#8B949E")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
            ax.tick_params(colors="#8B949E")

            txt = (f"min  {st['min']:.4f}\n"
                   f"max  {st['max']:.4f}\n"
                   f"std  {st['std']:.4f}\n"
                   f"IQR  {st['iqr']:.4f}")
            ax.text(0.02, 0.04, txt, transform=ax.transAxes,
                    fontsize=8, va="bottom", ha="left", family="monospace",
                    color="#8B949E",
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                              edgecolor="#30363D", alpha=0.95))
            ax.legend(loc="upper right", fontsize=8,
                      facecolor="#21262D", edgecolor="#30363D")

        for ax in axes[1]:
            ax.set_xlabel("Episode", fontsize=10, color="#8B949E")
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save(fig, "reward_curves.png")

    def _plot_queue_curves(self):
        if not any(self._history["queue"][tl] for tl in self.intersections):
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        fig.suptitle("Mean Queue Length per Episode",
                     fontsize=14, color="#E6EDF3", fontweight="bold")
        eps = self._episodes
        for tl in self.intersections:
            raw = self._history["queue"][tl]
            if not raw:
                continue
            smooth = _smooth(raw, window=max(3, len(raw) // 20 + 1))
            color  = _TL_COLORS[tl]
            glow   = _TL_GLOW[tl]
            ax.fill_between(eps, raw, alpha=0.10, color=_TL_FILL[tl], zorder=1)
            ax.plot(eps, raw, color=color, alpha=0.20, linewidth=0.8, zorder=2)
            _add_glow_line(ax, eps, smooth, color, glow, lw=2.3)
            ax.text(eps[-1], smooth[-1],
                    f"  {tl} (mu={np.mean(raw):.2f})",
                    va="center", ha="left", fontsize=9,
                    color=color, fontweight="bold", clip_on=False)
        ax.set_xlabel("Episode",          fontsize=11, color="#8B949E")
        ax.set_ylabel("Vehicles Halting", fontsize=11, color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "queue_curves.png")

    def _plot_wait_time_curves(self):
        if not any(self._history["wait_time"][tl] for tl in self.intersections):
            return
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        fig.suptitle("Mean Waiting Time per Episode",
                     fontsize=14, color="#E6EDF3", fontweight="bold")
        eps = self._episodes
        for tl in self.intersections:
            raw = self._history["wait_time"][tl]
            if not raw:
                continue
            smooth = _smooth(raw, window=max(3, len(raw) // 20 + 1))
            color  = _TL_COLORS[tl]
            glow   = _TL_GLOW[tl]
            ax.fill_between(eps, raw, alpha=0.10, color=_TL_FILL[tl], zorder=1)
            ax.plot(eps, raw, color=color, alpha=0.20, linewidth=0.8, zorder=2)
            _add_glow_line(ax, eps, smooth, color, glow, lw=2.3)
            ax.text(eps[-1], smooth[-1],
                    f"  {tl} (mu={np.mean(raw):.1f}s)",
                    va="center", ha="left", fontsize=9,
                    color=color, fontweight="bold", clip_on=False)
        ax.set_xlabel("Episode",                    fontsize=11, color="#8B949E")
        ax.set_ylabel("Waiting Time (s / vehicle)", fontsize=11, color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        self._save(fig, "wait_time_curves.png")

    def _plot_loss_curves(self):
        # `any(v > 0 ...)` guard meant all-zero agents showed "No loss data yet".
        has_episode_loss = any(
            len(self._history["loss"][tl]) > 0
            for tl in self.intersections
        )
        has_step_loss = any(
            len(self._step_data.get(tl, {}).get("loss", [])) > 0
            for tl in self.intersections
        )

        # When repeated consecutive values exceed 35% of total, we have a staircase.
        _is_parallel_mode = False
        for tl in self.intersections:
            _vals = self._history["loss"][tl]
            if len(_vals) > 3:
                _repeated = sum(1 for i in range(1, len(_vals)) if _vals[i] == _vals[i-1])
                if _repeated > len(_vals) * 0.35:
                    _is_parallel_mode = True
                    break

        # chart (bottom). The bar chart shows what % of episodes had zero loss per agent,
        # making the buffer-empty problem immediately visible without digging into CSVs.
        fig = plt.figure(figsize=(14, 7))
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("MAPPO-PIS Training Loss per Episode",
                     fontsize=14, color="#E6EDF3", fontweight="bold")

        # GridSpec: main loss plot takes 70% height, bar chart takes 30%
        gs_lc = gridspec.GridSpec(2, 1, figure=fig,
                                  height_ratios=[0.70, 0.30],
                                  hspace=0.42,
                                  top=0.91, bottom=0.08,
                                  left=0.07, right=0.95)
        ax     = fig.add_subplot(gs_lc[0])
        ax_bar = fig.add_subplot(gs_lc[1])

        for _ax in (ax, ax_bar):
            _ax.set_facecolor("#161B22")
            for spine in _ax.spines.values():
                spine.set_color("#30363D")

        if not has_episode_loss and not has_step_loss:
            ax.text(0.5, 0.5, "No loss data yet",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#8B949E", fontsize=12)
            ax.set_xlabel("Episode",  fontsize=11, color="#8B949E")
            ax.set_ylabel("PPO Loss", fontsize=11, color="#8B949E")
            ax.tick_params(colors="#8B949E")
            ax_bar.set_visible(False)
            self._save(fig, "loss_curves.png")
            return

        plotted_any   = False
        zero_pct_data = {}   # tl -> % of episodes with loss == 0

        for tl in self.intersections:
            all_ep_vals   = list(zip(self._episodes, self._history["loss"][tl]))
            step_vals_all = self._step_data.get(tl, {}).get("loss", [])

            if all_ep_vals:
                idxs = [e for e, _ in all_ep_vals]
                raw  = [v for _, v in all_ep_vals]
            elif step_vals_all:
                idxs = list(range(len(step_vals_all)))
                raw  = step_vals_all
            else:
                zero_pct_data[tl] = 0.0
                continue

            # Compute % zero for bar subplot
            n_total = len(raw)
            n_zero  = sum(1 for v in raw if v == 0)
            zero_pct_data[tl] = (n_zero / max(n_total, 1)) * 100.0

            color = _TL_COLORS[tl]
            glow  = _TL_GLOW[tl]

            # Zero episodes are shown as faint 'x' scatter markers at y=0 so they remain
            # visible without polluting the loss scale.
            nonzero_idxs = [i for i, v in zip(idxs, raw) if v > 0]
            nonzero_vals = [v for v in raw if v > 0]
            zero_idxs    = [i for i, v in zip(idxs, raw) if v == 0]

            # Plot zero-value episodes as semi-transparent 'x' markers at y=0
            if zero_idxs:
                ax.scatter(zero_idxs, [0] * len(zero_idxs),
                           color=color, alpha=0.25, s=18, marker='x',
                           zorder=2, linewidths=1.2)

            # (drawstyle='steps-post') instead of a smooth interpolated curve.
            # This correctly communicates that ONE value covers a whole batch of N episodes.
            if nonzero_idxs:
                smooth = _smooth(nonzero_vals, window=max(2, len(nonzero_vals) // 10 + 1))
                if _is_parallel_mode:
                    # Raw faint background trace as steps
                    ax.step(nonzero_idxs, nonzero_vals, where='post',
                            color=color, alpha=0.20, linewidth=0.8, zorder=2)
                    # Bold step line for the smoothed (batch-level) values
                    ax.step(nonzero_idxs, smooth, where='post',
                            color=color, linewidth=2.2, zorder=3,
                            label=tl)
                    # Glow effect on top
                    for _gi in range(3, 0, -1):
                        ax.step(nonzero_idxs, smooth, where='post',
                                color=glow, linewidth=2.2 + _gi * 2.5,
                                alpha=0.10 / _gi, zorder=2)
                else:
                    ax.plot(nonzero_idxs, nonzero_vals,
                            color=color, alpha=0.25, linewidth=0.8, zorder=2)
                    _add_glow_line(ax, nonzero_idxs, smooth, color, glow, lw=2.0)
                ax.text(nonzero_idxs[-1], smooth[-1], f"  {tl}",
                        va="center", ha="left", fontsize=9,
                        color=color, fontweight="bold", clip_on=False)
            elif zero_idxs:
                # All values are zero -- still label the agent
                ax.text(zero_idxs[-1], 0.0, f"  {tl} (all zero)",
                        va="bottom", ha="left", fontsize=9,
                        color=color, alpha=0.6, clip_on=False)
            plotted_any = True

        ax.text(0.01, 0.02,
                "⚠ zero values (×) = buffer empty / no training that episode",
                transform=ax.transAxes, fontsize=8, color="#8B949E",
                va="bottom", ha="left", style="italic")

        if _is_parallel_mode:
            ax.text(0.99, 0.98,
                    "Note: parallel mode -- one loss value per batch of N episodes\n"
                    "(step-line rendering -- horizontal = same batch update)",
                    transform=ax.transAxes, fontsize=8, color="#D29922",
                    va="top", ha="right", style="italic")

        if not plotted_any:
            ax.text(0.5, 0.5, "No loss values recorded",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#8B949E", fontsize=12)
        ax.set_xlabel("Episode",  fontsize=11, color="#8B949E")
        ax.set_ylabel("PPO Loss", fontsize=11, color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # This makes the buffer-empty / no-training problem immediately obvious.
        tl_labels = list(zero_pct_data.keys())
        tl_pcts   = [zero_pct_data[tl] for tl in tl_labels]
        tl_colors = [_TL_COLORS.get(tl, "#8B949E") for tl in tl_labels]
        x_pos = range(len(tl_labels))

        if tl_pcts:
            bars = ax_bar.bar(x_pos, tl_pcts,
                              color=tl_colors, alpha=0.75,
                              edgecolor="#0D1117", linewidth=1.2, zorder=3)
            for bar, pct, tl in zip(bars, tl_pcts, tl_labels):
                if pct > 0:
                    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.8,
                                f"{pct:.0f}%",
                                ha="center", va="bottom",
                                fontsize=8.5, fontweight="bold",
                                color=_TL_COLORS.get(tl, "#E6EDF3"))
            # Draw a red reference line at 100% (all episodes zero)
            ax_bar.axhline(100, color="#F78166", linestyle="--",
                           linewidth=0.9, alpha=0.5, zorder=1)
            ax_bar.set_ylim(0, max(max(tl_pcts) + 12, 15))
        ax_bar.set_xticks(list(x_pos))
        ax_bar.set_xticklabels(tl_labels, fontsize=10,
                               fontweight="bold", color="#E6EDF3")
        ax_bar.set_ylabel("% zero-loss\nepisodes", fontsize=9,
                          color="#8B949E", linespacing=1.2)
        ax_bar.set_title("  Buffer health -- % of episodes with zero loss (× markers above)",
                         fontsize=9, color="#8B949E", loc="left", pad=4)
        ax_bar.tick_params(colors="#8B949E")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.25, color="#30363D")
        ax_bar.text(0.99, 0.92,
                    "0% = always training  |  100% = never training",
                    transform=ax_bar.transAxes, fontsize=7.5, color="#8B949E",
                    va="top", ha="right", style="italic")

        self._save(fig, "loss_curves.png")

    def _plot_reward_comparison_bar(self):
        n = len(self._episodes)
        w = max(1, n // 3)
        windows = {
            f"Early (0-{w-1})":      slice(0,     w),
            f"Middle ({w}-{2*w-1})": slice(w,     2 * w),
            f"Late ({2*w}-{n-1})":   slice(2 * w, n),
        }
        x     = np.arange(len(self.intersections))
        width = 0.22
        fig, ax = plt.subplots(figsize=(13, 7))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        fig.suptitle("Average Back Pressure Reward - Training Progression",
                     fontsize=14, color="#E6EDF3", fontweight="bold")

        offsets = [-width, 0, width]
        alphas  = [0.55, 0.78, 1.00]
        for i, (label, sl) in enumerate(windows.items()):
            means  = [float(np.mean(self._history["reward"][tl][sl]))
                      if self._history["reward"][tl][sl] else 0.0
                      for tl in self.intersections]
            colors = [_TL_COLORS[tl] for tl in self.intersections]
            bars   = ax.bar(x + offsets[i], means, width, label=label,
                            color=colors, alpha=alphas[i],
                            edgecolor="#0D1117", linewidth=1.2, zorder=3)
            for bar, val in zip(bars, means):
                if val != 0.0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + abs(bar.get_height()) * 0.025,
                            f"{val:.4f}", ha="center", va="bottom",
                            fontsize=8, fontweight="bold", color="#E6EDF3")

        early_raw   = [self._history["reward"][tl][slice(0, w)]   for tl in self.intersections]
        late_raw    = [self._history["reward"][tl][slice(2*w, n)] for tl in self.intersections]
        early_means = [float(np.mean(v)) if v else None for v in early_raw]
        late_means  = [float(np.mean(v)) if v else None for v in late_raw]
        for xi, (e, l) in enumerate(zip(early_means, late_means)):
            if e is None or l is None or not np.isfinite(e) or not np.isfinite(l):
                continue
            delta = l - e
            sign  = "UP" if delta >= 0 else "DN"
            clr   = "#3FB950" if delta >= 0 else "#F78166"
            ax.annotate(f"{sign} {abs(delta):.4f}",
                        xy=(x[xi], 0), xycoords=("data", "axes fraction"),
                        xytext=(x[xi], 1.04), textcoords=("data", "axes fraction"),
                        ha="center", va="bottom", fontsize=9.5,
                        fontweight="bold", color=clr, annotation_clip=False)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="grey", alpha=0.55, label="Early third"),
            Patch(facecolor="grey", alpha=0.78, label="Middle third"),
            Patch(facecolor="grey", alpha=1.00, label="Late third"),
        ]
        ax.legend(handles=legend_els, fontsize=10, loc="lower right",
                  facecolor="#21262D", edgecolor="#30363D")
        ax.set_xticks(x)
        ax.set_xticklabels(self.intersections, fontsize=12, fontweight="bold",
                           color="#E6EDF3")
        ax.set_ylabel("Mean Reward",  fontsize=11, color="#8B949E")
        ax.set_xlabel("Intersection", fontsize=11, color="#8B949E")
        ax.tick_params(colors="#8B949E")
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#30363D")
        fig.tight_layout()
        self._save(fig, "reward_bar_comparison.png")

    def _plot_correlation_heatmap(self):
        reward_matrix = np.array([self._history["reward"][tl]
                                   for tl in self.intersections])
        if reward_matrix.shape[1] < 3:
            return
        corr = np.corrcoef(reward_matrix)
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        cmap = LinearSegmentedColormap.from_list(
            "gh_corr", ["#1F6FEB", "#21262D", "#3FB950"], N=256)
        im   = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="#8B949E")
        cbar.outline.set_edgecolor("#30363D")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E", fontsize=9)
        cbar.set_label("Pearson Correlation", color="#8B949E", fontsize=10)
        for i in range(len(self.intersections)):
            for j in range(len(self.intersections)):
                val       = corr[i, j]
                txt_color = "#E6EDF3" if abs(val) < 0.6 else "#0D1117"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=txt_color)
        ax.set_xticks(range(len(self.intersections)))
        ax.set_yticks(range(len(self.intersections)))
        ax.set_xticklabels(self.intersections, color="#E6EDF3",
                           fontsize=10, fontweight="bold")
        ax.set_yticklabels(self.intersections, color="#E6EDF3",
                           fontsize=10, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color("#30363D")
        ax.set_title("Reward Correlation Matrix", fontsize=13, color="#E6EDF3",
                     fontweight="bold", loc="left", pad=10)
        fig.tight_layout()
        self._save(fig, "correlation_heatmap.png")

    def _plot_distribution_violins(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Reward & Queue Distribution per Intersection",
                     fontsize=14, color="#E6EDF3", fontweight="bold")
        metrics = [
            ("reward", "Reward (Back Pressure)", axes[0]),
            ("queue",  "Queue Length (veh)",      axes[1]),
        ]
        n_eps = len(self._episodes)
        for metric, ylabel, ax in metrics:
            ax.set_facecolor("#161B22")
            for spine in ax.spines.values():
                spine.set_color("#30363D")
            data_list = [self._history[metric][tl] for tl in self.intersections]
            if not any(data_list):
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="#8B949E")
                continue
            positions = list(range(len(self.intersections)))
            if n_eps >= 3:
                parts = ax.violinplot(data_list, positions=positions,
                                      showmeans=True, showmedians=True,
                                      showextrema=True)
                for pc, tl in zip(parts["bodies"], self.intersections):
                    pc.set_facecolor(_TL_FILL[tl])
                    pc.set_edgecolor(_TL_COLORS[tl])
                    pc.set_alpha(0.65)
                for partname in ("cbars", "cmins", "cmaxes"):
                    parts[partname].set_edgecolor("#8B949E")
                    parts[partname].set_linewidth(1.2)
                parts["cmedians"].set_edgecolor("#FFD700")
                parts["cmedians"].set_linewidth(2.0)
                parts["cmeans"].set_edgecolor("#FF6B9D")
                parts["cmeans"].set_linewidth(2.0)
                for i, (vals, tl) in enumerate(zip(data_list, self.intersections)):
                    if vals:
                        jitter = np.random.uniform(-0.08, 0.08, len(vals))
                        ax.scatter(np.full(len(vals), i) + jitter, vals,
                                   color=_TL_COLORS[tl], alpha=0.25, s=8, zorder=4)
            else:
                bar_w = 0.5
                for i, (vals, tl) in enumerate(zip(data_list, self.intersections)):
                    if not vals:
                        continue
                    mean_v = float(np.mean(vals))
                    ax.bar(i, mean_v, width=bar_w, color=_TL_FILL[tl],
                           edgecolor=_TL_COLORS[tl], alpha=0.75, zorder=3, linewidth=1.5)
                    jitter = np.random.uniform(-0.12, 0.12, len(vals))
                    ax.scatter(np.full(len(vals), i) + jitter, vals,
                               color=_TL_COLORS[tl], s=60, zorder=5,
                               edgecolors="#E6EDF3", linewidths=0.8)
                    ax.hlines(mean_v, i - bar_w / 2, i + bar_w / 2,
                              colors="#FF6B9D", linewidth=2.0, zorder=6)
                    ax.text(i, mean_v, f"\n{mean_v:.4f}",
                            ha="center", va="top", fontsize=8,
                            color="#FF6B9D", fontweight="bold")
                ax.text(0.98, 0.98,
                        f"n={n_eps} ep - violin requires >=3 eps",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="#8B949E", fontstyle="italic")
            ax.set_xticks(positions)
            ax.set_xticklabels(self.intersections, fontsize=11,
                               fontweight="bold", color="#E6EDF3")
            ax.set_ylabel(ylabel, fontsize=10, color="#8B949E")
            ax.tick_params(colors="#8B949E")
            ax.set_title(f"  {ylabel}", fontsize=11, color="#8B949E",
                         loc="left", fontweight="bold")
            from matplotlib.lines import Line2D
            legend_handles = [Line2D([0], [0], color="#FF6B9D", lw=2, label="Mean")]
            if n_eps >= 3:
                legend_handles.insert(0, Line2D([0], [0], color="#FFD700",
                                                lw=2, label="Median"))
            ax.legend(handles=legend_handles, fontsize=9,
                      facecolor="#21262D", edgecolor="#30363D")
        fig.tight_layout()
        self._save(fig, "distribution_violins.png")

    def _plot_dashboard(self, use_steps: bool = False):
        fig = plt.figure(figsize=(20, 14), facecolor="#0D1117")
        fig.suptitle("OM Network  -  MAPPO-PIS + GAT (Multi-Agent PPO with Graph Attention Intent Sharing) Dashboard",
                     fontsize=18, color="#E6EDF3", fontweight="bold",
                     x=0.5, y=0.995, ha="center")
        gs = gridspec.GridSpec(3, 4, figure=fig,
                               hspace=0.60, wspace=0.40,
                               height_ratios=[2.2, 2, 1.2],
                               top=0.96, bottom=0.04,
                               left=0.06, right=0.97)

        ax_top = fig.add_subplot(gs[0, :])
        ax_top.set_facecolor("#161B22")
        for spine in ax_top.spines.values():
            spine.set_color("#30363D")

        if use_steps:
            ax_top.set_title(
                "Back Pressure Reward per Step (Eq. 4.92) - All Intersections",
                fontsize=13, fontweight="bold", color="#E6EDF3", loc="left")
            for tl in self.intersections:
                raw = self._step_data.get(tl, {}).get("reward", [])
                if not raw:
                    continue
                xs     = list(range(len(raw)))
                w      = max(5, len(raw) // 50)
                smooth = _smooth(raw, window=w)
                color  = _TL_COLORS[tl]
                glow   = _TL_GLOW[tl]
                ax_top.fill_between(xs, raw, alpha=0.07, color=_TL_FILL[tl])
                ax_top.plot(xs, raw, color=color, alpha=0.15, linewidth=0.6)
                _add_glow_line(ax_top, xs, smooth, color, glow, lw=2.4, alpha_glow=0.25)
                ax_top.text(xs[-1], smooth[-1],
                            f"  {tl} ({np.mean(raw):.4f})",
                            va="center", ha="left", fontsize=9,
                            color=color, fontweight="bold", clip_on=False)
            ax_top.set_xlabel("Step", fontsize=10, color="#8B949E")
        else:
            ax_top.set_title(
                "Back Pressure Reward Trend (Eq. 4.92) - All Intersections",
                fontsize=13, fontweight="bold", color="#E6EDF3", loc="left")
            eps = self._episodes
            for tl in self.intersections:
                raw    = self._history["reward"][tl]
                smooth = _smooth(raw, window=max(3, len(raw) // 20 + 1))
                color  = _TL_COLORS[tl]
                glow   = _TL_GLOW[tl]
                ax_top.fill_between(eps, raw, alpha=0.07, color=_TL_FILL[tl])
                ax_top.plot(eps, raw, color=color, alpha=0.15, linewidth=0.6)
                _add_glow_line(ax_top, eps, smooth, color, glow, lw=2.4, alpha_glow=0.25)
                ax_top.text(eps[-1], smooth[-1],
                            f"  {tl} ({raw[-1]:.4f})",
                            va="center", ha="left", fontsize=9,
                            color=color, fontweight="bold", clip_on=False)
            ax_top.set_xlabel("Episode", fontsize=10, color="#8B949E")
            ax_top.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax_top.set_ylabel("Reward", fontsize=10, color="#8B949E")
        ax_top.tick_params(colors="#8B949E")

        for col, tl in enumerate(self.intersections):
            ax = fig.add_subplot(gs[1, col])
            ax.set_facecolor("#161B22")
            for spine in ax.spines.values():
                spine.set_color("#30363D")

            color = _TL_COLORS[tl]
            glow  = _TL_GLOW[tl]

            if use_steps:
                raw = self._step_data.get(tl, {}).get("reward", [])
                if raw:
                    xs     = list(range(len(raw)))
                    w      = max(5, len(raw) // 50)
                    smooth = _smooth(raw, window=w)
                    ax.fill_between(xs, raw, alpha=0.15, color=_TL_FILL[tl], zorder=1)
                    ax.plot(xs, raw, color=color, alpha=0.20, linewidth=0.8, zorder=2)
                    _add_glow_line(ax, xs, smooth, color, glow, lw=2.0, alpha_glow=0.2)
                q_data  = self._step_data.get(tl, {}).get("queue", [])
                x_label = "Step"
            else:
                eps    = self._episodes
                raw    = self._history["reward"][tl]
                smooth = _smooth(raw, window=max(3, len(raw) // 20 + 1))
                ax.fill_between(eps, raw, alpha=0.15, color=_TL_FILL[tl], zorder=1)
                ax.plot(eps, raw, color=color, alpha=0.20, linewidth=0.8, zorder=2)
                _add_glow_line(ax, eps, smooth, color, glow, lw=2.0, alpha_glow=0.2)
                q_data  = self._history["queue"][tl]
                x_label = "Episode"

            if q_data:
                ax2 = ax.twinx()
                ax2.set_facecolor("#161B22")
                q_xs = list(range(len(q_data))) if use_steps else self._episodes
                ax2.plot(q_xs, q_data, color="#D29922", linewidth=1.2,
                         linestyle="--", alpha=0.70, label="Queue")
                ax2.set_ylabel("Queue", color="#D29922", fontsize=7.5, fontweight="bold")
                ax2.tick_params(axis="y", labelcolor="#D29922", labelsize=7.5, colors="#D29922")
                ax2.spines["right"].set_color("#D29922")

            ax.set_title(f" {tl}", fontweight="bold", fontsize=11,
                         color=color, loc="left", pad=6)
            ax.set_xlabel(x_label, fontsize=8, color="#8B949E")
            ax.set_ylabel("Reward", fontsize=8, color="#8B949E", fontweight="bold")
            ax.tick_params(labelsize=7.5, colors="#8B949E")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        ax_tbl = fig.add_subplot(gs[2, :])
        ax_tbl.set_facecolor("#0D1117")
        ax_tbl.axis("off")

        col_labels = ["Intersection", "Min", "Max", "Mean",
                      "Std Dev", "Median", "IQR", "Final", "Improvement",
                      "Skewness", "Best Ep", "Worst Ep"]
        table_data = []
        for tl in self.intersections:
            raw = self._history["reward"][tl]
            if raw:
                st = _stats(raw)
                table_data.append([
                    tl,
                    f"{st['min']:.4f}",  f"{st['max']:.4f}",
                    f"{st['mean']:.4f}", f"{st['std']:.4f}",
                    f"{st['median']:.4f}", f"{st['iqr']:.4f}",
                    f"{st['final']:.4f}",
                    ("+" if st["improvement"] >= 0 else "") + f"{st['improvement']:.4f}",
                    f"{st['skewness']:.3f}",
                    str(st["best_ep"]), str(st["worst_ep"]),
                ])

        if table_data:
            tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                               loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            tbl.scale(1.0, 1.9)
            tbl.auto_set_column_width(list(range(len(col_labels))))
            for j in range(len(col_labels)):
                cell = tbl[0, j]
                cell.set_facecolor("#1F6FEB")
                cell.set_text_props(color="white", fontweight="bold", fontsize=9)
                cell.set_edgecolor("#0D1117")
            for i in range(1, len(table_data) + 1):
                tl = self.intersections[i - 1]
                for j in range(len(col_labels)):
                    cell = tbl[i, j]
                    cell.set_facecolor("#161B22" if i % 2 == 0 else "#21262D")
                    cell.set_edgecolor("#30363D")
                    if j == 0:
                        cell.set_text_props(
                            color=_TL_COLORS.get(tl, "#58A6FF"),
                            fontweight="bold", fontsize=9)
                    elif j == 8:
                        try:
                            fv = float(table_data[i-1][j])
                            cell.set_text_props(
                                color="#3FB950" if fv >= 0 else "#F78166",
                                fontweight="bold", fontsize=9)
                        except Exception:
                            cell.set_text_props(color="#E6EDF3", fontsize=9)
                    else:
                        cell.set_text_props(color="#E6EDF3", fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.99])
        self._save(fig, "dashboard.png")

    def export_excel(self, filename="training_results.xlsx"):
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
            from openpyxl.chart import LineChart, Reference
            from openpyxl.formatting.rule import ColorScaleRule
        except ImportError:
            print("[Excel] openpyxl not found - run:  pip install openpyxl")
            return

        wb  = openpyxl.Workbook()
        eps = self._episodes

        C_BG_MID     = "161B22"
        C_BG_LIGHT   = "21262D"
        C_BORDER     = "30363D"
        C_TEXT_MAIN  = "E6EDF3"
        C_TEXT_MUTED = "8B949E"
        C_HEADER_BG  = "1565C0"
        C_HEADER_FG  = "FFFFFF"
        C_TL   = {"TL1": "58A6FF", "TL2": "3FB950",
                  "TL3": "F78166", "TL4": "D2A8FF"}
        C_ALT1 = "161B22"
        C_ALT2 = "21262D"
        C_POS  = "3FB950"
        C_NEG  = "F78166"

        thin   = Side(style="thin", color=C_BORDER)
        b_thin = Border(left=thin, right=thin, top=thin, bottom=thin)

        def fill(c):
            return PatternFill("solid", fgColor=c)
        def font(bold=False, color=C_TEXT_MAIN, size=10, italic=False):
            return Font(bold=bold, color=color, size=size, name="Calibri", italic=italic)
        def align(h="center", wrap=False):
            return Alignment(horizontal=h, vertical="center", wrap_text=wrap)
        def write_header(ws, row, cols, labels, bg=C_HEADER_BG, fg=C_HEADER_FG, sz=10):
            for c, lbl in zip(cols, labels):
                cell = ws.cell(row=row, column=c, value=lbl)
                cell.fill = fill(bg)
                cell.font = font(bold=True, color=fg, size=sz)
                cell.alignment = align("center")
                cell.border = b_thin
        def data_cell(cell, value=None, alt=False, h="center", accent=None, bold=False):
            cell.fill = fill(C_ALT2 if alt else C_ALT1)
            cell.font = font(bold=bold or (accent is not None),
                             color=accent or C_TEXT_MAIN)
            cell.alignment = align(h)
            cell.border = b_thin
            if value is not None:
                cell.value = value
        def col_w(ws, col, w):
            ws.column_dimensions[get_column_letter(col)].width = w
        def row_h(ws, row, h):
            ws.row_dimensions[row].height = h
        def sheet_banner(ws, title, subtitle="", row1=1):
            ws.merge_cells(f"A{row1}:R{row1}")
            c = ws[f"A{row1}"]
            c.value     = title
            c.font      = font(bold=True, color=C_TEXT_MAIN, size=15)
            c.alignment = align("left")
            c.fill      = fill(C_BG_MID)
            row_h(ws, row1, 30)
            if subtitle:
                ws.merge_cells(f"A{row1+1}:R{row1+1}")
                c2           = ws[f"A{row1+1}"]
                c2.value     = subtitle
                c2.font      = font(color=C_TEXT_MUTED, size=10, italic=True)
                c2.fill      = fill(C_BG_MID)
                c2.alignment = align("left")
                row_h(ws, row1+1, 18)
        def color_scale(ws, min_row, max_row, col,
                        low="F78166", mid="D29922", high="3FB950"):
            cl   = get_column_letter(col)
            rule = ColorScaleRule(
                start_type="min",      start_color=low,
                mid_type="percentile", mid_value=50, mid_color=mid,
                end_type="max",        end_color=high)
            ws.conditional_formatting.add(f"{cl}{min_row}:{cl}{max_row}", rule)

        ws0 = wb.active
        ws0.title = "Overview"
        ws0.sheet_view.showGridLines = False
        ws0.merge_cells("B2:L2")
        c = ws0["B2"]
        c.value     = "OM Network - MAPPO-PIS + GAT (Multi-Agent PPO with Graph Attention Intent Sharing)"
        c.font      = font(bold=True, color=C_TEXT_MAIN, size=20)
        c.fill      = fill(C_BG_MID)
        c.alignment = align("center")
        row_h(ws0, 2, 40)
        ws0.merge_cells("B3:L3")
        c3           = ws0["B3"]
        c3.value     = "PPO with GAE + Back Pressure Reward (Eq. 4.92) + Dynamic GAT Intent Sharing"
        c3.font      = font(color=C_TEXT_MUTED, size=11, italic=True)
        c3.fill      = fill(C_BG_MID)
        c3.alignment = align("center")
        row_h(ws0, 3, 22)

        info_rows = [
            ("Generated",      time.asctime()),
            ("Intersections",  ", ".join(self.intersections)),
            ("Total Episodes", str(len(eps))),
            ("Episode Range",  f"{eps[0]} to {eps[-1]}" if eps else "N/A"),
            ("Reward",         "Back Pressure Eq. 4.92"),
            ("Network",        "Greenshields Eq. 4.15 (v_free=13.89 m/s, k_jam=0.2 veh/m)"),
        ]
        for i, (k, v) in enumerate(info_rows):
            r     = 5 + i
            c_key = ws0.cell(row=r, column=2, value=k)
            c_key.font = font(bold=True, color=C_TEXT_MUTED, size=10)
            c_key.fill = fill(C_BG_LIGHT)
            c_key.alignment = align("left")
            c_key.border = b_thin
            col_w(ws0, 2, 22)
            ws0.merge_cells(start_row=r, start_column=3, end_row=r, end_column=9)
            c_val = ws0.cell(row=r, column=3, value=v)
            c_val.font = font(color=C_TEXT_MAIN, size=10)
            c_val.fill = fill(C_BG_MID)
            c_val.alignment = align("left")
            c_val.border = b_thin
            row_h(ws0, r, 20)

        leg_row = 12
        ws0.cell(row=leg_row, column=2, value="Intersection").font = \
            font(bold=True, color=C_TEXT_MUTED, size=10)
        row_h(ws0, leg_row, 20)
        for i, tl in enumerate(self.intersections):
            r        = leg_row + 1 + i
            c_swatch = ws0.cell(row=r, column=2, value=tl)
            c_swatch.fill      = fill(C_TL.get(tl, "58A6FF"))
            c_swatch.font      = font(bold=True, color="000000", size=10)
            c_swatch.alignment = align("center")
            c_swatch.border    = b_thin
            st      = (_stats(self._history["reward"][tl])
                       if self._history["reward"][tl] else {})
            summary = (f"Mean: {st.get('mean', 0):.4f}  "
                       f"Final: {st.get('final', 0):.4f}  "
                       f"Improvement: {st.get('improvement', 0):+.4f}") if st else ""
            ws0.merge_cells(start_row=r, start_column=3, end_row=r, end_column=10)
            c_sum = ws0.cell(row=r, column=3, value=summary)
            c_sum.font      = font(color=C_TEXT_MAIN, size=10)
            c_sum.fill      = fill(C_ALT1 if i % 2 == 0 else C_ALT2)
            c_sum.alignment = align("left")
            c_sum.border    = b_thin
            row_h(ws0, r, 20)
        for ci in range(1, 15):
            col_w(ws0, ci, 18 if ci > 1 else 3)

        ws1 = wb.create_sheet("Raw Data")
        ws1.sheet_view.showGridLines = False
        sheet_banner(ws1, "Episode-by-Episode Raw Metrics",
                     "Back Pressure Reward, Queue, Wait, Loss")
        ws1.freeze_panes = "B5"
        metric_cols = []
        for tl in self.intersections:
            metric_cols += [f"{tl} BP-Reward", f"{tl} Queue (veh)",
                            f"{tl} Wait (s)",  f"{tl} Loss"]
        headers = ["Episode"] + metric_cols
        write_header(ws1, 4, range(1, len(headers) + 1), headers, sz=9)
        col_w(ws1, 1, 12)
        for ci in range(2, len(headers) + 1):
            col_w(ws1, ci, 18)
        row_h(ws1, 4, 24)
        for ri, ep in enumerate(eps):
            row_idx = ri + 5
            alt     = (ri % 2 == 1)
            data_cell(ws1.cell(row=row_idx, column=1, value=ep), alt=alt)
            ci = 2
            for tl in self.intersections:
                for metric in ("reward", "queue", "wait_time", "loss"):
                    vals = self._history[metric][tl]
                    val  = round(vals[ri], 6) if ri < len(vals) else ""
                    data_cell(ws1.cell(row=row_idx, column=ci, value=val),
                              alt=alt, h="right")
                    ci += 1
        if eps:
            for i in range(len(self.intersections)):
                color_scale(ws1, 5, 4 + len(eps), 2 + i * 4)
        chart1 = LineChart()
        chart1.title        = "Back Pressure Reward per Episode"
        chart1.style        = 2
        chart1.y_axis.title = "Reward"
        chart1.x_axis.title = "Episode"
        chart1.width        = 20
        chart1.height       = 13
        for i in range(len(self.intersections)):
            col_idx = 2 + i * 4
            dr = Reference(ws1, min_col=col_idx,
                           min_row=4, max_row=4 + len(eps))
            chart1.add_data(dr, titles_from_data=True)
        ep_ref = Reference(ws1, min_col=1, min_row=5, max_row=4 + len(eps))
        chart1.set_categories(ep_ref)
        ws1.add_chart(chart1, f"A{len(eps) + 9}")

        ws2 = wb.create_sheet("Summary Stats")
        ws2.sheet_view.showGridLines = False
        sheet_banner(ws2, "Summary Statistics per Intersection")
        ws2.freeze_panes = "C5"
        s_cols = ["Intersection", "Metric",
                  "Min", "Max", "Mean", "Std Dev",
                  "Median", "Q25", "Q75", "IQR",
                  "Skewness", "% Positive",
                  "First Ep", "Final Ep", "Improvement",
                  "Best Ep", "Worst Ep"]
        write_header(ws2, 4, range(1, len(s_cols) + 1), s_cols, sz=9)
        widths = [16, 22, 12, 12, 12, 12, 12, 12, 12, 10, 10, 12, 12, 12, 14, 12, 12]
        for ci, cw in enumerate(widths, 1):
            col_w(ws2, ci, cw)
        row_h(ws2, 4, 24)
        metric_labels = {
            "reward":    "Back Pressure Reward (Eq. 4.92)",
            "queue":     "Queue Length (veh)",
            "wait_time": "Waiting Time (s)",
            "loss":      "PPO Loss",
        }
        row_idx = 5
        for tl in self.intersections:
            accent_c = C_TL.get(tl, "58A6FF")
            for mi, (metric, mlabel) in enumerate(metric_labels.items()):
                vals = self._history[metric][tl]
                alt  = (row_idx % 2 == 0)
                if vals:
                    st       = _stats(vals)
                    row_vals = [
                        tl if mi == 0 else "", mlabel,
                        round(st["min"],    4), round(st["max"],    4),
                        round(st["mean"],   4), round(st["std"],    4),
                        round(st["median"], 4), round(st["q25"],    4),
                        round(st["q75"],    4), round(st["iqr"],    4),
                        round(st["skewness"],    3),
                        round(st["pct_positive"], 1),
                        round(st["first"],  4), round(st["final"],  4),
                        round(st["improvement"], 4),
                        st["best_ep"], st["worst_ep"],
                    ]
                    for ci, val in enumerate(row_vals, 1):
                        cell = ws2.cell(row=row_idx, column=ci, value=val)
                        if ci == 1:
                            data_cell(cell, alt=alt, accent=accent_c, bold=True)
                        elif ci == 15:
                            data_cell(cell, alt=alt, h="right")
                            try:
                                fv = float(val)
                                cell.font = font(bold=True,
                                                 color=C_POS if fv >= 0 else C_NEG)
                            except Exception:
                                pass
                        else:
                            data_cell(cell, alt=alt, h="right")
                    row_idx += 1
            row_idx += 1

        ws3 = wb.create_sheet("Network Summary")
        ws3.sheet_view.showGridLines = False
        sheet_banner(ws3, "Network-Level Summary",
                     "Aggregated metrics across all intersections per episode")
        ws3.freeze_panes = "B5"
        ns_cols = ["Episode", "Total Reward", "Mean Reward",
                   "Total Queue", "Mean Queue",
                   "Total Wait",  "Mean Wait",
                   "Best TL",     "Worst TL"]
        write_header(ws3, 4, range(1, len(ns_cols) + 1), ns_cols, sz=9)
        for ci, cw in enumerate([12, 14, 14, 12, 12, 12, 12, 14, 14], 1):
            col_w(ws3, ci, cw)
        row_h(ws3, 4, 24)
        for ri, ep in enumerate(eps):
            row_idx = ri + 5
            alt     = (ri % 2 == 1)
            data_cell(ws3.cell(row=row_idx, column=1, value=ep), alt=alt)
            r_vals = [self._history["reward"][tl][ri]
                      if ri < len(self._history["reward"][tl]) else 0
                      for tl in self.intersections]
            q_vals = [self._history["queue"][tl][ri]
                      if ri < len(self._history["queue"][tl]) else 0
                      for tl in self.intersections]
            w_vals = [self._history["wait_time"][tl][ri]
                      if ri < len(self._history["wait_time"][tl]) else 0
                      for tl in self.intersections]
            summary_vals = [
                round(sum(r_vals), 4),
                round(float(np.mean(r_vals)), 4),
                round(sum(q_vals), 2),
                round(float(np.mean(q_vals)), 2),
                round(sum(w_vals), 2),
                round(float(np.mean(w_vals)), 2),
                self.intersections[int(np.argmax(r_vals))],
                self.intersections[int(np.argmin(r_vals))],
            ]
            for ci, val in enumerate(summary_vals, 2):
                c = ws3.cell(row=row_idx, column=ci, value=val)
                if ci in (8, 9):
                    data_cell(c, alt=alt, h="center", bold=True,
                              accent=C_TL.get(val, C_TEXT_MAIN))
                else:
                    data_cell(c, alt=alt, h="right")
        if eps:
            color_scale(ws3, 5, 4 + len(eps), 2)

        # -- Sheet: Episode Duration (ms) --------------------------------------
        if self._ep_duration_ms and any(v > 0 for v in self._ep_duration_ms):
            ws_ms = wb.create_sheet("Episode Timing (ms)")
            ws_ms.sheet_view.showGridLines = False
            sheet_banner(ws_ms, "Episode Duration -- Milliseconds",
                         "Wall-clock time per training episode in milliseconds")
            ws_ms.freeze_panes = "A5"
            ms_headers = ["Episode", "Duration (ms)", "Duration (s)", "Cumulative (s)"]
            write_header(ws_ms, 4, range(1, len(ms_headers) + 1), ms_headers, sz=9)
            for ci, cw in enumerate([12, 16, 14, 16], 1):
                col_w(ws_ms, ci, cw)
            row_h(ws_ms, 4, 24)
            cumulative_s = 0.0
            for ri, (ep, ms_val) in enumerate(zip(eps, self._ep_duration_ms)):
                row_idx = ri + 5
                alt     = (ri % 2 == 1)
                s_val   = round(ms_val / 1000.0, 3)
                cumulative_s += s_val
                data_cell(ws_ms.cell(row=row_idx, column=1, value=ep),            alt=alt)
                data_cell(ws_ms.cell(row=row_idx, column=2, value=round(ms_val, 1)), alt=alt, h="right")
                data_cell(ws_ms.cell(row=row_idx, column=3, value=s_val),          alt=alt, h="right")
                data_cell(ws_ms.cell(row=row_idx, column=4, value=round(cumulative_s, 3)), alt=alt, h="right")
            if eps:
                color_scale(ws_ms, 5, 4 + len(eps), 2,
                            low="3FB950", mid="D29922", high="F78166")
            ms_chart = LineChart()
            ms_chart.title        = "Episode Duration (ms) over Training"
            ms_chart.style        = 2
            ms_chart.y_axis.title = "Duration (ms)"
            ms_chart.x_axis.title = "Episode"
            ms_chart.width        = 20
            ms_chart.height       = 13
            ms_dr  = Reference(ws_ms, min_col=2, min_row=4, max_row=4 + len(eps))
            ms_chart.add_data(ms_dr, titles_from_data=True)
            ep_ref = Reference(ws_ms, min_col=1, min_row=5, max_row=4 + len(eps))
            ms_chart.set_categories(ep_ref)
            ws_ms.add_chart(ms_chart, f"A{len(eps) + 9}")

        path = os.path.join(self.save_dir, filename)
        wb.save(path)
        print(f"[TrainingPlotter] Excel workbook saved -> {path}")
        return path

    def _plot_by_global_step(self):
        metrics = [
            ("reward",    "Reward (Back Pressure)", "reward_by_step.png",    "Reward per Global Simulation Step"),
            ("queue",     "Vehicles Halting",        "queue_by_step.png",     "Queue Length per Global Simulation Step"),
            ("wait_time", "Waiting Time (s/veh)",    "wait_by_step.png",      "Mean Wait Time per Global Simulation Step"),
        ]
        for key, ylabel, fname, title in metrics:
            tl_data = self._step_history.get(key, {})
            has_data = any(tl_data.get(tl) for tl in self.intersections)
            if not has_data:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
            fig.patch.set_facecolor("#0D1117")
            fig.suptitle(title, fontsize=14, color="#E6EDF3",
                         fontweight="bold", x=0.5, y=1.01, ha="center")

            for ax, tl in zip(axes.flat, self.intersections):
                ax.set_facecolor("#161B22")
                for spine in ax.spines.values():
                    spine.set_color("#30363D")

                pairs = tl_data.get(tl, [])
                if not pairs:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", color="#8B949E", fontsize=12)
                    ax.set_title(f"  {tl}", color=_TL_COLORS[tl],
                                 fontweight="bold", fontsize=13, loc="left")
                    continue

                xs_all  = [p[0] for p in pairs]
                ys_all  = [p[1] for p in pairs]
                color   = _TL_COLORS[tl]
                fill    = _TL_FILL[tl]
                glow    = _TL_GLOW[tl]
                w       = max(10, len(ys_all) // 80)
                smooth  = _smooth(ys_all, window=w)
                mean_v  = float(np.mean(ys_all))
                st      = _stats(ys_all)

                ax.plot(xs_all, ys_all, color=color, alpha=0.12, linewidth=0.5, zorder=2)
                ax.fill_between(xs_all, ys_all, alpha=0.08, color=fill, zorder=1)
                _add_glow_line(ax, xs_all, smooth, color, glow, lw=2.2)

                if len(ys_all) > 50:
                    ax.axhspan(st["q25"], st["q75"], alpha=0.06, color=color, zorder=0)

                ax.axhline(mean_v, color="#8B949E", linestyle=":", linewidth=1.1, zorder=4)
                ax.text(xs_all[-1], mean_v, f"  μ={mean_v:.3f}",
                        va="center", ha="left", fontsize=8, color="#8B949E",
                        fontweight="bold", clip_on=False)

                total_steps = xs_all[-1] - xs_all[0] + 1
                n_eps = len(self._episodes)
                ax.set_title(f"  {tl}", fontweight="bold", fontsize=13,
                             color=color, loc="left", pad=8)
                ax.set_xlabel("Global Simulation Step", fontsize=9, color="#8B949E")
                ax.set_ylabel(ylabel, fontsize=9, color="#8B949E")
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
                ax.tick_params(colors="#8B949E")

                info = (f"steps  {total_steps:,}\n"
                        f"eps    {n_eps}\n"
                        f"min    {st['min']:.3f}\n"
                        f"max    {st['max']:.3f}\n"
                        f"std    {st['std']:.3f}")
                ax.text(0.02, 0.04, info, transform=ax.transAxes,
                        fontsize=7.5, va="bottom", ha="left", family="monospace",
                        color="#8B949E",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                                  edgecolor="#30363D", alpha=0.95))

                for ep_idx, ep in enumerate(self._episodes):
                    gs_start = ep_idx * (total_steps // max(len(self._episodes), 1))
                    if ep_idx > 0:
                        ax.axvline(gs_start, color="#30363D", linewidth=0.6,
                                   linestyle="--", alpha=0.5, zorder=0)

            fig.tight_layout(rect=[0, 0, 1, 0.98])
            self._save(fig, fname)

    def _plot_ep_duration_ms(self):
        eps = self._episodes
        ms  = self._ep_duration_ms
        if not eps or not ms or len(eps) != len(ms):
            return

        ms_arr  = np.array(ms, dtype=float)
        eps_arr = np.array(eps)
        smooth  = _smooth(list(ms_arr), window=max(3, len(ms_arr) // 10 + 1))
        mean_v  = float(np.mean(ms_arr))
        st      = _stats(list(ms_arr))

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_color("#30363D")

        color = "#58A6FF"
        fill  = "#1F6FEB"
        glow  = "#0D419D"

        ax.fill_between(eps_arr, ms_arr, alpha=0.12, color=fill, zorder=1)
        _add_glow_line(ax, list(eps_arr), smooth, color, glow, lw=2.5)
        ax.plot(eps_arr, ms_arr, color=color, alpha=0.25, linewidth=0.8, zorder=2)

        ax.axhline(mean_v, color="#8B949E", linestyle=":", linewidth=1.1, zorder=4)
        ax.text(eps_arr[-1], mean_v, f"  mean={mean_v:.0f} ms",
                va="center", ha="left", fontsize=8, color="#8B949E",
                fontweight="bold", clip_on=False)

        if len(ms_arr) > 5:
            ax.axhspan(st["q25"], st["q75"], alpha=0.06, color=color, zorder=0)

        txt = (f"min   {st['min']:.0f} ms\n"
               f"max   {st['max']:.0f} ms\n"
               f"mean  {mean_v:.0f} ms\n"
               f"std   {st['std']:.0f} ms\n"
               f"IQR   {st['iqr']:.0f} ms")
        ax.text(0.02, 0.96, txt, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", family="monospace",
                color="#8B949E",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                          edgecolor="#30363D", alpha=0.95))

        ax.set_title("  Episode Duration (ms)", fontweight="bold", fontsize=13,
                     color=color, loc="left", pad=8)
        ax.set_xlabel("Episode", fontsize=9, color="#8B949E")
        ax.set_ylabel("Duration (ms)", fontsize=9, color="#8B949E")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(colors="#8B949E")

        fig.tight_layout()
        self._save(fig, "ep_duration_ms.png")

    def _plot_by_wall_time(self):
        # Use episode-level wall times: cumulative sum of ep_duration_ms
        # per-step wall times are all identical (measured post-episode) and produce vertical lines
        if not self._ep_duration_ms or not self._episodes:
            return
        eps = self._episodes
        cum_ms  = np.cumsum([max(v, 0.0) for v in self._ep_duration_ms[:len(eps)]])
        cum_min = cum_ms / 60000.0
        if cum_min[-1] < 1e-6:
            return

        metrics = [
            ("reward",    "Reward (Back Pressure)", "reward_by_time.png",  "Reward vs Wall-Clock Time"),
            ("queue",     "Vehicles Halting",        "queue_by_time.png",   "Queue Length vs Wall-Clock Time"),
            ("wait_time", "Waiting Time (s/veh)",    "wait_by_time.png",    "Mean Wait Time vs Wall-Clock Time"),
        ]

        for key, ylabel, fname, title in metrics:
            has_data = any(self._history[key].get(tl) for tl in self.intersections)
            if not has_data:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
            fig.patch.set_facecolor("#0D1117")
            fig.suptitle(title, fontsize=14, color="#E6EDF3",
                         fontweight="bold", x=0.5, y=1.01, ha="center")

            for ax, tl in zip(axes.flat, self.intersections):
                ax.set_facecolor("#161B22")
                for spine in ax.spines.values():
                    spine.set_color("#30363D")

                ys_raw = self._history[key].get(tl, [])
                ts_raw = list(cum_min[:len(ys_raw)])

                if not ts_raw or not ys_raw:
                    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                            ha="center", va="center", color="#8B949E", fontsize=12)
                    ax.set_title(f"  {tl}", color=_TL_COLORS[tl],
                                 fontweight="bold", fontsize=13, loc="left")
                    continue

                color  = _TL_COLORS[tl]
                fill   = _TL_FILL[tl]
                glow   = _TL_GLOW[tl]
                w      = max(3, len(ys_raw) // 20 + 1)
                smooth = _smooth(ys_raw, window=w)
                mean_v = float(np.mean(ys_raw))
                st     = _stats(ys_raw)
                total_min = float(ts_raw[-1]) if ts_raw else 0.0

                ax.plot(ts_raw, ys_raw, color=color, alpha=0.12, linewidth=0.5, zorder=2)
                ax.fill_between(ts_raw, ys_raw, alpha=0.08, color=fill, zorder=1)
                _add_glow_line(ax, ts_raw, smooth, color, glow, lw=2.2)

                if len(ys_raw) > 10:
                    ax.axhspan(st["q25"], st["q75"], alpha=0.06, color=color, zorder=0)

                ax.axhline(mean_v, color="#8B949E", linestyle=":", linewidth=1.1, zorder=4)
                ax.text(ts_raw[-1], mean_v, f"  μ={mean_v:.3f}",
                        va="center", ha="left", fontsize=8, color="#8B949E",
                        fontweight="bold", clip_on=False)

                ax.set_title(f"  {tl}", fontweight="bold", fontsize=13,
                             color=color, loc="left", pad=8)
                ax.set_xlabel("Wall-Clock Time (minutes)", fontsize=9, color="#8B949E")
                ax.set_ylabel(ylabel, fontsize=9, color="#8B949E")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                ax.tick_params(colors="#8B949E")

                hrs  = int(total_min // 60)
                mins = int(total_min % 60)
                info = (f"total  {hrs}h {mins}m\n"
                        f"eps    {len(self._episodes)}\n"
                        f"min    {st['min']:.3f}\n"
                        f"max    {st['max']:.3f}\n"
                        f"std    {st['std']:.3f}")
                ax.text(0.02, 0.04, info, transform=ax.transAxes,
                        fontsize=7.5, va="bottom", ha="left", family="monospace",
                        color="#8B949E",
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262D",
                                  edgecolor="#30363D", alpha=0.95))

                for t_boundary in ts_raw[1:]:
                    ax.axvline(t_boundary, color="#30363D", linewidth=0.4,
                               linestyle="--", alpha=0.35, zorder=0)

            fig.tight_layout(rect=[0, 0, 1, 0.98])
            self._save(fig, fname)

    def _plot_intent_visualization(self):
        """Three-panel figure visualising what the GAT intent vectors encode.

        Panel row 1 — Intent Heatmap (one subplot per agent):
            X-axis : intent dimension (0 … intent_dim-1)
            Y-axis : training episode (earliest at top)
            Color  : tanh-scaled intent value in [-1, +1]
            Shows  : how the intent representation evolves during training.
                     A completely flat heatmap = dead intents (GAT collapsed).
                     Structured patterns = agents encoding meaningful traffic state.

        Panel row 2, left — Inter-Agent Cosine Similarity:
            4×4 matrix averaged over all recorded episodes.
            High off-diagonal values = agents share similar intent representations
            (strong coordination signal). Low values = each agent encodes independent
            local state (weak or no coordination).

        Panel row 2, center — GAT Alpha Evolution (local ↔ global mix):
            One line per intersection node.
            alpha → 0 : node ignores global cluster context (pure local attention).
            alpha → 1 : node relies entirely on global inter-cluster context.
            Watching this converge (or not) reveals whether the hierarchical GAT
            is actually leveraging global topology.

        Panel row 2, right — Intent L2 Norm per Agent:
            How "active" each agent's intent signal is over training.
            Norm ≈ 0 throughout = dead intent channel; consistently growing norm
            = the agent is encoding increasingly rich traffic state.
        """
        eps = self._episodes
        if not eps:
            return

        n_agents  = len(self.intersections)
        # Build intent matrix: shape (n_episodes, n_agents, intent_dim)
        # Use the minimum length across agents so the matrix is rectangular.
        n_eps_per_tl = [len(self._intent_history[tl]) for tl in self.intersections]
        if not all(n_eps_per_tl):
            return
        n_rec = min(n_eps_per_tl)
        if n_rec == 0:
            return

        # intent_mat[ep_idx, agent_idx, dim] = intent value
        intent_mat = np.array(
            [[self._intent_history[tl][ei] for tl in self.intersections]
             for ei in range(n_rec)],
            dtype=np.float32,
        )  # (n_rec, n_agents, intent_dim)

        intent_dim = intent_mat.shape[2]
        ep_labels  = eps[:n_rec]

        # ── figure layout ────────────────────────────────────────────────────
        fig = plt.figure(figsize=(20, 14), facecolor="#0D1117")
        fig.suptitle(
            "GAT Intent Representation Analysis",
            fontsize=16, color="#E6EDF3", fontweight="bold",
            x=0.5, y=0.995, ha="center",
        )
        gs = gridspec.GridSpec(
            2, 4, figure=fig,
            hspace=0.55, wspace=0.38,
            height_ratios=[1.6, 1.0],
            top=0.95, bottom=0.06,
            left=0.06, right=0.97,
        )

        # ── row 0: per-agent intent heatmaps ─────────────────────────────────
        # and ensure small signals are visible via dynamic scaling.
        cmap_intent = plt.get_cmap("RdBu_r")

        for ai, tl in enumerate(self.intersections):
            ax = fig.add_subplot(gs[0, ai])
            ax.set_facecolor("#0D1117")
            for spine in ax.spines.values():
                spine.set_color("#30363D")

            # agent_intents: (n_rec, intent_dim)
            agent_intents = intent_mat[:, ai, :]  # row=episode, col=dim
            
            # Dynamic scaling: if max value is very small, scale up so features are visible.
            v_max = float(np.max(np.abs(agent_intents)))
            v_limit = max(0.1, v_max) # at least 0.1 for visibility

            im = ax.imshow(
                agent_intents,
                aspect="auto", cmap=cmap_intent,
                vmin=-v_limit, vmax=v_limit,
                interpolation="bilinear", origin="upper",
            )

            # colorbar
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="#8B949E", labelsize=7)
            cbar.outline.set_edgecolor("#30363D")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8B949E")

            ax.set_title(f"  {tl}  Intent Heatmap",
                         fontweight="bold", fontsize=11,
                         color=_TL_COLORS[tl], loc="left", pad=6)
            ax.set_xlabel("Intent Dimension", fontsize=9, color="#8B949E")
            ax.set_ylabel("Episode", fontsize=9, color="#8B949E")

            # y-ticks: show episode numbers at a sparse interval
            n_yticks = min(8, n_rec)
            ytick_idxs = np.linspace(0, n_rec - 1, n_yticks, dtype=int)
            ax.set_yticks(ytick_idxs)
            ax.set_yticklabels([str(ep_labels[i]) for i in ytick_idxs],
                               fontsize=7, color="#8B949E")

            # x-ticks
            n_xticks = min(8, intent_dim)
            xtick_idxs = np.linspace(0, intent_dim - 1, n_xticks, dtype=int)
            ax.set_xticks(xtick_idxs)
            ax.set_xticklabels([str(i) for i in xtick_idxs],
                               fontsize=7, color="#8B949E")

            # annotation: mean norm per episode shown as right-side line
            norms = np.linalg.norm(agent_intents, axis=1)
            mean_norm = float(np.mean(norms))
            ax.text(0.98, 0.01,
                    f"mean‖intent‖={mean_norm:.3f}",
                    transform=ax.transAxes, fontsize=7.5,
                    va="bottom", ha="right", color="#8B949E",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D",
                              edgecolor="#30363D", alpha=0.9))

        # ── row 1, col 0: inter-agent cosine similarity ───────────────────────
        ax_sim = fig.add_subplot(gs[1, 0])
        ax_sim.set_facecolor("#161B22")
        for spine in ax_sim.spines.values():
            spine.set_color("#30363D")

        # Average cosine similarity matrix over all recorded episodes
        sim_sum = np.zeros((n_agents, n_agents), dtype=np.float64)
        valid_eps = 0
        for ei in range(n_rec):
            vecs = intent_mat[ei]  # (n_agents, intent_dim)
            norms_ = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms_ = np.where(norms_ < 1e-8, 1e-8, norms_)
            normed = vecs / norms_
            sim_sum += normed @ normed.T
            valid_eps += 1
        sim_mat = sim_sum / max(valid_eps, 1)

        cmap_sim = LinearSegmentedColormap.from_list(
            "sim", ["#1F6FEB", "#21262D", "#3FB950"], N=256)
        im_sim = ax_sim.imshow(sim_mat, cmap=cmap_sim,
                               vmin=-1.0, vmax=1.0, aspect="auto")
        cbar_sim = fig.colorbar(im_sim, ax=ax_sim, fraction=0.046, pad=0.04)
        cbar_sim.ax.yaxis.set_tick_params(color="#8B949E", labelsize=7)
        cbar_sim.outline.set_edgecolor("#30363D")
        plt.setp(cbar_sim.ax.yaxis.get_ticklabels(), color="#8B949E")

        for i in range(n_agents):
            for j in range(n_agents):
                val = sim_mat[i, j]
                txt_c = "#E6EDF3" if abs(val) < 0.5 else "#0D1117"
                ax_sim.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold", color=txt_c)

        ax_sim.set_xticks(range(n_agents))
        ax_sim.set_yticks(range(n_agents))
        ax_sim.set_xticklabels(self.intersections, fontsize=9,
                               color="#E6EDF3", fontweight="bold")
        ax_sim.set_yticklabels(self.intersections, fontsize=9,
                               color="#E6EDF3", fontweight="bold")
        ax_sim.set_title("  Inter-Agent Cosine Similarity\n  (avg over episodes)",
                         fontsize=10, color="#E6EDF3", fontweight="bold",
                         loc="left", pad=6)
        ax_sim.text(0.5, -0.18,
                    "High off-diagonal = strong coordination",
                    transform=ax_sim.transAxes, ha="center",
                    fontsize=8, color="#8B949E", style="italic")

        # ── row 1, col 1-2 (span 2): GAT alpha evolution ──────────────────────
        ax_alpha = fig.add_subplot(gs[1, 1:3])
        ax_alpha.set_facecolor("#161B22")
        for spine in ax_alpha.spines.values():
            spine.set_color("#30363D")

        if self._alpha_history:
            alpha_arr = np.array(self._alpha_history, dtype=np.float32)
            # alpha_arr: (n_alpha_records, n_agents)
            n_alpha = alpha_arr.shape[0]
            alpha_eps = np.linspace(ep_labels[0], ep_labels[-1], n_alpha)
            for ai, tl in enumerate(self.intersections):
                if ai < alpha_arr.shape[1]:
                    color = _TL_COLORS[tl]
                    glow  = _TL_GLOW[tl]
                    vals  = alpha_arr[:, ai].tolist()
                    sm    = _smooth(vals, window=max(3, n_alpha // 15 + 1))
                    ax_alpha.plot(alpha_eps, vals, color=color,
                                  alpha=0.20, linewidth=0.8, zorder=2)
                    _add_glow_line(ax_alpha, list(alpha_eps), sm, color, glow, lw=2.0)
                    ax_alpha.text(alpha_eps[-1], sm[-1],
                                  f"  {tl} ({sm[-1]:.3f})",
                                  va="center", ha="left", fontsize=8.5,
                                  color=color, fontweight="bold", clip_on=False)
            ax_alpha.axhline(0.5, color="#8B949E", linestyle=":",
                             linewidth=1.0, alpha=0.6, zorder=1)
            ax_alpha.text(alpha_eps[0], 0.51, "  balance (0.5)",
                          fontsize=7.5, color="#8B949E", va="bottom")
            ax_alpha.set_ylim(-0.05, 1.05)
        else:
            ax_alpha.text(0.5, 0.5, "No alpha data recorded\n(GAT not parametric?)",
                          transform=ax_alpha.transAxes, ha="center", va="center",
                          color="#8B949E", fontsize=11)

        ax_alpha.set_title("  GAT Dynamic Alpha: Context Gating Trend",
                           fontsize=10, color="#E6EDF3", fontweight="bold",
                           loc="left", pad=6)
        ax_alpha.set_xlabel("Episode", fontsize=9, color="#8B949E")
        ax_alpha.set_ylabel("Gate Value (0=local · 1=global)", fontsize=9,
                            color="#8B949E")
        ax_alpha.tick_params(colors="#8B949E")
        ax_alpha.xaxis.set_major_locator(MaxNLocator(integer=True))

        # ── row 1, col 3: intent L2 norm evolution ────────────────────────────
        ax_norm = fig.add_subplot(gs[1, 3])
        ax_norm.set_facecolor("#161B22")
        for spine in ax_norm.spines.values():
            spine.set_color("#30363D")

        for ai, tl in enumerate(self.intersections):
            agent_intents = intent_mat[:, ai, :]  # (n_rec, intent_dim)
            norms = np.linalg.norm(agent_intents, axis=1)
            color = _TL_COLORS[tl]
            glow  = _TL_GLOW[tl]
            sm    = _smooth(norms.tolist(), window=max(3, n_rec // 15 + 1))
            ax_norm.plot(ep_labels, norms, color=color,
                         alpha=0.18, linewidth=0.7, zorder=2)
            _add_glow_line(ax_norm, ep_labels, sm, color, glow, lw=2.0)
            ax_norm.text(ep_labels[-1], sm[-1],
                         f"  {tl} ({sm[-1]:.2f})",
                         va="center", ha="left", fontsize=8.5,
                         color=color, fontweight="bold", clip_on=False)

        ax_norm.set_title("  Intent L2 Norm per Agent",
                          fontsize=10, color="#E6EDF3", fontweight="bold",
                          loc="left", pad=6)
        ax_norm.set_xlabel("Episode", fontsize=9, color="#8B949E")
        ax_norm.set_ylabel("‖intent‖₂", fontsize=9, color="#8B949E")
        ax_norm.tick_params(colors="#8B949E")
        ax_norm.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax_norm.text(0.02, 0.97,
                     "norm≈0 = dead intent\ngrowing = richer encoding",
                     transform=ax_norm.transAxes, fontsize=7.5,
                     va="top", ha="left", color="#8B949E", style="italic",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D",
                               edgecolor="#30363D", alpha=0.85))

        fig.tight_layout(rect=[0, 0, 1, 0.98])
        self._save(fig, "intent_visualization.png")

    def _save(self, fig, filename):
        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  [plot] saved -> {path}")

class TBComparisonLogger:
    """
    Logs MAPPO-PIS+GAT  vs  Fixed-Time  metrics to TensorBoard side-by-side.

    Directory layout inside save_dir:
        tensorboard/
            MAPPO_PIS_GAT/   <- one SummaryWriter
            Fixed_Time/      <- one SummaryWriter
    """

    def __init__(self, save_dir: str):
        self.enabled = TB_AVAILABLE
        if not self.enabled:
            return
        tb_root = os.path.join(save_dir, "tensorboard")
        self.writer_rl    = TBWriter(log_dir=os.path.join(tb_root, "MAPPO_PIS_GAT"))
        self.writer_fixed = TBWriter(log_dir=os.path.join(tb_root, "Fixed_Time"))
        print(f"  [TensorBoard] Logs -> {tb_root}")
        print(f"  [TensorBoard] Run:  tensorboard --logdir {tb_root}")

    # -- per-episode logging ---------------------------------------------------
    def log_episode_rl(self, ep: int,
                       ep_reward: dict, ep_queue: dict,
                       ep_wait: dict,  ep_loss: dict):
        if not self.enabled:
            return
        for tl in INTERSECTIONS:
            self.writer_rl.add_scalar(f"Reward/{tl}",    ep_reward.get(tl, 0), ep)
            self.writer_rl.add_scalar(f"Queue/{tl}",     ep_queue.get(tl, 0),  ep)
            self.writer_rl.add_scalar(f"WaitTime/{tl}",  ep_wait.get(tl, 0),   ep)
            self.writer_rl.add_scalar(f"Loss/{tl}",      ep_loss.get(tl, 0),   ep)
        self.writer_rl.add_scalar("Avg/Reward",   np.mean(list(ep_reward.values())), ep)
        self.writer_rl.add_scalar("Avg/Queue",    np.mean(list(ep_queue.values())),  ep)
        self.writer_rl.add_scalar("Avg/WaitTime", np.mean(list(ep_wait.values())),   ep)
        self.writer_rl.add_scalar("Avg/Loss",     np.mean(list(ep_loss.values())),   ep)
        self.writer_rl.flush()

    def log_episode_fixed(self, ep: int, controllers: dict):
        if not self.enabled:
            return
        ep_reward, ep_queue, ep_wait = {}, {}, {}
        for tl, ctrl in controllers.items():
            ep_reward[tl] = float(np.mean(ctrl.step_rewards)) if ctrl.step_rewards else 0.0
            ep_queue[tl]  = float(np.mean(ctrl.step_queues))  if ctrl.step_queues  else 0.0
            ep_wait[tl]   = float(np.mean(ctrl.step_waits))   if ctrl.step_waits   else 0.0
            self.writer_fixed.add_scalar(f"Reward/{tl}",    ep_reward[tl], ep)
            self.writer_fixed.add_scalar(f"Queue/{tl}",     ep_queue[tl],  ep)
            self.writer_fixed.add_scalar(f"WaitTime/{tl}",  ep_wait[tl],   ep)
        self.writer_fixed.add_scalar("Avg/Reward",   np.mean(list(ep_reward.values())), ep)
        self.writer_fixed.add_scalar("Avg/Queue",    np.mean(list(ep_queue.values())),  ep)
        self.writer_fixed.add_scalar("Avg/WaitTime", np.mean(list(ep_wait.values())),   ep)
        self.writer_fixed.flush()
        return ep_reward, ep_queue, ep_wait

    # -- combined comparison scalars (same writer key for easy overlay) --------
    def log_comparison(self, ep: int,
                       rl_vals: dict, fixed_vals: dict, metric: str):
        """
        Writes RL and Fixed values under the same tag group so TensorBoard
        can overlay them on one chart.
        Example metric = 'Compare/Avg_Queue'
        """
        if not self.enabled:
            return
        self.writer_rl.add_scalar(metric,    rl_vals,    ep)
        self.writer_fixed.add_scalar(metric, fixed_vals, ep)

    # These scalars appear on separate TensorBoard charts from training metrics,
    # making it easy to see the gap between training performance (optimistic,
    # measures memorisation) and eval performance (realistic, measures generalisation).
    def log_evaluation(self, ep: int, eval_metrics: dict):
        """Write eval scalars to the RL writer under the Eval/ prefix.

        Parameters
        ----------
        ep           : training episode number at which eval was run
        eval_metrics : dict with keys avg_reward, avg_queue, avg_wait, and optional
                       overfitting_gap (training_reward - eval_reward)
        """
        if not self.enabled:
            return
        self.writer_rl.add_scalar("Eval/Avg_Reward", eval_metrics.get("avg_reward", 0.0), ep)
        self.writer_rl.add_scalar("Eval/Avg_Queue",  eval_metrics.get("avg_queue",  0.0), ep)
        self.writer_rl.add_scalar("Eval/Avg_Wait",   eval_metrics.get("avg_wait",   0.0), ep)
        # memorises training seeds instead of generalising.
        # Positive gap = training reward > eval reward = overfitting.
        if "overfitting_gap" in eval_metrics:
            self.writer_rl.add_scalar("Eval/Overfitting_Gap",
                                      eval_metrics["overfitting_gap"], ep)
        self.writer_rl.flush()

    def close(self):
        if not self.enabled:
            return
        self.writer_rl.close()
        self.writer_fixed.close()

def export_comparison_excel(
        save_dir: str,
        rl_history:    dict,
        fixed_history: dict,
        episodes:      list,
        ep_duration_ms: list = None,
        *,
        verbose: bool = True,
) -> "str | None":
    """
    Export an enhanced RL-vs-Fixed-Time comparison Excel workbook.

    Primary sheet: "Full_Comparison"
      - Per-TL columns (TL1-TL4): RL_Reward_{TL}, FT_Reward_{TL},
        RL_Queue_{TL}, FT_Queue_{TL}, RL_Wait_{TL}, FT_Wait_{TL},
        Imp_Queue_pct_{TL}, Imp_Wait_pct_{TL}
      - Network averages: Avg_RL_Reward, Avg_FT_Reward, Avg_RL_Queue,
        Avg_FT_Queue, Avg_RL_Wait, Avg_FT_Wait, Avg_Imp_Queue_pct,
        Avg_Imp_Wait_pct
      - EV section (if data exists): RL_EV_Cleared, FT_EV_Cleared,
        RL_EV_Avg_Wait, FT_EV_Avg_Wait, EV_Improvement_pct
      - Throughput (if data exists): RL_Cars_Passed, FT_Cars_Passed,
        Throughput_Improvement_pct
      - Emissions (if data exists): RL_CO2_kg, FT_CO2_kg, CO2_Saving_pct,
        RL_NOx_g, FT_NOx_g, NOx_Saving_pct, RL_Fuel_kg, FT_Fuel_kg,
        Fuel_Saving_pct
      - Summary columns: Traffic_Improvement_Score, Sustainability_Score, Best_RL_Episode,
        Best_EV_Episode

    Additional supplementary sheets: "Overview", "Emissions Summary"
    (conditional), "Episode Timing" (conditional).

    Written atomically (temp-file -> rename).  A PermissionError (Excel open)
    falls back to a timestamped filename so no data is ever lost.

    Parameters
    ----------
    save_dir       : directory that receives ``RL_vs_FixedTime_comparison.xlsx``
    rl_history     : training history dict
    fixed_history  : same structure for the Fixed-Time baseline
    episodes       : list of episode numbers recorded (x-axis)
    ep_duration_ms : optional list of wall-clock ms per episode
    verbose        : print success/skip messages when True

    Returns
    -------
    Absolute path to the saved file, or None on failure / skipped.
    """

    # -- 0. dependency check ------------------------------------------------
    try:
        import pandas as pd
        import openpyxl                                          # noqa: F401
        from openpyxl                import load_workbook
        from openpyxl.styles         import PatternFill, Font, Alignment, Border, Side
        from openpyxl.utils          import get_column_letter
        from openpyxl.chart          import LineChart, BarChart, Reference
    except ImportError as _ie:
        print(f"[Excel] Skipping -- missing dependency: {_ie}. "
              f"Install with: pip install pandas openpyxl")
        return None

    # -- 1. guard: need at least some RL data ------------------------------
    if not episodes:
        if verbose:
            print("[Excel] Skipping comparison export: no episodes recorded.")
        return None

    _has_rl = any(
        len(rl_history.get(m, {}).get(tl, [])) > 0
        for m in ("reward", "queue", "wait")
        for tl in INTERSECTIONS
    )
    if not _has_rl:
        if verbose:
            print("[Excel] Skipping comparison export: rl_history has no data.")
        return None

    # -- 2. helper functions ------------------------------------------------
    def _v(hist: dict, metric: str, tl: str, i: int,
           default: float = 0.0) -> float:
        """Return hist[metric][tl][i] safely."""
        seq = hist.get(metric, {}).get(tl, [])
        if seq and i < len(seq):
            v = seq[i]
            return float(v) if v is not None else default
        return default

    def _flat(hist: dict, key: str, i: int, default: float = 0.0) -> float:
        """Return a flat (non-per-TL) list value safely."""
        seq = hist.get(key, [])
        if seq and i < len(seq):
            v = seq[i]
            return float(v) if v is not None else default
        return default

    def _mean_tl(hist: dict, metric: str, i: int) -> float:
        vals = [_v(hist, metric, tl, i) for tl in INTERSECTIONS]
        return float(np.mean(vals)) if vals else 0.0

    def _pct_imp(rl_v: float, fx_v: float) -> float:
        """
        Improvement %: positive = RL is better (lower for queue/wait/emissions,
        higher for reward/ev_cleared/throughput).
        For queue/wait/emissions: (fx - rl) / |fx| * 100
        Caller must negate for reward/cleared/throughput if needed; here we
        compute (fx - rl) / |fx| * 100 uniformly and let callers decide sign.
        """
        denom = abs(fx_v) if abs(fx_v) > 1e-9 else 1e-9
        return ((fx_v - rl_v) / denom) * 100.0

    def _pct_reward(rl_v: float, fx_v: float) -> float:
        """For reward/throughput: (rl - fx) / |fx| * 100 (positive = RL better)."""
        denom = abs(fx_v) if abs(fx_v) > 1e-9 else 1e-9
        return ((rl_v - fx_v) / denom) * 100.0

    # -- 3. detect optional metric availability -----------------------------
    _has_ev   = bool(rl_history.get("ev_cleared"))
    _has_cars = bool(rl_history.get("cars_passed"))
    _has_em   = bool(rl_history.get("co2_total"))
    _has_dur  = bool(ep_duration_ms) and any(v > 0 for v in ep_duration_ms)

    # -- Check for existing cumulative workbook -----------------------------
    dest_path = os.path.join(save_dir, "RL_vs_FixedTime_comparison.xlsx")
    _existing_ep_set: set = set()
    _old_full_df = pd.DataFrame()
    if os.path.exists(dest_path):
        try:
            _old_full_df = pd.read_excel(dest_path, sheet_name="Full_Comparison")
            if "Episode" in _old_full_df.columns:
                _existing_ep_set = set(int(e) for e in _old_full_df["Episode"].tolist())
            # Backward compat: rename old _mg/_ml columns to new _kg/_g
            _col_migrate = {
                "RL_CO2_mg": "RL_CO2_kg", "FT_CO2_mg": "FT_CO2_kg",
                "RL_NOx_mg": "RL_NOx_g",  "FT_NOx_mg": "FT_NOx_g",
                "RL_Fuel_mg": "RL_Fuel_kg", "FT_Fuel_mg": "FT_Fuel_kg",
            }
            for old_c, new_c in _col_migrate.items():
                if old_c in _old_full_df.columns and new_c not in _old_full_df.columns:
                    if old_c.startswith(("RL_CO2", "FT_CO2", "RL_Fuel", "FT_Fuel")):
                        _old_full_df[new_c] = _old_full_df[old_c] / _MG_TO_KG
                    elif old_c.startswith(("RL_NOx", "FT_NOx")):
                        _old_full_df[new_c] = _old_full_df[old_c] / _MG_TO_G
                    _old_full_df.drop(columns=[old_c], inplace=True)
        except Exception:
            _old_full_df = pd.DataFrame()
            _existing_ep_set = set()

    # Only build rows for episodes not yet in the workbook
    _new_indices = [i for i, ep in enumerate(episodes) if ep not in _existing_ep_set]
    if not _new_indices:
        if _existing_ep_set:
            if verbose:
                print(f"[Excel] All {len(episodes)} episodes already exported — nothing new to add.")
            return dest_path
        if verbose:
            print("[Excel] Skipping: no episodes to export.")
        return None

    # -- 4. build rows only for NEW episodes --------------------------------
    rows = []
    for idx in _new_indices:
        i, ep = idx, episodes[idx]
        row: dict = {"Episode": ep}

        # -- per-TL block --------------------------------------------------
        for tl in INTERSECTIONS:
            rl_rew = _v(rl_history,    "reward", tl, i)
            ft_rew = _v(fixed_history, "reward", tl, i)
            rl_q   = _v(rl_history,    "queue",  tl, i)
            ft_q   = _v(fixed_history, "queue",  tl, i)
            rl_w   = _v(rl_history,    "wait",   tl, i)
            ft_w   = _v(fixed_history, "wait",   tl, i)

            row[f"RL_Reward_{tl}"]      = round(rl_rew, 6)
            row[f"FT_BP_Score_{tl}"]    = round(ft_rew, 6)
            row[f"Imp_Reward_pct_{tl}"] = round(_pct_reward(rl_rew, ft_rew), 2)
            row[f"RL_Queue_{tl}"]       = round(rl_q,   4)
            row[f"FT_Queue_{tl}"]       = round(ft_q,   4)
            row[f"RL_Wait_{tl}"]        = round(rl_w,   4)
            row[f"FT_Wait_{tl}"]        = round(ft_w,   4)
            row[f"Imp_Queue_pct_{tl}"]  = round(_pct_imp(rl_q, ft_q),    2)
            row[f"Imp_Wait_pct_{tl}"]   = round(_pct_imp(rl_w, ft_w),    2)

        # -- network averages ----------------------------------------------
        avg_rl_rew = _mean_tl(rl_history,    "reward", i)
        avg_ft_rew = _mean_tl(fixed_history, "reward", i)
        avg_rl_q   = _mean_tl(rl_history,    "queue",  i)
        avg_ft_q   = _mean_tl(fixed_history, "queue",  i)
        avg_rl_w   = _mean_tl(rl_history,    "wait",   i)
        avg_ft_w   = _mean_tl(fixed_history, "wait",   i)

        row["Avg_RL_Reward"]     = round(avg_rl_rew, 6)
        row["Avg_FT_Reward"]     = round(avg_ft_rew, 6)
        row["Avg_RL_Queue"]      = round(avg_rl_q,   4)
        row["Avg_FT_Queue"]      = round(avg_ft_q,   4)
        row["Avg_RL_Wait"]       = round(avg_rl_w,   4)
        row["Avg_FT_Wait"]       = round(avg_ft_w,   4)
        avg_q_imp  = round(_pct_imp(avg_rl_q, avg_ft_q),   2)
        avg_w_imp  = round(_pct_imp(avg_rl_w, avg_ft_w),   2)
        row["Avg_Imp_Queue_pct"] = avg_q_imp
        row["Avg_Imp_Wait_pct"]  = avg_w_imp
        avg_rew_imp = round(_pct_reward(avg_rl_rew, avg_ft_rew), 2)
        row["Avg_Imp_Reward_pct"] = avg_rew_imp

        # -- EV section ----------------------------------------------------
        # We propagate None to Excel and skip improvement when either side has no EV data.
        ev_imp_pct = None
        if _has_ev:
            rl_ev_c = int(_flat(rl_history,    "ev_cleared",  i))
            ft_ev_c = int(_flat(fixed_history, "ev_cleared",  i))
            # Retrieve raw values preserving None
            _rl_ev_seq = rl_history.get("ev_avg_wait", [])
            _ft_ev_seq = fixed_history.get("ev_avg_wait", [])
            rl_ev_w_raw = _rl_ev_seq[i] if i < len(_rl_ev_seq) else None
            ft_ev_w_raw = _ft_ev_seq[i] if i < len(_ft_ev_seq) else None
            row["RL_EV_Cleared"]  = rl_ev_c
            row["FT_EV_Cleared"]  = ft_ev_c
            row["RL_EV_Avg_Wait"] = round(rl_ev_w_raw, 3) if rl_ev_w_raw is not None else None
            row["FT_EV_Avg_Wait"] = round(ft_ev_w_raw, 3) if ft_ev_w_raw is not None else None
            # Only compute improvement when both sides actually had EVs
            if rl_ev_w_raw is not None and ft_ev_w_raw is not None:
                ev_imp_pct = round(_pct_imp(rl_ev_w_raw, ft_ev_w_raw), 2)
            row["EV_Improvement_pct"] = ev_imp_pct

        # -- throughput ----------------------------------------------------
        thr_imp_pct = None
        if _has_cars:
            rl_cars  = int(_flat(rl_history,    "cars_passed", i))
            ft_cars  = int(_flat(fixed_history, "cars_passed", i))
            thr_imp_pct = round(_pct_reward(float(rl_cars), float(ft_cars)), 2)
            row["RL_Cars_Passed"]           = rl_cars
            row["FT_Cars_Passed"]           = ft_cars
            row["Throughput_Improvement_pct"] = thr_imp_pct

        # -- emissions -----------------------------------------------------
        co2_imp_pct = nox_imp_pct = fuel_imp_pct = None
        if _has_em:
            rl_co2  = round(_flat(rl_history,    "co2_total",  i), 2)
            ft_co2  = round(_flat(fixed_history, "co2_total",  i), 2)
            rl_nox  = round(_flat(rl_history,    "nox_total",  i), 2)
            ft_nox  = round(_flat(fixed_history, "nox_total",  i), 2)
            rl_fuel = round(_flat(rl_history,    "fuel_total", i), 2)
            ft_fuel = round(_flat(fixed_history, "fuel_total", i), 2)
            co2_imp_pct  = round(_pct_imp(rl_co2,  ft_co2),  2)
            nox_imp_pct  = round(_pct_imp(rl_nox,  ft_nox),  2)
            fuel_imp_pct = round(_pct_imp(rl_fuel, ft_fuel), 2)
            row["RL_CO2_kg"]      = round(rl_co2  / _MG_TO_KG, 4)
            row["FT_CO2_kg"]      = round(ft_co2  / _MG_TO_KG, 4)
            row["CO2_Saving_pct"] = co2_imp_pct
            row["RL_NOx_g"]       = round(rl_nox  / _MG_TO_G,  4)
            row["FT_NOx_g"]       = round(ft_nox  / _MG_TO_G,  4)
            row["NOx_Saving_pct"] = nox_imp_pct
            row["RL_Fuel_kg"]     = round(rl_fuel / _MG_TO_KG, 4)
            row["FT_Fuel_kg"]     = round(ft_fuel / _MG_TO_KG, 4)
            row["Fuel_Saving_pct"] = fuel_imp_pct

        # -- summary columns -----------------------------------------------
        # composite scores that use meaningful weights and avoid mixing incompatible units.
        #
        # Traffic efficiency score: queue 40%, wait 40%, throughput 20%
        _thr_part = thr_imp_pct if thr_imp_pct is not None else 0.0
        traffic_score = 0.4 * avg_q_imp + 0.4 * avg_w_imp + 0.2 * _thr_part
        row["Traffic_Improvement_Score"] = round(traffic_score, 3)

        # Sustainability score: CO2, NOx, Fuel -- equal weight (only when all available)
        if co2_imp_pct is not None and nox_imp_pct is not None and fuel_imp_pct is not None:
            sustain_score = (co2_imp_pct + nox_imp_pct + fuel_imp_pct) / 3.0
            row["Sustainability_Score"] = round(sustain_score, 3)
        else:
            row["Sustainability_Score"] = None

        ft_list = fixed_history.get("ft_baseline_ep", [])
        row["FT_Baseline_Episode"] = ft_list[i] if i < len(ft_list) else 0

        # Best_RL/Best_EV markers — recomputed below across ALL episodes
        row["Best_RL_Episode"] = False
        row["Best_EV_Episode"] = False

        # Duration_ms needed for Episode Timing sheet
        dur_val = ep_duration_ms[i] if ep_duration_ms and i < len(ep_duration_ms) else 0.0
        row["Duration_ms"] = round(float(dur_val), 1)

        rows.append(row)

    new_df = pd.DataFrame(rows)
    if new_df.empty and _old_full_df.empty:
        if verbose:
            print("[Excel] Skipping: empty DataFrame after building rows.")
        return None

    # -- Combine old + new --------------------------------------------------
    df_full = pd.concat([_old_full_df, new_df], ignore_index=True)
    df_full = df_full.sort_values("Episode").reset_index(drop=True)
    n = len(df_full)
    episodes_all = df_full["Episode"].tolist()

    # Recompute best-episode markers across ALL episodes
    if "Avg_RL_Reward" in df_full.columns:
        best_rl_idx = int(np.argmax(df_full["Avg_RL_Reward"].values))
        best_rl_ep_num = df_full.iloc[best_rl_idx]["Episode"]
    else:
        best_rl_ep_num = 0
    if _has_ev and "RL_EV_Avg_Wait" in df_full.columns:
        ev_waits_raw = df_full["RL_EV_Avg_Wait"].values
        if ev_waits_raw is None or len(ev_waits_raw) == 0:
            best_ev_ep_num = 0
        else:
            ev_waits_all = np.array([v if v is not None else np.nan for v in ev_waits_raw], dtype=float)
            nonzero_mask = (ev_waits_all > 0)
            if nonzero_mask.any():
                ev_waits_masked = np.where(nonzero_mask, ev_waits_all, np.inf)
                best_ev_idx = int(np.argmin(ev_waits_masked))
                best_ev_ep_num = df_full.iloc[best_ev_idx]["Episode"]
            else:
                best_ev_ep_num = 0
    else:
        best_ev_ep_num = 0

    df_full["Best_RL_Episode"] = df_full["Episode"] == best_rl_ep_num
    df_full["Best_EV_Episode"] = (_has_ev) & (df_full["Episode"] == best_ev_ep_num)

    # -- Console summary ----------------------------------------------------
    n_old = len(_old_full_df) if not _old_full_df.empty else 0
    n_new = len(new_df) if not new_df.empty else 0
    if _existing_ep_set:
        print(f"Existing workbook found.")
        print(f"Loaded {n_old} previous episodes.")
        print(f"Appended {n_new} new episodes.")
        print(f"Final workbook contains {n} episodes.")
    elif verbose:
        print(f"  [Excel] ({n} episodes)")

    # -- 5. atomic write setup ----------------------------------------------
    dest_path = os.path.join(save_dir, "RL_vs_FixedTime_comparison.xlsx")
    import tempfile
    import shutil as _shutil
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xlsx", dir=save_dir)
        os.close(tmp_fd)
    except OSError as e:
        print(f"[Excel] Cannot create temp file in {save_dir}: {e}")
        return None

    try:
        # -- 5a. write all sheets with pandas ExcelWriter ------------------
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:

            # Sheet 1: Full_Comparison -- primary data sheet
            df_full.to_excel(writer, sheet_name="Full_Comparison", index=False)

            # Sheet 2: Overview -- per-TL summary statistics (from df_full)
            overview_rows = []
            for tl in INTERSECTIONS:
                rl_q_all = df_full[f"RL_Queue_{tl}"].tolist()
                fx_q_all = df_full[f"FT_Queue_{tl}"].tolist()
                rl_w_all = df_full[f"RL_Wait_{tl}"].tolist()
                fx_w_all = df_full[f"FT_Wait_{tl}"].tolist()
                rl_r_all = df_full[f"RL_Reward_{tl}"].tolist()
                fx_r_all = df_full[f"FT_BP_Score_{tl}"].tolist()
                overview_rows.append({
                    "Intersection":    tl,
                    "Mean_RL_Reward":  round(float(np.mean(rl_r_all)), 4),
                    "Mean_FT_BP_Score": round(float(np.mean(fx_r_all)), 4),
                    "Mean_RL_Queue":   round(float(np.mean(rl_q_all)), 3),
                    "Mean_FT_Queue":   round(float(np.mean(fx_q_all)), 3),
                    "Queue_Imp_pct":   round(_pct_imp(float(np.mean(rl_q_all)),
                                                       float(np.mean(fx_q_all))), 2),
                    "Mean_RL_Wait":    round(float(np.mean(rl_w_all)), 3),
                    "Mean_FT_Wait":    round(float(np.mean(fx_w_all)), 3),
                    "Wait_Imp_pct":    round(_pct_imp(float(np.mean(rl_w_all)),
                                                       float(np.mean(fx_w_all))), 2),
                })
            net_cols = ["Avg_RL_Reward", "Avg_FT_Reward", "Avg_RL_Queue",
                        "Avg_FT_Queue", "Avg_RL_Wait", "Avg_FT_Wait"]
            if all(c in df_full.columns for c in net_cols):
                net_rl_r = float(np.mean(df_full["Avg_RL_Reward"]))
                net_fx_r = float(np.mean(df_full["Avg_FT_Reward"]))
                net_rl_q = float(np.mean(df_full["Avg_RL_Queue"]))
                net_fx_q = float(np.mean(df_full["Avg_FT_Queue"]))
                net_rl_w = float(np.mean(df_full["Avg_RL_Wait"]))
                net_fx_w = float(np.mean(df_full["Avg_FT_Wait"]))
            else:
                net_rl_q = net_fx_q = net_rl_w = net_fx_w = net_rl_r = net_fx_r = 0.0
            overview_rows.append({
                "Intersection":    "NETWORK AVG",
                "Mean_RL_Reward":  round(net_rl_r, 4),
                "Mean_FT_BP_Score": round(net_fx_r, 4),
                "Mean_RL_Queue":   round(net_rl_q, 3),
                "Mean_FT_Queue":   round(net_fx_q, 3),
                "Queue_Imp_pct":   round(_pct_imp(net_rl_q, net_fx_q), 2),
                "Mean_RL_Wait":    round(net_rl_w, 3),
                "Mean_FT_Wait":    round(net_fx_w, 3),
                "Wait_Imp_pct":    round(_pct_imp(net_rl_w, net_fx_w), 2),
            })
            pd.DataFrame(overview_rows).to_excel(
                writer, sheet_name="Overview", index=False)

            # Sheet 3: Emissions Summary (from df_full)
            if _has_em:
                em_rows = []
                for rl_col, ft_col, label, unit in [
                    ("RL_CO2_kg",  "FT_CO2_kg",  "CO₂",  "kg"),
                    ("RL_NOx_g",   "FT_NOx_g",   "NOx",  "g"),
                    ("RL_Fuel_kg", "FT_Fuel_kg", "Fuel", "kg"),
                ]:
                    if rl_col not in df_full.columns:
                        continue
                    rl_vals = df_full[rl_col].tolist()
                    fx_vals = df_full[ft_col].tolist()
                    mean_rl = float(np.mean(rl_vals)) if rl_vals else 0.0
                    mean_fx = float(np.mean(fx_vals)) if fx_vals else 0.0
                    em_row = {
                        "Metric":           f"{label} ({unit})",
                        "RL_Mean_per_Ep":   round(mean_rl, 4),
                        "FT_Mean_per_Ep":   round(mean_fx, 4),
                        "Mean_Saving_pct":  round(_pct_imp(mean_rl, mean_fx), 2),
                        "RL_Total":         round(float(np.sum(rl_vals)), 4),
                        "FT_Total":         round(float(np.sum(fx_vals)), 4),
                        "Total_Saved":      round(float(np.sum(fx_vals)) -
                                                  float(np.sum(rl_vals)), 4),
                        "Total_Saving_pct": round(_pct_imp(float(np.sum(rl_vals)),
                                                             float(np.sum(fx_vals))), 2),
                    }
                    if _has_cars and "RL_Cars_Passed" in df_full.columns:
                        cars_rl = df_full["RL_Cars_Passed"].tolist()
                        cars_fx = df_full["FT_Cars_Passed"].tolist()
                        pv_rl = [rv / max(cv, 1) for rv, cv in zip(rl_vals, cars_rl)]
                        pv_fx = [fv / max(cv, 1) for fv, cv in zip(fx_vals, cars_fx)]
                        em_row["RL_per_Vehicle"]  = round(float(np.mean(pv_rl)), 6)
                        em_row["FT_per_Vehicle"]  = round(float(np.mean(pv_fx)), 6)
                        em_row["PerVeh_Saving_pct"] = round(_pct_imp(float(np.mean(pv_rl)),
                                                                       float(np.mean(pv_fx))), 2)
                    em_rows.append(em_row)
                pd.DataFrame(em_rows).to_excel(
                    writer, sheet_name="Emissions Summary", index=False)

            # Sheet 4: Episode Timing (from df_full)
            if _has_dur and "Duration_ms" in df_full.columns:
                dfe = df_full[["Episode", "Duration_ms"]].copy()
                dfe["Duration_s"]   = (dfe["Duration_ms"] / 1000.0).round(3)
                dfe["Cumulative_s"] = dfe["Duration_s"].cumsum().round(3)
                dfe.to_excel(writer, sheet_name="Episode Timing", index=False)

            # Sheet 5: Statistics — aggregate stats from df_full
            _stat_rows = []

            def _stats(label, rl_seq, fx_seq, is_pct=False):
                rl_arr = np.array(rl_seq, dtype=float)
                fx_arr = np.array(fx_seq, dtype=float)
                def _s(a):
                    if len(a) == 0:
                        return [0.0]*7
                    return [
                        float(np.mean(a)), float(np.median(a)), float(np.std(a)),
                        float(np.min(a)), float(np.max(a)),
                        float(np.percentile(a, 5)), float(np.percentile(a, 95)),
                    ]
                rl_s = _s(rl_arr)
                fx_s = _s(fx_arr)
                row = {
                    "Metric": label,
                    "RL_Mean":   round(rl_s[0], 4), "RL_Median": round(rl_s[1], 4),
                    "RL_Std":    round(rl_s[2], 4), "RL_Min":    round(rl_s[3], 4),
                    "RL_Max":    round(rl_s[4], 4), "RL_P5":     round(rl_s[5], 4),
                    "RL_P95":    round(rl_s[6], 4),
                    "FT_Mean":   round(fx_s[0], 4), "FT_Median": round(fx_s[1], 4),
                    "FT_Std":    round(fx_s[2], 4), "FT_Min":    round(fx_s[3], 4),
                    "FT_Max":    round(fx_s[4], 4), "FT_P5":     round(fx_s[5], 4),
                    "FT_P95":    round(fx_s[6], 4),
                }
                if not is_pct:
                    imp = _pct_imp(float(np.mean(rl_arr)) if len(rl_arr) else 0.0,
                                   float(np.mean(fx_arr)) if len(fx_arr) else 0.0)
                    row["Mean_Improvement_pct"] = round(imp, 2)
                _stat_rows.append(row)

            for tl in INTERSECTIONS:
                rl_q = df_full[f"RL_Queue_{tl}"].tolist()
                fx_q = df_full[f"FT_Queue_{tl}"].tolist()
                _stats(f"{tl} Queue", rl_q, fx_q)
                rl_w = df_full[f"RL_Wait_{tl}"].tolist()
                fx_w = df_full[f"FT_Wait_{tl}"].tolist()
                _stats(f"{tl} Wait(s)", rl_w, fx_w)
                rl_r = df_full[f"RL_Reward_{tl}"].tolist()
                fx_r = df_full[f"FT_BP_Score_{tl}"].tolist()
                _stats(f"{tl} Reward", rl_r, fx_r)

            if _has_cars and "RL_Cars_Passed" in df_full.columns:
                _stats("Cars Passed",
                       df_full["RL_Cars_Passed"].tolist(),
                       df_full["FT_Cars_Passed"].tolist())

            if _has_ev and "RL_EV_Cleared" in df_full.columns:
                _stats("EVs Cleared",
                       df_full["RL_EV_Cleared"].tolist(),
                       df_full["FT_EV_Cleared"].tolist())
                _stats("EV Avg Wait(s)",
                       df_full["RL_EV_Avg_Wait"].tolist(),
                       df_full["FT_EV_Avg_Wait"].tolist())

            if _has_em:
                for rl_col, label in [
                    ("RL_CO2_kg",  "CO2 (kg)"),
                    ("RL_NOx_g",   "NOx (g)"),
                    ("RL_Fuel_kg", "Fuel (kg)"),
                ]:
                    ft_col = rl_col.replace("RL_", "FT_")
                    if rl_col in df_full.columns:
                        _stats(label,
                               df_full[rl_col].tolist(),
                               df_full[ft_col].tolist())

            pd.DataFrame(_stat_rows).to_excel(
                writer, sheet_name="Statistics", index=False)

        # -- 5b. re-open with openpyxl to apply dark-theme formatting ------
        wb = load_workbook(tmp_path)

        # colour palette (GitHub dark-mode inspired)
        _C = {
            "hdr_rl":     "1F6FEB",   # RL columns          -- blue
            "hdr_ft":     "CF4B22",   # Fixed-Time columns  -- deep coral
            "hdr_imp":    "1565C0",   # Improvement %       -- dark blue
            "hdr_ev":     "9D4EDD",   # EV section          -- purple
            "hdr_thr":    "0D7377",   # Throughput          -- teal
            "hdr_em":     "2D6A4F",   # Emissions           -- dark green
            "hdr_meta":   "30363D",   # Episode/summary     -- grey
            "hdr_best":   "B45309",   # Best-episode cols   -- amber
            "row_alt":    "1C2128",   # alternating row bg
            "row_norm":   "161B22",   # normal row bg
            "fg":         "E6EDF3",   # main text
            "fg_pos":     "3FB950",   # positive improvement (green)
            "fg_neg":     "F85149",   # negative improvement (red)
            "fg_best":    "FFD700",   # best-episode flag
            "border":     "30363D",
        }
        _thin = Side(style="thin", color=_C["border"])
        _bdr  = Border(left=_thin, right=_thin, top=_thin, bottom=_thin)

        def _fill(hex_c):
            return PatternFill("solid", fgColor=hex_c)

        def _fnt(bold=False, color="E6EDF3", sz=10):
            return Font(bold=bold, color=color, size=sz, name="Calibri")

        def _align(h="center"):
            return Alignment(horizontal=h, vertical="center", wrap_text=False)

        def _hdr_color(col_name: str) -> str:
            """Return palette hex for a column header string."""
            name = col_name or ""
            if name in ("Best_RL_Episode", "Best_EV_Episode",
                        "Traffic_Improvement_Score", "Sustainability_Score"):
                return _C["hdr_best"]
            if any(k in name for k in ("FT_BP_Score", "FT_", "_FT", "FT_Queue",
                                        "FT_Wait", "FT_EV", "FT_Cars",
                                        "FT_CO2", "FT_NOx", "FT_Fuel",
                                        "FT_Baseline_Episode")):
                return _C["hdr_ft"]
            if any(k in name for k in ("Imp_Queue", "Imp_Wait",
                                        "Avg_Imp_Queue", "Avg_Imp_Wait",
                                        "Queue_Imp", "Wait_Imp")):
                return _C["hdr_imp"]
            if any(k in name for k in ("EV_Cleared", "EV_Avg_Wait",
                                        "EV_Improvement", "EV_Wait")):
                return _C["hdr_ev"]
            if any(k in name for k in ("Cars_Passed", "Throughput")):
                return _C["hdr_thr"]
            if any(k in name for k in ("CO2", "NOx", "Fuel", "Emission",
                                        "CO₂", "Saving_pct", "PerVeh")):
                return _C["hdr_em"]
            if any(k in name for k in ("Episode", "Duration", "Cumulative",
                                        "Intersection", "Ep_", "NETWORK")):
                return _C["hdr_meta"]
            # default: RL blue
            return _C["hdr_rl"]

        def _is_pct_col(col_name: str) -> bool:
            return any(k in (col_name or "")
                       for k in ("_pct", "Imp_", "Saving_", "Improvement",
                                  "Score", "Queue_Imp", "Wait_Imp"))

        def _is_best_col(col_name: str) -> bool:
            return (col_name or "") in ("Best_RL_Episode", "Best_EV_Episode")

        def _style_ws(ws):
            ws.sheet_view.showGridLines = False
            n_cols = ws.max_column
            n_rows = ws.max_row

            # header row styling
            ws.row_dimensions[1].height = 28
            for c in range(1, n_cols + 1):
                cell   = ws.cell(row=1, column=c)
                bg_hex = _hdr_color(str(cell.value or ""))
                cell.fill      = _fill(bg_hex)
                cell.font      = _fnt(bold=True, color="FFFFFF", sz=10)
                cell.alignment = _align("center")
                cell.border    = _bdr

            # data rows
            for r in range(2, n_rows + 1):
                bg = _C["row_alt"] if r % 2 == 0 else _C["row_norm"]
                for c in range(1, n_cols + 1):
                    cell     = ws.cell(row=r, column=c)
                    hdr_name = str(ws.cell(row=1, column=c).value or "")

                    # best-episode flag rows: highlight entire row in amber
                    if _is_best_col(hdr_name):
                        try:
                            is_best = bool(cell.value)
                        except (TypeError, ValueError):
                            is_best = False
                        if is_best:
                            # highlight this cell in amber
                            cell.fill = _fill("2D2000")
                            cell.font = _fnt(bold=True, color=_C["fg_best"], sz=10)
                        else:
                            cell.fill = _fill(bg)
                            cell.font = _fnt(bold=False, color=_C["fg"], sz=10)
                        cell.alignment = _align("center")
                        cell.border    = _bdr
                        continue

                    # percentage / improvement columns: colour by sign
                    if _is_pct_col(hdr_name):
                        try:
                            v = float(cell.value or 0)
                            fg = _C["fg_pos"] if v >= 0 else _C["fg_neg"]
                        except (TypeError, ValueError):
                            fg = _C["fg"]
                    else:
                        fg = _C["fg"]

                    halign = "center" if c <= 1 else "right"
                    cell.fill      = _fill(bg)
                    cell.font      = _fnt(bold=False, color=fg, sz=10)
                    cell.alignment = _align(halign)
                    cell.border    = _bdr

            # freeze panes after header
            ws.freeze_panes = "B2"

            # auto-width (sample up to 300 rows for speed)
            for c in range(1, n_cols + 1):
                letter  = get_column_letter(c)
                max_len = max(
                    (len(str(ws.cell(row=r, column=c).value or ""))
                     for r in range(1, min(n_rows + 1, 300))),
                    default=8,
                )
                ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 36)

        for ws in wb.worksheets:
            _style_ws(ws)

        # -- 5c. embed line charts on Full_Comparison sheet ----------------
        try:
            ws_fc   = wb["Full_Comparison"]
            headers = [ws_fc.cell(row=1, column=c).value
                       for c in range(1, ws_fc.max_column + 1)]
            last_row = ws_fc.max_row

            def _col(name):
                try:
                    return headers.index(name) + 1
                except ValueError:
                    return None

            def _add_line_chart(title, rl_col_name, ft_col_name, anchor,
                                y_label="Value", w=22, h=13):
                rl_ci = _col(rl_col_name)
                ft_ci = _col(ft_col_name)
                if rl_ci is None or ft_ci is None:
                    return
                ch = LineChart()
                ch.title = title
                ch.style = 2
                ch.y_axis.title = y_label
                ch.x_axis.title = "Episode"
                ch.width  = w
                ch.height = h
                for ci, stitle in ((rl_ci, "MAPPO-PIS+GAT"),
                                    (ft_ci, "Fixed-Time")):
                    ref = Reference(ws_fc, min_col=ci,
                                    min_row=1, max_row=last_row)
                    ch.add_data(ref, titles_from_data=True)
                ep_ref = Reference(ws_fc, min_col=_col("Episode"),
                                   min_row=2, max_row=last_row)
                ch.set_categories(ep_ref)
                ws_fc.add_chart(ch, anchor)

            chart_start = last_row + 3
            _add_line_chart("Avg Queue -- RL vs Fixed-Time",
                            "Avg_RL_Queue", "Avg_FT_Queue",
                            f"A{chart_start}", y_label="Queue (vehicles)")
            _add_line_chart("Avg Wait -- RL vs Fixed-Time",
                            "Avg_RL_Wait", "Avg_FT_Wait",
                            f"M{chart_start}", y_label="Wait (s/vehicle)")
            _add_line_chart("Avg Reward -- RL vs Fixed-Time",
                            "Avg_RL_Reward", "Avg_FT_BP_Score",
                            f"A{chart_start + 22}", y_label="Episode Reward")

            imp_ci = _col("Traffic_Improvement_Score")
            if imp_ci:
                bar = BarChart()
                bar.title = "Traffic Improvement Score per Episode (queue 40%, wait 40%, throughput 20%)"
                bar.style = 2
                bar.grouping = "clustered"
                bar.y_axis.title = "Score (%)"
                bar.x_axis.title = "Episode"
                bar.width  = 22
                bar.height = 13
                bar.add_data(Reference(ws_fc, min_col=imp_ci,
                                       min_row=1, max_row=last_row),
                             titles_from_data=True)
                bar.set_categories(Reference(ws_fc, min_col=_col("Episode"),
                                              min_row=2, max_row=last_row))
                ws_fc.add_chart(bar, f"M{chart_start + 22}")
            # sustainability score chart (if available)
            sus_ci = _col("Sustainability_Score")
            if sus_ci:
                sus_bar = BarChart()
                sus_bar.title = "Sustainability Score per Episode (CO2+NOx+Fuel equal weight)"
                sus_bar.style = 2
                sus_bar.grouping = "clustered"
                sus_bar.y_axis.title = "Score (%)"
                sus_bar.x_axis.title = "Episode"
                sus_bar.width  = 22
                sus_bar.height = 13
                sus_bar.add_data(Reference(ws_fc, min_col=sus_ci,
                                           min_row=1, max_row=last_row),
                                 titles_from_data=True)
                sus_bar.set_categories(Reference(ws_fc, min_col=_col("Episode"),
                                                  min_row=2, max_row=last_row))
                ws_fc.add_chart(sus_bar, f"M{chart_start + 37}")

            # EV wait chart if available
            if _has_ev:
                _add_line_chart("EV Avg Wait -- RL vs Fixed-Time",
                                "RL_EV_Avg_Wait", "FT_EV_Avg_Wait",
                                f"A{chart_start + 44}", y_label="EV Wait (s)")

        except Exception:
            pass   # chart errors are non-fatal; data is always correct

        wb.save(tmp_path)
        wb.close()

        # -- 5d. atomic rename -- handle file-open-in-Excel gracefully ------
        if os.path.exists(dest_path):
            try:
                os.remove(dest_path)
            except PermissionError:
                import datetime as _dt
                stamp    = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                fallback = os.path.join(
                    save_dir,
                    f"RL_vs_FixedTime_comparison_{stamp}.xlsx")
                _shutil.move(tmp_path, fallback)
                if verbose:
                    print(f"  [Excel] File open in Excel - saved timestamped copy: {fallback}")
                return fallback

        _shutil.move(tmp_path, dest_path)
        if verbose:
            size_kb  = os.path.getsize(dest_path) / 1024
            n_cols   = len(df_full.columns)
            n_sheets = 3 + int(_has_em) + int(_has_dur)
            print(f"  [Excel] Saved -> {dest_path}")
            print(f"          ({size_kb:.1f} KB | {n} episodes | "
                  f"{n_cols} columns | {n_sheets} sheets)")
            print(f"          Best RL ep: {best_rl_ep_num} | "
                  f"Best EV ep: {best_ev_ep_num if _has_ev else 'N/A'}")
        return dest_path

    except Exception as _exc:
        try:
            if "tmp_path" in dir() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        print(f"  [Excel] Export FAILED: {_exc}")
        import traceback as _tb
        _tb.print_exc()
        return None

def plot_comparison(
        save_dir: str,
        rl_history:    dict,
        fixed_history: dict,
        episodes:      list,
        ep_duration_ms: list = None,
):
    """
    Generates a set of comparison PNG plots:
      1. Avg Queue      RL vs Fixed  (all 4 TLs + combined)
      2. Avg Wait Time  RL vs Fixed
      3. Avg Reward     RL vs Fixed
      4. Improvement %  (queue + wait) -- bar chart
      5. Per-TL comparison grid (2×4)
    """
    if not episodes:
        return

    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    DARK_BG    = "#0D1117"
    MID_BG     = "#161B22"
    RL_COLOR   = "#1F6FEB"
    FIX_COLOR  = "#F78166"
    MUTED      = "#8B949E"
    GRID_COLOR = "#21262D"
    TL_COLORS  = {"TL1": "#58A6FF", "TL2": "#3FB950",
                  "TL3": "#F78166", "TL4": "#D2A8FF"}

    def _smooth(arr, w=5):
        if len(arr) < w:
            return arr
        return np.convolve(arr, np.ones(w)/w, mode="valid")

    def _apply_style(fig, axes_flat):
        fig.patch.set_facecolor(DARK_BG)
        for ax in axes_flat:
            ax.set_facecolor(MID_BG)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.xaxis.label.set_color(MUTED)
            ax.yaxis.label.set_color(MUTED)
            for s in ax.spines.values():
                s.set_color(GRID_COLOR)
            ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.6)

    eps_arr = np.array(episodes)

    # -- 1. Overview: Avg Queue, Wait, Reward  (3 rows × 1 col) ----------
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle("MAPPO-PIS+GAT  vs  Fixed-Time -- Network Average",
                 color="#E6EDF3", fontsize=14, fontweight="bold", y=1.01)
    _apply_style(fig, axes)

    for ax, key, ylabel, title in zip(
            axes,
            ["queue", "wait", "reward"],
            ["Queue (veh/TL)", "Wait Time (s/veh)", "Reward (Back-Pressure)"],
            ["Average Queue Length", "Average Waiting Time", "Average Reward"],
    ):
        rl_avg  = np.array([np.mean([rl_history[key].get(tl, [0]*9999)[i]
                                     if i < len(rl_history[key].get(tl,[]))
                                     else 0 for tl in INTERSECTIONS])
                            for i in range(len(episodes))])
        fx_avg  = np.array([np.mean([fixed_history[key].get(tl, [0]*9999)[i]
                                     if i < len(fixed_history[key].get(tl,[]))
                                     else 0 for tl in INTERSECTIONS])
                            for i in range(len(episodes))])
        ax.plot(eps_arr, rl_avg,  color=RL_COLOR,  alpha=0.25, linewidth=0.8)
        ax.plot(eps_arr, fx_avg,  color=FIX_COLOR, alpha=0.25, linewidth=0.8)
        w = max(3, len(episodes) // 15)
        if len(rl_avg) >= w:
            sm_rl = _smooth(rl_avg, w); sm_eps = eps_arr[w-1:]
            ax.plot(sm_eps, sm_rl, color=RL_COLOR,  linewidth=2.2,
                    label="MAPPO-PIS+GAT")
        if len(fx_avg) >= w:
            sm_fx = _smooth(fx_avg, w); sm_eps = eps_arr[w-1:]
            ax.plot(sm_eps, sm_fx, color=FIX_COLOR, linewidth=2.2,
                    linestyle="--", label="Fixed-Time (single run, cached)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, color="#E6EDF3", fontsize=11, fontweight="bold", loc="left")
        ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                  labelcolor="#E6EDF3", fontsize=8, loc="upper right")
        ax.text(0.01, 0.01,
                "Fixed baseline = one run with random seed",
                transform=ax.transAxes, fontsize=7, color=MUTED,
                va="bottom", ha="left", style="italic")

    axes[-1].set_xlabel("Episode", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(os.path.join(plot_dir, "comparison_overview.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  [plot] comparison_overview.png")

    # -- 2. Per-TL comparison grid (4 TLs × 2 metrics) --------------------
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("Per-Intersection: RL vs Fixed-Time (Queue & Wait)",
                 color="#E6EDF3", fontsize=14, fontweight="bold")
    _apply_style(fig, axes.flat)

    for row_idx, tl in enumerate(INTERSECTIONS):
        for col_idx, (key, ylabel) in enumerate([("queue","Queue (veh)"),
                                                  ("wait", "Wait (s/veh)")]):
            ax = axes[row_idx][col_idx]
            rl_vals  = rl_history[key].get(tl, [])
            fx_vals  = fixed_history[key].get(tl, [])
            eps_rl   = eps_arr[:len(rl_vals)]
            eps_fx   = eps_arr[:len(fx_vals)]
            tl_color = TL_COLORS[tl]

            ax.plot(eps_rl, rl_vals,  color=RL_COLOR,  alpha=0.2, linewidth=0.7)
            ax.plot(eps_fx, fx_vals,  color=FIX_COLOR, alpha=0.2, linewidth=0.7)
            w = max(3, len(episodes) // 15)
            if len(rl_vals) >= w:
                ax.plot(eps_rl[w-1:], _smooth(np.array(rl_vals), w),
                        color=RL_COLOR,  linewidth=2.0, label="RL")
            if len(fx_vals) >= w:
                ax.plot(eps_fx[w-1:], _smooth(np.array(fx_vals), w),
                        color=FIX_COLOR, linewidth=2.0, linestyle="--", label="Fixed")

            ax.set_title(f"{tl} -- {ylabel}", color=tl_color,
                         fontsize=10, fontweight="bold", loc="left")
            ax.set_xlabel("Episode", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                      labelcolor="#E6EDF3", fontsize=7)
            ax.text(0.01, 0.01,
                    "Fixed = single cached run",
                    transform=ax.transAxes, fontsize=6.5, color=MUTED,
                    va="bottom", ha="left", style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "comparison_per_tl.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  [plot] comparison_per_tl.png")

    # -- 3. Improvement bar chart ------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(DARK_BG)
    _apply_style(fig, axes)
    fig.suptitle("RL Improvement over Fixed-Time (%)",
                 color="#E6EDF3", fontsize=14, fontweight="bold")

    for ax, key, ylabel in zip(axes,
                                ["queue", "wait"],
                                ["Queue Reduction (%)", "Wait Reduction (%)"]):
        improvements = []
        tl_labels    = []
        colors_bar   = []
        for tl in INTERSECTIONS:
            rl_vals = rl_history[key].get(tl, [0])
            fx_vals = fixed_history[key].get(tl, [0])
            # Full-history mean includes early high-error episodes, making RL look worse than it is.
            tail    = max(1, len(rl_vals) // 5)
            rl_mean = np.mean(rl_vals[-tail:])
            fx_mean = np.mean(fx_vals[-tail:]) if fx_vals else 0.0
            imp      = ((fx_mean - rl_mean) / max(fx_mean, 1e-6)) * 100
            improvements.append(imp)
            tl_labels.append(tl)
            colors_bar.append("#3FB950" if imp >= 0 else "#F78166")

        bars = ax.bar(tl_labels, improvements, color=colors_bar, width=0.5,
                      edgecolor=GRID_COLOR, linewidth=0.8)
        ax.axhline(0, color=MUTED, linewidth=1.0, linestyle="--")
        for bar, imp in zip(bars, improvements):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.5 if imp >= 0 else -1.5),
                    f"{imp:+.1f}%",
                    ha="center", va="bottom" if imp >= 0 else "top",
                    fontsize=10, color="#E6EDF3", fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, color="#E6EDF3", fontsize=11, loc="left")
        ax.tick_params(colors=MUTED)
        ax.text(0.99, 0.01,
                "Improvement computed over last 20% of episodes (converged tail)",
                transform=ax.transAxes, fontsize=7, color=MUTED,
                va="bottom", ha="right", style="italic")

    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "comparison_improvement_bar.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  [plot] comparison_improvement_bar.png")

    # -- 4. Reward comparison (overlaid, all 4 TLs) -----------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    fig.suptitle("Reward: MAPPO-PIS+GAT vs Fixed-Time",
                 color="#E6EDF3", fontsize=14, fontweight="bold")
    _apply_style(fig, axes.flat)

    for ax, tl in zip(axes.flat, INTERSECTIONS):
        rl_r = np.array(rl_history["reward"].get(tl, []))
        fx_r = np.array(fixed_history["reward"].get(tl, []))
        eps_rl = eps_arr[:len(rl_r)]; eps_fx = eps_arr[:len(fx_r)]
        ax.plot(eps_rl, rl_r, color=RL_COLOR,  alpha=0.15, linewidth=0.7)
        ax.plot(eps_fx, fx_r, color=FIX_COLOR, alpha=0.15, linewidth=0.7)
        w = max(3, len(episodes) // 15)
        if len(rl_r) >= w:
            ax.plot(eps_rl[w-1:], _smooth(rl_r, w),
                    color=RL_COLOR, linewidth=2.2, label="MAPPO-PIS+GAT")
        if len(fx_r) >= w:
            ax.plot(eps_fx[w-1:], _smooth(fx_r, w),
                    color=FIX_COLOR, linewidth=2.2, linestyle="--", label="Fixed-Time")
        ax.set_title(tl, color=TL_COLORS[tl], fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel("Reward", fontsize=9)
        ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                  labelcolor="#E6EDF3", fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel("Episode", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "comparison_reward.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print("  [plot] comparison_reward.png")

    # -- 5. Episode duration (ms) -----------------------------------------
    if ep_duration_ms and any(v > 0 for v in ep_duration_ms):
        ms_arr  = np.array(ep_duration_ms[:len(episodes)], dtype=float)
        eps_ms  = eps_arr[:len(ms_arr)]
        w       = max(3, len(ms_arr) // 10 + 1)
        smooth  = _smooth(ms_arr, w) if len(ms_arr) >= w else ms_arr
        sm_eps  = eps_ms[w-1:] if len(ms_arr) >= w else eps_ms
        mean_v  = float(np.mean(ms_arr))

        fig, ax = plt.subplots(figsize=(14, 5))
        _apply_style(fig, [ax])
        fig.patch.set_facecolor(DARK_BG)
        fig.suptitle("Episode Duration (ms) -- Wall-Clock Training Time per Episode",
                     color="#E6EDF3", fontsize=13, fontweight="bold", y=1.01)

        ax.fill_between(eps_ms, ms_arr, alpha=0.12, color="#1F6FEB", zorder=1)
        ax.plot(eps_ms, ms_arr, color="#58A6FF", alpha=0.25, linewidth=0.8, zorder=2)
        ax.plot(sm_eps, smooth,  color="#58A6FF", linewidth=2.5, zorder=3,
                label=f"Duration (smooth w={w})")
        ax.axhline(mean_v, color=MUTED, linestyle=":", linewidth=1.1, zorder=4)
        ax.text(eps_ms[-1], mean_v, f"  mean={mean_v:.0f} ms",
                va="center", ha="left", fontsize=8, color=MUTED,
                fontweight="bold", clip_on=False)

        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylabel("Duration (ms)", fontsize=9)
        ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                  labelcolor="#E6EDF3", fontsize=8, loc="upper right")

        stats_txt = (f"min   {ms_arr.min():.0f} ms\n"
                     f"max   {ms_arr.max():.0f} ms\n"
                     f"mean  {mean_v:.0f} ms\n"
                     f"std   {ms_arr.std():.0f} ms")
        ax.text(0.02, 0.96, stats_txt, transform=ax.transAxes,
                fontsize=8, va="top", ha="left", family="monospace", color=MUTED,
                bbox=dict(boxstyle="round,pad=0.5", facecolor=MID_BG,
                          edgecolor=GRID_COLOR, alpha=0.95))

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "ep_duration_ms.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  [plot] ep_duration_ms.png")

    # -- 6. Emission comparison plots (CO2 / NOx / Fuel) ---------------------------
    co2_rl   = np.array(rl_history.get("co2_total",  []), dtype=float)
    co2_fx   = np.array(fixed_history.get("co2_total",  []), dtype=float)
    nox_rl   = np.array(rl_history.get("nox_total",  []), dtype=float)
    nox_fx   = np.array(fixed_history.get("nox_total",  []), dtype=float)
    fuel_rl  = np.array(rl_history.get("fuel_total", []), dtype=float)
    fuel_fx  = np.array(fixed_history.get("fuel_total", []), dtype=float)

    _has_emissions = (co2_rl.size > 0 and co2_rl.sum() > 0) or (co2_fx.size > 0 and co2_fx.sum() > 0)
    if _has_emissions:
        # - 6a. 3-panel line chart: CO2 / NOx / Fuel over episodes ---------------------
        fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
        fig.suptitle("Emission Totals per Episode -- RL vs Fixed-Time",
                     color="#E6EDF3", fontsize=14, fontweight="bold", y=1.01)
        _apply_style(fig, axes)

        _em_series = [
            (co2_rl,  co2_fx,  "CO₂ Total (mg)",  "CO₂ Emissions"),
            (nox_rl,  nox_fx,  "NOx Total (mg)",   "NOx Emissions"),
            (fuel_rl, fuel_fx, "Fuel Total (mg)",  "Fuel Consumption"),
        ]
        w_em = max(3, len(episodes) // 15)
        for ax, (rl_arr, fx_arr, ylabel, title) in zip(axes, _em_series):
            eps_rl = eps_arr[:len(rl_arr)]
            eps_fx = eps_arr[:len(fx_arr)]
            ax.plot(eps_rl, rl_arr, color=RL_COLOR,  alpha=0.20, linewidth=0.8)
            ax.plot(eps_fx, fx_arr, color=FIX_COLOR, alpha=0.20, linewidth=0.8)
            if len(rl_arr) >= w_em:
                ax.plot(eps_rl[w_em-1:], _smooth(rl_arr, w_em),
                        color=RL_COLOR,  linewidth=2.2, label="MAPPO-PIS+GAT")
            if len(fx_arr) >= w_em:
                ax.plot(eps_fx[w_em-1:], _smooth(fx_arr, w_em),
                        color=FIX_COLOR, linewidth=2.2, linestyle="--",
                        label="Fixed-Time (cached)")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, color="#E6EDF3", fontsize=11,
                         fontweight="bold", loc="left")
            ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                      labelcolor="#E6EDF3", fontsize=8, loc="upper right")
            ax.text(0.01, 0.01,
                    "Lower is better  |  Fixed baseline = single run with random seed",
                    transform=ax.transAxes, fontsize=7, color=MUTED,
                    va="bottom", ha="left", style="italic")
        axes[-1].set_xlabel("Episode", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(plot_dir, "comparison_emissions.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  [plot] comparison_emissions.png")

        # - 6b. Emission improvement bar chart ------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor(DARK_BG)
        _apply_style(fig, axes)
        fig.suptitle("Emission Reduction: RL vs Fixed-Time (%)",
                     color="#E6EDF3", fontsize=14, fontweight="bold")

        _bar_data = [
            (co2_rl,  co2_fx,  "CO₂ Saving (%)",  "CO₂"),
            (nox_rl,  nox_fx,  "NOx Saving (%)",   "NOx"),
            (fuel_rl, fuel_fx, "Fuel Saving (%)",  "Fuel"),
        ]
        for ax, (rl_arr, fx_arr, ylabel, label) in zip(axes, _bar_data):
            mean_rl = float(np.mean(rl_arr)) if rl_arr.size else 0.0
            mean_fx = float(np.mean(fx_arr)) if fx_arr.size else 0.0
            imp     = ((mean_fx - mean_rl) / max(mean_fx, 1e-6)) * 100
            color   = "#3FB950" if imp >= 0 else "#F78166"
            bar = ax.bar([label], [imp], color=color, width=0.45,
                         edgecolor=GRID_COLOR, linewidth=0.8)
            ax.axhline(0, color=MUTED, linewidth=1.0, linestyle="--")
            ax.text(0, imp + (0.3 if imp >= 0 else -0.8),
                    f"{imp:+.1f}%",
                    ha="center", va="bottom" if imp >= 0 else "top",
                    fontsize=14, color="#E6EDF3", fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{label} Reduction", color="#E6EDF3",
                         fontsize=11, loc="left")
            ax.text(0.99, 0.01, "vs single cached run",
                    transform=ax.transAxes, fontsize=7, color=MUTED,
                    va="bottom", ha="right", style="italic")
            # annotate mean values below bar
            ax.text(0, -abs(imp) * 0.12 if imp >= 0 else imp * 1.15,
                    f"RL {mean_rl:,.0f}  vs  FT {mean_fx:,.0f}",
                    ha="center", fontsize=8, color=MUTED)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, "comparison_emission_bar.png"),
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close(fig)
        print("  [plot] comparison_emission_bar.png")

        # - 6c. Per-vehicle emission efficiency (mg per car) --------------
        cp_rl_arr  = np.array(rl_history.get("cars_passed", []), dtype=float)
        cp_fx_arr  = np.array(fixed_history.get("cars_passed", []), dtype=float)
        # Guard against division by zero with where
        co2_per_rl   = np.where(cp_rl_arr > 0, co2_rl  / np.maximum(cp_rl_arr,  1), 0.0)
        co2_per_fx   = np.where(cp_fx_arr > 0, co2_fx  / np.maximum(cp_fx_arr,  1), 0.0)
        nox_per_rl   = np.where(cp_rl_arr > 0, nox_rl  / np.maximum(cp_rl_arr,  1), 0.0)
        nox_per_fx   = np.where(cp_fx_arr > 0, nox_fx  / np.maximum(cp_fx_arr,  1), 0.0)
        fuel_per_rl  = np.where(cp_rl_arr > 0, fuel_rl / np.maximum(cp_rl_arr,  1), 0.0)
        fuel_per_fx  = np.where(cp_fx_arr > 0, fuel_fx / np.maximum(cp_fx_arr,  1), 0.0)

        _has_per_car = (co2_per_rl.sum() + co2_per_fx.sum()) > 0
        if _has_per_car:
            fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)
            fig.suptitle("Per-Vehicle Emissions -- RL vs Fixed-Time",
                         color="#E6EDF3", fontsize=14, fontweight="bold", y=1.01)
            _apply_style(fig, axes)
            _pv_series = [
                (co2_per_rl,  co2_per_fx,  "CO₂ per vehicle (mg)",  "CO₂ per Car"),
                (nox_per_rl,  nox_per_fx,  "NOx per vehicle (mg)",   "NOx per Car"),
                (fuel_per_rl, fuel_per_fx, "Fuel per vehicle (mg)",  "Fuel per Car"),
            ]
            for ax, (rl_arr, fx_arr, ylabel, title) in zip(axes, _pv_series):
                eps_rl = eps_arr[:len(rl_arr)]
                eps_fx = eps_arr[:len(fx_arr)]
                ax.plot(eps_rl, rl_arr, color=RL_COLOR,  alpha=0.20, linewidth=0.8)
                ax.plot(eps_fx, fx_arr, color=FIX_COLOR, alpha=0.20, linewidth=0.8)
                if len(rl_arr) >= w_em:
                    ax.plot(eps_rl[w_em-1:], _smooth(rl_arr, w_em),
                            color=RL_COLOR,  linewidth=2.2, label="RL per car")
                if len(fx_arr) >= w_em:
                    ax.plot(eps_fx[w_em-1:], _smooth(fx_arr, w_em),
                            color=FIX_COLOR, linewidth=2.2,
                            linestyle="--", label="Fixed per car")
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_title(title, color="#E6EDF3", fontsize=11,
                             fontweight="bold", loc="left")
                ax.legend(facecolor=MID_BG, edgecolor=GRID_COLOR,
                          labelcolor="#E6EDF3", fontsize=8, loc="upper right")
            axes[-1].set_xlabel("Episode", fontsize=9)
            fig.tight_layout(rect=[0, 0, 1, 0.99])
            fig.savefig(os.path.join(plot_dir, "comparison_emissions_per_car.png"),
                        dpi=150, bbox_inches="tight", facecolor=DARK_BG)
            plt.close(fig)
            print("  [plot] comparison_emissions_per_car.png")

def safe_improvement_pct(rl_val: float, fx_val: float,
                         min_denom: float = 0.01) -> float:
    """
    Compute improvement percentage of RL over a Fixed-Time baseline.

    Formula: (fx_val - rl_val) / max(|fx_val|, min_denom) * 100

    A positive result means RL is better (lower queue / wait / emissions).
    For reward / throughput the caller should negate or use (rl - fx) instead.

    Parameters
    ----------
    rl_val    : RL metric value for this episode.
    fx_val    : Fixed-Time metric value for this episode.
    min_denom : Minimum absolute denominator to avoid division explosion when
                the baseline is near-zero.  Default 0.01 (much safer than 1e-6
                which can produce results like -49,990,000 %).

    Returns
    -------
    Improvement percentage, clamped to [-9999.9, 9999.9] to keep the UI
    readable if the baseline is legitimately very small.
    """
    denom = max(abs(fx_val), min_denom)
    raw   = ((fx_val - rl_val) / denom) * 100.0
    return max(-9999.9, min(9999.9, raw))

class RichUI:
    """
    Professional Rich-based terminal UI.

    Layout:
    ┌-----------------------------------------------------┐
    |  Header banner (algorithm name, device, mode)       |
    ├-----------------------------------------------------┤
    |  Episode progress bar  [====>  ]  12/50  ETA 4m32s  |
    |  Step progress bar     [=====> ]  3420/3600  12s     |
    |  Fixed-Time bar        [=====> ]  3420/3600  running |
    ├-----------------------------------------------------┤
    |  Live metrics table  (RL vs Fixed-Time per TL)      |
    ├-----------------------------------------------------┤
    |  Status / log line                                   |
    └-----------------------------------------------------┘
    """

    # TL accent colours (Rich colour names)
    _TL_STYLE = {
        "TL1": "bold dodger_blue2",
        "TL2": "bold green3",
        "TL3": "bold indian_red1",
        "TL4": "bold medium_purple1",
    }

    def __init__(self, n_episodes: int, sim_max_steps: int,
                 device: str, mode: str, intersections: list,
                 reward_name: str = "BackPressure"):
        self.n_episodes    = n_episodes
        self.sim_max_steps = sim_max_steps
        self.intersections = intersections
        self._device       = device
        self._mode         = mode
        self._reward_name  = reward_name
        self._ep           = 0
        self._step         = 0
        self._phase        = "init"   # "rl" | "fixed" | "train" | "plot"
        self._ep_start_t   = time.time()
        self._train_start  = time.time()
        self._last_rl:  dict = {tl: {"r": 0.0, "q": 0.0, "w": 0.0, "l": 0.0}
                                for tl in intersections}
        self._last_fx:  dict = {tl: {"r": 0.0, "q": 0.0, "w": 0.0}
                                for tl in intersections}
        # Episode-level emission totals (set by record_rl / record_fixed)
        self._last_rl_em: dict = {"co2": 0.0, "nox": 0.0, "fuel": 0.0}
        self._last_fx_em: dict = {"co2": 0.0, "nox": 0.0, "fuel": 0.0}
        self._best_ep   = 0
        self._best_r    = -1e9
        self._ep_times: list[float] = []
        # not the mean of all historical times (which may include timings from a
        # previous session on different hardware / a slow resume checkpoint load).
        self._ETA_WINDOW = 10   # number of recent episodes to use for ETA

        if not RICH_AVAILABLE:
            self._console = None
            self._progress = None
            self._live     = None
            self._ep_task  = None
            self._step_task = None
            self._fx_task   = None
            return

        self._console = Console(highlight=False)

        # -- Build Progress bar (compatible with rich ≥ 10.x) --------------
        _progress_kwargs: dict = dict(console=self._console, expand=False)
        # refresh_per_second added in rich 10.2 -- guard it
        try:
            import inspect as _inspect
            if "refresh_per_second" in _inspect.signature(Progress.__init__).parameters:
                _progress_kwargs["refresh_per_second"] = 8
        except Exception:
            pass

        # BarColumn style kwargs added in rich 10.x -- use safe subset
        try:
            _bar = BarColumn(
                bar_width=38,
                style="bar.back",
                complete_style="bold cyan",
                finished_style="bold green",
                pulse_style="bold magenta",
            )
        except TypeError:
            _bar = BarColumn(bar_width=38)

        self._progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold white]{task.description}"),
            _bar,
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            **_progress_kwargs,
        )
        self._ep_task   = self._progress.add_task(
            "[cyan]Episodes   ", total=n_episodes)
        self._step_task = self._progress.add_task(
            "[green]RL Step    ", total=sim_max_steps, visible=False)
        self._fx_task   = self._progress.add_task(
            "[yellow]Fixed Step ", total=sim_max_steps, visible=False)

        # -- Live display (compatible with rich ≥ 9.x) --------------------
        _live_kwargs: dict = dict(console=self._console, screen=False)
        try:
            import inspect as _inspect
            _live_sig = _inspect.signature(Live.__init__).parameters
            if "refresh_per_second" in _live_sig:
                _live_kwargs["refresh_per_second"] = 6
            elif "refresh_rate" in _live_sig:
                _live_kwargs["refresh_rate"] = 6
            if "transient" in _live_sig:
                _live_kwargs["transient"] = False
        except Exception:
            pass

        self._live = Live(self._build_layout(), **_live_kwargs)

    # -- public lifecycle ------------------------------------------------------
    def start(self):
        if not RICH_AVAILABLE or self._live is None:
            return
        self._print_header()
        try:
            self._live.start(refresh=True)
        except Exception:
            pass

    def stop(self):
        if not RICH_AVAILABLE or self._live is None:
            return
        try:
            self._live.stop()
        except Exception:
            pass

    def __enter__(self):
        self.start(); return self

    def __exit__(self, *_):
        self.stop()

    # -- update helpers --------------------------------------------------------
    def begin_episode(self, ep: int):
        self._ep        = ep
        self._ep_start_t = time.time()
        if not RICH_AVAILABLE:
            return
        self._progress.update(self._ep_task,   completed=ep)
        self._progress.update(self._step_task, completed=0, visible=True)
        self._progress.update(self._fx_task,   completed=0, visible=False)
        self._phase = "rl"
        self._refresh()

    def update_step(self, step: int):
        self._step = step
        if not RICH_AVAILABLE:
            return
        self._progress.update(self._step_task, completed=step)
        self._refresh()

    def begin_fixed(self, ep: int):
        if not RICH_AVAILABLE:
            return
        self._progress.update(self._step_task, visible=False)
        self._progress.update(self._fx_task,   completed=0, visible=True)
        self._phase = "fixed"
        self._refresh()

    def update_fixed_step(self, step: int):
        if not RICH_AVAILABLE:
            return
        self._progress.update(self._fx_task, completed=step)
        self._refresh()

    def set_phase(self, phase: str):
        """phase: 'train' | 'checkpoint' | 'plot' | 'done'"""
        self._phase = phase
        if not RICH_AVAILABLE:
            return
        self._refresh()

    def record_rl(self, ep: int, ep_reward: dict, ep_queue: dict,
                  ep_wait: dict, ep_loss: dict, ep_stats: dict = None):
        """Record end-of-episode RL metrics and update UI state.

        Best-episode definition (RichUI)
        ---------------------------------
        The status panel tracks the episode with the highest *network-average*
        reward (mean across all TLs).  This is intentionally different from the
        per-TL best-episode tracking inside ``plotter`` and the Excel summary,
        which use per-intersection maxima.  The label in the status panel reads
        "Best Ep (avg reward)" to make this distinction clear.

        Bug-6 note: previously both were silently using different definitions
        with no indication in the UI. The status panel now explicitly says
        "(avg reward)" so operators are not confused when the two numbers differ.
        """
        avg_r = float(np.mean(list(ep_reward.values())))
        if avg_r > self._best_r:
            self._best_r  = avg_r
            self._best_ep = ep
        self._ep_times.append(time.time() - self._ep_start_t)
        for tl in self.intersections:
            self._last_rl[tl] = {
                "r": ep_reward.get(tl, 0.0),
                "q": ep_queue.get(tl, 0.0),
                "w": ep_wait.get(tl, 0.0),
                "l": ep_loss.get(tl, 0.0),
            }
        if ep_stats is not None:
            self._last_rl_em = {
                "co2":  round(ep_stats.get("co2_sum",  0.0), 1),
                "nox":  round(ep_stats.get("nox_sum",  0.0), 1),
                "fuel": round(ep_stats.get("fuel_sum", 0.0), 1),
            }
        if not RICH_AVAILABLE:
            # fallback plain print
            avg_q = float(np.mean(list(ep_queue.values())))
            avg_l = float(np.mean(list(ep_loss.values())))
            print(f"  Ep {ep:>4d}/{self.n_episodes}"
                  f"  avg_reward={avg_r:+.4f}"
                  f"  avg_queue={avg_q:.2f}"
                  f"  avg_loss={avg_l:.5f}", flush=True)
            return
        self._progress.update(self._ep_task, completed=ep + 1)
        self._refresh()

    def record_fixed(self, fx_reward: dict, fx_queue: dict, fx_wait: dict,
                     fx_stats: dict = None):
        for tl in self.intersections:
            self._last_fx[tl] = {
                "r": fx_reward.get(tl, 0.0),
                "q": fx_queue.get(tl, 0.0),
                "w": fx_wait.get(tl, 0.0),
            }
        if fx_stats is not None:
            self._last_fx_em = {
                "co2":  round(fx_stats.get("__co2_total",  0.0), 1),
                "nox":  round(fx_stats.get("__nox_total",  0.0), 1),
                "fuel": round(fx_stats.get("__fuel_total", 0.0), 1),
            }
        if not RICH_AVAILABLE:
            return
        self._refresh()

    def log(self, msg: str, style: str = "dim white"):
        """Print a timestamped log line that appears above the live panel."""
        ts = time.strftime("%H:%M:%S")
        if not RICH_AVAILABLE or self._console is None:
            print(f"{ts}  {msg}", flush=True)
            return
        # Use the live's console so output is correctly interleaved
        console = self._live.console if (self._live and hasattr(self._live, "console")) \
                  else self._console
        console.print(f"[dim]{ts}[/dim]  {msg}", style=style, highlight=False)

    # -- internal rendering ----------------------------------------------------
    def _print_header(self):
        if not RICH_AVAILABLE:
            return
        self._console.print()
        self._console.print(Rule(
            "[bold cyan]OM Network  ·  MAPPO-PIS + GAT  ·  Fixed-Time Comparison",
            style="cyan"))
        meta = Table.grid(padding=(0, 2))
        meta.add_column(style="bold dim white")
        meta.add_column(style="white")
        meta.add_row("Device",   self._device)
        meta.add_row("Mode",     self._mode)
        meta.add_row("Episodes", str(self.n_episodes))
        meta.add_row("Steps",    str(self.sim_max_steps))
        meta.add_row("TLS",      ", ".join(self.intersections))
        self._console.print(Panel(meta, title="[bold white]Configuration",
                                  border_style="dim cyan", expand=False))
        self._console.print()

    def _build_metrics_table(self):
        # SIMPLE_HEAVY may not exist in older rich
        try:
            _box = box.SIMPLE_HEAVY
        except (AttributeError, TypeError):
            try:
                _box = box.SIMPLE
            except (AttributeError, TypeError):
                _box = None

        tbl = Table(
            title="[bold white]Live Metrics -- Episode {:>4d}/{:>4d}".format(
                self._ep, self.n_episodes),
            box=_box,
            border_style="dim cyan",
            header_style="bold white",
            show_lines=True,
            expand=False,
            padding=(0, 1),
        )
        tbl.add_column("TLS",          style="bold white",   justify="center", width=6)
        tbl.add_column("RL Reward",    style="cyan",         justify="right",  width=11)
        tbl.add_column("RL Queue",     style="cyan",         justify="right",  width=10)
        tbl.add_column("RL Wait(s)",   style="cyan",         justify="right",  width=10)
        tbl.add_column("RL Loss",      style="dim cyan",     justify="right",  width=10)
        tbl.add_column("FT BP-Score",  style="yellow",       justify="right",  width=11)  # BUG-9 FIX: was "FT Reward" -- FT has no policy reward, this is the BackPressure value under fixed schedule
        tbl.add_column("FT Queue",     style="yellow",       justify="right",  width=10)
        tbl.add_column("FT Wait(s)",   style="yellow",       justify="right",  width=10)
        tbl.add_column("Q Δ%",         justify="right",      width=8)
        tbl.add_column("W Δ%",         justify="right",      width=8)
        # Emission summary (network totals for this episode)
        tbl.add_column("RL CO₂(kg)",   style="green4",      justify="right",  width=12)
        tbl.add_column("FT CO₂(kg)",   style="dark_orange", justify="right",  width=12)
        tbl.add_column("CO₂ Δ%",      justify="right",      width=9)
        tbl.add_column("RL NOx(g)",    style="green4",      justify="right",  width=11)
        tbl.add_column("FT NOx(g)",    style="dark_orange", justify="right",  width=11)
        tbl.add_column("NOx Δ%",      justify="right",      width=9)
        tbl.add_column("RL Fuel(kg)",  style="green4",      justify="right",  width=12)
        tbl.add_column("FT Fuel(kg)",  style="dark_orange", justify="right",  width=12)
        tbl.add_column("Fuel Δ%",     justify="right",      width=9)

        for tl in self.intersections:
            rl = self._last_rl.get(tl, {})
            fx = self._last_fx.get(tl, {})
            rl_q = rl.get("q", 0.0); fx_q = fx.get("q", 0.0)
            rl_w = rl.get("w", 0.0); fx_w = fx.get("w", 0.0)
            # With 1e-6 a near-zero FT baseline produced values like -49,990,000 % in the UI.
            imp_q = safe_improvement_pct(rl_q, fx_q)
            imp_w = safe_improvement_pct(rl_w, fx_w)
            q_style = "bold green3"   if imp_q >= 0 else "bold indian_red1"
            w_style = "bold green3"   if imp_w >= 0 else "bold indian_red1"
            r_style = "bold green3"   if rl.get("r", 0) >= 0 else "bold indian_red1"
            # Emission cols: only show on first TL row (network total, not per-TL)
            _show_em = (tl == self.intersections[0])
            _rl_co2  = self._last_rl_em.get("co2",  0.0) / _MG_TO_KG
            _fx_co2  = self._last_fx_em.get("co2",  0.0) / _MG_TO_KG
            _rl_nox  = self._last_rl_em.get("nox",  0.0) / _MG_TO_G
            _fx_nox  = self._last_fx_em.get("nox",  0.0) / _MG_TO_G
            _rl_fuel = self._last_rl_em.get("fuel", 0.0) / _MG_TO_KG
            _fx_fuel = self._last_fx_em.get("fuel", 0.0) / _MG_TO_KG
            _co2_imp  = safe_improvement_pct(_rl_co2,  _fx_co2)  if _fx_co2  else 0.0
            _nox_imp  = safe_improvement_pct(_rl_nox,  _fx_nox)  if _fx_nox  else 0.0
            _fuel_imp = safe_improvement_pct(_rl_fuel, _fx_fuel) if _fx_fuel else 0.0
            _co2_s    = "bold green3" if _co2_imp  >= 0 else "bold indian_red1"
            _nox_s    = "bold green3" if _nox_imp  >= 0 else "bold indian_red1"
            _fuel_s   = "bold green3" if _fuel_imp >= 0 else "bold indian_red1"
            tbl.add_row(
                Text(tl, style=self._TL_STYLE.get(tl, "bold white")),
                Text(f"{rl.get('r',0.0):+.4f}", style=r_style),
                f"{rl_q:.2f}",
                f"{rl_w:.2f}",
                Text(f"{rl.get('l',0.0):.5f}", style="dim cyan"),
                Text(f"{fx.get('r',0.0):+.4f}", style="yellow"),
                f"{fx_q:.2f}",
                f"{fx_w:.2f}",
                Text(f"{imp_q:+.1f}%", style=q_style),
                Text(f"{imp_w:+.1f}%", style=w_style),
                # Emission columns: show on first TL only, blank on others
                Text(f"{_rl_co2:,.2f}"   if _show_em else "", style="green4"),
                Text(f"{_fx_co2:,.2f}"   if _show_em else "", style="dark_orange"),
                Text(f"{_co2_imp:+.1f}%" if _show_em else "", style=_co2_s if _show_em else ""),
                Text(f"{_rl_nox:,.2f}"   if _show_em else "", style="green4"),
                Text(f"{_fx_nox:,.2f}"   if _show_em else "", style="dark_orange"),
                Text(f"{_nox_imp:+.1f}%" if _show_em else "", style=_nox_s if _show_em else ""),
                Text(f"{_rl_fuel:,.2f}"  if _show_em else "", style="green4"),
                Text(f"{_fx_fuel:,.2f}"  if _show_em else "", style="dark_orange"),
                Text(f"{_fuel_imp:+.1f}%" if _show_em else "", style=_fuel_s if _show_em else ""),
            )

        def _avg(d, k): return float(np.mean([d.get(t, {}).get(k, 0) for t in self.intersections]))
        rl_avg_r = _avg(self._last_rl, "r"); fx_avg_r = _avg(self._last_fx, "r")
        rl_avg_q = _avg(self._last_rl, "q"); fx_avg_q = _avg(self._last_fx, "q")
        rl_avg_w = _avg(self._last_rl, "w"); fx_avg_w = _avg(self._last_fx, "w")
        rl_avg_l = _avg(self._last_rl, "l")
        imp_q_avg = safe_improvement_pct(rl_avg_q, fx_avg_q)
        imp_w_avg = safe_improvement_pct(rl_avg_w, fx_avg_w)
        # Emission summary for AVG row
        _avg_rl_co2  = self._last_rl_em.get("co2",  0.0) / _MG_TO_KG
        _avg_fx_co2  = self._last_fx_em.get("co2",  0.0) / _MG_TO_KG
        _avg_rl_nox  = self._last_rl_em.get("nox",  0.0) / _MG_TO_G
        _avg_fx_nox  = self._last_fx_em.get("nox",  0.0) / _MG_TO_G
        _avg_rl_fuel = self._last_rl_em.get("fuel", 0.0) / _MG_TO_KG
        _avg_fx_fuel = self._last_fx_em.get("fuel", 0.0) / _MG_TO_KG
        _avg_co2_imp  = safe_improvement_pct(_avg_rl_co2,  _avg_fx_co2)  if _avg_fx_co2  else 0.0
        _avg_nox_imp  = safe_improvement_pct(_avg_rl_nox,  _avg_fx_nox)  if _avg_fx_nox  else 0.0
        _avg_fuel_imp = safe_improvement_pct(_avg_rl_fuel, _avg_fx_fuel) if _avg_fx_fuel else 0.0
        tbl.add_section()
        tbl.add_row(
            Text("AVG", style="bold white"),
            Text(f"{rl_avg_r:+.4f}", style="bold green3" if rl_avg_r >= 0 else "bold red"),
            Text(f"{rl_avg_q:.2f}",  style="bold white"),
            Text(f"{rl_avg_w:.2f}",  style="bold white"),
            Text(f"{rl_avg_l:.5f}",  style="dim white"),
            Text(f"{fx_avg_r:+.4f}", style="bold yellow"),
            Text(f"{fx_avg_q:.2f}",  style="bold yellow"),
            Text(f"{fx_avg_w:.2f}",  style="bold yellow"),
            Text(f"{imp_q_avg:+.1f}%", style="bold green3" if imp_q_avg >= 0 else "bold red"),
            Text(f"{imp_w_avg:+.1f}%", style="bold green3" if imp_w_avg >= 0 else "bold red"),
            Text(f"{_avg_rl_co2:,.2f}",  style="bold green4"),
            Text(f"{_avg_fx_co2:,.2f}",  style="bold dark_orange"),
            Text(f"{_avg_co2_imp:+.1f}%",  style="bold green3" if _avg_co2_imp  >= 0 else "bold red"),
            Text(f"{_avg_rl_nox:,.2f}", style="bold green4"),
            Text(f"{_avg_fx_nox:,.2f}", style="bold dark_orange"),
            Text(f"{_avg_nox_imp:+.1f}%", style="bold green3" if _avg_nox_imp >= 0 else "bold red"),
            Text(f"{_avg_rl_fuel:,.2f}", style="bold green4"),
            Text(f"{_avg_fx_fuel:,.2f}", style="bold dark_orange"),
            Text(f"{_avg_fuel_imp:+.1f}%", style="bold green3" if _avg_fuel_imp >= 0 else "bold red"),
        )
        return tbl

    def _build_status_panel(self):
        phase_labels = {
            "init":       ("[bold cyan]●[/] Initialising …",                   "cyan"),
            "rl":         (f"[bold green]▶[/] Running RL episode {self._ep}",  "green"),
            "fixed":      ("[bold yellow]▶[/] Running Fixed-Time baseline",    "yellow"),
            "train":      ("[bold magenta]⚙[/] Training on rollouts …",        "magenta"),
            "checkpoint": ("[bold blue]💾[/] Saving checkpoint …",             "blue"),
            "plot":       ("[bold white]📊[/] Generating plots & Excel …",     "white"),
            "done":       ("[bold green]✓[/] Training complete!",              "green"),
        }
        label, border = phase_labels.get(self._phase, ("…", "white"))

        elapsed   = time.time() - self._train_start
        # instead of the mean of ALL historical times.  Old sessions on different
        # hardware, or a slow first-episode checkpoint load, would inflate the
        # mean and make ETA wildly inaccurate after a resume.
        _recent_times = self._ep_times[-self._ETA_WINDOW:] if self._ep_times else []
        avg_ep_t  = float(np.mean(_recent_times)) if _recent_times else 0.0
        remaining = avg_ep_t * max(self.n_episodes - self._ep - 1, 0)

        stats = Table.grid(padding=(0, 3))
        stats.add_column(style="dim white")
        stats.add_column(style="white")
        stats.add_row("Status",     label)
        stats.add_row("Elapsed",    _fmt_time(elapsed))
        stats.add_row("ETA",        _fmt_time(remaining) if remaining > 0 else "--")
        stats.add_row("Reward",     self._reward_name)
        # when this differs from the per-TL best-episode in plotter / Excel summary.
        stats.add_row("Best Ep (avg reward)", f"{self._best_ep}  r={self._best_r:+.4f}")
        stats.add_row("Avg ep/s",   f"{1/avg_ep_t:.2f} ep/s" if avg_ep_t > 0 else "--")
        return Panel(stats, title="[bold white]Training Status",
                     border_style=border, expand=False, padding=(0, 1))

    def _build_layout(self):
        # rich.console.Group added in rich 12+ -- fall back gracefully
        try:
            from rich.console import Group as RGroup
            return RGroup(
                self._progress,
                self._build_metrics_table(),
                self._build_status_panel(),
            )
        except ImportError:
            # older rich: return just the progress bar;
            # table/panel will be printed via console.print separately
            return self._progress

    def _refresh(self):
        if self._live is None:
            return
        try:
            self._live.update(self._build_layout())
        except Exception:
            pass