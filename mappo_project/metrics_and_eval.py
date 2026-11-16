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
from . import xml_metrics

__all__ = ['CheckpointManager', 'load_checkpoint', 'EpisodeAutoSaver', 'FixedTimeController', '_run_fixed_time_episode', '_run_evaluation_episode', 'xml_metrics']


class CheckpointManager:
    def __init__(self, save_dir: str):
        self.save_dir  = save_dir
        self.ckpt_dir  = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._agents:    dict = {}
        self._plotter         = None
        self._writers:   list = []
        self._proc_ref:  list = [None]
        self._current_ep: int = 0

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._emergency_save)
            except (OSError, ValueError):
                pass

    def register(self, agents: dict, plotter, writers: list, proc_ref: list):
        self._agents   = agents
        self._plotter  = plotter
        self._writers  = writers
        self._proc_ref = proc_ref

    def update_episode(self, ep: int):
        self._current_ep = ep

    def _emergency_save(self, signum=None, frame=None):
        print(f"\n[CHECKPOINT] Signal received -- saving at ep={self._current_ep}...",
              flush=True)
        self.save_now(emergency=True)
        if self._proc_ref and self._proc_ref[0] is not None:
            try:
                _close_sumo(self._proc_ref[0])
            except Exception:
                pass
        sys.exit(0)

    def save_now(self, emergency: bool = False):
        tag = "emergency" if emergency else f"ep{self._current_ep:04d}"
        for tls_id, agent in self._agents.items():
            path = os.path.join(self.ckpt_dir, f"{tls_id}_{tag}.pt")
            torch.save({
                "episode":             self._current_ep,
                "actor_state":         agent.actor.state_dict(),
                # critic may be None for centralised-critic IPPO agents
                "critic_state":        (agent.critic.state_dict()
                                        if agent.critic is not None else None),
                "actor_optimizer":     agent.actor_optimizer.state_dict(),
                "critic_optimizer":    (agent.critic_optimizer.state_dict()
                                        if agent.critic_optimizer is not None else None),
                # Scheduler state_dicts include T_max and last_epoch so the
                # cosine curve can be restored exactly on resume.
                "actor_scheduler":     (agent._actor_scheduler.state_dict()
                                        if agent._actor_scheduler is not None else None),
                "critic_scheduler":    (agent._critic_scheduler.state_dict()
                                        if agent._critic_scheduler is not None else None),
                # GAT / intent fields only present on MAPPOAgent
                "gat_state":           (agent.gat_aggregator.state_dict()
                                        if hasattr(agent, "gat_aggregator") else None),
                "gat_optimizer":       (agent.gat_optimizer.state_dict()
                                        if hasattr(agent, "gat_optimizer") else None),
                "current_intent":      (agent.current_intent.copy()
                                        if hasattr(agent, "current_intent") else None),
                "neighbor_intents":    ({k: v.copy()
                                         for k, v in agent.neighbor_intents.items()}
                                        if hasattr(agent, "neighbor_intents") else {}),
                "current_action_idx":  agent.current_action_idx,
                "last_switch_step":    agent.last_switch_step,
                "_current_step":       agent._current_step,
                # Entropy collapse detection state
                "entropy_history":        list(agent._entropy_history),
                "entropy_boost_remaining": agent._entropy_boost_remaining,
                "entropy_coeff":           agent.entropy_coeff,
                "base_entropy_coeff":      agent._base_entropy_coeff,
                # step_rewards/queues/losses/waits are per-episode accumulators
                # and must start empty at the beginning of each episode after resume.
            }, path)

        if self._plotter:
            hist_path = os.path.join(self.ckpt_dir, f"plotter_history_{tag}.json")
            with open(hist_path, "w") as f:
                json.dump({
                    "episodes":            self._plotter._episodes,
                    "history":             {k: dict(v) for k, v in self._plotter._history.items()},
                    "ep_duration_ms":      self._plotter._ep_duration_ms,
                    "global_step_counter": self._plotter._global_step_counter,
                    "global_steps":        self._plotter._global_steps,
                    "wall_times":          self._plotter._wall_times,
                    "step_history":        {k: {tl: pairs
                                                for tl, pairs in tl_data.items()}
                                            for k, tl_data in self._plotter._step_history.items()},
                }, f, indent=2)

        # Update latest.json pointer (works on all platforms, no symlink needed)
        if not emergency:
            try:
                ptr_path = os.path.join(self.ckpt_dir, "latest.json")
                with open(ptr_path, "w") as _pf:
                    json.dump({"latest_ep": self._current_ep,
                               "tag": tag,
                               "path": os.path.join(self.ckpt_dir, tag)}, _pf)
            except OSError:
                pass

        for writer in self._writers:
            try:
                writer.f.flush()
            except Exception:
                pass

        if self._plotter and self._plotter._episodes:
            try:
                self._plotter.plot_all()
                print(f"[CHECKPOINT] Plots saved ({tag})", flush=True)
            except Exception as e:
                print(f"[CHECKPOINT] Plot failed: {e}", flush=True)

        print(f"[CHECKPOINT] Saved -> {self.ckpt_dir} ({tag})", flush=True)

def load_checkpoint(agents: dict, save_dir: str, episode: str = "emergency") -> int:
    """Load agent checkpoints and stash scheduler/entropy state for later restore.

    Supported checkpoint layouts
    ----------------------------
    * EpDir  (EpisodeAutoSaver)  : checkpoints/ep{N:04d}/{TLS_ID}.pt
    * Flat   (CheckpointManager) : checkpoints/{TLS_ID}_{tag}.pt
    * Latest pointer             : checkpoints/latest.json  (preferred)
                                   checkpoints/latest       (symlink fallback)

    Scheduler restore protocol
    --------------------------
    Schedulers are re-created by ``setup_lr_schedulers()`` *after* this function
    returns (because T_max is not yet known here for fresh starts).  We therefore
    stash the raw state_dicts on the agent as
    ``_pending_actor_scheduler_sd`` / ``_pending_critic_scheduler_sd``.
    ``setup_lr_schedulers()`` reads the original ``T_max`` stored inside the
    state_dict and calls ``load_state_dict()`` -- giving a perfectly seamless
    cosine curve continuation with no LR jump at the resume boundary.

    Returns
    -------
    int
        The episode number stored in the checkpoint (the loop will start from
        ``returned_ep + 1``).  Returns 0 if no files were found.
    """
    ckpt_dir = os.path.join(save_dir, "checkpoints")

    # -- Resolve "latest" -> concrete tag ---------------------------------------
    tag = episode
    if tag == "latest":
        _ptr_json = os.path.join(ckpt_dir, "latest.json")
        _ptr_link = os.path.join(ckpt_dir, "latest")
        if os.path.isfile(_ptr_json):
            try:
                with open(_ptr_json) as _pf:
                    _ptr = json.load(_pf)
                tag = f"ep{int(_ptr['latest_ep']):04d}"
                print(f"  [RESTORE] 'latest' resolved via latest.json -> {tag}", flush=True)
            except Exception as _je:
                print(f"  [RESTORE] WARNING: Could not parse latest.json ({_je}); "
                      f"trying symlink fallback.", flush=True)
        if tag == "latest" and os.path.islink(_ptr_link):
            tag = os.path.basename(os.readlink(_ptr_link))
            print(f"  [RESTORE] 'latest' resolved via symlink -> {tag}", flush=True)
        if tag == "latest":
            print(f"  [RESTORE] WARNING: No 'latest' pointer found in {ckpt_dir}. "
                  f"No checkpoint loaded.", flush=True)
            return 0

    print(f"\n  [RESTORE] -- Loading checkpoint tag='{tag}' ------------------",
          flush=True)

    restored_ep = 0
    any_loaded  = False

    for tls_id, agent in agents.items():
        # Prefer EpDir layout; fall back to flat layout.
        path_epdir = os.path.join(ckpt_dir, tag, f"{tls_id}.pt")
        path_flat  = os.path.join(ckpt_dir, f"{tls_id}_{tag}.pt")

        if os.path.exists(path_epdir):
            path = path_epdir
        elif os.path.exists(path_flat):
            path = path_flat
        else:
            print(f"  [RESTORE] WARNING: No checkpoint for {tls_id} "
                  f"(tried {path_epdir} and {path_flat})", flush=True)
            continue

        try:
            data = torch.load(path, map_location=agent.device, weights_only=False)
        except Exception as _le:
            print(f"  [RESTORE] ERROR: Could not load {path}: {_le}", flush=True)
            continue

        any_loaded = True

        # -- Actor / Critic weights -----------------------------------------
        for _net_name, _net, _key in (
            ("actor",  agent.actor,  "actor_state"),
            ("critic", agent.critic, "critic_state"),
        ):
            if _net is None:
                # centralised-critic IPPO: skip restoring local critic
                continue
            if _key in data and data[_key] is not None:
                try:
                    _net.load_state_dict(data[_key])
                except Exception as _e:
                    print(f"  [RESTORE] WARNING: {tls_id} {_net_name} weights failed: {_e}",
                          flush=True)
            else:
                print(f"  [RESTORE] WARNING: {tls_id} missing key '{_key}' in checkpoint.",
                      flush=True)

        # -- Optimisers -----------------------------------------------------
        for _opt, _key in (
            (agent.actor_optimizer,  "actor_optimizer"),
            (agent.critic_optimizer, "critic_optimizer"),
        ):
            if _opt is None:
                # centralised-critic IPPO: no local critic optimizer
                continue
            if _key in data and data[_key] is not None:
                try:
                    _opt.load_state_dict(data[_key])
                except Exception as _e:
                    print(f"  [RESTORE] WARNING: {tls_id} optimizer '{_key}' failed: {_e}",
                          flush=True)

        # -- GAT aggregator + its optimizer --------------------------------
        if "gat_state" in data and data["gat_state"] is not None:
            try:
                if hasattr(agent, "gat_aggregator"):
                    agent.gat_aggregator.load_state_dict(data["gat_state"])
            except Exception as _e:
                print(f"  [RESTORE] WARNING: {tls_id} GAT weights failed: {_e}", flush=True)
        if "gat_optimizer" in data and data["gat_optimizer"] is not None:
            try:
                if hasattr(agent, "gat_optimizer"):
                    agent.gat_optimizer.load_state_dict(data["gat_optimizer"])
            except Exception as _e:
                print(f"  [RESTORE] WARNING: {tls_id} GAT optimizer failed: {_e}", flush=True)

        # -- Agent state ----------------------------------------------------
        if hasattr(agent, "current_intent"):
            _idim = getattr(agent, "intent_dim", 1)
            agent.current_intent = np.array(
                data.get("current_intent", np.zeros(_idim, dtype=np.float32)),
                dtype=np.float32,
            ).copy()
        if hasattr(agent, "neighbor_intents"):
            raw_ni = data.get("neighbor_intents", {})
            agent.neighbor_intents = {k: np.array(v, dtype=np.float32).copy()
                                      for k, v in raw_ni.items()}
        agent.current_action_idx = int(data.get("current_action_idx", 0))
        agent.last_switch_step   = int(data.get("last_switch_step",
                                                 -agent.min_green_steps))
        agent._current_step      = int(data.get("_current_step", 0))

        # -- LR scheduler state -- stash for setup_lr_schedulers() ----------
        # We intentionally do NOT create the scheduler here: T_max = n_episodes
        # is only available in run_training().  setup_lr_schedulers() will pick
        # up these pending state_dicts, recover the original T_max from inside
        # them, and call load_state_dict() so the cosine continues seamlessly.
        _actor_sched_sd  = data.get("actor_scheduler")
        _critic_sched_sd = data.get("critic_scheduler")
        agent._pending_actor_scheduler_sd  = _actor_sched_sd
        agent._pending_critic_scheduler_sd = _critic_sched_sd
        _sched_info = ("scheduler SDs stashed for restore"
                       if _actor_sched_sd is not None
                       else "no scheduler SD in checkpoint -- will create fresh")

        # -- Entropy state --------------------------------------------------
        agent._entropy_history         = list(data.get("entropy_history", []))
        agent._entropy_boost_remaining = int(data.get("entropy_boost_remaining", 0))
        _saved_coeff      = data.get("entropy_coeff")
        _saved_base_coeff = data.get("base_entropy_coeff")

        if _saved_coeff is not None:
            agent.entropy_coeff = float(_saved_coeff)
        if _saved_base_coeff is not None:
            agent._base_entropy_coeff = float(_saved_base_coeff)
        elif _saved_coeff is not None:
            # Backward compat: old checkpoints only stored entropy_coeff.
            # Safe to use as base only when not mid-boost (otherwise base ≠ live).
            if agent._entropy_boost_remaining == 0:
                agent._base_entropy_coeff = float(_saved_coeff)
            else:
                print(f"  [RESTORE] WARNING: {tls_id} mid-boost entropy state without "
                      f"base_entropy_coeff key -- base_entropy_coeff left at __init__ default.",
                      flush=True)

        # -- Per-episode accumulators -- always start fresh ------------------
        # These are step-level lists that must be empty at episode start.
        # Do NOT restore them: a resumed episode is a new episode.
        agent.step_rewards = []
        agent.step_queues  = []
        agent.step_losses  = []
        agent.step_waits   = []

        _ep_in_file = int(data.get("episode", 0))
        restored_ep = max(restored_ep, _ep_in_file)

        print(
            f"  [RESTORE] {tls_id:4s} <- ep={_ep_in_file:4d}"
            f"  action={agent.current_action_idx}"
            f"  last_switch={agent.last_switch_step}"
            f"  step={agent._current_step}"
            f"  entropy={agent.entropy_coeff:.4f}"
            f"  ({_sched_info})",
            flush=True,
        )

    if not any_loaded:
        print(
            f"  [RESTORE] ERROR: No checkpoint files were loaded "
            f"(tag='{tag}', ckpt_dir='{ckpt_dir}'). "
            f"Training will start from ep 0 with uninitialised weights. "
            f"Verify --resume_from and --save_dir.",
            flush=True,
        )

    print(f"  [RESTORE] -- Done. restored_ep={restored_ep} ------------------\n",
          flush=True)
    return restored_ep

class EpisodeAutoSaver:
    """
    Saves every artefact produced during training after each episode:

      checkpoints/
        ep{N}/
          TL1_ep{N}.pt   TL2_ep{N}.pt …   <- model weights + optimiser
          onnx/
            actor_TL1.onnx  …              <- ONNX exports (refreshed each ep)
          plotter_history.json             <- full plotter state
          comparison_history.json          <- RL + Fixed histories
          metrics_ep{N}.json               <- per-episode scalar summary
      plots/                               <- PNG plots (already existing)
      episode_rewards.csv  …              <- already written per-step
      RL_vs_FixedTime_comparison.xlsx      <- written here too every N eps
    """

    def __init__(self, save_dir: str,
                 agents: dict,
                 plotter,
                 writers: list,
                 rl_history:    dict,
                 fixed_history: dict,
                 onnx_export_interval: int = 5,
                 excel_export_interval: int = 5,
                 save_every: int = 10,
                 rich_ui=None):
        self.save_dir   = save_dir
        self.agents     = agents
        self.plotter    = plotter
        self.writers    = writers
        self.rl_history    = rl_history
        self.fixed_history = fixed_history
        self.onnx_interval   = onnx_export_interval
        self.excel_interval  = excel_export_interval
        self.save_every      = save_every
        self._ui         = rich_ui
        self._onnx_dir   = os.path.join(save_dir, "checkpoints", "onnx")
        os.makedirs(self._onnx_dir, exist_ok=True)

    def _log(self, msg: str, style: str = "dim white"):
        if self._ui:
            self._ui.log(msg, style=style)
        else:
            print(msg, flush=True)

    # -- called after every episode -----------------------------------------
    def save_episode(self,
                     ep: int,
                     ep_reward: dict,
                     ep_queue:  dict,
                     ep_wait:   dict,
                     ep_loss:   dict,
                     fx_reward: dict,
                     fx_queue:  dict,
                     fx_wait:   dict,
                     rl_xml_metrics:  dict | None = None,
                     rl_xml_validation: dict | None = None):

        ep_ckpt_dir = os.path.join(self.save_dir, "checkpoints", f"ep{ep:04d}")
        os.makedirs(ep_ckpt_dir, exist_ok=True)

        _should_save = (ep % self.save_every == 0) or (ep == 0)

        # 1. -- Model weights -- every save_every episodes ------------------
        if _should_save:
            for tls_id, agent in self.agents.items():
                pt_path = os.path.join(ep_ckpt_dir, f"{tls_id}.pt")
                torch.save({
                    "episode":             ep,
                    "actor_state":         agent.actor.state_dict(),
                    # critic may be None for centralised-critic IPPO agents
                    "critic_state":        (agent.critic.state_dict()
                                            if agent.critic is not None else None),
                    "actor_optimizer":     agent.actor_optimizer.state_dict(),
                    "critic_optimizer":    (agent.critic_optimizer.state_dict()
                                            if agent.critic_optimizer is not None else None),
                    "actor_scheduler":     agent._actor_scheduler.state_dict()
                                          if agent._actor_scheduler is not None else None,
                    "critic_scheduler":    agent._critic_scheduler.state_dict()
                                          if agent._critic_scheduler is not None else None,
                    # GAT / intent fields only present on MAPPOAgent
                    "gat_state":           (agent.gat_aggregator.state_dict()
                                            if hasattr(agent, "gat_aggregator") else None),
                    "gat_optimizer":       (agent.gat_optimizer.state_dict()
                                            if hasattr(agent, "gat_optimizer") else None),
                    "current_intent":      (agent.current_intent.copy()
                                            if hasattr(agent, "current_intent") else None),
                    "neighbor_intents":    ({k: v.copy()
                                             for k, v in agent.neighbor_intents.items()}
                                            if hasattr(agent, "neighbor_intents") else {}),
                    "current_action_idx":  agent.current_action_idx,
                    "last_switch_step":    agent.last_switch_step,
                    "_current_step":       agent._current_step,
                    "entropy_history":         list(agent._entropy_history),
                    "entropy_boost_remaining":  agent._entropy_boost_remaining,
                    "entropy_coeff":            agent.entropy_coeff,
                    "base_entropy_coeff":       agent._base_entropy_coeff,
                }, pt_path)

        # 2. -- ONNX export (every onnx_interval episodes) ----------------
        if _should_save and ONNX_AVAILABLE and (ep % self.onnx_interval == 0 or ep == 0):
            for tls_id, agent in self.agents.items():
                try:
                    onnx_path = agent.export_onnx(self._onnx_dir)
                    if onnx_path:
                        # also copy to ep checkpoint dir for traceability
                        import shutil
                        ep_onnx = os.path.join(ep_ckpt_dir, f"actor_{tls_id}.onnx")
                        shutil.copy2(onnx_path, ep_onnx)
                except Exception as oe:
                    self._log(f"[ONNX] Export failed {tls_id}: {oe}", style="bold red")

        # 3. -- Plotter + comparison history JSON -- every save_every episodes
        if _should_save:
            hist_path = os.path.join(ep_ckpt_dir, "plotter_history.json")
            try:
                with open(hist_path, "w") as f:
                    json.dump({
                        "episodes":            self.plotter._episodes,
                        "history":             {k: dict(v)
                                                for k, v in self.plotter._history.items()},
                        "ep_duration_ms":      self.plotter._ep_duration_ms,
                        "global_step_counter": self.plotter._global_step_counter,
                        "global_steps":        self.plotter._global_steps,
                        "wall_times":          self.plotter._wall_times,
                        "step_history":        {k: {tl: pairs
                                                    for tl, pairs in tl_data.items()}
                                                for k, tl_data
                                                in self.plotter._step_history.items()},
                    }, f, separators=(",", ":"))
            except Exception as e:
                self._log(f"[AutoSave] plotter JSON failed: {e}", style="bold red")

            cmp_path = os.path.join(ep_ckpt_dir, "comparison_history.json")
            try:
                with open(cmp_path, "w") as f:
                    json.dump({
                        "rl_history":    self.rl_history,
                        "fixed_history": self.fixed_history,
                        "episodes":      self.plotter._episodes,
                    }, f, separators=(",", ":"))
            except Exception as e:
                self._log(f"[AutoSave] comparison JSON failed: {e}", style="bold red")

        # 5. -- Per-episode scalar summary JSON ----------------------------
        metrics_path = os.path.join(ep_ckpt_dir, "metrics.json")
        try:
            avg_rl_q  = float(np.mean(list(ep_queue.values())))
            avg_rl_w  = float(np.mean(list(ep_wait.values())))
            avg_rl_r  = float(np.mean(list(ep_reward.values())))
            avg_fx_q  = float(np.mean(list(fx_queue.values())))
            avg_fx_w  = float(np.mean(list(fx_wait.values())))
            avg_fx_r  = float(np.mean(list(fx_reward.values())))
            # Snapshot emissions from the episode-level stats accumulator
            _m_stats = _snapshot_ep_stats()
            with open(metrics_path, "w") as f:
                json.dump({
                    "episode":      ep,
                    "timestamp":    time.asctime(),
                    "rl": {
                        "reward":  {k: round(v, 6) for k, v in ep_reward.items()},
                        "queue":   {k: round(v, 4) for k, v in ep_queue.items()},
                        "wait":    {k: round(v, 4) for k, v in ep_wait.items()},
                        "loss":    {k: round(v, 6) for k, v in ep_loss.items()},
                        "avg_reward": round(avg_rl_r, 6),
                        "avg_queue":  round(avg_rl_q, 4),
                        "avg_wait":   round(avg_rl_w, 4),
                        # Emission totals for this episode (mg)
                        "co2_total_mg":  round(_m_stats["co2_sum"],  2),
                        "nox_total_mg":  round(_m_stats["nox_sum"],  2),
                        "fuel_total_mg": round(_m_stats["fuel_sum"], 2),
                    },
                    "fixed_time": {
                        "reward":  {k: round(v, 6) for k, v in fx_reward.items()},
                        "queue":   {k: round(v, 4) for k, v in fx_queue.items()},
                        "wait":    {k: round(v, 4) for k, v in fx_wait.items()},
                        "avg_reward": round(avg_fx_r, 6),
                        "avg_queue":  round(avg_fx_q, 4),
                        "avg_wait":   round(avg_fx_w, 4),
                        # Emission totals from cached deterministic FT baseline (mg)
                        "co2_total_mg":  round(self.fixed_history.get("co2_total",  [0])[-1] if self.fixed_history.get("co2_total") else 0.0, 2),
                        "nox_total_mg":  round(self.fixed_history.get("nox_total",  [0])[-1] if self.fixed_history.get("nox_total") else 0.0, 2),
                        "fuel_total_mg": round(self.fixed_history.get("fuel_total", [0])[-1] if self.fixed_history.get("fuel_total") else 0.0, 2),
                    },
                    "improvement": {
                        "queue_pct": round(((avg_fx_q - avg_rl_q) / max(avg_fx_q, 1e-6)) * 100, 2),
                        "wait_pct":  round(((avg_fx_w - avg_rl_w) / max(avg_fx_w, 1e-6)) * 100, 2),
                        "co2_pct":   round(((self.fixed_history.get("co2_total",  [0])[-1] - _m_stats["co2_sum"])  / max(self.fixed_history.get("co2_total",  [1e-6])[-1], 1e-6)) * 100, 2) if self.fixed_history.get("co2_total") else 0.0,
                        "nox_pct":   round(((self.fixed_history.get("nox_total",  [0])[-1] - _m_stats["nox_sum"])  / max(self.fixed_history.get("nox_total",  [1e-6])[-1], 1e-6)) * 100, 2) if self.fixed_history.get("nox_total") else 0.0,
                        "fuel_pct":  round(((self.fixed_history.get("fuel_total", [0])[-1] - _m_stats["fuel_sum"]) / max(self.fixed_history.get("fuel_total", [1e-6])[-1], 1e-6)) * 100, 2) if self.fixed_history.get("fuel_total") else 0.0,
                    },
                    # Offline XML-derived metrics / TraCI-vs-XML validation,
                    # present only when config.ENABLE_XML_OUTPUT was True for
                    # this episode; otherwise both remain null.
                    "xml_metrics":    rl_xml_metrics,
                    "xml_validation": rl_xml_validation,
                }, f, indent=2)
        except Exception as e:
            self._log(f"[AutoSave] metrics JSON failed: {e}", style="bold red")

        # 6. -- Flush all CSV writers ---------------------------------------
        for w in self.writers:
            try:
                w.f.flush()
            except Exception:
                pass

        # 7. -- Excel comparison ----------------------------------------
        # Runs every save_every episodes (same cadence as checkpoints).
        # export_comparison_excel() is fully defensive: it handles all errors
        # internally, returns None on failure, and prints its own messages.
        if _should_save and self.plotter._episodes:
            _excel_path = export_comparison_excel(
                self.save_dir,
                self.rl_history,
                self.fixed_history,
                self.plotter._episodes,
                ep_duration_ms=self.plotter._ep_duration_ms,
                verbose=True,
            )
            if _excel_path is None:
                self._log(
                    "[AutoSave] Excel not saved this episode "
                    "(see console message above for reason).",
                    style="bold yellow")

        # 8. -- Plots -- every save_every episodes -------------------------
        if _should_save and self.plotter._episodes:
            try:
                self.plotter.plot_all()
            except Exception as pe:
                self._log(f"[AutoSave] plot_all failed: {pe}", style="bold red")

        self._log(
            f"[AutoSave] ep{ep:04d} -> {ep_ckpt_dir}  "
            f"(RL q={float(np.mean(list(ep_queue.values()))):.2f}  "
            f"FT q={float(np.mean(list(fx_queue.values()))):.2f})",
            style="dim cyan",
        )

    # -- symlink/copy latest checkpoint ----------------------------------------
# ═══════════════════════════════════════════════════════════════════════════════
# === 3. Updated tag_latest() -- inside EpisodeAutoSaver ===
# ═══════════════════════════════════════════════════════════════════════════════

    def tag_latest(self, ep: int) -> None:
        """Record which episode is the current 'latest' checkpoint.

        ``latest.json`` is the single, authoritative pointer used by the resume
        logic (``--resume_from=latest``).  It is always written first so the
        pointer is never in an inconsistent state.

        On non-Windows systems a ``latest`` symlink is also created for
        shell convenience (``ls -la checkpoints/latest``), but the resume code
        never reads the symlink -- only ``latest.json``.

        Atomic write strategy
        ---------------------
        The JSON is written to a temporary file in the same directory, then
        renamed (atomic on POSIX, best-effort on Windows) so a crash mid-write
        never leaves a truncated pointer file.
        """
        ep_tag   = f"ep{ep:04d}"
        ep_dir   = os.path.join(self.save_dir, "checkpoints", ep_tag)
        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        ptr_path = os.path.join(ckpt_dir, "latest.json")

        # -- Atomic JSON write ----------------------------------------------
        _payload = json.dumps(
            {
                "latest_ep":  ep,
                "tag":        ep_tag,
                "path":       ep_dir,
                "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            indent=2,
        )
        try:
            # Write to a temp file then rename for atomicity.
            import tempfile as _tempfile
            _fd, _tmp = _tempfile.mkstemp(dir=ckpt_dir, suffix=".json.tmp")
            try:
                with os.fdopen(_fd, "w") as _tf:
                    _tf.write(_payload)
                os.replace(_tmp, ptr_path)   # atomic on POSIX; best-effort on Windows
            except Exception:
                # Clean up tmp file if rename failed.
                try:
                    os.unlink(_tmp)
                except OSError:
                    pass
                raise
        except Exception as _e:
            # Non-fatal: training continues even if the pointer cannot be written.
            print(f"  [CKPT] WARNING: Could not write latest.json: {_e}", flush=True)
            return

        # -- Symlink (Linux / macOS only, best-effort) ----------------------
        import platform as _plat
        if _plat.system() != "Windows":
            lnk = os.path.join(ckpt_dir, "latest")
            try:
                # Remove stale link/file before creating new one.
                if os.path.islink(lnk) or os.path.exists(lnk):
                    os.remove(lnk)
                os.symlink(os.path.abspath(ep_dir), lnk)
            except OSError:
                pass  # Symlink is cosmetic; JSON is the authoritative pointer.

class FixedTimeController:
    """Runs a pre-programmed fixed-time signal plan for one TLS junction."""

    def __init__(self, tls_id: str):
        self.tls_id = tls_id
        plan = FIXED_TIME_PLANS.get(tls_id)
        if plan is None:
            raise ValueError(f"No fixed-time plan defined for {tls_id}")
        self.phases       = plan["phases"]          # [(dur, state), ...]
        self.cycle_length = plan["cycle_length"]
        self.offset       = plan["offset"]
        self._phase_idx   = 0
        self._phase_elapsed = 0
        self.step_queues: list[float] = []
        self.step_waits:  list[float] = []
        self.step_rewards: list[float] = []
        # Get lanes for pressure calculation
        self.incoming = UPSTREAM_LANES.get(tls_id, [])
        self.outgoing = DOWNSTREAM_LANES.get(tls_id, [])

    def reset(self):
        self._phase_idx     = 0
        self._phase_elapsed = 0
        self.step_queues.clear()
        self.step_waits.clear()
        self.step_rewards.clear()

    def step(self, sim_step: int):
        """Advance the fixed controller by one simulation step."""
        dur, state = self.phases[self._phase_idx]
        try:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
        except Exception:
            pass

        self._phase_elapsed += 1
        if self._phase_elapsed >= dur:
            self._phase_idx     = (self._phase_idx + 1) % len(self.phases)
            self._phase_elapsed = 0

        # collect metrics
        q    = _get_tls_queue_from_detectors(self.tls_id)
        wait = _get_tls_mean_wait(self.tls_id)

        qcap = float(UNIFIED_BP_CAPACITY)
        n_pairs = max(len(self.incoming), 1)
        pressure = 0.0
        for k, in_lane in enumerate(self.incoming):
            qi = _lane_queue(in_lane) / qcap
            out_lane = self.outgoing[k % len(self.outgoing)] if self.outgoing else None
            qo = (_lane_queue(out_lane) / qcap) if out_lane else 0.0
            pressure += (qi - qo)

        rew = float(np.clip(-(pressure / n_pairs), -REWARD_CLIP, REWARD_CLIP))

        self.step_queues.append(q)
        self.step_waits.append(wait)
        self.step_rewards.append(rew)

def _run_fixed_time_episode(
        sumo_cmd:      list,
        sim_max_steps: int,
        port:          int,
        controllers:   dict,
        rich_ui:       "RichUI | None" = None,
) -> dict:
    """
    Runs one SUMO episode with Fixed-Time controllers.
    Returns dict of {tl_id: {"queues":[], "waits":[], "rewards":[]}}
    Also includes top-level keys "ev_cleared", "ev_avg_wait", "cars_passed".
    """
    for ctrl in controllers.values():
        ctrl.reset()

    _reset_ep_stats()   # zero EV/throughput counters for this baseline run

    # Optionally append tripinfo/emissions/fcd output flags (no-op unless
    # config.ENABLE_XML_OUTPUT is True).
    #
    # NOTE: this module does `from .config import *`, which copies these
    # names at IMPORT time. If a caller sets config.ENABLE_XML_OUTPUT = True
    # at runtime (e.g. a verification script), the bare name here would
    # otherwise still read the stale, unpatched copy. Read live from the
    # `config` module object to make runtime toggling actually work.
    from . import config as _cfg
    _xml_enabled = getattr(_cfg, "ENABLE_XML_OUTPUT", False)
    _xml_out_dir = os.path.join(getattr(_cfg, "XML_OUTPUT_DIR", "."), "fixed_time") if _xml_enabled else None
    sumo_cmd = list(sumo_cmd) + _xml_output_args(_xml_out_dir)

    proc = _start_sumo(sumo_cmd, port=port)
    all_lanes = set()
    for tl_id in INTERSECTIONS:
        all_lanes.update(UPSTREAM_LANES.get(tl_id, []))
    _subscribe_lanes(list(all_lanes))
    all_dets = set()
    for dets in TLS_DETECTOR_MAP.values():
        all_dets.update(dets)
    _subscribe_detectors(list(all_dets))

    try:
        for step in range(1, sim_max_steps + 1):
            if step % 50 == 0 and traci.simulation.getMinExpectedNumber() == 0:
                if rich_ui is None:
                    print(f"    [FixedTime] early stop at step {step}, closing SUMO …", flush=True)
                break
            # Call every step (not every 10) so arrived-vehicle EV stats are complete
            _subscribe_vehicles_bulk()
            if step % 10 == 0:
                if rich_ui is not None:
                    rich_ui.update_fixed_step(step)
                elif step % 500 == 0:
                    print(f"    [FixedTime] step {step}/{sim_max_steps} …", flush=True)
            _STEP_CACHE.clear()
            for ctrl in controllers.values():
                ctrl.step(step)
            traci.simulationStep()
            _STEP_CACHE.bulk_load_lanes()
            _STEP_CACHE.bulk_load_detectors()
            _STEP_CACHE.bulk_load_vehicles()
    except Exception as e:
        if rich_ui is not None:
            rich_ui.log(f"[FixedTime] Episode error: {e}", style="bold red")
        else:
            print(f"  [FixedTime] Episode error: {e}", flush=True)
    finally:
        _collect_final_emissions()
        _close_sumo(proc)

    ep_stats = _snapshot_ep_stats()
    result = {}
    for tl, ctrl in controllers.items():
        result[tl] = {
            "queues":   list(ctrl.step_queues),
            "waits":    list(ctrl.step_waits),
            "rewards":  list(ctrl.step_rewards),
        }
    # Attach network-level EV & throughput stats as top-level keys
    result["__ev_cleared"]  = ep_stats["ev_cleared"]
    result["__ev_avg_wait"] = (round(ep_stats["ev_wait_sum"] / ep_stats["ev_cleared"], 3)
                               if ep_stats["ev_cleared"] > 0 else None)
    result["__cars_passed"] = ep_stats["cars_passed"]
    # Emission totals from this baseline episode (mg for CO2/NOx, ml for fuel)
    result["__co2_total"]   = round(ep_stats["co2_sum"],  2)
    result["__nox_total"]   = round(ep_stats["nox_sum"],  2)
    result["__fuel_total"]  = round(ep_stats["fuel_sum"], 2)

    # -- Optional XML cross-validation -----------------------------------
    result["__xml_metrics"]     = None
    result["__xml_validation"]  = None
    if _xml_enabled:
        xml_m = _load_xml_metrics_for_episode(_xml_out_dir)
        result["__xml_metrics"] = xml_m
        if getattr(_cfg, "ENABLE_XML_VALIDATION", False) and xml_m is not None:
            try:
                from . import xml_metrics
                result["__xml_validation"] = xml_metrics.validate_traci_vs_xml(
                    ep_stats, xml_m,
                    rel_tolerance=getattr(_cfg, "XML_VALIDATION_REL_TOL", 0.05),
                    abs_tolerance=getattr(_cfg, "XML_VALIDATION_ABS_TOL", 1.0),
                )
                if not result["__xml_validation"]["all_consistent"]:
                    _msg = f"[FixedTime] TraCI/XML mismatch detected: {result['__xml_validation']}"
                    if rich_ui is not None:
                        rich_ui.log(_msg, style="bold yellow")
                    else:
                        print(f"    {_msg}", flush=True)
            except Exception as ve:
                print(f"    [xml_metrics] validation failed: {ve}", flush=True)

    return result

def _run_evaluation_episode(
        sumo_cmd:         list,
        agents:           dict,
        sim_max_steps:    int,
        port:             int,
        deterministic:    bool = True,
        step_callback=None,
) -> dict:
    """Run one SUMO episode in pure evaluation mode and return metrics.


    #
    #   No training  -- train_step() is NEVER called inside this function.
    #                  Gradients do not flow; rollout buffers are not touched.
    #                  The function is a pure read of the current policy.
    #
    # Returns a dict with keys: avg_reward, avg_queue, avg_wait,
    #   per_tl_reward, per_tl_queue, per_tl_wait, cars_passed,
    #   ev_cleared, ev_avg_wait, co2_total, nox_total, fuel_total.
    """
    # Save per-episode agent state so we can restore it after eval.
    # The evaluation must not corrupt training bookkeeping (last_switch_step,
    # current_action_idx, neighbor_intents, EV potentials, etc.).
    _saved_state = {}
    for tid, ag in agents.items():
        _saved_state[tid] = {
            "state":              ag.state.copy(),
            "current_intent":     ag.current_intent.copy(),
            "current_action_idx": ag.current_action_idx,
            "last_switch_step":   ag.last_switch_step,
            "_current_step":      ag._current_step,
            "neighbor_intents":   {k: v.copy() for k, v in ag.neighbor_intents.items()},
            "_ev_phi_prev":       ag._ev_phi_prev,
            "_ev_phi_curr":       ag._ev_phi_curr,
        }

    # Reset agent episode state for the eval episode
    for ag in agents.values():
        ag.last_switch_step   = -ag.min_green_steps
        ag.current_action_idx = 0
        ag._current_step      = 0
        ag.neighbor_intents.clear()
        ag._ev_phi_prev = 0.0
        ag._ev_phi_curr = 0.0

    MAPPOAgent.clear_lane_length_cache()
    _reset_ep_stats()

    # Step-level accumulators
    eval_step_rewards: dict[str, list] = {tid: [] for tid in agents}
    eval_step_queues:  dict[str, list] = {tid: [] for tid in agents}
    eval_step_waits:   dict[str, list] = {tid: [] for tid in agents}

    # Optionally append tripinfo/emissions/fcd output flags (no-op unless
    # config.ENABLE_XML_OUTPUT is True).
    #
    # NOTE: this module does `from .config import *`, which copies these
    # names at IMPORT time. Read live from the `config` module object so
    # runtime toggling (e.g. by a verification script) actually works.
    from . import config as _cfg
    _xml_enabled = getattr(_cfg, "ENABLE_XML_OUTPUT", False)
    _xml_out_dir = os.path.join(getattr(_cfg, "XML_OUTPUT_DIR", "."), "eval") if _xml_enabled else None
    sumo_cmd = list(sumo_cmd) + _xml_output_args(_xml_out_dir)

    proc = None
    try:
        proc = _start_sumo(sumo_cmd, port=port)
        _setup_all_subscriptions(agents)

        # Refresh states from the new SUMO instance
        for ag in agents.values():
            try:
                ag.state = ag._get_state()
            except Exception:
                ag.state = np.zeros(ag.state_dim, dtype=np.float32)

        _min_expected_interval = 50
        for step in range(1, sim_max_steps + 1):
            if step % _min_expected_interval == 0:
                if traci.simulation.getMinExpectedNumber() == 0:
                    break

            if step_callback is not None and step % 10 == 0:
                try:
                    step_callback(step)
                except Exception:
                    pass

            _subscribe_vehicles_bulk()
            _STEP_CACHE.clear()
            _broadcast_intents(agents)

            for tid, ag in agents.items():
                agg     = ag.aggregate_neighbor_intents(step)
                combined = (ag.current_intent + agg) / 2.0
                action = ag._choose_action_with_intent(combined, deterministic=deterministic)
                ag.current_action_idx = action
                ag.apply_action(action, step)

            traci.simulationStep()
            _STEP_CACHE.bulk_load_lanes()
            _STEP_CACHE.bulk_load_detectors()
            _STEP_CACHE.bulk_load_vehicles()

            for tid, ag in agents.items():
                ns     = ag._get_state()
                reward = ag.get_reward()
                ag.state = ns
                eval_step_rewards[tid].append(reward)
                eval_step_queues[tid].append(_get_tls_queue_from_detectors(tid))
                eval_step_waits[tid].append(_get_tls_mean_wait(tid))

    except Exception as exc:
        print(f"  [EVAL] Warning: episode error: {exc}", flush=True)
    finally:
        if proc is not None:
            _collect_final_emissions()
            _close_sumo(proc)
        _ALL_SUBSCRIBED_LANES.clear()
        _ALL_SUBSCRIBED_DETECTORS.clear()

    # Restore agent state so training is unaffected
    for tid, ag in agents.items():
        sv = _saved_state[tid]
        ag.state              = sv["state"]
        ag.current_intent     = sv["current_intent"]
        ag.current_action_idx = sv["current_action_idx"]
        ag.last_switch_step   = sv["last_switch_step"]
        ag._current_step      = sv["_current_step"]
        ag.neighbor_intents   = sv["neighbor_intents"]
        ag._ev_phi_prev       = sv["_ev_phi_prev"]
        ag._ev_phi_curr       = sv["_ev_phi_curr"]

    ep_stats = _snapshot_ep_stats()

    # Build per-TL summary
    per_tl_reward = {}
    per_tl_queue  = {}
    per_tl_wait   = {}
    for tid in INTERSECTIONS:
        sr = eval_step_rewards.get(tid, [])
        sq = eval_step_queues.get(tid,  [])
        sw = eval_step_waits.get(tid,   [])
        per_tl_reward[tid] = float(np.mean(sr)) if sr else 0.0
        per_tl_queue[tid]  = float(np.mean(sq)) if sq else 0.0
        per_tl_wait[tid]   = float(np.mean(sw)) if sw else 0.0

    avg_reward = float(np.mean(list(per_tl_reward.values())))
    avg_queue  = float(np.mean(list(per_tl_queue.values())))
    avg_wait   = float(np.mean(list(per_tl_wait.values())))

    ev_avg_wait = None
    if ep_stats["ev_cleared"] > 0:
        ev_avg_wait = round(ep_stats["ev_wait_sum"] / ep_stats["ev_cleared"], 3)

    result = {
        "avg_reward":   avg_reward,
        "avg_queue":    avg_queue,
        "avg_wait":     avg_wait,
        "per_tl_reward": per_tl_reward,
        "per_tl_queue":  per_tl_queue,
        "per_tl_wait":   per_tl_wait,
        "cars_passed":  ep_stats["cars_passed"],
        "ev_cleared":   ep_stats["ev_cleared"],
        "ev_avg_wait":  ev_avg_wait,
        "co2_total":    round(ep_stats["co2_sum"],  2),
        "nox_total":    round(ep_stats["nox_sum"],  2),
        "fuel_total":   round(ep_stats["fuel_sum"], 2),
        "xml_metrics":    None,
        "xml_validation": None,
    }

    # -- Optional XML cross-validation -----------------------------------
    if _xml_enabled:
        xml_m = _load_xml_metrics_for_episode(_xml_out_dir)
        result["xml_metrics"] = xml_m
        if getattr(_cfg, "ENABLE_XML_VALIDATION", False) and xml_m is not None:
            try:
                from . import xml_metrics
                result["xml_validation"] = xml_metrics.validate_traci_vs_xml(
                    ep_stats, xml_m,
                    rel_tolerance=getattr(_cfg, "XML_VALIDATION_REL_TOL", 0.05),
                    abs_tolerance=getattr(_cfg, "XML_VALIDATION_ABS_TOL", 1.0),
                )
                if not result["xml_validation"]["all_consistent"]:
                    print(f"  [EVAL] TraCI/XML mismatch detected: {result['xml_validation']}",
                          flush=True)
            except Exception as ve:
                print(f"  [EVAL] xml_metrics validation failed: {ve}", flush=True)

    return result