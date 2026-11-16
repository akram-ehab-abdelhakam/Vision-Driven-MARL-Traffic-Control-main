"""
evaluate.py  –  Evaluation script for MAPPO-PIS+GAT / IPPO traffic signal control
===============================================================================

Modes (--mode):
  rl           – RL-only evaluation (original behaviour)
  compare      – Webster-Pressure vs RL  (default)
  webster      – Webster-Pressure only (no checkpoint required)

Algorithm (--algorithm):
  mappo        – MAPPO-PIS+GAT (default)
  ippo         – IPPO baseline (no intent sharing)

Webster-Pressure controller
----------------------------
At every decision point (after MIN_GREEN_STEPS have elapsed for the
current phase), the controller computes a back-pressure score for every
possible phase and selects the one with the highest score:

    pressure_i = Σ incoming_queue_i − Σ outgoing_queue_i

This approximates Webster's optimal cycle condition in an online,
adaptive fashion (Varaiya 2013; Wei et al. 2019) and is a standard
strong baseline in adaptive signal control literature.

Rich terminal UI
----------------
When ``rich`` is installed, a live panel is displayed with:
  • Evaluation step progress bar
  • Per-TLS metrics table  (Webster-Pressure | RL) with Δ% columns
  • Summary status panel

Usage examples
--------------
    python evall.py --sim_file OM.sumocfg --mode compare
    python evall.py --sim_file OM.sumocfg --mode rl --gui enable
    python evall.py --sim_file OM.sumocfg --mode webster
    python evall.py --sim_file OM.sumocfg --mode compare --report

    # IPPO evaluation:
    python evall.py --sim_file OM.sumocfg --mode rl --algorithm ippo --ckpt_dir ./RES/ippo/checkpoints/ep0240

    # Point directly at a checkpoint folder (skips save_dir/checkpoints/ convention):
    python evall.py --sim_file OM.sumocfg --ckpt_dir ./results/checkpoints/ep0150
    python evall.py --sim_file OM.sumocfg --ckpt_dir ./my_run/checkpoints --checkpoint ep0200

Notes
-----
* Webster-Pressure is now the primary baseline, replacing the previous Fixed-Time logic.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import time
import webbrowser
from pathlib import Path

import numpy as np
import torch

from mappo_project.config import (
    BASE_PORT, INTERSECTIONS, UPSTREAM_LANES, DOWNSTREAM_LANES,
    TLS_DETECTOR_MAP, FIXED_TIME_PLANS, UNIFIED_BP_CAPACITY,
    REWARD_CLIP, MIN_GREEN_STEPS, STEP_LENGTH,
)
from mappo_project.sumo_env import (
    _start_sumo, _close_sumo,
    _setup_all_subscriptions, _subscribe_vehicles_bulk,
    _STEP_CACHE, _reset_ep_stats, _snapshot_ep_stats,
    _get_tls_queue_from_detectors, _get_tls_mean_wait,
    _collect_final_emissions,
)
from mappo_project.agent import MAPPOAgent
from mappo_project.ippo import IPPOAgent
from mappo_project.metrics_and_eval import _run_evaluation_episode, load_checkpoint

# ── optional rich ─────────────────────────────────────────────────────────────
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
        class MofNCompleteColumn:          # type: ignore
            def __init__(self, *a, **kw): pass
        class TaskProgressColumn:          # type: ignore
            def __init__(self, *a, **kw): pass
    try:
        from rich.console import Group as RGroup
    except ImportError:
        RGroup = None
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Console = Live = Table = Panel = Text = Rule = box = None
    Progress = BarColumn = TextColumn = None
    TimeElapsedColumn = TimeRemainingColumn = SpinnerColumn = None
    MofNCompleteColumn = TaskProgressColumn = None
    RGroup = None

try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    tc = None


# ==============================================================================
#  WEBSTER-PRESSURE CONTROLLER  (redesigned from sumolights + sumo-rl patterns)
# ==============================================================================

class WebsterPressureController:
    """
    Max-pressure traffic signal controller for SUMO via TraCI.

    Design follows two canonical open-source references:
      • docwza/sumolights  — phase_lanes construction from getControlledLinks,
                             intermediate yellow/all-red transition logic.
      • LucasAlegre/sumo-rl — getControlledLanes for incoming lane discovery,
                               capacity = lane_length / 7.5 (veh length + gap),
                               green-bit check via getCompleteRedYellowGreenDefinition.

    Algorithm (Varaiya 2013 / Wei et al. 2019 PressLight):
        For each green phase p:
            score(p) = Σ_{l ∈ green_lanes(p)}  queue(l)/cap(l)
                     − Σ_{l ∈ green_lanes(p)}  queue(downstream(l))/cap(downstream(l))
        Select phase with max score; ties broken by hysteresis (keep current).

    Transition sequence on phase change:
        current_green → yellow → all-red → next_green
        (yellow = YELLOW_DUR steps,  all-red = ALLRED_DUR steps)
        _idx is committed to the new phase only after the all-red clears,
        which prevents any state mismatch between Python logic and SUMO.

    Lane discovery:
        Incoming lanes  → traci.trafficlight.getControlledLanes  (no duplicates)
        Outgoing lanes  → traci.trafficlight.getControlledLinks  [0][1]
        Lane capacity   → traci.lane.getLength / 7.5  (sumolights formula)
        Phase→lane map  → built once from getCompleteRedYellowGreenDefinition
                          (sumolights pattern: link index → from_lane)

    All of this is computed once in __init__ / _build_phase_structure() so
    step() contains no traci metadata calls — only live traffic queries.
    """

    YELLOW_DUR  = 3   # steps of yellow between any two green phases
    ALLRED_DUR  = 1   # steps of all-red after yellow (safety clearance)

    def __init__(self, tls_id: str):
        self.tls_id    = tls_id
        self._min_green = int(MIN_GREEN_STEPS)

        # ── lane lists (sumo-rl: getControlledLanes, dedup, preserve order) ──
        try:
            self.incoming: list[str] = list(
                dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id))
            )
        except Exception:
            self.incoming = list(UPSTREAM_LANES.get(tls_id, []))

        # ── outgoing lanes from getControlledLinks (sumo-rl pattern) ─────────
        try:
            links = traci.trafficlight.getControlledLinks(tls_id)
            # links[i] = [(from_lane, to_lane, via), ...]  for link index i
            self.outgoing: list[str] = list(dict.fromkeys(
                link[0][1] for link in links if link
            ))
        except Exception:
            self.outgoing = list(DOWNSTREAM_LANES.get(tls_id, []))

        # ── lane capacities: length / 7.5  (sumolights formula) ──────────────
        #    7.5 m = default vehicle length (5 m) + min gap (2.5 m)
        self._cap: dict[str, float] = {}
        for lane in self.incoming + self.outgoing:
            try:
                self._cap[lane] = max(traci.lane.getLength(lane) / 7.5, 1.0)
            except Exception:
                self._cap[lane] = float(UNIFIED_BP_CAPACITY)

        # ── green phases + phase→lane map (sumolights pattern) ────────────────
        self._green_phases: list[str]       = []  # sorted green state strings
        self._phase_lanes:  dict[str, list[str]] = {}  # state → [incoming lanes]
        self._yellow_state: str             = ""  # all-yellow mask
        self._allred_state: str             = ""  # all-red mask
        self._build_phase_structure(tls_id)

        # ── runtime state ─────────────────────────────────────────────────────
        self._phase_idx:   int  = 0     # index into self._green_phases
        self._elapsed:     int  = 0     # steps spent in current green phase
        self._transition:  str  = ""    # "" | "yellow" | "allred"
        self._trans_rem:   int  = 0     # steps remaining in current transition
        self._pending_idx: int  = 0     # green phase to commit after transition

        # ── metric accumulators ───────────────────────────────────────────────
        self.step_queues:  list[float] = []
        self.step_waits:   list[float] = []
        self.step_rewards: list[float] = []

    # ── one-time structure builder ─────────────────────────────────────────────

    def _build_phase_structure(self, tls_id: str) -> None:
        """
        Discover green phases and build the phase→incoming-lane map.

        Follows the sumolights approach:
          1. Read all phases from getCompleteRedYellowGreenDefinition.
          2. Keep only phases without 'y' that have at least one green signal.
          3. For each green phase, map signal index → from_lane via getControlledLinks.
          4. A lane is "green in phase p" if its link index carries g/G/s in p.state.
          5. Pure-green lanes (green in p, NOT red in p for any other movement)
             are preferred; fall back to any green lane if pure set is empty.
        """
        n_links = 0
        link_to_from: dict[int, str] = {}  # link_idx → from_lane

        try:
            links = traci.trafficlight.getControlledLinks(tls_id)
            n_links = len(links)
            for idx, link_tuple in enumerate(links):
                if link_tuple:
                    link_to_from[idx] = link_tuple[0][0]
        except Exception:
            pass

        all_green: list[str] = []
        try:
            logic  = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            phases = logic.getPhases()
            # Collect green phases (no yellow signal present)
            for ph in phases:
                s = ph.state
                if 'y' not in s.lower() and ('g' in s.lower() or 'G' in s):
                    all_green.append(s)
            # Build mask strings for transitions
            if phases:
                ref = phases[0].state
                self._allred_state  = 'r' * len(ref)
                self._yellow_state  = 'y' * len(ref)   # used as fallback only
        except Exception:
            pass

        # Deduplicate and sort (reproducibility across runs, matching sumo-rl)
        self._green_phases = sorted(set(all_green))

        # Fall back to FIXED_TIME_PLANS if SUMO gave nothing usable
        if not self._green_phases:
            plan = FIXED_TIME_PLANS.get(tls_id, {})
            for _dur, state in plan.get("phases", []):
                if 'y' not in state.lower() and ('g' in state.lower()):
                    if state not in self._green_phases:
                        self._green_phases.append(state)
            if not self._allred_state and self._green_phases:
                self._allred_state = 'r' * len(self._green_phases[0])

        # Build phase → lanes map (sumolights phase_lanes logic)
        incoming_set = set(self.incoming)
        for state in self._green_phases:
            green_lanes: set[str] = set()
            red_lanes:   set[str] = set()
            for link_idx, bit in enumerate(state):
                lane = link_to_from.get(link_idx)
                if lane is None or lane not in incoming_set:
                    continue
                b = bit.lower()
                if b in ('g', 's'):
                    green_lanes.add(lane)
                elif b == 'r':
                    red_lanes.add(lane)
            # Pure green: lanes that are green in this phase but NOT red
            # (i.e. they have no conflicting movement that stays red)
            pure_green = [l for l in green_lanes if l not in red_lanes]
            self._phase_lanes[state] = pure_green if pure_green else list(green_lanes)

        # If map is empty (no link data), fall back to full incoming list for all phases
        if not any(self._phase_lanes.values()):
            for state in self._green_phases:
                self._phase_lanes[state] = list(self.incoming)

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        self._phase_idx  = 0
        self._elapsed    = 0
        self._transition = ""
        self._trans_rem  = 0
        self._pending_idx = 0
        self.step_queues.clear()
        self.step_waits.clear()
        self.step_rewards.clear()

    # ── per-step interface ────────────────────────────────────────────────────

    def step(self, sim_step: int) -> None:
        """
        Advance the controller by one simulation step.

        State machine:
            GREEN  → (min_green elapsed & pressure says switch) → YELLOW → ALLRED → GREEN(new)
                   → (no switch needed)                         → GREEN (same, elapsed++)
        """
        if self._transition == "yellow":
            self._trans_rem -= 1
            if self._trans_rem <= 0:
                # Enter all-red clearance
                self._transition = "allred"
                self._trans_rem  = self.ALLRED_DUR
                self._apply_state(self._allred_state)

        elif self._transition == "allred":
            self._trans_rem -= 1
            if self._trans_rem <= 0:
                # Commit to new green phase
                self._phase_idx  = self._pending_idx
                self._elapsed    = 0
                self._transition = ""
                self._apply_green(self._phase_idx)

        else:
            # Currently in green phase — enforce it
            self._apply_green(self._phase_idx)
            self._elapsed += 1

            if self._elapsed >= self._min_green:
                best = self._best_phase()
                if best != self._phase_idx:
                    self._pending_idx = best
                    self._transition  = "yellow"
                    self._trans_rem   = self.YELLOW_DUR
                    self._apply_yellow(self._phase_idx)

        # ── metrics collection ────────────────────────────────────────────────
        q    = self._get_queue()
        wait = self._get_mean_wait()
        rew  = self._compute_reward()

        self.step_queues.append(q)
        self.step_waits.append(wait)
        self.step_rewards.append(rew)

    # ── traci state application ───────────────────────────────────────────────

    def _apply_green(self, phase_idx: int) -> None:
        if not self._green_phases:
            return
        state = self._green_phases[phase_idx % len(self._green_phases)]
        self._apply_state(state)

    def _apply_yellow(self, phase_idx: int) -> None:
        """
        Derive a yellow state from the current green phase (sumolights pattern):
          each green bit (g/G/s) → y,  red bits stay r.
        Falls back to the stored all-yellow mask if derivation fails.
        """
        if not self._green_phases:
            return
        state = self._green_phases[phase_idx % len(self._green_phases)]
        yellow = ''.join('y' if c.lower() in ('g', 's') else 'r' for c in state)
        self._apply_state(yellow)

    def _apply_state(self, state: str) -> None:
        if not state:
            return
        try:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
        except Exception:
            pass

    # ── pressure scoring ──────────────────────────────────────────────────────

    def _best_phase(self) -> int:
        if not self._green_phases:
            return self._phase_idx

        scores: list[float] = []
        for idx, state in enumerate(self._green_phases):
            lanes  = self._phase_lanes.get(state, [])
            score  = 0.0
            for lane in lanes:
                lane_cap = self._cap.get(lane, 1.0)
                q_in  = self._halting(lane) / lane_cap
                
                q_out = 0.0
                try:
                    links = traci.trafficlight.getControlledLinks(self.tls_id)
                    connected_out_lanes = []
                    for link_tuple in links:
                        if link_tuple and link_tuple[0][0] == lane:
                            connected_out_lanes.append(link_tuple[0][1])
                    
                    if connected_out_lanes:
                        q_out = sum(self._halting(out) / self._cap.get(out, 1.0) for out in connected_out_lanes) / len(connected_out_lanes)
                except Exception:
                    if self.outgoing:
                        out = self.outgoing[0]
                        q_out = self._halting(out) / self._cap.get(out, 1.0)

                score += q_in - q_out
            
            hysteresis = 0.001 if idx == self._phase_idx else 0.0
            scores.append(score + hysteresis)

        return int(np.argmax(scores))

    def _apply_yellow(self, phase_idx: int) -> None:
        if not self._green_phases:
            return
        state = self._green_phases[phase_idx % len(self._green_phases)]
        yellow = ''.join('y' if c.lower() in ('g', 's', 'o') else 'r' for c in state)
        self._apply_state(yellow)

    # ── low-level traci getters ───────────────────────────────────────────────

    def _halting(self, lane_id: str) -> float:
        try:
            return float(traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            return 0.0

    def _get_queue(self) -> float:
        """Halting vehicles: area detectors preferred, lane fallback."""
        dets = TLS_DETECTOR_MAP.get(self.tls_id, [])
        if dets:
            total = 0.0
            for d in dets:
                try:
                    total += traci.lanearea.getLastStepHaltingNumber(d)
                except Exception:
                    pass
            return total
        return sum(self._halting(l) for l in self.incoming)

    def _get_mean_wait(self) -> float:
        """Mean accumulated waiting time across all vehicles on incoming lanes."""
        total, n = 0.0, 0
        for lane_id in self.incoming:
            try:
                for veh in traci.lane.getLastStepVehicleIDs(lane_id):
                    total += traci.vehicle.getAccumulatedWaitingTime(veh)
                    n += 1
            except Exception:
                pass
        return total / n if n > 0 else 0.0

    def _compute_reward(self) -> float:
        """Back-pressure reward over all incoming lanes (same formula as RL agent)."""
        qcap    = float(UNIFIED_BP_CAPACITY)
        n_pairs = max(len(self.incoming), 1)
        pressure = 0.0
        for k, lane in enumerate(self.incoming):
            qi = self._halting(lane) / qcap
            out_lane = self.outgoing[k % len(self.outgoing)] if self.outgoing else None
            qo = (self._halting(out_lane) / qcap) if out_lane else 0.0
            pressure += qi - qo
        return float(np.clip(-(pressure / n_pairs), -REWARD_CLIP, REWARD_CLIP))


# ==============================================================================
#  EVALUATION RICH UI
# ==============================================================================

class EvalRichUI:
    """
    Rich terminal UI — matches the Live Metrics screenshot layout.

    Columns (compare mode):
      TLS | RL Reward | RL Queue | RL Wait(s) | RL Loss |
      FT BP-Score | FT Queue | FT Wait(s) |
      Q Δ% | W Δ% |
      RL CO₂ | FT CO₂ | CO₂ Δ% |
      RL Fuel | FT Fuel | Fuel Δ%
    """

    _TL_STYLE = {
        "TL1": "bold dodger_blue2",
        "TL2": "bold green3",
        "TL3": "bold indian_red1",
        "TL4": "bold medium_purple1",
    }

    def __init__(self, sim_max_steps: int, mode: str, intersections: list,
                 algorithm: str = "mappo"):
        self.sim_max_steps = sim_max_steps
        self.mode          = mode
        self.algorithm     = algorithm
        self.intersections = intersections
        self._step         = 0
        self._phase        = "init"
        self._episode      = 10
        self._total_ep     = 0
        self._start_t      = time.time()
        self._best_ep      = 0
        self._best_r       = 0.0

        self._rl: dict[str, dict] = {
            tl: {"r": 0.0, "q": 0.0, "w": 0.0, "loss": 0.0, "co2": 0.0, "fuel": 0.0}
            for tl in intersections
        }
        self._ft: dict[str, dict] = {
            tl: {"r": 0.0, "q": 0.0, "w": 0.0, "co2": 0.0, "fuel": 0.0}
            for tl in intersections
        }

        if not RICH_AVAILABLE:
            self._console   = None
            self._progress  = None
            self._live      = None
            self._step_task = None
            return

        self._console = Console(highlight=False)

        _pkw: dict = dict(console=self._console, expand=True)
        try:
            import inspect as _i
            if "refresh_per_second" in _i.signature(Progress.__init__).parameters:
                _pkw["refresh_per_second"] = 8
        except Exception:
            pass

        try:
            _bar = BarColumn(
                bar_width=None,
                style="bar.back",
                complete_style="bold cyan",
                finished_style="bold green",
                pulse_style="bold magenta",
            )
        except TypeError:
            _bar = BarColumn()

        self._progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold italic cyan]{task.description}"),
            _bar,
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            **_pkw,
        )
        self._step_task = self._progress.add_task(
            "Live Metrics", total=sim_max_steps)

        _lkw: dict = dict(console=self._console, screen=False)
        try:
            import inspect as _i
            _ls = _i.signature(Live.__init__).parameters
            if "refresh_per_second" in _ls:
                _lkw["refresh_per_second"] = 6
            if "transient" in _ls:
                _lkw["transient"] = False
        except Exception:
            pass

        self._live = Live(self._build_layout(), **_lkw)

    # ── lifecycle ─────────────────────────────────────────────────────────────

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
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ── updates ───────────────────────────────────────────────────────────────

    def set_phase(self, phase: str):
        self._phase = phase
        self._step  = 0
        if self._progress and self._step_task is not None:
            self._progress.reset(self._step_task)
        self._refresh()

    def update_step(self, step: int):
        self._step = step
        if self._progress and self._step_task is not None:
            desc = (
                f"Live Metrics -- Episode {self._episode:>3d}/{self._total_ep}"
                if self._total_ep else "Live Metrics"
            )
            self._progress.update(self._step_task, completed=step, description=desc)

    def set_episode(self, episode: int, total: int, best_ep: int = 0, best_r: float = 0.0):
        self._episode  = episode
        self._total_ep = total
        self._best_ep  = best_ep
        self._best_r   = best_r

    def log(self, msg: str, style: str = "white"):
        if self._console:
            self._console.print(f"[{style}]  {msg}[/]")

    def record_rl(self, metrics: dict):
        for tl in self.intersections:
            self._rl[tl] = {
                "r":    metrics["per_tl_reward"].get(tl, 0.0),
                "q":    metrics["per_tl_queue"].get(tl, 0.0),
                "w":    metrics["per_tl_wait"].get(tl, 0.0),
                "loss": metrics.get("per_tl_loss", {}).get(tl, 0.0),
                "co2":  metrics.get("per_tl_co2",  {}).get(tl, 0.0),
                "fuel": metrics.get("per_tl_fuel", {}).get(tl, 0.0),
            }
        self._refresh()

    def record_wp(self, controllers: dict):
        for tl, ctrl in controllers.items():
            self._ft[tl] = {
                "r":    float(np.mean(ctrl.step_rewards)) if ctrl.step_rewards else 0.0,
                "q":    float(np.mean(ctrl.step_queues))  if ctrl.step_queues  else 0.0,
                "w":    float(np.mean(ctrl.step_waits))   if ctrl.step_waits   else 0.0,
                "co2":  0.0,
                "fuel": 0.0,
            }
        self._refresh()

    # ── layout building ───────────────────────────────────────────────────────

    def _print_header(self):
        if not self._console:
            return
        algo_label = "IPPO" if self.algorithm == "ippo" else "MAPPO-PIS+GAT"
        self._console.print(Rule(style="dim white"))
        self._console.print(
            f"[bold cyan]{algo_label}[/]  [dim white]|[/]  [bold white]Evaluation Session[/]"
        )
        self._console.print(
            f"Mode: [bold yellow]{self.mode}[/]   "
            f"Intersections: [bold white]{len(self.intersections)}[/]"
        )
        self._console.print(Rule(style="dim white"))

    def _pct(self, val: float, ref: float) -> float:
        if abs(ref) < 0.01:
            return 0.0
        return 100.0 * (ref - val) / abs(ref)

    def _delta_text(self, val: float, ref: float, lower_is_better: bool = True) -> Text:
        pct = self._pct(val, ref) if lower_is_better else -self._pct(val, ref)
        colour = "bold green3" if pct >= 0 else "bold indian_red1"
        arrow  = "▲" if pct >= 0 else "▼"
        return Text(f"{arrow}{abs(pct):.1f}%", style=colour)

    def _build_metrics_table(self):
        mode = self.mode

        tbl = Table(
            box=box.SIMPLE_HEAD,
            border_style="dim white",
            header_style="bold white",
            show_edge=False,
            padding=(0, 1),
            expand=True,
        )

        tbl.add_column("TLS",        style="bold white",     width=5,  no_wrap=True)

        if mode in ("rl", "compare"):
            tbl.add_column("RL Reward",  justify="right", style="indian_red1",  width=11)
            tbl.add_column("RL Queue",   justify="right", style="cyan",          width=10)
            tbl.add_column("RL\nWait(s)",justify="right", style="cyan",          width=10)

        if mode in ("webster", "compare"):
            tbl.add_column("FT Queue",     justify="right", style="yellow3",    width=10)
            tbl.add_column("FT\nWait(s)",  justify="right", style="yellow3",    width=10)

        if mode == "compare":
            tbl.add_column("Q Δ%",    justify="right", width=9)
            tbl.add_column("W Δ%",    justify="right", width=9)

        def _build_row(tl: str) -> list:
            rl = self._rl.get(tl, {})
            ft = self._ft.get(tl, {})
            row: list = [Text(tl, style=self._TL_STYLE.get(tl, "bold white"))]

            if mode in ("rl", "compare"):
                r_style = "bold indian_red1" if rl.get("r", 0) < 0 else "bold green3"
                row += [
                    Text(f"{rl.get('r', 0.0):.4f}", style=r_style),
                    Text(f"{rl.get('q', 0.0):.2f}", style="cyan"),
                    Text(f"{rl.get('w', 0.0):.2f}", style="cyan"),
                ]

            if mode in ("webster", "compare"):
                row += [
                    Text(f"{ft.get('q', 0.0):.2f}", style="yellow3"),
                    Text(f"{ft.get('w', 0.0):.2f}", style="yellow3"),
                ]

            if mode == "compare":
                row.append(self._delta_text(rl.get("q", 0), ft.get("q", 0), lower_is_better=True))
                row.append(self._delta_text(rl.get("w", 0), ft.get("w", 0), lower_is_better=True))

            return row

        for tl in self.intersections:
            tbl.add_row(*_build_row(tl))

        def _avg(d: dict, k: str) -> float:
            return float(np.mean([d.get(t, {}).get(k, 0.0) for t in self.intersections]))

        tbl.add_section()

        avg_rl = {k: _avg(self._rl, k) for k in ("r", "q", "w", "loss", "co2", "fuel")}
        avg_ft = {k: _avg(self._ft, k) for k in ("r", "q", "w", "co2", "fuel")}

        avg_row: list = [Text("AVG", style="bold white")]

        if mode in ("rl", "compare"):
            r_style = "bold indian_red1" if avg_rl["r"] < 0 else "bold green3"
            avg_row += [
                Text(f"{avg_rl['r']:.4f}",    style=r_style),
                Text(f"{avg_rl['q']:.2f}",    style="bold cyan"),
                Text(f"{avg_rl['w']:.2f}",    style="bold cyan"),
            ]

        if mode in ("webster", "compare"):
            avg_row += [
                Text(f"{avg_ft['q']:.2f}", style="bold yellow3"),
                Text(f"{avg_ft['w']:.2f}", style="bold yellow3"),
            ]

        if mode == "compare":
            avg_row.append(self._delta_text(avg_rl["q"],    avg_ft["q"],    lower_is_better=True))
            avg_row.append(self._delta_text(avg_rl["w"],    avg_ft["w"],    lower_is_better=True))

        tbl.add_row(*avg_row)
        return tbl

    def _build_status_panel(self):
        phase_labels = {
            "init": ("[bold cyan]●[/] Initialising …",              "cyan"),
            "wp":   ("[bold green3]▶[/] Running FT baseline …",     "green"),
            "rl":   ("[bold cyan]▶[/] Running RL episode …",        "cyan"),
            "done": ("[bold green]✓[/] Evaluation complete!",        "green"),
        }
        label, border = phase_labels.get(self._phase, ("…", "white"))
        elapsed = time.time() - self._start_t
        h  = int(elapsed // 3600)
        m  = int((elapsed % 3600) // 60)
        s  = int(elapsed % 60)
        ep_str = (
            f"{self._episode}/{self._total_ep}" if self._total_ep else "--"
        )
        avg_eps = self._episode / elapsed if elapsed > 1 and self._episode else 0.0

        stats = Table.grid(padding=(0, 3))
        stats.add_column(style="dim white", width=22)
        stats.add_column(style="white")
        stats.add_row("Status",               label)
        stats.add_row("Elapsed",              f"{h}h {m:02d}m {s:02d}s")
        stats.add_row("ETA",                  "--")
        stats.add_row("Reward",               "BackPressure")
        stats.add_row("Best Ep (avg reward)", f"{self._best_ep}  r={self._best_r:.4f}")
        stats.add_row("Avg ep/s",             f"{avg_eps:.2f} ep/s")

        return Panel(
            stats,
            title="[bold white] Evaluation Status ",
            border_style=border,
            expand=False,
            padding=(0, 2),
        )

    def _build_layout(self):
        if RGroup is not None:
            return RGroup(
                self._progress,
                self._build_metrics_table(),
                self._build_status_panel(),
            )
        return self._progress

    def _refresh(self):
        if self._live is None:
            return
        try:
            self._live.update(self._build_layout())
            self._live.refresh()
        except Exception:
            pass


# ==============================================================================
#  WEBSTER-PRESSURE EPISODE RUNNER
# ==============================================================================

def _run_webster_episode(sumo_cmd: list, max_steps: int, port: int,
                         ui: EvalRichUI | None = None) -> dict:
    """
    Run one SUMO episode with Webster-Pressure controllers.

    Returns a dict compatible with `_run_evaluation_episode` output:
      avg_reward, avg_queue, avg_wait, per_tl_reward/queue/wait,
      cars_passed, ev_cleared, ev_avg_wait.
    """
    controllers = {tl: WebsterPressureController(tl) for tl in INTERSECTIONS}
    for ctrl in controllers.values():
        ctrl.reset()

    proc = _start_sumo(sumo_cmd, port=port)
    cars_passed = 0

    try:
        # Subscribe lanes & detectors
        all_lanes = set()
        for tl in INTERSECTIONS:
            all_lanes.update(UPSTREAM_LANES.get(tl, []))
        for lane in all_lanes:
            try:
                traci.lane.subscribe(lane, [
                    traci.constants.LAST_STEP_VEHICLE_ID_LIST,
                    traci.constants.LAST_STEP_HALTING_NUMBER,
                    traci.constants.LAST_STEP_VEHICLE_NUMBER,
                ])
            except Exception:
                pass
        all_dets = set()
        for dets in TLS_DETECTOR_MAP.values():
            all_dets.update(dets)
        for det in all_dets:
            try:
                traci.lanearea.subscribe(det, [
                    traci.constants.LAST_STEP_VEHICLE_HALTING_NUMBER,
                ])
            except Exception:
                pass

        for step in range(1, max_steps + 1):
            if step % 50 == 0:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        if ui:
                            ui.log("FT: early stop (network empty)", "dim yellow3")
                        break
                except Exception:
                    pass

            if step % 10 == 0 and ui:
                ui.update_step(step)

            for ctrl in controllers.values():
                ctrl.step(step)

            traci.simulationStep()

            try:
                cars_passed += traci.simulation.getArrivedNumber()
            except Exception:
                pass

    except Exception as e:
        if ui:
            ui.log(f"[FT] Episode error: {e}", "bold red")
        else:
            print(f"[FT] Episode error: {e}", flush=True)
    finally:
        try:
            _close_sumo(proc)
        except Exception:
            pass

    # ── aggregate metrics ────────────────────────────────────────────────────
    per_tl_reward, per_tl_queue, per_tl_wait = {}, {}, {}
    for tl, ctrl in controllers.items():
        per_tl_reward[tl] = float(np.mean(ctrl.step_rewards)) if ctrl.step_rewards else 0.0
        per_tl_queue[tl]  = float(np.mean(ctrl.step_queues))  if ctrl.step_queues  else 0.0
        per_tl_wait[tl]   = float(np.mean(ctrl.step_waits))   if ctrl.step_waits   else 0.0

    return {
        "avg_reward":    float(np.mean(list(per_tl_reward.values()))),
        "avg_queue":     float(np.mean(list(per_tl_queue.values()))),
        "avg_wait":      float(np.mean(list(per_tl_wait.values()))),
        "per_tl_reward": per_tl_reward,
        "per_tl_queue":  per_tl_queue,
        "per_tl_wait":   per_tl_wait,
        "cars_passed":   cars_passed,
        "ev_cleared":    0,
        "ev_avg_wait":   None,
        "co2_total":     0.0,
        "nox_total":     0.0,
        "fuel_total":    0.0,
    }, controllers   # also return controllers so caller can inspect step lists


# ==============================================================================
#  IPPO EVALUATION EPISODE RUNNER
# ==============================================================================

def _run_ippo_evaluation_episode(
    sumo_cmd:      list,
    agents:        dict,
    sim_max_steps: int,
    port:          int,
    deterministic: bool = True,
    step_callback=None,
) -> dict:
    """Run one SUMO episode evaluating IPPO agents (no intent sharing)."""

    _saved_state = {}
    for tid, ag in agents.items():
        _saved_state[tid] = {
            "state":              ag.state.copy(),
            "current_action_idx": ag.current_action_idx,
            "last_switch_step":   ag.last_switch_step,
            "_current_step":      ag._current_step,
            "_ev_phi_prev":       ag._ev_phi_prev,
            "_ev_phi_curr":       ag._ev_phi_curr,
        }

    for ag in agents.values():
        ag.last_switch_step   = -ag.min_green_steps
        ag.current_action_idx = 0
        ag._current_step      = 0
        ag._ev_phi_prev = 0.0
        ag._ev_phi_curr = 0.0

    IPPOAgent.clear_lane_length_cache()
    _reset_ep_stats()

    eval_step_rewards: dict[str, list] = {tid: [] for tid in agents}
    eval_step_queues:  dict[str, list] = {tid: [] for tid in agents}
    eval_step_waits:   dict[str, list] = {tid: [] for tid in agents}

    proc = None
    try:
        proc = _start_sumo(sumo_cmd, port=port)
        _setup_all_subscriptions(agents)

        for ag in agents.values():
            try:
                ag.state = ag._get_state()
            except Exception:
                ag.state = np.zeros(ag.state_dim, dtype=np.float32)

        for step in range(1, sim_max_steps + 1):
            if step % 50 == 0:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        break
                except Exception:
                    pass

            if step_callback is not None and step % 10 == 0:
                try:
                    step_callback(step)
                except Exception:
                    pass

            _subscribe_vehicles_bulk()
            _STEP_CACHE.clear()

            for tid, ag in agents.items():
                action = ag.choose_action(deterministic=deterministic)
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
        print(f"  [EVAL-IPPO] Warning: episode error: {exc}", flush=True)
    finally:
        if proc is not None:
            _collect_final_emissions()
            _close_sumo(proc)

    for tid, ag in agents.items():
        sv = _saved_state[tid]
        ag.state              = sv["state"]
        ag.current_action_idx = sv["current_action_idx"]
        ag.last_switch_step   = sv["last_switch_step"]
        ag._current_step      = sv["_current_step"]
        ag._ev_phi_prev       = sv["_ev_phi_prev"]
        ag._ev_phi_curr       = sv["_ev_phi_curr"]

    ep_stats = _snapshot_ep_stats()

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

    ev_avg_wait = None
    if ep_stats["ev_cleared"] > 0:
        ev_avg_wait = round(ep_stats["ev_wait_sum"] / ep_stats["ev_cleared"], 3)

    return {
        "avg_reward":    float(np.mean(list(per_tl_reward.values()))),
        "avg_queue":     float(np.mean(list(per_tl_queue.values()))),
        "avg_wait":      float(np.mean(list(per_tl_wait.values()))),
        "per_tl_reward": per_tl_reward,
        "per_tl_queue":  per_tl_queue,
        "per_tl_wait":   per_tl_wait,
        "cars_passed":   ep_stats["cars_passed"],
        "ev_cleared":    ep_stats["ev_cleared"],
        "ev_avg_wait":   ev_avg_wait,
        "co2_total":     round(ep_stats["co2_sum"],  2),
        "nox_total":     round(ep_stats["nox_sum"],  2),
        "fuel_total":    round(ep_stats["fuel_sum"], 2),
    }


# ==============================================================================
#  CHECKPOINT PATH RESOLVER  (handles full paths, latest.json, episode dirs …)
# ==============================================================================

import re as _re

_EP_DIR_RE = _re.compile(r"^ep(\d+)$")


def _resolve_checkpoint_arg(
    ckpt_path: Path,
    save_dir:  str,
    agents:    dict,
) -> tuple[str, str]:
    """Resolve an arbitrary --checkpoint value to (save_dir, episode_tag).

    Handles all of these:
      1. Plain tag            "ep0283"  or  "latest"
      2. Full path to latest.json
      3. Full path to any file inside an ep dir  (e.g. metrics.json, TL1.pt)
      4. Full path to an ep dir                   (e.g. …/checkpoints/ep0283)
      5. Full path to the checkpoints dir          (e.g. …/checkpoints)
      6. Full path to a single .pt file
    """

    sample_tls = next(iter(agents), "TL1") if agents else "TL1"

    # ── helpers ────────────────────────────────────────────────────────────
    def _read_latest_ep(json_path: Path) -> int | None:
        """Return latest_ep from a JSON pointer file, or None."""
        try:
            data = json.loads(json_path.read_text())
            return int(data["latest_ep"])
        except Exception:
            return None

    def _ep_from_name(name: str) -> str | None:
        """Return 'ep0283' if name matches epNNNN, else None."""
        m = _EP_DIR_RE.match(name)
        return name if m else None

    def _resolve_dir_as_ckpt_dir(d: Path) -> tuple[str, str]:
        """Given a checkpoints/ directory, find the latest episode."""
        # try latest.json inside it
        lj = d / "latest.json"
        if lj.is_file():
            ep = _read_latest_ep(lj)
            if ep is not None:
                tag = f"ep{ep:04d}"
                print(f"  [CKPT] latest.json resolved -> {tag}", flush=True)
                return str(d.parent), tag
        # fallback: scan for ep dirs
        ep_dirs = sorted(
            [x.name for x in d.iterdir()
             if x.is_dir() and _EP_DIR_RE.match(x.name)],
        )
        if ep_dirs:
            tag = ep_dirs[-1]
            print(f"  [CKPT] Scanned dir -> {tag}", flush=True)
            return str(d.parent), tag
        # fallback: scan for flat TL_*.pt files
        flat = sorted(d.glob(f"{sample_tls}_ep*.pt"))
        if flat:
            m = _EP_DIR_RE.search(flat[-1].stem)
            tag = m.group(0) if m else flat[-1].stem
            print(f"  [CKPT] Scanned flat files -> {tag}", flush=True)
            return str(d.parent), tag
        return str(d.parent), "latest"

    # -- Case 1: not a real path -> treat as plain tag
    if not ckpt_path.exists():
        print(f"  [CKPT] Path not found, treating '{ckpt_path}' as tag", flush=True)
        return save_dir, str(ckpt_path)

    # ── Case 2: .json pointer file ────────────────────────────────────────
    if ckpt_path.suffix == ".json" and ckpt_path.is_file():
        ep = _read_latest_ep(ckpt_path)
        if ep is not None:
            tag = f"ep{ep:04d}"
            # latest.json lives inside checkpoints/
            parent = ckpt_path.parent          # checkpoints/
            print(f"  [CKPT] latest.json resolved -> {tag}", flush=True)
            return str(parent.parent), tag
        else:
            # some other .json -- check if parent dir looks like an ep dir
            parent = ckpt_path.parent
            tag = _ep_from_name(parent.name)
            if tag:
                print(f"  [CKPT] File in ep dir -> {tag}", flush=True)
                return str(parent.parent.parent), tag
            # check grandparent (file might be in a subdir of ep dir)
            gp = parent.parent
            tag = _ep_from_name(gp.name)
            if tag:
                print(f"  [CKPT] File in ep subdir -> {tag}", flush=True)
                return str(gp.parent.parent), tag
            # give up — fall through
            print(f"  [CKPT] WARNING: Cannot extract episode from {ckpt_path}", flush=True)
            return save_dir, str(ckpt_path)

    # ── Case 3: single .pt file ───────────────────────────────────────────
    if ckpt_path.suffix == ".pt" and ckpt_path.is_file():
        parent = ckpt_path.parent
        tag = _ep_from_name(parent.name)
        if tag:
            print(f"  [CKPT] .pt in ep dir -> {tag}", flush=True)
            return str(parent.parent.parent), tag
        print(f"  [CKPT] Single .pt file -> {ckpt_path}", flush=True)
        return str(parent.parent), ckpt_path.stem

    # ── Case 4: directory ─────────────────────────────────────────────────
    if ckpt_path.is_dir():
        # Is this an ep dir?  (contains TL1.pt directly)
        if (ckpt_path / f"{sample_tls}.pt").exists():
            # Check if there's a 'checkpoints' dir above this ep dir
            parent_has_checkpoints = (ckpt_path.parent / "checkpoints").is_dir()
            if parent_has_checkpoints:
                # Standard layout: save_dir/checkpoints/epXXXX/
                print(f"  [CKPT] EpDir -> {ckpt_path.name}", flush=True)
                return str(ckpt_path.parent.parent), ckpt_path.name
            else:
                # Direct path to ep dir -- create a temp structure for load_checkpoint
                import tempfile, shutil as _shutil
                _tmp = tempfile.mkdtemp(prefix="ckpt_epdir_")
                _ck = os.path.join(_tmp, "checkpoints", ckpt_path.name)
                os.makedirs(_ck, exist_ok=True)
                for pt in ckpt_path.glob("*.pt"):
                    _shutil.copy2(str(pt), os.path.join(_ck, pt.name))
                print(f"  [CKPT] EpDir (direct) -> {ckpt_path.name} [tmp={_tmp}]", flush=True)
                return _tmp, ckpt_path.name

        # Is this the checkpoints/ dir itself?
        if (ckpt_path / "latest.json").is_file() or list(ckpt_path.glob(f"{sample_tls}_ep*.pt")):
            return _resolve_dir_as_ckpt_dir(ckpt_path)

        # Is its name an ep dir name?
        tag = _ep_from_name(ckpt_path.name)
        if tag:
            print(f"  [CKPT] Named ep dir -> {tag}", flush=True)
            return str(ckpt_path.parent.parent), tag

        # fallback: treat dir as checkpoints dir
        return _resolve_dir_as_ckpt_dir(ckpt_path)

    # ── fallback ──────────────────────────────────────────────────────────
    return save_dir, str(ckpt_path)


# ==============================================================================
#  MAIN EVALUATION FUNCTION
# ==============================================================================

def _average_metrics(all_metrics: list[dict]) -> dict:
    """Average a list of metric dicts (from multiple episodes)."""
    if not all_metrics:
        return {}
    if len(all_metrics) == 1:
        return all_metrics[0]

    result = {}
    # top-level scalar keys
    scalar_keys = [k for k in all_metrics[0] if not k.startswith("per_tl_")
                   and isinstance(all_metrics[0].get(k), (int, float))]
    for k in scalar_keys:
        vals = [m[k] for m in all_metrics if k in m and isinstance(m[k], (int, float))]
        result[k] = round(sum(vals) / len(vals), 6) if vals else 0

    # per_tl_ dicts
    for k in all_metrics[0]:
        if not k.startswith("per_tl_"):
            continue
        d0 = all_metrics[0].get(k, {})
        if not isinstance(d0, dict):
            continue
        merged = {}
        for tl in d0:
            vals = [m.get(k, {}).get(tl, 0) for m in all_metrics if isinstance(m.get(k), dict)]
            merged[tl] = round(sum(vals) / len(vals), 6) if vals else 0
        result[k] = merged

    # preserve non-numeric keys from first entry
    for k in all_metrics[0]:
        if k not in result:
            result[k] = all_metrics[0][k]
    return result


def _resolve_ckpt(ckpt_arg: str, fallback_save: str, agents: dict) -> tuple[str, str]:
    """Resolve a single --checkpoint / --ckpt_dir value -> (save_dir, tag)."""
    p = Path(ckpt_arg).expanduser().resolve()
    if p.is_file() or p.is_dir():
        return _resolve_checkpoint_arg(p, fallback_save, agents)
    return fallback_save, ckpt_arg


def run_evaluation(
    sim_file:      str,
    save_dir:      str,
    max_steps:     int,
    gui:           str,
    checkpoint:    str  = "latest",
    deterministic: bool = True,
    report:        bool = False,
    open_report:   bool = False,
    mode:          str  = "compare",
    ckpt_dir:      str | None = None,
    algorithm:     str  = "mappo",
    ckpt_ippo:     str | None = None,
    ckpt_mappo:    str | None = None,
    excel:         bool = False,
    num_episodes:  int  = 1,
    seed:          int | None = None,
) -> dict:
    """Run evaluation in the requested mode and return a results dict.

    mode
    ----
    "rl"            - RL only
    "compare"       - Webster-Pressure + RL  (default)
    "webster"       - Webster-Pressure only
    "algo_compare"  - IPPO vs MAPPO_hierGAT head-to-head (no Webster)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── build base SUMO command ───────────────────────────────────────────────
    sumo_binary = "sumo-gui" if gui == "enable" else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", sim_file,
        "--step-length", str(int(STEP_LENGTH)),
        "--no-step-log",
        "--time-to-teleport", "100",
        "--collision.action", "warn",
    ]
    if seed is not None:
        base_seed = seed
    else:
        base_seed = None

    # ── initialise agents ─────────────────────────────────────────────────────
    rl_metrics:   dict | None = None
    ippo_metrics: dict | None = None
    mappo_metrics:dict | None = None
    wp_metrics:   dict | None = None
    wp_controllers: dict      = {}

    ippo_agents:  dict = {}
    mappo_agents: dict = {}
    agents:       dict = {}          # used for single-algo modes

    need_rl = mode in ("rl", "compare", "algo_compare") or algorithm == "both"

    if need_rl:
        import traci as _traci
        proc = _start_sumo(["sumo", "-c", sim_file], port=BASE_PORT)
        tls_ids = list(_traci.trafficlight.getIDList())

        if algorithm == "both" or mode == "algo_compare":
            print(f"Loading IPPO + MAPPO_hierGAT agents ...", flush=True)
            for tls_id in tls_ids:
                ippo_agents[tls_id]  = IPPOAgent(tls_id, device)
                mappo_agents[tls_id] = MAPPOAgent(tls_id, device)
        elif algorithm == "ippo":
            print(f"Loading agents (algorithm=ippo) ...", flush=True)
            for tls_id in tls_ids:
                agents[tls_id] = IPPOAgent(tls_id, device)
        else:
            print(f"Loading agents (algorithm=mappo) ...", flush=True)
            for tls_id in tls_ids:
                agents[tls_id] = MAPPOAgent(tls_id, device)

        _close_sumo(proc)

        # -- resolve & load checkpoints -------------------------------------------
        if algorithm == "both" or mode == "algo_compare":
            # --- IPPO checkpoint ---
            ippo_src = ckpt_ippo or checkpoint
            ippo_save, ippo_tag = _resolve_ckpt(ippo_src, save_dir, ippo_agents)
            ep_i = load_checkpoint(ippo_agents, ippo_save, episode=ippo_tag)
            print(f"  [IPPO] Loaded checkpoint from episode {ep_i}", flush=True)

            # --- MAPPO checkpoint ---
            mappo_src = ckpt_mappo or checkpoint
            mappo_save, mappo_tag = _resolve_ckpt(mappo_src, save_dir, mappo_agents)
            ep_m = load_checkpoint(mappo_agents, mappo_save, episode=mappo_tag)
            print(f"  [MAPPO] Loaded checkpoint from episode {ep_m}", flush=True)
        else:
            # single-algorithm path (original logic)
            if ckpt_dir is not None:
                ckpt_dir_abs = os.path.abspath(ckpt_dir)
                _sample_tls = next(iter(agents), "TL1")
                _base_name  = os.path.basename(ckpt_dir_abs)
                _parent_dir = os.path.dirname(ckpt_dir_abs)
                _cleanup_tmp = None

                _has_pt_direct = os.path.isfile(
                    os.path.join(ckpt_dir_abs, f"{_sample_tls}.pt"))
                _has_pt_flat = os.path.isfile(
                    os.path.join(_parent_dir, f"{_sample_tls}_{_base_name}.pt"))

                if _has_pt_direct:
                    _eff_save = os.path.dirname(_parent_dir)
                    _eff_tag  = _base_name
                elif _has_pt_flat:
                    _eff_save = os.path.dirname(_parent_dir)
                    _eff_tag  = _base_name
                else:
                    _eff_save = os.path.dirname(ckpt_dir_abs)
                    _eff_tag  = checkpoint
            else:
                _cleanup_tmp = None
                _ckpt_path   = Path(checkpoint).expanduser().resolve()
                _eff_save, _eff_tag = _resolve_checkpoint_arg(
                    _ckpt_path, save_dir, agents,
                )

            try:
                loaded_ep = load_checkpoint(agents, _eff_save, episode=_eff_tag)
            finally:
                if "_cleanup_tmp" in locals() and _cleanup_tmp and os.path.isdir(_cleanup_tmp):
                    import shutil
                    shutil.rmtree(_cleanup_tmp, ignore_errors=True)

            if loaded_ep == 0:
                print("Warning: No checkpoint found – evaluating with random weights.", flush=True)
            else:
                print(f"Loaded checkpoint from episode {loaded_ep}", flush=True)

    # ── Rich UI ───────────────────────────────────────────────────────────────
    ui_label = algorithm if algorithm != "both" else "IPPO + MAPPO_hierGAT"
    if mode == "algo_compare":
        ui_label = "IPPO vs MAPPO_hierGAT"
    ui = EvalRichUI(max_steps, mode, INTERSECTIONS, algorithm=ui_label)

    wp_all:   list[dict] = []
    ippo_all: list[dict] = []
    mappo_all: list[dict] = []
    rl_all:   list[dict] = []

    with ui:
        # ── Webster-Pressure phase (Run First) ────────────────────────────────
        if mode in ("compare", "webster"):
            wp_all = []
            for ep_idx in range(num_episodes):
                ep_seed = base_seed + ep_idx if base_seed is not None else None
                ep_cmd = sumo_cmd + (["--seed", str(ep_seed)] if ep_seed is not None else [])
                ui.set_phase("wp")
                label = f"FT episode {ep_idx+1}/{num_episodes}" if num_episodes > 1 else "FT baseline episode"
                ui.log(f"Starting {label} ...", "bold yellow3")
                wp_result = _run_webster_episode(
                    ep_cmd, max_steps, BASE_PORT + 3, ui=ui,
                )
                wp_m, wp_c = wp_result
                wp_all.append(wp_m)
                ui.record_wp(wp_c)
                ui.log(
                    f"FT ep{ep_idx+1} done  reward={wp_m['avg_reward']:+.4f}"
                    f"  queue={wp_m['avg_queue']:.2f}"
                    f"  wait={wp_m['avg_wait']:.2f}s"
                    f"  cars={wp_m['cars_passed']}",
                    "yellow3",
                )
            wp_metrics = _average_metrics(wp_all)

        # ── RL phase ──────────────────────────────────────────────────────────
        if algorithm == "both" or mode == "algo_compare":
            ippo_all = []
            mappo_all = []
            for ep_idx in range(num_episodes):
                ep_seed = base_seed + ep_idx if base_seed is not None else None
                ep_cmd = sumo_cmd + (["--seed", str(ep_seed)] if ep_seed is not None else [])

                # Run IPPO episode
                ui.set_phase("rl")
                label = f"IPPO episode {ep_idx+1}/{num_episodes}" if num_episodes > 1 else "IPPO evaluation episode"
                ui.log(f"Starting {label} ...", "bold cyan")
                im = _run_ippo_evaluation_episode(
                    ep_cmd, ippo_agents, max_steps,
                    BASE_PORT + 1, deterministic=deterministic,
                    step_callback=ui.update_step,
                )
                ippo_all.append(im)
                ui.record_rl(im)
                ui.log(
                    f"IPPO ep{ep_idx+1} done  reward={im['avg_reward']:+.4f}"
                    f"  queue={im['avg_queue']:.2f}"
                    f"  wait={im['avg_wait']:.2f}s"
                    f"  cars={im['cars_passed']}",
                    "cyan",
                )

                # Run MAPPO episode
                ui.set_phase("rl")
                label = f"MAPPO episode {ep_idx+1}/{num_episodes}" if num_episodes > 1 else "MAPPO_hierGAT evaluation episode"
                ui.log(f"Starting {label} ...", "bold magenta")
                mm = _run_evaluation_episode(
                    ep_cmd, mappo_agents, max_steps,
                    BASE_PORT + 2, deterministic=deterministic,
                    step_callback=ui.update_step,
                )
                mappo_all.append(mm)
                ui.record_rl(mm)
                ui.log(
                    f"MAPPO ep{ep_idx+1} done  reward={mm['avg_reward']:+.4f}"
                    f"  queue={mm['avg_queue']:.2f}"
                    f"  wait={mm['avg_wait']:.2f}s"
                    f"  cars={mm['cars_passed']}",
                    "magenta",
                )
            ippo_metrics  = _average_metrics(ippo_all)
            mappo_metrics = _average_metrics(mappo_all)

        elif mode in ("rl", "compare"):
            rl_all = []
            for ep_idx in range(num_episodes):
                ep_seed = base_seed + ep_idx if base_seed is not None else None
                ep_cmd = sumo_cmd + (["--seed", str(ep_seed)] if ep_seed is not None else [])
                ui.set_phase("rl")
                label = f"RL episode {ep_idx+1}/{num_episodes}" if num_episodes > 1 else "RL evaluation episode"
                ui.log(f"Starting {label} ...", "bold cyan")
                if algorithm == "ippo":
                    rm = _run_ippo_evaluation_episode(
                        ep_cmd, agents, max_steps,
                        BASE_PORT + 1, deterministic=deterministic,
                        step_callback=ui.update_step,
                    )
                else:
                    rm = _run_evaluation_episode(
                        ep_cmd, agents, max_steps,
                        BASE_PORT + 1, deterministic=deterministic,
                        step_callback=ui.update_step,
                    )
                rl_all.append(rm)
                ui.record_rl(rm)
                ui.log(
                    f"RL ep{ep_idx+1} done  reward={rm['avg_reward']:+.4f}"
                    f"  queue={rm['avg_queue']:.2f}"
                    f"  wait={rm['avg_wait']:.2f}s"
                    f"  cars={rm['cars_passed']}",
                    "cyan",
                )
            rl_metrics = _average_metrics(rl_all)

        ui.set_phase("done")

    # ── plain-text summary ────────────────────────────────────────────────────
    _ep_label = f" (avg over {num_episodes} episodes)" if num_episodes > 1 else ""
    if algorithm == "both" or mode == "algo_compare":
        print(f"\n=== Evaluation Results (IPPO vs MAPPO_hierGAT){_ep_label} ===")
        for tag, m in [("Webster", wp_metrics), ("IPPO", ippo_metrics), ("MAPPO_hierGAT", mappo_metrics)]:
            if m:
                print(f"\n[{tag}]")
                print(f"  Average Reward : {m['avg_reward']:+.4f}")
                print(f"  Average Queue  : {m['avg_queue']:.4f}")
                print(f"  Average Wait   : {m['avg_wait']:.4f} s")
                print(f"  Cars Passed    : {m['cars_passed']}")
        # Pairwise comparisons
        pairs = [
            ("IPPO", ippo_metrics, "MAPPO_hierGAT", mappo_metrics),
        ]
        if wp_metrics:
            pairs.insert(0, ("IPPO", ippo_metrics, "Webster", wp_metrics))
            pairs.insert(1, ("MAPPO_hierGAT", mappo_metrics, "Webster", wp_metrics))
        for a_name, a_m, b_name, b_m in pairs:
            if a_m and b_m:
                dq = 100.0 * (b_m['avg_queue'] - a_m['avg_queue']) / max(b_m['avg_queue'], 0.01)
                dw = 100.0 * (b_m['avg_wait']  - a_m['avg_wait'])  / max(b_m['avg_wait'],  0.01)
                print(f"\n[{a_name} vs {b_name}]")
                print(f"  Queue improvement : {dq:+.1f}%")
                print(f"  Wait  improvement : {dw:+.1f}%")
    else:
        algo_label = "IPPO" if algorithm == "ippo" else "MAPPO"
        print(f"\n=== Evaluation Results ({algo_label}){_ep_label} ===")
        if wp_metrics:
            print(f"[FT]")
            print(f"  Average Reward : {wp_metrics['avg_reward']:+.4f}")
            print(f"  Average Queue  : {wp_metrics['avg_queue']:.4f}")
            print(f"  Average Wait   : {wp_metrics['avg_wait']:.4f} s")
            print(f"  Cars Passed    : {wp_metrics['cars_passed']}")

        if rl_metrics:
            print(f"\n[RL]")
            print(f"  Average Reward : {rl_metrics['avg_reward']:+.4f}")
            print(f"  Average Queue  : {rl_metrics['avg_queue']:.4f}")
            print(f"  Average Wait   : {rl_metrics['avg_wait']:.4f} s")
            print(f"  Cars Passed    : {rl_metrics['cars_passed']}")
            if rl_metrics.get('ev_cleared', 0) > 0:
                print(f"  EVs Cleared    : {rl_metrics['ev_cleared']}"
                      f" (Avg Wait: {rl_metrics['ev_avg_wait']} s)")

        if rl_metrics and wp_metrics:
            imp_q = 100.0 * (wp_metrics['avg_queue'] - rl_metrics['avg_queue']) / max(wp_metrics['avg_queue'], 0.01)
            imp_w = 100.0 * (wp_metrics['avg_wait']  - rl_metrics['avg_wait'])  / max(wp_metrics['avg_wait'],  0.01)
            print(f"\n[RL vs FT]")
            print(f"  Queue improvement : {imp_q:+.1f}%")
            print(f"  Wait  improvement : {imp_w:+.1f}%")
    print("==========================\n")

    # ── build combined results dict ───────────────────────────────────────────
    combined = {
        "wp":      wp_metrics,
        "rl":      rl_metrics,
        "ippo":    ippo_metrics,
        "mappo":   mappo_metrics,
        "_per_episode": {
            "wp":    wp_all,
            "ippo":  ippo_all,
            "mappo": mappo_all,
            "rl":    rl_all,
        },
        "_num_episodes": num_episodes,
        "_seed":         seed,
    }

    # ── optional reports ──────────────────────────────────────────────────────
    report_path = None
    if report:
        report_path = generate_report(combined, save_dir)
    if open_report and report_path:
        webbrowser.open(f"file://{report_path}")

    # ── optional Excel export ─────────────────────────────────────────────────
    if excel:
        generate_excel_report(combined, save_dir)

    return combined


# ==============================================================================
#  HTML REPORT GENERATOR
# ==============================================================================

def _metric_row(label: str, rl_val, wp_val,
                fmt: str = ".4f", lower_is_better: bool = True) -> str:
    """Build one <tr> with optional improvement badges."""
    def _badge(new_v, ref_v):
        if ref_v is None or new_v is None:
            return ""
        denom = abs(ref_v) if abs(ref_v) > 0.01 else 0.01
        imp = (100.0 * (ref_v - new_v) / denom
               if lower_is_better
               else 100.0 * (new_v - ref_v) / denom)
        colour = "#27ae60" if imp >= 0 else "#e74c3c"
        arrow  = "▲" if imp >= 0 else "▼"
        return (f'<span style="color:{colour};font-size:0.85em;">'
                f' {arrow} {abs(imp):.1f}%</span>')

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, int):
            return str(v)
        try:
            return format(v, fmt)
        except Exception:
            return str(v)

    rl_s  = _fmt(rl_val)
    wp_s  = _fmt(wp_val)
    # badge on RL vs wp
    rl_b  = _badge(rl_val, wp_val) if wp_val is not None else ""
    return f"<tr><td>{label}</td><td>{rl_s}{rl_b}</td><td>{wp_s}</td></tr>"


def generate_excel_report(results: dict, save_dir: str) -> str:
    """Generate a detailed Excel comparison report with per-episode data,
    summary statistics (mean, std, min, max, median, SEM, 95% CI),
    per-intersection stats, and pairwise improvement analysis.

    ``results`` may contain keys: ``"wp"``, ``"rl"``, ``"ippo"``, ``"mappo"``.
    Each value is a metrics dict or None.  The ``_per_episode`` key holds
    lists of per-episode metric dicts for each algorithm.
    """
    import math
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx_path = os.path.join(save_dir, f"eval_comparison_{ts}.xlsx")

    wb = Workbook()

    # ── colour palette ────────────────────────────────────────────────────
    hdr_fill    = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    hdr_font    = Font(bold=True, color="FFFFFF", size=11)
    green_fill  = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    red_fill    = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    green_font  = Font(color="006100")
    red_font    = Font(color="9C0006")
    bold_font   = Font(bold=True)
    bold11      = Font(bold=True, size=11)
    bold12      = Font(bold=True, size=12)
    title_font  = Font(bold=True, size=14, color="1F4E79")
    subtitle_font = Font(bold=True, size=11, color="4472C4")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )
    center = Alignment(horizontal="center", vertical="center")
    left   = Alignment(horizontal="left",   vertical="center")

    def _style_header(ws, row, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = hdr_fill
            cell.font = hdr_font
            cell.alignment = center
            cell.border = thin_border

    def _style_cells(ws, row, ncols):
        for c in range(1, ncols + 1):
            ws.cell(row=row, column=c).border = thin_border

    def _auto_width(ws, ncols, min_w=14, max_w=30):
        for c in range(1, ncols + 1):
            col_letter = get_column_letter(c)
            max_len = min_w
            for row_cells in ws.iter_cols(min_col=c, max_col=c):
                for cell in row_cells:
                    if cell.value is not None:
                        max_len = max(max_len, min(len(str(cell.value)) + 3, max_w))
            ws.column_dimensions[col_letter].width = max_len

    # ── statistics helpers ────────────────────────────────────────────────
    def _stats(values: list[float]) -> dict:
        """Compute mean, std, min, max, median, sem, ci95 for a list of floats."""
        n = len(values)
        if n == 0:
            return {"mean": None, "std": None, "min": None, "max": None,
                    "median": None, "sem": None, "ci95_lo": None, "ci95_hi": None, "n": 0}
        mean = sum(values) / n
        if n == 1:
            return {"mean": mean, "std": 0, "min": mean, "max": mean,
                    "median": mean, "sem": 0, "ci95_lo": mean, "ci95_hi": mean, "n": 1}
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        sem = std / math.sqrt(n)
        sorted_v = sorted(values)
        if n % 2 == 0:
            median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
        else:
            median = sorted_v[n // 2]
        ci95 = 1.96 * sem
        return {"mean": mean, "std": std, "min": min(values), "max": max(values),
                "median": median, "sem": sem, "ci95_lo": mean - ci95, "ci95_hi": mean + ci95, "n": n}

    def _g(d, k, default=None):
        return d.get(k, default) if d else default

    # ── extract per-episode data ──────────────────────────────────────────
    per_ep = results.get("_per_episode", {})
    ippo_eps  = per_ep.get("ippo",  [])
    mappo_eps = per_ep.get("mappo", [])
    wp_eps    = per_ep.get("wp",    [])
    rl_eps    = per_ep.get("rl",    [])
    num_eps   = results.get("_num_episodes", 1)
    seed_val  = results.get("_seed")

    # Collect which algorithms are present
    algos = []
    algo_eps_map = {}   # name -> per-episode list
    if results.get("wp") is not None:
        algos.append(("Webster", results["wp"]))
        algo_eps_map["Webster"] = wp_eps
    if results.get("ippo") is not None:
        algos.append(("IPPO", results["ippo"]))
        algo_eps_map["IPPO"] = ippo_eps
    if results.get("mappo") is not None:
        algos.append(("MAPPO_hierGAT", results["mappo"]))
        algo_eps_map["MAPPO_hierGAT"] = mappo_eps
    if results.get("rl") is not None and not any(
        k in ("ippo", "mappo") for k in results if results[k] is not None
    ):
        algos.append(("RL", results["rl"]))
        algo_eps_map["RL"] = rl_eps

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 1: Run Info
    # ═══════════════════════════════════════════════════════════════════════
    ws0 = wb.active
    ws0.title = "Run Info"
    info_rows = [
        ("Timestamp",          datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("Episodes",           num_eps),
        ("SUMO Seed Range",   f"{seed_val} .. {seed_val + num_eps - 1}" if seed_val is not None else "None (random)" if True else ""),
        ("Algorithms",         ", ".join(a[0] for a in algos)),
        ("Intersections",      ", ".join(INTERSECTIONS)),
    ]
    ws0.cell(row=1, column=1, value="Run Information").font = title_font
    for r, (k, v) in enumerate(info_rows, 3):
        ws0.cell(row=r, column=1, value=k).font = bold_font
        ws0.cell(row=r, column=2, value=str(v))
        ws0.cell(row=r, column=1).border = thin_border
        ws0.cell(row=r, column=2).border = thin_border
    ws0.column_dimensions["A"].width = 22
    ws0.column_dimensions["B"].width = 60

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 2: Summary Statistics
    # ═══════════════════════════════════════════════════════════════════════
    ws1 = wb.create_sheet("Summary Statistics")

    summary_metrics = [
        ("Avg Reward",       "avg_reward",       False),
        ("Avg Queue (veh)",  "avg_queue",         True),
        ("Avg Wait (s)",     "avg_wait",          True),
        ("Cars Passed",      "cars_passed",       False),
        ("CO2 Total (mg)",   "co2_total",         True),
        ("NOx Total (mg)",   "nox_total",         True),
        ("Fuel Total (ml)",  "fuel_total",        True),
        ("EVs Cleared",      "ev_cleared",        False),
    ]

    # Build multi-level header
    # Col 1: Metric, then for each algo: Mean | Std | Min | Max | Median | SEM | 95% CI
    stat_names = ["Mean", "Std", "Min", "Max", "Median", "SEM", "95% CI"]
    headers1 = ["Metric"]
    for aname, _ in algos:
        for sn in stat_names:
            headers1.append(f"{aname} {sn}")
    ncols1 = len(headers1)

    ws1.cell(row=1, column=1, value="Summary Statistics (averaged over episodes)")
    ws1.cell(row=1, column=1).font = title_font

    # algo name row
    row = 3
    ws1.cell(row=row, column=1, value="Metric").font = bold_font
    col = 2
    for aname, _ in algos:
        span = len(stat_names)
        ws1.cell(row=row, column=col, value=aname)
        ws1.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + span - 1)
        ws1.cell(row=row, column=col).font = hdr_font
        ws1.cell(row=row, column=col).fill = hdr_fill
        ws1.cell(row=row, column=col).alignment = center
        for sc in range(col, col + span):
            ws1.cell(row=row, column=sc).fill = hdr_fill
            ws1.cell(row=row, column=sc).border = thin_border
        col += span

    # stat names sub-header
    row = 4
    ws1.cell(row=row, column=1, value="").font = bold_font
    col = 2
    for _ in algos:
        for sn in stat_names:
            ws1.cell(row=row, column=col, value=sn)
            ws1.cell(row=row, column=col).font = Font(bold=True, color="FFFFFF", size=10)
            ws1.cell(row=row, column=col).fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
            ws1.cell(row=row, column=col).alignment = center
            ws1.cell(row=row, column=col).border = thin_border
            col += 1

    # data rows
    row = 5
    for label, key, lower_better in summary_metrics:
        ws1.cell(row=row, column=1, value=label).font = bold_font
        col = 2
        for aname, am in algos:
            # gather per-episode values for this key
            ep_list = algo_eps_map.get(aname, [])
            vals = []
            for ep_m in ep_list:
                v = ep_m.get(key)
                if v is not None and isinstance(v, (int, float)):
                    vals.append(float(v))
            s = _stats(vals)
            display_vals = [s["mean"], s["std"], s["min"], s["max"], s["median"], s["sem"], f'{s["ci95_lo"]:.4f}-{s["ci95_hi"]:.4f}']
            for i, dv in enumerate(display_vals):
                cell = ws1.cell(row=row, column=col + i)
                if dv is None:
                    cell.value = "--"
                elif isinstance(dv, str):
                    cell.value = dv
                else:
                    cell.value = round(dv, 6)
                cell.alignment = center
                cell.border = thin_border
            col += len(stat_names)
        row += 1

    # Pairwise improvement section
    row += 2
    ws1.cell(row=row, column=1, value="Pairwise Improvement (%)").font = bold12
    row += 1
    imp_headers = ["Metric"]
    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            imp_headers.append(f"{algos[j][0]} vs {algos[i][0]}")
    for c, h in enumerate(imp_headers, 1):
        ws1.cell(row=row, column=c, value=h)
    _style_header(ws1, row, len(imp_headers))
    row += 1

    for label, key, lower_better in summary_metrics:
        ws1.cell(row=row, column=1, value=label).font = bold_font
        col_offset = 2
        for i in range(len(algos)):
            for j in range(i + 1, len(algos)):
                aname_i, am_i = algos[i]
                aname_j, am_j = algos[j]
                vi = _g(am_i, key)
                vj = _g(am_j, key)
                if vi is None or vj is None or vi == 0:
                    ws1.cell(row=row, column=col_offset, value="--")
                else:
                    denom = abs(vi) if abs(vi) > 0.01 else 0.01
                    imp = 100.0 * (vi - vj) / denom if lower_better else 100.0 * (vj - vi) / denom
                    cell = ws1.cell(row=row, column=col_offset, value=round(imp, 2))
                    cell.alignment = center
                    if imp >= 0:
                        cell.fill = green_fill
                        cell.font = green_font
                    else:
                        cell.fill = red_fill
                        cell.font = red_font
                col_offset += 1
        _style_cells(ws1, row, len(imp_headers))
        row += 1

    _auto_width(ws1, ncols1)

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 3: Per-Episode Raw Data
    # ═══════════════════════════════════════════════════════════════════════
    ws2 = wb.create_sheet("Per-Episode Data")

    scalar_keys = [
        "avg_reward", "avg_queue", "avg_wait", "cars_passed",
        "co2_total", "nox_total", "fuel_total", "ev_cleared",
    ]
    key_labels = {
        "avg_reward":  "Avg Reward",
        "avg_queue":   "Avg Queue (veh)",
        "avg_wait":    "Avg Wait (s)",
        "cars_passed": "Cars Passed",
        "co2_total":   "CO2 Total (mg)",
        "nox_total":   "NOx Total (mg)",
        "fuel_total":  "Fuel Total (ml)",
        "ev_cleared":  "EVs Cleared",
    }

    ws2.cell(row=1, column=1, value="Per-Episode Raw Data").font = title_font
    row = 3

    # For each algorithm, create a block
    for aname, _ in algos:
        ep_list = algo_eps_map.get(aname, [])
        ws2.cell(row=row, column=1, value=aname).font = bold12
        row += 1

        # header
        hdr = ["Episode"] + [key_labels.get(k, k) for k in scalar_keys]
        for c, h in enumerate(hdr, 1):
            ws2.cell(row=row, column=c, value=h)
        _style_header(ws2, row, len(hdr))
        row += 1

        for ep_idx, ep_m in enumerate(ep_list):
            ws2.cell(row=row, column=1, value=f"Episode {ep_idx + 1}")
            ws2.cell(row=row, column=1).font = bold_font
            for ci, k in enumerate(scalar_keys):
                v = ep_m.get(k)
                if v is None:
                    ws2.cell(row=row, column=ci + 2, value="--")
                elif isinstance(v, int):
                    ws2.cell(row=row, column=ci + 2, value=v)
                else:
                    ws2.cell(row=row, column=ci + 2, value=round(v, 6))
                ws2.cell(row=row, column=ci + 2).alignment = center
            _style_cells(ws2, row, len(hdr))
            row += 1

        # Summary row
        ws2.cell(row=row, column=1, value="MEAN").font = bold_font
        ws2.cell(row=row, column=1).fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
        for ci, k in enumerate(scalar_keys):
            vals = [float(ep_m.get(k, 0)) for ep_m in ep_list if isinstance(ep_m.get(k), (int, float))]
            if vals:
                ws2.cell(row=row, column=ci + 2, value=round(sum(vals) / len(vals), 6))
            else:
                ws2.cell(row=row, column=ci + 2, value="--")
            ws2.cell(row=row, column=ci + 2).alignment = center
            ws2.cell(row=row, column=ci + 2).fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
            ws2.cell(row=row, column=ci + 2).font = bold_font
        _style_cells(ws2, row, len(hdr))
        row += 1

        ws2.cell(row=row, column=1, value="STD").font = bold_font
        ws2.cell(row=row, column=1).fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
        for ci, k in enumerate(scalar_keys):
            vals = [float(ep_m.get(k, 0)) for ep_m in ep_list if isinstance(ep_m.get(k), (int, float))]
            if len(vals) >= 2:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
                ws2.cell(row=row, column=ci + 2, value=round(math.sqrt(var), 6))
            else:
                ws2.cell(row=row, column=ci + 2, value="--")
            ws2.cell(row=row, column=ci + 2).alignment = center
            ws2.cell(row=row, column=ci + 2).fill = PatternFill(start_color="D9E2F3", end_color="D9E2F3", fill_type="solid")
            ws2.cell(row=row, column=ci + 2).font = bold_font
        _style_cells(ws2, row, len(hdr))
        row += 1
        row += 1  # gap between algorithms

    _auto_width(ws2, len(scalar_keys) + 1)

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 4: Per-Intersection Statistics
    # ═══════════════════════════════════════════════════════════════════════
    ws3 = wb.create_sheet("Per-Intersection Stats")

    tl_metrics = [
        ("Queue (halting)",  "per_tl_queue",  True),
        ("Wait (s)",         "per_tl_wait",   True),
        ("Reward (BP)",      "per_tl_reward", False),
    ]

    ws3.cell(row=1, column=1, value="Per-Intersection Statistics (mean over episodes)").font = title_font
    row = 3

    headers3 = ["Intersection", "Metric"]
    for aname, _ in algos:
        for sn in ["Mean", "Std", "Min", "Max"]:
            headers3.append(f"{aname} {sn}")
    ncols3 = len(headers3)

    # algo header row
    ws3.cell(row=row, column=1, value="").font = bold_font
    col = 3
    for aname, _ in algos:
        span = 4
        ws3.cell(row=row, column=col, value=aname)
        ws3.merge_cells(start_row=row, start_column=col, end_row=row, end_column=col + span - 1)
        ws3.cell(row=row, column=col).font = hdr_font
        ws3.cell(row=row, column=col).fill = hdr_fill
        ws3.cell(row=row, column=col).alignment = center
        for sc in range(col, col + span):
            ws3.cell(row=row, column=sc).fill = hdr_fill
            ws3.cell(row=row, column=sc).border = thin_border
        col += span
    row += 1

    # stat sub-header
    col = 3
    for _ in algos:
        for sn in ["Mean", "Std", "Min", "Max"]:
            ws3.cell(row=row, column=col, value=sn)
            ws3.cell(row=row, column=col).font = Font(bold=True, color="FFFFFF", size=10)
            ws3.cell(row=row, column=col).fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
            ws3.cell(row=row, column=col).alignment = center
            ws3.cell(row=row, column=col).border = thin_border
            col += 1
    row += 1

    for tl in INTERSECTIONS:
        for label, key, lower_better in tl_metrics:
            ws3.cell(row=row, column=1, value=tl).font = bold_font
            ws3.cell(row=row, column=2, value=label)
            col = 3
            for aname, _ in algos:
                ep_list = algo_eps_map.get(aname, [])
                vals = []
                for ep_m in ep_list:
                    d = ep_m.get(key, {})
                    v = d.get(tl) if isinstance(d, dict) else None
                    if v is not None and isinstance(v, (int, float)):
                        vals.append(float(v))
                s = _stats(vals)
                for sv in [s["mean"], s["std"], s["min"], s["max"]]:
                    cell = ws3.cell(row=row, column=col)
                    if sv is None:
                        cell.value = "--"
                    else:
                        cell.value = round(sv, 6)
                    cell.alignment = center
                    cell.border = thin_border
                    col += 1
            _style_cells(ws3, row, ncols3)
            row += 1
        row += 1  # gap between intersections

    _auto_width(ws3, ncols3)

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 5: Pairwise Improvement Detail
    # ═══════════════════════════════════════════════════════════════════════
    ws4 = wb.create_sheet("Improvement Analysis")

    ws4.cell(row=1, column=1, value="Pairwise Improvement Analysis").font = title_font
    row = 3

    imp_metrics = [
        ("Avg Reward",       "avg_reward",  False),
        ("Avg Queue",        "avg_queue",   True),
        ("Avg Wait",         "avg_wait",    True),
        ("Cars Passed",      "cars_passed", False),
        ("CO2 Total",        "co2_total",   True),
        ("NOx Total",        "nox_total",   True),
        ("Fuel Total",       "fuel_total",  True),
    ]

    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a_name, a_data = algos[i]
            b_name, b_data = algos[j]

            ws4.cell(row=row, column=1, value=f"{b_name} vs {a_name}").font = bold12
            row += 1

            hdr4 = ["Metric", f"{a_name} Mean", f"{b_name} Mean", "Difference", "Improvement (%)", "Better?"]
            for c, h in enumerate(hdr4, 1):
                ws4.cell(row=row, column=c, value=h)
            _style_header(ws4, row, len(hdr4))
            row += 1

            for label, key, lower_better in imp_metrics:
                va = _g(a_data, key)
                vb = _g(b_data, key)
                ws4.cell(row=row, column=1, value=label).font = bold_font
                ws4.cell(row=row, column=2, value=round(va, 6) if va is not None else "--")
                ws4.cell(row=row, column=3, value=round(vb, 6) if vb is not None else "--")
                ws4.cell(row=row, column=2).alignment = center
                ws4.cell(row=row, column=3).alignment = center

                if va is not None and vb is not None:
                    diff = vb - va
                    ws4.cell(row=row, column=4, value=round(diff, 6))
                    ws4.cell(row=row, column=4).alignment = center
                    if abs(va) > 0.01:
                        imp = 100.0 * (va - vb) / abs(va) if lower_better else 100.0 * (vb - va) / abs(va)
                        cell = ws4.cell(row=row, column=5, value=round(imp, 2))
                        cell.alignment = center
                        better = b_name if imp > 0 else a_name
                        ws4.cell(row=row, column=6, value=better)
                        if imp >= 0:
                            cell.fill = green_fill
                            cell.font = green_font
                        else:
                            cell.fill = red_fill
                            cell.font = red_font
                    else:
                        ws4.cell(row=row, column=5, value="--")
                        ws4.cell(row=row, column=6, value="--")
                else:
                    ws4.cell(row=row, column=4, value="--")
                    ws4.cell(row=row, column=5, value="--")
                    ws4.cell(row=row, column=6, value="--")
                _style_cells(ws4, row, len(hdr4))
                row += 1

            # per-intersection improvement sub-table
            row += 1
            ws4.cell(row=row, column=1, value="Per-Intersection Queue Improvement (%)").font = subtitle_font
            row += 1
            hdr_tl = ["Intersection", f"{a_name} Queue", f"{b_name} Queue", "Improvement (%)"]
            for c, h in enumerate(hdr_tl, 1):
                ws4.cell(row=row, column=c, value=h)
            _style_header(ws4, row, len(hdr_tl))
            row += 1

            for tl in INTERSECTIONS:
                qa = _g(a_data, "per_tl_queue", {}).get(tl) if isinstance(_g(a_data, "per_tl_queue"), dict) else None
                qb = _g(b_data, "per_tl_queue", {}).get(tl) if isinstance(_g(b_data, "per_tl_queue"), dict) else None
                ws4.cell(row=row, column=1, value=tl).font = bold_font
                ws4.cell(row=row, column=2, value=round(qa, 4) if qa is not None else "--")
                ws4.cell(row=row, column=3, value=round(qb, 4) if qb is not None else "--")
                ws4.cell(row=row, column=2).alignment = center
                ws4.cell(row=row, column=3).alignment = center
                if qa is not None and qb is not None and abs(qa) > 0.01:
                    imp = 100.0 * (qa - qb) / abs(qa)
                    cell = ws4.cell(row=row, column=4, value=round(imp, 2))
                    cell.alignment = center
                    if imp >= 0:
                        cell.fill = green_fill
                        cell.font = green_font
                    else:
                        cell.fill = red_fill
                        cell.font = red_font
                else:
                    ws4.cell(row=row, column=4, value="--")
                _style_cells(ws4, row, len(hdr_tl))
                row += 1

            # per-intersection wait improvement
            row += 1
            ws4.cell(row=row, column=1, value="Per-Intersection Wait Improvement (%)").font = subtitle_font
            row += 1
            hdr_wt = ["Intersection", f"{a_name} Wait", f"{b_name} Wait", "Improvement (%)"]
            for c, h in enumerate(hdr_wt, 1):
                ws4.cell(row=row, column=c, value=h)
            _style_header(ws4, row, len(hdr_wt))
            row += 1

            for tl in INTERSECTIONS:
                wa = _g(a_data, "per_tl_wait", {}).get(tl) if isinstance(_g(a_data, "per_tl_wait"), dict) else None
                wb_ = _g(b_data, "per_tl_wait", {}).get(tl) if isinstance(_g(b_data, "per_tl_wait"), dict) else None
                ws4.cell(row=row, column=1, value=tl).font = bold_font
                ws4.cell(row=row, column=2, value=round(wa, 4) if wa is not None else "--")
                ws4.cell(row=row, column=3, value=round(wb_, 4) if wb_ is not None else "--")
                ws4.cell(row=row, column=2).alignment = center
                ws4.cell(row=row, column=3).alignment = center
                if wa is not None and wb_ is not None and abs(wa) > 0.01:
                    imp = 100.0 * (wa - wb_) / abs(wa)
                    cell = ws4.cell(row=row, column=4, value=round(imp, 2))
                    cell.alignment = center
                    if imp >= 0:
                        cell.fill = green_fill
                        cell.font = green_font
                    else:
                        cell.fill = red_fill
                        cell.font = red_font
                else:
                    ws4.cell(row=row, column=4, value="--")
                _style_cells(ws4, row, len(hdr_wt))
                row += 1

            row += 2  # gap between pairs

    _auto_width(ws4, 7)

    # ═══════════════════════════════════════════════════════════════════════
    #  Sheet 6: Full Metrics Dump
    # ═══════════════════════════════════════════════════════════════════════
    ws5 = wb.create_sheet("Full Metrics")
    row = 1
    ws5.cell(row=row, column=1, value="Full Metric Dump (averaged values)").font = title_font
    row += 1

    headers5 = ["Key"] + [a[0] for a in algos]
    for c, h in enumerate(headers5, 1):
        ws5.cell(row=row, column=c, value=h)
    _style_header(ws5, row, len(headers5))
    row += 1

    all_keys = set()
    for _, ad in algos:
        if ad:
            all_keys.update(ad.keys())
    for key in sorted(all_keys):
        if key.startswith("per_tl_"):
            continue
        ws5.cell(row=row, column=1, value=key).font = bold_font
        for ci, (aname, am) in enumerate(algos):
            v = _g(am, key)
            if v is None:
                ws5.cell(row=row, column=ci + 2, value="--")
            elif isinstance(v, (int, float)):
                ws5.cell(row=row, column=ci + 2, value=round(v, 6))
            else:
                ws5.cell(row=row, column=ci + 2, value=str(v))
            ws5.cell(row=row, column=ci + 2).alignment = center
        _style_cells(ws5, row, len(headers5))
        row += 1

    # Per-TL full breakdown
    row += 1
    ws5.cell(row=row, column=1, value="-- Per-Intersection Detail --").font = bold11
    row += 1
    for tl in INTERSECTIONS:
        for key_prefix in ["per_tl_queue", "per_tl_wait", "per_tl_reward"]:
            ws5.cell(row=row, column=1, value=f"{tl} / {key_prefix}").font = bold_font
            for ci, (aname, am) in enumerate(algos):
                d = _g(am, key_prefix, {})
                v = d.get(tl) if isinstance(d, dict) else None
                ws5.cell(row=row, column=ci + 2, value=round(v, 4) if v is not None else "--")
                ws5.cell(row=row, column=ci + 2).alignment = center
            _style_cells(ws5, row, len(headers5))
            row += 1

    _auto_width(ws5, len(headers5))

    # ── save ──────────────────────────────────────────────────────────────
    wb.save(xlsx_path)
    print(f"  [EXCEL] Report saved -> {xlsx_path}", flush=True)
    return xlsx_path


def generate_report(results: dict, save_dir: str) -> str:
    """
    Generate a styled HTML report comparing RL and Webster-Pressure results.
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = Path(save_dir) / "evaluation_report.html"

    rl = results.get("rl") or {}
    wp = results.get("wp") or {}

    def _g(d, k, default=None):
        return d.get(k, default) if d else default

    # per-TL section
    per_tl_html = ""
    if rl or wp:
        tl_rows = ""
        for tl in INTERSECTIONS:
            rl_q = _g(rl, "per_tl_queue",  {}).get(tl)
            wp_q = _g(wp, "per_tl_queue",  {}).get(tl)
            rl_w = _g(rl, "per_tl_wait",   {}).get(tl)
            wp_w = _g(wp, "per_tl_wait",   {}).get(tl)
            rl_r = _g(rl, "per_tl_reward", {}).get(tl)
            wp_r = _g(wp, "per_tl_reward", {}).get(tl)
            tl_rows += f"""
            <tr class="tl-header"><td colspan="3"><strong>{tl}</strong></td></tr>
            {_metric_row("  Queue (halting veh.)", rl_q, wp_q, ".2f",  True)}
            {_metric_row("  Wait (s)",             rl_w, wp_w, ".2f",  True)}
            {_metric_row("  BP-Score (reward)",    rl_r, wp_r, "+.4f", False)}
            """
        per_tl_html = f"""
        <h2>Per-Intersection Breakdown</h2>
        <table>
        <tr><th>Metric</th><th>RL</th><th>Webster-Pressure</th></tr>
        {tl_rows}
        </table>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MAPPO-PIS+GAT Evaluation Report</title>
<style>
  body    {{font-family:'Segoe UI',Tahoma,sans-serif;background:#0d1117;
           color:#e6edf3;padding:30px;max-width:960px;margin:auto;}}
  h1      {{color:#58a6ff;border-bottom:1px solid #30363d;padding-bottom:10px;}}
  h2      {{color:#79c0ff;margin-top:32px;}}
  table   {{border-collapse:collapse;width:100%;margin-top:16px;}}
  th      {{background:#161b22;color:#79c0ff;padding:10px 14px;text-align:left;border:1px solid #30363d;}}
  td      {{padding:10px 14px;border:1px solid #30363d;}}
  tr:nth-child(even) {{background:#161b22;}}
  .tl-header {{background:#21262d !important;}}
  .summary-card {{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:20px;margin-top:20px;}}
  .timestamp {{color:#8b949e;font-size:0.9em;margin-top:40px;text-align:center;border-top:1px solid #30363d;padding-top:20px;}}
</style>
</head>
<body>
  <h1>Evaluation Report</h1>
  <div class="summary-card">
    <h2>Summary Results</h2>
    <table>
      <tr><th>Metric</th><th>RL</th><th>Webster-Pressure</th></tr>
      {_metric_row("Avg Reward", _g(rl, "avg_reward"), _g(wp, "avg_reward"), "+.4f", False)}
      {_metric_row("Avg Queue",  _g(rl, "avg_queue"),  _g(wp, "avg_queue"),  ".4f",  True)}
      {_metric_row("Avg Wait",   _g(rl, "avg_wait"),   _g(wp, "avg_wait"),   ".4f",  True)}
      {_metric_row("Cars Passed",_g(rl, "cars_passed"),_g(wp, "cars_passed"), "d",    False)}
    </table>
  </div>

  {per_tl_html}

  <div class="timestamp">
    Generated on {timestamp} | MAPPO-PIS+GAT Evaluation System
  </div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return str(report_path)


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MAPPO-PIS+GAT / IPPO against Webster-Pressure.")
    parser.add_argument("--sim_file",    type=str, default="OM.sumocfg",  help="SUMO config file")
    parser.add_argument("--save_dir",    type=str, default="./results",   help="Root results directory")
    parser.add_argument("--max_steps",   type=int, default=1800,          help="Simulation steps")
    parser.add_argument("--gui",         type=str, default="disable",     choices=["enable", "disable"])
    parser.add_argument("--checkpoint",  type=str, default="latest",      help="Checkpoint tag / path / 'latest'")
    parser.add_argument("--num_episodes",type=int, default=1,              help="Number of evaluation episodes to run (averaged)")
    parser.add_argument("--seed",        type=int, default=None,          help="SUMO random seed (same seed = same traffic for all algos)")
    parser.add_argument("--mode",        type=str, default="compare",
                        choices=["rl", "compare", "webster", "algo_compare"])
    parser.add_argument("--algorithm",   type=str, default="mappo",       choices=["mappo", "ippo", "both"],
                        help="mappo (hierGAT), ippo (baseline), or both")
    parser.add_argument("--ckpt_ippo",   type=str, default=None,
                        help="Path to IPPO checkpoint dir (for --algorithm both / --mode algo_compare)")
    parser.add_argument("--ckpt_mappo",  type=str, default=None,
                        help="Path to MAPPO_hierGAT checkpoint dir")
    parser.add_argument("--excel",       action="store_true",             help="Export full Excel comparison report")
    parser.add_argument("--report",      action="store_true",             help="Generate HTML report")
    parser.add_argument("--open",        action="store_true",             help="Open report in browser")
    parser.add_argument("--ckpt_dir",    type=str, default=None,          help="Direct path to a checkpoint folder")

    args = parser.parse_args()

    # -- validate algo_compare / both modes -----------------------------------
    if args.mode == "algo_compare" or args.algorithm == "both":
        if not args.ckpt_ippo:
            parser.error("--ckpt_ippo is required for algo_compare / both modes")
        if not args.ckpt_mappo:
            parser.error("--ckpt_mappo is required for algo_compare / both modes")
        if args.algorithm != "both":
            args.algorithm = "both"

    run_evaluation(
        sim_file    = args.sim_file,
        save_dir    = args.save_dir,
        max_steps   = args.max_steps,
        gui         = args.gui,
        checkpoint  = args.checkpoint,
        report      = args.report,
        open_report = args.open,
        mode        = args.mode,
        ckpt_dir    = args.ckpt_dir,
        algorithm   = args.algorithm,
        ckpt_ippo   = args.ckpt_ippo,
        ckpt_mappo  = args.ckpt_mappo,
        excel       = args.excel,
        num_episodes = args.num_episodes,
        seed         = args.seed,
    )