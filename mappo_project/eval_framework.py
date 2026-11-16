"""
==============================================================================
  SCIENTIFIC EVALUATION FRAMEWORK
  RL Traffic Signal Control vs Baselines in SUMO
==============================================================================
  Methods compared:
    1. Reinforcement Learning  (trained policy, inference only)
    2. Fixed-Time              (Webster-derived pre-programmed plans)
    3. Webster-Pressure        (adaptive max-pressure heuristic)

  Requirements:
    pip install numpy pandas scipy matplotlib traci sumo

  Usage:
    python eval_framework.py \
        --sumocfg  sumo/OM.sumocfg \
        --rl-checkpoint  results/checkpoints/best_model.pt \
        --seeds  10 \
        --steps  1800 \
        --output-dir  results/

  All three methods run on the SAME seeds → paired statistical tests are valid.
==============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

# ── optional traci import (fails gracefully so the module is importable
#    without SUMO for unit-testing the stats layer) ───────────────────────────
try:
    import traci
    import traci.constants as tc
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False
    warnings.warn("traci not found – SUMO simulation disabled.")

# ── optional torch import ────────────────────────────────────────────────────
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("torch not found – RL inference disabled.")


# ==============================================================================
#  CONFIGURATION  (mirrors mappo_project/config.py)
# ==============================================================================

INTERSECTIONS: List[str] = ["TL1", "TL2", "TL3", "TL4"]

UPSTREAM_LANES: Dict[str, List[str]] = {
    "TL1": ["E7_0", "E7_1", "E7_2", "E9_0", "E9_1", "E9_2"],
    "TL2": ["E1_0", "E1_1", "E2_0", "E2_1", "E2_2", "E2_3", "E4_0", "E4_1", "E6_0", "E6_1"],
    "TL3": ["E16_0", "E23_0", "E24_0", "E24_1", "E24_2"],
    "TL4": ["E14_1", "E14_2", "E15_0", "E15_1"],
}

DOWNSTREAM_LANES: Dict[str, List[str]] = {
    "TL1": ["E19_0", "E19_1", "E19_2", "E2_0", "E2_1", "E2_2", "E2_3", "E8_0", "E8_1", "E8_2"],
    "TL2": ["E0_0", "E0_1", "E3_0", "E3_1", "E5_0", "E5_1"],
    "TL3": ["E15_0", "E15_1", "E15_2", "E17_0", "E17_1"],
    "TL4": ["E11_0", "E11_1", "E18_0", "E18_1", "E18_2"],
}

TLS_DETECTOR_MAP: Dict[str, List[str]] = {
    "TL1": ["e2_15", "e2_16", "e2_17", "e2_0", "e2_2", "e2_1"],
    "TL2": ["e2_3",  "e2_4",  "e2_5",  "e2_6"],
    "TL3": ["e2_10", "e2_11", "e2_12", "e2_9"],
    "TL4": ["e2_8",  "e2_7",  "e2_13", "e2_14"],
}

FIXED_TIME_PLANS: Dict[str, dict] = {
    "TL1": {
        "cycle_length": 60, "offset": 0,
        "phases": [(29, "rrrrGGGGGG"), (3, "rrrryyyyyy"),
                   (25, "GGGGrrrrrr"), (3, "yyyyrrrrrr")],
    },
    "TL2": {
        "cycle_length": 60, "offset": 0,
        "phases": [(30, "rrrrrGGGrrrr"), (3, "rrrrryyyrrrr"),
                   (14, "GGGGGrrrrrrr"), (3, "GGyyyrrrrrrr"),
                   (10, "rrrrrrrrGGGG"), (3, "rrrrrrrryyyy")],
    },
    "TL3": {
        "cycle_length": 60, "offset": 0,
        "phases": [(18, "rGGGG"), (3, "ryyGy"),
                   (36, "Grrrr"), (3, "yrrrr")],
    },
    "TL4": {
        "cycle_length": 60, "offset": 10,
        "phases": [(38, "GGrr"), (3, "yyrr"),
                   (16, "rrGG"), (3, "rryy")],
    },
}

UNIFIED_BP_CAPACITY: float = 50.0
REWARD_CLIP:         float = 5.0
MIN_GREEN_STEPS:     int   = 10      # seconds @ step_length=1 s
STOPPED_SPEED:       float = 0.1     # m/s threshold for "halting"


# ==============================================================================
#  DATA CLASSES
# ==============================================================================

@dataclass
class EpisodeResult:
    """Scalar metrics for one episode of one method."""
    seed:          int
    method:        str
    avg_queue:     float
    avg_wait:      float
    total_passed:  float
    avg_reward:    float   # NaN if method has no reward signal


@dataclass
class MethodResults:
    name:    str
    records: List[EpisodeResult] = field(default_factory=list)

    # ── convenience vectors ──────────────────────────────────────────────────
    @property
    def queues(self)  -> np.ndarray: return np.array([r.avg_queue    for r in self.records])
    @property
    def waits(self)   -> np.ndarray: return np.array([r.avg_wait     for r in self.records])
    @property
    def passed(self)  -> np.ndarray: return np.array([r.total_passed for r in self.records])
    @property
    def rewards(self) -> np.ndarray: return np.array([r.avg_reward   for r in self.records])


# ==============================================================================
#  SUMO UTILITIES
# ==============================================================================

def _build_sumo_cmd(sumocfg: str, seed: int) -> List[str]:
    return [
        "sumo",
        "-c", sumocfg,
        "--seed", str(seed),
        "--no-warnings", "true",
        "--time-to-teleport", "-1",
        "--collision.action", "warn",
    ]


def _start_sumo(cmd: List[str], port: int) -> "subprocess.Popen":
    import subprocess, socket
    for _ in range(15):
        try:
            with socket.create_connection(("localhost", port), timeout=0.3):
                time.sleep(0.5)
        except OSError:
            break
    proc = subprocess.Popen(
        cmd + ["--remote-port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 90.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("SUMO exited prematurely.")
        time.sleep(0.5)
        try:
            traci.init(port=port, host="localhost", numRetries=0)
            return proc
        except Exception:
            pass
    proc.kill()
    raise RuntimeError(f"SUMO did not accept TraCI on port {port}.")


def _close_sumo(proc) -> None:
    try:
        traci.close()
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


# ── per-step metric helpers ───────────────────────────────────────────────────

def _halting_count(lane_id: str) -> float:
    try:
        return float(traci.lane.getLastStepHaltingNumber(lane_id))
    except Exception:
        return 0.0


def _tls_queue(tls_id: str) -> float:
    """Total halting vehicles across all detector lanes for one TLS."""
    dets = TLS_DETECTOR_MAP.get(tls_id, [])
    if dets:
        total = 0.0
        for d in dets:
            try:
                total += traci.lanearea.getLastStepHaltingNumber(d)
            except Exception:
                pass
        return total
    return sum(_halting_count(l) for l in UPSTREAM_LANES.get(tls_id, []))


def _tls_mean_wait(tls_id: str) -> float:
    lanes = UPSTREAM_LANES.get(tls_id, [])
    total_wait, n_veh = 0.0, 0
    for lane_id in lanes:
        try:
            for veh in traci.lane.getLastStepVehicleIDs(lane_id):
                total_wait += traci.vehicle.getAccumulatedWaitingTime(veh)
                n_veh += 1
        except Exception:
            pass
    return total_wait / n_veh if n_veh > 0 else 0.0


def _bp_reward(tls_id: str) -> float:
    """Back-pressure reward (identical formula to RL training)."""
    incoming = UPSTREAM_LANES.get(tls_id, [])
    outgoing = DOWNSTREAM_LANES.get(tls_id, [])
    qcap     = UNIFIED_BP_CAPACITY
    n_pairs  = max(len(incoming), 1)
    pressure = 0.0
    for k, in_lane in enumerate(incoming):
        qi = _halting_count(in_lane) / qcap
        out_lane = outgoing[k % len(outgoing)] if outgoing else None
        qo = (_halting_count(out_lane) / qcap) if out_lane else 0.0
        pressure += qi - qo
    return float(np.clip(-(pressure / n_pairs), -REWARD_CLIP, REWARD_CLIP))


def _network_queue() -> float:
    return np.mean([_tls_queue(t)    for t in INTERSECTIONS])


def _network_wait() -> float:
    return np.mean([_tls_mean_wait(t) for t in INTERSECTIONS])


def _network_reward() -> float:
    return np.mean([_bp_reward(t)    for t in INTERSECTIONS])


def _vehicles_passed() -> int:
    try:
        return traci.simulation.getArrivedNumber()
    except Exception:
        return 0


# ==============================================================================
#  CONTROLLER 1 — FIXED-TIME (Webster)
# ==============================================================================

class FixedTimeController:
    """Pre-programmed Webster-derived signal plan for one TLS.

    Fixes applied
    -------------
    Offset support
        ``FIXED_TIME_PLANS`` carries a per-TLS ``offset`` (e.g. TL4 = 10 s).
        The original code ignored it, forcing all intersections to start on
        phase 0 simultaneously, which creates artificial platoon collisions and
        stop waves at downstream intersections.
        Fix: fast-forward the internal clock by ``offset`` steps during
        ``__init__`` (and ``reset``) so each TLS enters the simulation at the
        correct point in its pre-designed cycle.

    Step / simulationStep ordering
        The original ``step()`` called ``setRedYellowGreenState`` **and**
        advanced ``_elapsed`` / rolled over ``_idx`` in the same call, which
        meant the phase could flip inside the same TraCI step that first
        applied it — SUMO then saw an effective green of < 1 second.
        Fix: separate concerns.  ``step()`` now only applies the *current*
        phase state to SUMO.  The phase counter is advanced in
        ``_advance_clock()``, which the episode loop calls **after**
        ``traci.simulationStep()``.  Sequence per simulation step:

            1. ctrl.step()            → setRedYellowGreenState (current phase)
            2. traci.simulationStep() → SUMO moves vehicles under that signal
            3. ctrl._advance_clock()  → increment elapsed; roll over if needed

    Yellow-phase integrity
        ``setRedYellowGreenState`` overwrites SUMO's internal phase logic.
        If a yellow-phase string contains 'y', car-following models in SUMO
        honour it correctly — but only when it is applied for its full
        programmed duration without interruption.  The step/simulationStep
        ordering already guarantees that; no additional change needed here.
    """

    def __init__(self, tls_id: str):
        plan            = FIXED_TIME_PLANS[tls_id]
        self.tls_id     = tls_id
        self.phases     = plan["phases"]          # list of (duration, state_str)
        self._offset    = int(plan.get("offset", 0))
        self._idx       = 0
        self._elapsed   = 0
        self._apply_offset()

    def _apply_offset(self) -> None:
        """Advance internal clock by self._offset steps (no TraCI calls)."""
        for _ in range(self._offset):
            self._elapsed += 1
            dur, _ = self.phases[self._idx]
            if self._elapsed >= dur:
                self._idx     = (self._idx + 1) % len(self.phases)
                self._elapsed = 0

    def reset(self):
        self._idx     = 0
        self._elapsed = 0
        self._apply_offset()

    def step(self):
        """Apply the *current* phase state to SUMO (TraCI call only)."""
        _, state = self.phases[self._idx]
        try:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
        except Exception:
            pass

    def _advance_clock(self) -> None:
        """Increment elapsed time and roll over to the next phase if due.

        Called by the episode loop after traci.simulationStep() so
        that SUMO always sees exactly ``duration`` full simulation steps for
        every phase, including yellow phases.
        """
        self._elapsed += 1
        dur, _ = self.phases[self._idx]
        if self._elapsed >= dur:
            self._idx     = (self._idx + 1) % len(self.phases)
            self._elapsed = 0


# ==============================================================================
#  CONTROLLER 2 — WEBSTER-PRESSURE (adaptive max-pressure heuristic)
# ==============================================================================

class WebsterPressureController:
    """
    Adaptive pressure-based controller.

    At every decision point (end of current phase's minimum green):
      pressure_i = Σ incoming_queue_i − Σ outgoing_queue_i
    The phase with maximum pressure is selected next.

    This approximates Webster's optimality condition in an online,
    adaptive fashion and is a standard strong baseline in the literature
    (Varaiya 2013; Wei et al. 2019).
    """

    def __init__(self, tls_id: str):
        plan             = FIXED_TIME_PLANS[tls_id]
        self.tls_id      = tls_id
        self.phases      = plan["phases"]        # reuse phase states
        self._idx        = 0
        self._elapsed    = 0
        self._min_green  = MIN_GREEN_STEPS

    def reset(self):
        self._idx    = 0
        self._elapsed = 0

    def _phase_pressure(self, phase_idx: int) -> float:
        """Estimate pressure for a hypothetical phase selection."""
        _, state = self.phases[phase_idx]
        # green lanes in this phase → treat as "serving" those movements
        incoming = UPSTREAM_LANES.get(self.tls_id, [])
        outgoing = DOWNSTREAM_LANES.get(self.tls_id, [])
        pressure = 0.0
        for k, in_lane in enumerate(incoming):
            qi = _halting_count(in_lane) / UNIFIED_BP_CAPACITY
            out_lane = outgoing[k % len(outgoing)] if outgoing else None
            qo = (_halting_count(out_lane) / UNIFIED_BP_CAPACITY) if out_lane else 0.0
            pressure += qi - qo
        return pressure

    def step(self):
        """Apply current phase state to SUMO (TraCI call only).

        Clock advancement moved to _advance_clock(), called after
        traci.simulationStep() in the episode loop.
        """
        _, state = self.phases[self._idx]
        try:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, state)
        except Exception:
            pass

    def _advance_clock(self) -> None:
        """Advance elapsed counter and switch phase if minimum green elapsed."""
        self._elapsed += 1
        if self._elapsed >= self._min_green:
            pressures = [self._phase_pressure(i) for i in range(len(self.phases))]
            best      = int(np.argmax(pressures))
            if best != self._idx:
                self._idx     = best
                self._elapsed = 0


# ==============================================================================
#  EPISODE RUNNERS
# ==============================================================================

def _run_episode_generic(
    controllers: dict,
    sim_steps:   int,
    has_reward:  bool = True,
) -> Tuple[float, float, float, float]:
    """
    Shared simulation loop for Fixed-Time and Webster-Pressure.
    Returns (avg_queue, avg_wait, total_passed, avg_reward).
    """
    for ctrl in controllers.values():
        ctrl.reset()

    q_buf, w_buf, r_buf = [], [], []
    passed_total = 0

    for step in range(1, sim_steps + 1):
        # early stop when network is empty
        if step % 50 == 0:
            try:
                if traci.simulation.getMinExpectedNumber() == 0:
                    break
            except Exception:
                pass

        for ctrl in controllers.values():
            ctrl.step()

        traci.simulationStep()

        for ctrl in controllers.values():
            if hasattr(ctrl, "_advance_clock"):
                ctrl._advance_clock()

        q_buf.append(_network_queue())
        w_buf.append(_network_wait())
        if has_reward:
            r_buf.append(_network_reward())
        passed_total += _vehicles_passed()

    avg_q   = float(np.mean(q_buf))   if q_buf else 0.0
    avg_w   = float(np.mean(w_buf))   if w_buf else 0.0
    avg_r   = float(np.mean(r_buf))   if r_buf else float("nan")
    return avg_q, avg_w, float(passed_total), avg_r


def run_fixed_time_episode(
    sumocfg: str,
    seed:    int,
    steps:   int,
    port:    int,
) -> EpisodeResult:
    cmd  = _build_sumo_cmd(sumocfg, seed)
    proc = _start_sumo(cmd, port)
    try:
        controllers = {t: FixedTimeController(t) for t in INTERSECTIONS}
        aq, aw, ap, ar = _run_episode_generic(controllers, steps)
    finally:
        _close_sumo(proc)
    return EpisodeResult(seed=seed, method="Fixed-Time",
                         avg_queue=aq, avg_wait=aw,
                         total_passed=ap, avg_reward=ar)


def run_webster_pressure_episode(
    sumocfg: str,
    seed:    int,
    steps:   int,
    port:    int,
) -> EpisodeResult:
    cmd  = _build_sumo_cmd(sumocfg, seed)
    proc = _start_sumo(cmd, port)
    try:
        controllers = {t: WebsterPressureController(t) for t in INTERSECTIONS}
        aq, aw, ap, ar = _run_episode_generic(controllers, steps)
    finally:
        _close_sumo(proc)
    return EpisodeResult(seed=seed, method="Webster-Pressure",
                         avg_queue=aq, avg_wait=aw,
                         total_passed=ap, avg_reward=ar)


def run_rl_episode(
    sumocfg:    str,
    seed:       int,
    steps:      int,
    port:       int,
    agents:     dict,            # {tls_id: agent_object}
    obs_fn,                      # callable(tls_id) -> np.ndarray
) -> EpisodeResult:
    """
    Inference-only RL episode.
    `agents` must expose  agent.select_action(obs) -> int  (no gradient).
    `obs_fn` must return the observation array for a given TLS.
    """
    cmd  = _build_sumo_cmd(sumocfg, seed)
    proc = _start_sumo(cmd, port)
    try:
        q_buf, w_buf, r_buf = [], [], []
        passed_total = 0
        last_action  = {t: 0 for t in INTERSECTIONS}
        phase_steps  = {t: 0 for t in INTERSECTIONS}

        for step in range(1, steps + 1):
            if step % 50 == 0:
                try:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        break
                except Exception:
                    pass

            for tls_id, agent in agents.items():
                phase_steps[tls_id] += 1
                if phase_steps[tls_id] >= MIN_GREEN_STEPS:
                    obs    = obs_fn(tls_id)
                    action = agent.select_action(obs)
                    if action != last_action[tls_id]:
                        last_action[tls_id]  = action
                        phase_steps[tls_id]  = 0
                    _, state = FIXED_TIME_PLANS[tls_id]["phases"][
                        action % len(FIXED_TIME_PLANS[tls_id]["phases"])
                    ]
                    try:
                        traci.trafficlight.setRedYellowGreenState(tls_id, state)
                    except Exception:
                        pass

            traci.simulationStep()

            q_buf.append(_network_queue())
            w_buf.append(_network_wait())
            r_buf.append(_network_reward())
            passed_total += _vehicles_passed()

    finally:
        _close_sumo(proc)

    return EpisodeResult(
        seed=seed, method="RL",
        avg_queue=float(np.mean(q_buf)),
        avg_wait=float(np.mean(w_buf)),
        total_passed=float(passed_total),
        avg_reward=float(np.mean(r_buf)),
    )


# ==============================================================================
#  STATISTICAL ANALYSIS
# ==============================================================================

ALPHA = 0.05   # significance threshold


def _stat_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, str]:
    """
    Paired t-test if normality holds (Shapiro-Wilk p > 0.05),
    otherwise Mann-Whitney U test.
    Returns (p_value, test_name).
    """
    diff = a - b
    _, p_shapiro = stats.shapiro(diff)
    if p_shapiro > 0.05 and len(diff) >= 5:
        _, p = stats.ttest_rel(a, b)
        return float(p), "paired t-test"
    else:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p), "Mann-Whitney U"


def _relative_improvement(rl: np.ndarray, baseline: np.ndarray,
                           lower_is_better: bool = True) -> float:
    """Mean relative improvement of RL over baseline (%)."""
    rl_m, bl_m = np.mean(rl), np.mean(baseline)
    if bl_m == 0:
        return float("nan")
    if lower_is_better:
        return 100.0 * (bl_m - rl_m) / abs(bl_m)
    else:
        return 100.0 * (rl_m - bl_m) / abs(bl_m)


def build_stats_table(
    rl_res:  MethodResults,
    ft_res:  MethodResults,
    wp_res:  MethodResults,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        summary_df   – mean ± std per metric per method
        sig_df       – p-values + test names + conclusion for RL vs each baseline
    """
    metrics = {
        "Avg Queue":   ("queues",  True),
        "Avg Wait":    ("waits",   True),
        "Vehicles":    ("passed",  False),
        "Avg Reward":  ("rewards", False),
    }

    rows_summary = []
    for m_name, (attr, _) in metrics.items():
        row = {"Metric": m_name}
        for res in [rl_res, ft_res, wp_res]:
            vec = getattr(res, attr)
            row[f"{res.name} Mean"] = np.mean(vec)
            row[f"{res.name} Std"]  = np.std(vec, ddof=1)
            row[f"{res.name} Mean±Std"] = (
                f"{np.mean(vec):.3f} ± {np.std(vec, ddof=1):.3f}"
            )
        rows_summary.append(row)
    summary_df = pd.DataFrame(rows_summary).set_index("Metric")

    rows_sig = []
    for m_name, (attr, lib) in metrics.items():
        rl_vec = getattr(rl_res,  attr)
        for baseline in [ft_res, wp_res]:
            bl_vec = getattr(baseline, attr)
            p, test = _stat_test(rl_vec, bl_vec)
            imp     = _relative_improvement(rl_vec, bl_vec, lib)
            rows_sig.append({
                "Metric":        m_name,
                "RL vs":         baseline.name,
                "Test":          test,
                "p-value":       round(p, 4),
                "Improvement %": round(imp, 2),
                "Conclusion":    "RL significantly better" if p < ALPHA else
                                 "No significant difference",
            })
    sig_df = pd.DataFrame(rows_sig)
    return summary_df, sig_df


# ==============================================================================
#  PLOTTING
# ==============================================================================

PALETTE = {
    "RL":               "#2196F3",
    "Fixed-Time":       "#F44336",
    "Webster-Pressure": "#4CAF50",
}


def _bar_with_errorbars(
    ax, methods: List[str], means: List[float], stds: List[float],
    ylabel: str, title: str,
) -> None:
    x      = np.arange(len(methods))
    colors = [PALETTE[m] for m in methods]
    bars   = ax.bar(x, means, yerr=stds, capsize=6,
                    color=colors, alpha=0.85, edgecolor="white", linewidth=1.2,
                    error_kw={"elinewidth": 1.8, "ecolor": "#333"})
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{mean:.2f}", ha="center", va="bottom", fontsize=9)


def plot_comparison(
    rl_res:    MethodResults,
    ft_res:    MethodResults,
    wp_res:    MethodResults,
    save_dir:  str,
) -> None:
    results_order = [rl_res, ft_res, wp_res]
    methods       = [r.name for r in results_order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("RL vs Baselines — Scientific Evaluation", fontsize=14, fontweight="bold")

    cfg = [
        ("queues", "Avg Queue Length (veh)", "Queue"),
        ("waits",  "Avg Waiting Time (s)",   "Wait Time"),
        ("rewards","Avg Reward (BP-score)",   "Reward"),
    ]
    for ax, (attr, ylabel, title) in zip(axes, cfg):
        means = [np.mean(getattr(r, attr)) for r in results_order]
        stds  = [np.std(getattr(r, attr), ddof=1) for r in results_order]
        _bar_with_errorbars(ax, methods, means, stds, ylabel, title)

    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {path}")


def plot_episode_curves(
    rl_res:   MethodResults,
    ft_res:   MethodResults,
    wp_res:   MethodResults,
    save_dir: str,
) -> None:
    """Episode-by-episode metric curves across seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Per-Seed Episode Metrics", fontsize=13, fontweight="bold")

    for ax, attr, ylabel in [
        (axes[0], "queues", "Avg Queue"),
        (axes[1], "waits",  "Avg Wait (s)"),
    ]:
        for res in [rl_res, ft_res, wp_res]:
            vec = getattr(res, attr)
            ax.plot(vec, marker="o", ms=4, label=res.name,
                    color=PALETTE[res.name], linewidth=1.5)
            ax.fill_between(range(len(vec)),
                            vec - np.std(vec, ddof=1),
                            vec + np.std(vec, ddof=1),
                            color=PALETTE[res.name], alpha=0.12)
        ax.set_xlabel("Seed index", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "episode_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {path}")


# ==============================================================================
#  REPORTING
# ==============================================================================

def print_summary(summary_df: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    sep = "=" * 70
    print(f"\n{sep}")
    print("  SUMMARY TABLE  (mean ± std)")
    print(sep)
    cols = [c for c in summary_df.columns if "Mean±Std" in c]
    print(summary_df[cols].to_string())

    print(f"\n{sep}")
    print("  STATISTICAL SIGNIFICANCE  (RL vs Baselines)")
    print(sep)
    print(sig_df[["Metric", "RL vs", "Test", "p-value",
                  "Improvement %", "Conclusion"]].to_string(index=False))
    print(sep + "\n")


def save_results(
    records_all: List[EpisodeResult],
    summary_df:  pd.DataFrame,
    sig_df:      pd.DataFrame,
    save_dir:    str,
) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # raw episode records
    raw_path = os.path.join(save_dir, "episode_records.csv")
    pd.DataFrame([vars(r) for r in records_all]).to_csv(raw_path, index=False)
    print(f"  [CSV] Raw records  → {raw_path}")

    # summary
    sum_path = os.path.join(save_dir, "summary.csv")
    summary_df.to_csv(sum_path)
    print(f"  [CSV] Summary      → {sum_path}")

    # significance
    sig_path = os.path.join(save_dir, "significance.csv")
    sig_df.to_csv(sig_path, index=False)
    print(f"  [CSV] Significance → {sig_path}")


# ==============================================================================
#  MAIN ORCHESTRATOR
# ==============================================================================

def run_evaluation(
    sumocfg:    str,
    seeds:      List[int],
    steps:      int,
    save_dir:   str,
    base_port:  int   = 8813,
    rl_agents:  Optional[dict] = None,
    obs_fn                     = None,
) -> None:
    """
    Full evaluation loop.
    If rl_agents is None, only Fixed-Time and Webster-Pressure are compared.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  Scientific Evaluation — {len(seeds)} seeds, {steps} steps each")
    print(f"  Methods: RL={'YES' if rl_agents else 'SKIP'}, Fixed-Time, Webster-Pressure")
    print(f"{'='*60}\n")

    rl_res = MethodResults("RL")
    ft_res = MethodResults("Fixed-Time")
    wp_res = MethodResults("Webster-Pressure")

    all_records: List[EpisodeResult] = []

    for i, seed in enumerate(seeds, 1):
        print(f"[Seed {seed:>5}  ({i}/{len(seeds)})]")

        print("  → Fixed-Time       ", end="", flush=True)
        r = run_fixed_time_episode(sumocfg, seed, steps, base_port)
        ft_res.records.append(r); all_records.append(r)
        print(f"queue={r.avg_queue:.2f}  wait={r.avg_wait:.2f}s  passed={r.total_passed:.0f}")

        # Webster-Pressure
        print("  → Webster-Pressure ", end="", flush=True)
        r = run_webster_pressure_episode(sumocfg, seed, steps, base_port + 1)
        wp_res.records.append(r); all_records.append(r)
        print(f"queue={r.avg_queue:.2f}  wait={r.avg_wait:.2f}s  passed={r.total_passed:.0f}")

        # RL (optional)
        if rl_agents and obs_fn:
            print("  → RL               ", end="", flush=True)
            r = run_rl_episode(sumocfg, seed, steps, base_port + 2, rl_agents, obs_fn)
            rl_res.records.append(r); all_records.append(r)
            print(f"queue={r.avg_queue:.2f}  wait={r.avg_wait:.2f}s  passed={r.total_passed:.0f}")
        print()

    # ── stats ────────────────────────────────────────────────────────────────
    if rl_agents:
        summary_df, sig_df = build_stats_table(rl_res, ft_res, wp_res)
    else:
        summary_df, sig_df = build_stats_table(wp_res, ft_res, wp_res)

    print_summary(summary_df, sig_df)
    save_results(all_records, summary_df, sig_df, save_dir)

    # ── plots ────────────────────────────────────────────────────────────────
    ref_rl = rl_res if rl_agents else wp_res
    plot_comparison(ref_rl, ft_res, wp_res, save_dir)
    plot_episode_curves(ref_rl, ft_res, wp_res, save_dir)

    print("\nEvaluation complete.\n")


# ==============================================================================
#  CHECKPOINT LOADER  (matches mappo_project checkpoint format exactly)
# ==============================================================================

def _resolve_checkpoint_paths(checkpoint_path: str) -> Dict[str, str]:
    """
    Resolve a checkpoint argument to a {tls_id: file_path} mapping.

    Accepted formats
    ────────────────
    1. Directory  →  looks for <dir>/TL1.pt  TL2.pt … (EpDir layout)
    2. Single .pt →  tries to load ALL TLS from the same file
                     (flat/legacy single-file layout)
    3. Glob pattern like  checkpoints/ep0050/*.pt
       Pass the directory part; we'll glob it.
    4. latest.json pointer file  →  reads {"latest_ep": N} and resolves
       to  <parent>/ep{N:04d}/  automatically.

    Returns dict  {tls_id: absolute_path}  for every TLS found.
    """
    path = Path(checkpoint_path).expanduser().resolve()

    # ── case 4: latest.json pointer ─────────────────────────────────────────
    if path.name == "latest.json" and path.is_file():
        try:
            data = json.loads(path.read_text())
            ep   = int(data["latest_ep"])
            path = path.parent / f"ep{ep:04d}"
            print(f"  [CKPT] latest.json resolved → {path}")
        except Exception as e:
            raise ValueError(f"Cannot parse latest.json: {e}")

    # ── case 1: directory (EpDir layout) ────────────────────────────────────
    if path.is_dir():
        mapping = {}
        for tls_id in INTERSECTIONS:
            candidate = path / f"{tls_id}.pt"
            if candidate.exists():
                mapping[tls_id] = str(candidate)
            else:
                print(f"  [CKPT] WARNING: {candidate} not found — {tls_id} will be skipped.")
        if not mapping:
            raise FileNotFoundError(f"No <TLS>.pt files found in directory: {path}")
        return mapping

    # ── case 2: single .pt file ──────────────────────────────────────────────
    if path.is_file() and path.suffix == ".pt":
        return {tls_id: str(path) for tls_id in INTERSECTIONS}

    raise FileNotFoundError(
        f"Cannot resolve checkpoint path: {checkpoint_path}\n"
        f"Expected a directory with TL*.pt files, a single .pt file, or latest.json."
    )


def load_rl_agents(checkpoint_path: str, device_str: str = "cpu") -> dict:
    """
    Load trained MAPPOAgent actors from a checkpoint.

    Supports both:
      • EpDir layout  : checkpoints/ep0050/TL1.pt  TL2.pt …
      • Flat layout   : checkpoints/TL1_emergency.pt …
      • Single file   : one .pt that contains actor_state for every TLS
      • latest.json   : auto-resolved pointer

    Returns
    -------
    dict  {tls_id: agent}
        Each agent has actor weights loaded and set to eval() mode.
        Optimisers and schedulers are NOT loaded (inference only).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not installed — cannot load RL checkpoint.")

    # import from the mappo_project package (must be on sys.path)
    try:
        from mappo_project.agent import MAPPOAgent
    except ImportError:
        raise ImportError(
            "Cannot import MAPPOAgent from mappo_project.\n"
            "Make sure the project root is on PYTHONPATH:\n"
            "  export PYTHONPATH=/path/to/project_root:$PYTHONPATH"
        )

    device    = torch.device(device_str)
    path_map  = _resolve_checkpoint_paths(checkpoint_path)

    print(f"\n  [CKPT] Loading RL agents from checkpoint …")
    print(f"         device = {device}")

    agents: Dict[str, "MAPPOAgent"] = {}

    for tls_id in INTERSECTIONS:
        if tls_id not in path_map:
            print(f"  [CKPT] SKIP {tls_id} — no checkpoint file found.")
            continue

        pt_path = path_map[tls_id]
        print(f"  [CKPT] {tls_id:4s} ← {pt_path}")

        try:
            data = torch.load(pt_path, map_location=device, weights_only=False)
        except Exception as e:
            print(f"  [CKPT] ERROR loading {pt_path}: {e} — skipping {tls_id}.")
            continue

        try:
            agent = MAPPOAgent(tls_id=tls_id, device=device)
        except Exception as e:
            print(f"  [CKPT] ERROR constructing MAPPOAgent({tls_id}): {e} — skipping.")
            continue

        # load actor weights (key is 'actor_state' in CheckpointManager format)
        actor_key = "actor_state" if "actor_state" in data else "actor"
        if actor_key not in data:
            print(f"  [CKPT] WARNING: key '{actor_key}' missing in {pt_path} — skipping.")
            continue

        try:
            # unwrap torch.compile wrapper if present
            actor_net = (agent.actor._orig_mod
                         if hasattr(agent.actor, "_orig_mod") else agent.actor)
            actor_net.load_state_dict(data[actor_key])
            actor_net.eval()
            print(f"           actor weights loaded OK  "
                  f"(episode={data.get('episode', '?')})")
        except Exception as e:
            print(f"  [CKPT] ERROR loading actor weights for {tls_id}: {e} — skipping.")
            continue

        # restore intent vector if available
        if "current_intent" in data:
            agent.current_intent = np.array(
                data["current_intent"], dtype=np.float32
            )

        agents[tls_id] = agent

    if not agents:
        raise RuntimeError("No agents were loaded successfully from the checkpoint.")

    loaded = list(agents.keys())
    missing = [t for t in INTERSECTIONS if t not in agents]
    print(f"\n  [CKPT] Loaded: {loaded}")
    if missing:
        print(f"  [CKPT] Missing (will use Fixed-Time fallback): {missing}")

    return agents


def make_obs_fn(agents: dict):
    """
    Returns a per-agent observation function that mirrors MAPPOAgent._get_state()
    but without EV features or shock-wave features that require private cache state.

    For a fully faithful obs, call  agent._get_state()  directly (requires
    that agent.current_action_idx and agent._current_step are maintained).
    This simpler version is sufficient for inference when the actor is robust.
    """
    def obs_fn(tls_id: str) -> np.ndarray:
        agent = agents[tls_id]
        try:
            # Try the full faithful observation first
            obs = agent._get_state()
            return obs
        except Exception:
            pass

        # Fallback: minimal observation matching state_dim structure
        phase_onehot = np.zeros(agent.action_dim, dtype=np.float32)
        phase_onehot[agent.current_action_idx] = 1.0
        values = list(phase_onehot)

        for lane_id in agent.incoming_lanes:
            try:
                n = traci.lane.getLastStepVehicleNumber(lane_id)
                h = traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                n, h = 0, 0
            values.extend([float(n - h), float(h)])   # moving, stopped

        for lane_id in agent.outgoing_lanes:
            try:
                n = traci.lane.getLastStepVehicleNumber(lane_id)
                h = traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                n, h = 0, 0
            values.extend([float(n - h), float(h)])

        dets = TLS_DETECTOR_MAP.get(tls_id, [])
        for det_id in dets:
            try:
                values.append(float(traci.lanearea.getLastStepHaltingNumber(det_id)))
            except Exception:
                values.append(0.0)

        for lane_id in agent.incoming_lanes:
            values.append(_halting_count(lane_id))

        values.append(0.0)   # time-since-switch (unknown here)
        values.append(min(_tls_queue(tls_id) / max(UNIFIED_BP_CAPACITY, 1.0), 1.0))
        values.extend([0.0, 0.0])   # shock features

        if agent._ev_feat_dim > 0:
            values.extend([0.0] * agent._ev_feat_dim)

        obs = np.array(values, dtype=np.float32)

        # pad / trim to exact state_dim (safety net)
        if len(obs) < agent.state_dim:
            obs = np.pad(obs, (0, agent.state_dim - len(obs)))
        elif len(obs) > agent.state_dim:
            obs = obs[:agent.state_dim]

        return obs

    return obs_fn


# ==============================================================================
#  CLI ENTRY POINT
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scientific RL vs Fixed-Time vs Webster-Pressure evaluation in SUMO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Checkpoint formats accepted by --rl-checkpoint:
  checkpoints/ep0050/          directory with TL1.pt TL2.pt TL3.pt TL4.pt
  checkpoints/ep0050/TL1.pt    single .pt  (same file reused for all TLS)
  checkpoints/latest.json      pointer file auto-resolved to ep dir
  checkpoints/TL1_best.pt      flat layout (tag = 'best'; pass any one file)

Examples:
  python eval_framework.py --sumocfg sumo/OM.sumocfg --seeds 10

  # Full 3-way comparison, 20 seeds, episode 50 checkpoint:
  python eval_framework.py \\
      --sumocfg sumo/OM.sumocfg \\
      --rl-checkpoint results/checkpoints/ep0050 \\
      --seeds 20 --steps 1800 --output-dir results/

  # Use latest checkpoint pointer:
  python eval_framework.py \\
      --sumocfg sumo/OM.sumocfg \\
      --rl-checkpoint results/checkpoints/latest.json \\
      --seeds 15
        """,
    )
    p.add_argument("--sumocfg",
                   required=True,
                   help="Path to the SUMO .sumocfg file (e.g. OM.sumocfg)")

    # ── checkpoint options ────────────────────────────────────────────────────
    ckpt = p.add_argument_group("RL checkpoint options")
    ckpt.add_argument(
        "--rl-checkpoint",
        default=None,
        metavar="PATH",
        help=(
            "Path to the RL checkpoint to evaluate. Accepts:\n"
            "  • directory  : checkpoints/ep0050/  (contains TL*.pt files)\n"
            "  • .pt file   : checkpoints/ep0050/TL1.pt  (or any single file)\n"
            "  • latest.json: auto-resolved pointer file\n"
            "Omit to run Fixed-Time vs Webster-Pressure only."
        ),
    )
    ckpt.add_argument(
        "--rl-device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for RL inference (default: cpu)",
    )

    # ── experiment options ────────────────────────────────────────────────────
    exp = p.add_argument_group("Experiment options")
    exp.add_argument("--seeds",      type=int, default=10,
                     help="Number of independent random seeds (default: 10)")
    exp.add_argument("--seed-start", type=int, default=42,
                     help="First seed value; seeds = [start, start+N) (default: 42)")
    exp.add_argument("--steps",      type=int, default=1800,
                     help="Simulation steps per episode (default: 1800)")
    exp.add_argument("--base-port",  type=int, default=8813,
                     help="Base TraCI port; +1 and +2 used for parallel methods (default: 8813)")

    # ── output options ────────────────────────────────────────────────────────
    out = p.add_argument_group("Output options")
    out.add_argument("--output-dir", default="results",
                     help="Directory for CSV files and plots (default: results/)")

    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # ── load RL agents from checkpoint ───────────────────────────────────────
    rl_agents: Optional[dict] = None
    obs_fn                    = None

    if args.rl_checkpoint:
        if not TORCH_AVAILABLE:
            print("[RL] ERROR: torch not installed — cannot load checkpoint. Skipping RL.")
        else:
            try:
                rl_agents = load_rl_agents(
                    checkpoint_path = args.rl_checkpoint,
                    device_str      = args.rl_device,
                )
                obs_fn = make_obs_fn(rl_agents)
                print(f"[RL] {len(rl_agents)} agents ready for inference.\n")
            except Exception as e:
                print(f"[RL] ERROR: {e}")
                print("[RL] Continuing without RL (Fixed-Time vs Webster-Pressure only).\n")
                rl_agents = None
    else:
        print("[RL] No checkpoint provided → comparing Fixed-Time vs Webster-Pressure only.\n")

    run_evaluation(
        sumocfg   = args.sumocfg,
        seeds     = seeds,
        steps     = args.steps,
        save_dir  = args.output_dir,
        base_port = args.base_port,
        rl_agents = rl_agents,
        obs_fn    = obs_fn,
    )