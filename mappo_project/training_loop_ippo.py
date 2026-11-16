"""
training_loop_ippo.py
=====================
IPPO training loop for SmartTraffic-OM.

Import and usage
----------------
The existing ``training_loop.py`` is untouched.
This file adds:

  run_training_ippo(...)          — full training entry-point (mirrors run_training)
  _run_sequential_ippo(...)       — inner sequential episode loop
  _run_parallel_ippo(...)         — batched parallel episode loop
  _fill_ippo_rollout_buffer(...)  — batch-compute values/log-probs and push to buffer

The ``main()`` function in ``training_loop.py`` is patched (see Patch A at
the bottom of this file) to honour the ``--algorithm`` CLI flag:
  python main.py --sim_file sumo/OM.sumocfg --algorithm ippo

  python main.py --sim_file sumo/OM.sumocfg --algorithm mappo_gat  (default)

Centralised vs local critic
----------------------------
Pass ``--ippo_critic centralised`` to share one critic fed by all agents'
concatenated states.  Default is ``--ippo_critic local`` (decentralised).

All other hyperparameters (GAMMA, CLIP_RATIO, …) are read from config.py,
identical to the MAPPO+GAT run so comparisons are fair.
"""

from __future__ import annotations

import os
import sys
import time
import csv
import json
import random
import signal
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ── project imports ────────────────────────────────────────────────────────────
from .config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_RATIO, BATCH_SIZE, N_EPOCHS, HIDDEN_DIM, MEMORY_CAPACITY,
    TOTAL_STEPS, INTERSECTIONS, TLS_DETECTOR_MAP, UPSTREAM_LANES,
    DOWNSTREAM_LANES, NEIGHBOR_MAP, QUEUE_CAPACITY, UNIFIED_BP_CAPACITY,
    _QUEUE_CAPACITY_PER_TLS, EV_PRIORITY_ENABLED, MIN_GREEN_STEPS,
    STEP_LENGTH, REWARD_CLIP, BASE_PORT, FIXED_TIME_PLANS,
    RICH_AVAILABLE,
)
from .sumo_env import (
    _start_sumo, _close_sumo, _kill_sumo,
    _setup_all_subscriptions, _subscribe_lanes, _subscribe_detectors,
    _subscribe_vehicles_bulk,
    _STEP_CACHE, _reset_ep_stats, _snapshot_ep_stats, _collect_final_emissions,
    _get_tls_queue_from_detectors, _get_tls_mean_wait, _null_ctx,
    _ALL_SUBSCRIBED_LANES, _ALL_SUBSCRIBED_DETECTORS,
    _xml_output_args,
    TRACI_AVAILABLE,
)
from .networks import _TLS_INDEX
from .rewards import BackPressureReward, WaitingTimeReward, QueueReductionReward
from .ev_priority import DefaultEVPriority, NoEVPriority
from .visualization import (
    TrainingPlotter, ResultsWriter, TBComparisonLogger, RichUI,
    plot_comparison, export_comparison_excel,
)
from .metrics_and_eval import (
    CheckpointManager, EpisodeAutoSaver, FixedTimeController,
    _run_fixed_time_episode,
)
from .ippo import (
    IPPOAgent, IPPOBuffer, IPPOCentralCritic,
    save_ippo_checkpoint, load_ippo_checkpoint, build_global_state_vec,
)

try:
    import traci
    import traci.constants as tc
except ImportError:
    traci = None  # type: ignore

__all__ = [
    "run_training_ippo",
    "_run_sequential_ippo",
    "_run_parallel_ippo",
    "_fill_ippo_rollout_buffer",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helper: fill rollout buffer from raw transitions
# ══════════════════════════════════════════════════════════════════════════════

def _fill_ippo_rollout_buffer(
    ag:           IPPOAgent,
    transitions:  list,
    global_states_list: "list | None" = None,
) -> None:
    """
    Batch-compute V(s) and log π(a|s) for every transition, then push to
    the agent's rollout buffer.

    Parameters
    ----------
    ag : IPPOAgent
    transitions : list of (state, action, reward, next_state, done)
        Produced by the episode loop.
    global_states_list : list of np.ndarray | None
        Required when ag.critic_mode == "centralised".  Each entry is the
        concatenated global state at that timestep.
    """
    if not transitions:
        return

    all_s   = np.stack([t[0] for t in transitions])
    all_a   = np.array([t[1] for t in transitions], dtype=np.int64)
    all_ns  = np.stack([t[3] for t in transitions])
    all_d   = np.array([t[4] for t in transitions], dtype=np.float32)

    # Optional global states for centralised critic
    all_gs  = (np.stack(global_states_list)
               if global_states_list is not None else None)
    all_ngs = None
    if global_states_list is not None and len(global_states_list) > 0:
        # next-global-states: shift by 1, repeat last for terminal
        all_ngs = np.roll(all_gs, -1, axis=0)
        all_ngs[-1] = all_gs[-1]

    # Dimension guard
    if all_s.shape[1] != ag.state_dim:
        print(f"  [IPPO-DIM-ERROR] {ag.tls_id}: "
              f"state_dim={all_s.shape[1]} != {ag.state_dim} -- skipping",
              flush=True)
        return

    # Batch value and log-prob inference
    all_values    = ag.get_values_batch(all_s, all_gs)
    all_log_probs = ag.get_log_probs_batch(all_s, all_a)

    # Next-state values (0 for terminal steps)
    all_nv = np.zeros(len(transitions), dtype=np.float32)
    non_terminal = np.where(all_d == 0.0)[0]
    if len(non_terminal) > 0:
        all_nv[non_terminal] = ag.get_values_batch(
            all_ns[non_terminal],
            all_ngs[non_terminal] if all_ngs is not None else None,
        )

    for i, (s, a, r, ns, d) in enumerate(transitions):
        ag.rollout_buffer.push(
            s,
            a,
            float(r),
            float(all_values[i]),
            float(all_log_probs[i]),
            float(d),
            float(all_nv[i]),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Sequential training loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_sequential_ippo(
    agents:           Dict[str, IPPOAgent],
    sumo_cmd:         list,
    n_episodes:       int,
    sim_max_steps:    int,
    start_ep:         int,
    device:           torch.device,
    step_interval:    int,
    plotter,
    reward_writer,
    queue_writer,
    wait_writer,
    loss_writer,
    ckpt,
    proc_ref:         list,
    tb_logger         = None,
    fixed_controllers = None,
    cached_fx_result: "dict | None" = None,
    rl_history:       "dict | None" = None,
    fixed_history:    "dict | None" = None,
    rich_ui           = None,
    auto_saver        = None,
    emissions_writer  = None,
    # Centralised critic bookkeeping (passed through for checkpoint)
    shared_critic:               "IPPOCentralCritic | None" = None,
    shared_critic_optimizer:     "optim.Adam | None"              = None,
    shared_critic_scheduler      = None,
    seed_enable:      bool = False,
    base_seed:        int  = 42,
    fixed_seed:       bool = False,
) -> None:
    """Mirror of _run_training_sequential but for IPPOAgent."""

    _ui = rich_ui

    def _log(msg, style="dim white"):
        if _ui and RICH_AVAILABLE:
            _ui.log(msg, style=style)
        else:
            ts = time.strftime("%H:%M:%S")
            print(f"{ts}  {msg}", flush=True)

    is_first_ep = True
    for ep in range(start_ep, n_episodes):
        ckpt.update_episode(ep)

        _is_sumo_disconnect  = False
        _sumo_closed_by_user = False

        if _ui:
            _ui.begin_episode(ep)
        _ep_start_ms = time.time() * 1000.0

        # Reset per-episode buffers
        for ag in agents.values():
            ag.step_rewards.clear()
            ag.step_queues.clear()
            ag.step_losses.clear()
            ag.step_waits.clear()
            ag._ev_phi_prev = 0.0
            ag._ev_phi_curr = 0.0

        IPPOAgent.clear_lane_length_cache()

        ep_port = BASE_PORT + (ep % 500) + 1

        if seed_enable:
            ep_seed_val = base_seed if fixed_seed else base_seed + ep
            random.seed(ep_seed_val)
            np.random.seed(ep_seed_val)
            torch.manual_seed(ep_seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(ep_seed_val)
                torch.cuda.manual_seed_all(ep_seed_val)
            _ep_sumo_cmd = sumo_cmd + ["--seed", str(ep_seed_val)]
            _log(f"[IPPO Ep {ep}] Deterministic seed = {ep_seed_val}"
                 + (" (fixed)" if fixed_seed else f" (base+ep={base_seed}+{ep})"))
        else:
            _ep_sumo_cmd = sumo_cmd
            _log(f"[IPPO Ep {ep}] Starting SUMO on port {ep_port} …")

        _ep_sumo_cmd = list(_ep_sumo_cmd) + _xml_output_args(os.path.join("results", "ippo_xml"))
        try:
            _proc = _start_sumo(_ep_sumo_cmd, port=ep_port)
            proc_ref[0] = _proc
        except Exception as exc:
            _log(f"[ERROR] Ep {ep}: Cannot start SUMO: {exc}", style="bold red")
            import traceback; traceback.print_exc()
            break

        _setup_all_subscriptions(agents)

        for ag in agents.values():
            if is_first_ep and start_ep > 0:
                ag._current_step = 0
            else:
                ag.last_switch_step   = -ag.min_green_steps
                ag.current_action_idx = 0
                ag._current_step      = 0
            try:
                ag.state = ag._get_state()
            except Exception:
                ag.state = np.zeros(ag.state_dim, dtype=np.float32)
        is_first_ep = False

        _reset_ep_stats()
        _log(f"[IPPO Ep {ep}] Running {sim_max_steps} steps …")

        # Raw transitions: (state, action, reward, next_state, done)
        raw: Dict[str, list] = {tid: [] for tid in agents}
        # Global state per step (centralised critic)
        global_states_per_step: Dict[str, list] = {tid: [] for tid in agents}

        try:
            for step in range(1, sim_max_steps + 1):
                if step % 50 == 0:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        _log(f"[IPPO Ep {ep}] Early stop at step {step}")
                        break

                if step % step_interval == 0 and _ui:
                    _ui.update_step(step)

                _subscribe_vehicles_bulk()
                _STEP_CACHE.clear()

                # ── choose & apply actions ────────────────────────────────
                cur_states: Dict[str, np.ndarray] = {}
                actions:    Dict[str, int]         = {}

                # Build global state before step (needed for centralised critic)
                # Use current agent states (from previous step)
                first_ag = next(iter(agents.values()))
                gs_vec = (build_global_state_vec(agents, first_ag.state_dim)
                          if first_ag.critic_mode == "centralised" else None)

                for tid, ag in agents.items():
                    cur_states[tid] = ag.state.copy()
                    a = ag.choose_action(deterministic=False)
                    ag.current_action_idx = a
                    ag.apply_action(a, step)
                    actions[tid] = a

                traci.simulationStep()

                _STEP_CACHE.bulk_load_lanes()
                _STEP_CACHE.bulk_load_detectors()
                _STEP_CACHE.bulk_load_vehicles()

                done = 1.0 if step >= sim_max_steps else 0.0

                for tid, ag in agents.items():
                    ns     = ag._get_state()
                    reward = ag.get_reward()

                    raw[tid].append((
                        cur_states[tid],
                        actions[tid],
                        float(reward),
                        ns.copy(),
                        float(done),
                    ))
                    if gs_vec is not None:
                        global_states_per_step[tid].append(gs_vec.copy())

                    ag.state = ns
                    ag.step_rewards.append(reward)
                    ag.step_queues.append(_get_tls_queue_from_detectors(tid))
                    ag.step_waits.append(_get_tls_mean_wait(tid))

        except Exception as step_exc:
            _is_sumo_disconnect  = (
                "Connection closed by SUMO" in str(step_exc) or
                "FatalTraCIError" in type(step_exc).__name__)
            _sumo_closed_by_user = _is_sumo_disconnect
            if _is_sumo_disconnect:
                _log(f"[IPPO Ep {ep}] SUMO closed by user.", style="bold yellow")
            else:
                _log(f"[IPPO Ep {ep}] Step error: {step_exc}", style="bold red")

        # ── fill rollout buffers & train ───────────────────────────────────
        if _ui:
            _ui.set_phase("train")
        _log(f"[IPPO Ep {ep}] Training on rollouts …")

        for tid, ag in agents.items():
            gs_list = (global_states_per_step[tid]
                       if ag.critic_mode == "centralised" else None)
            _fill_ippo_rollout_buffer(ag, raw[tid], gs_list)

            # Global states array for train_epoch (centralised critic)
            gs_arr = (np.stack(gs_list) if gs_list else None)

            _use_shared = (ag.critic_mode == "centralised"
                           and shared_critic_optimizer is not None)
            if _use_shared:
                shared_critic_optimizer.zero_grad(set_to_none=True)

            loss = ag.train_step(global_states=gs_arr)

            if _use_shared:
                shared_critic_optimizer.step()

            if not ag.step_losses:
                ag.step_losses.append(loss)
            elif ag.step_losses[-1] == 0.0 and loss > 0.0:
                ag.step_losses[-1] = loss

        if shared_critic_scheduler is not None:
            shared_critic_scheduler.step()

        if _ui:
            _ui.set_phase("checkpoint")
        _log(f"[IPPO Ep {ep}] Closing SUMO …")
        _close_sumo(_proc)
        proc_ref[0] = None

        if _sumo_closed_by_user:
            if not any(ag.step_rewards for ag in agents.values()):
                ckpt.save_now(emergency=True)
                break
            _log(f"[IPPO Ep {ep}] Partial data — saving.")

        # ── episode metrics ────────────────────────────────────────────────
        ep_reward, ep_queue, ep_wait, ep_loss = {}, {}, {}, {}
        ep_queue_peak, ep_wait_p95 = {}, {}
        for tl in INTERSECTIONS:
            ag = agents.get(tl)
            if ag and ag.step_rewards:
                ep_reward[tl] = float(np.mean(ag.step_rewards))
                ep_queue[tl]  = float(np.mean(ag.step_queues))
                ep_wait[tl]   = (float(np.mean(ag.step_waits))
                                 if ag.step_waits else 0.0)
                ep_loss[tl]   = (float(np.mean(ag.step_losses))
                                 if ag.step_losses else 0.0)
                ep_queue_peak[tl] = float(np.max(ag.step_queues))
                ep_wait_p95[tl]   = float(np.percentile(ag.step_waits, 95))
            else:
                ep_reward[tl] = ep_queue[tl] = ep_wait[tl] = ep_loss[tl] = 0.0
                ep_queue_peak[tl] = ep_wait_p95[tl] = 0.0

        ep_duration_ms = time.time() * 1000.0 - _ep_start_ms

        plotter.record(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                       ep_duration_ms=ep_duration_ms)
        plotter.record_steps(agents)

        reward_writer.write_row({"episode": ep, **ep_reward,
                                 "ep_duration_ms": round(ep_duration_ms, 1)})
        _q_peak = {f"{tl}_peak": ep_queue_peak.get(tl, 0.0)
                   for tl in INTERSECTIONS}
        _w_p95  = {f"{tl}_p95": ep_wait_p95.get(tl, 0.0)
                   for tl in INTERSECTIONS}
        queue_writer.write_row({"episode": ep, **ep_queue,
                                "ep_duration_ms": round(ep_duration_ms, 1),
                                **_q_peak})
        wait_writer.write_row({"episode": ep, **ep_wait,
                               "ep_duration_ms": round(ep_duration_ms, 1),
                               **_w_p95})
        loss_writer.write_row({"episode": ep, **ep_loss,
                               "ep_duration_ms": round(ep_duration_ms, 1),
                               "batch_no": 0,
                               "baseline_is_cached": True,
                               "parallel_batch_loss": False})

        # ── emissions CSV ──────────────────────────────────────────────────
        if emissions_writer is not None:
            _collect_final_emissions()
            _em_rl   = _snapshot_ep_stats()
            _co2_fx  = (cached_fx_result.get("__co2_total",  0.0)
                        if cached_fx_result else 0.0)
            _nox_fx  = (cached_fx_result.get("__nox_total",  0.0)
                        if cached_fx_result else 0.0)
            _fuel_fx = (cached_fx_result.get("__fuel_total", 0.0)
                        if cached_fx_result else 0.0)
            _co2_rl  = round(_em_rl["co2_sum"],  2)
            _nox_rl  = round(_em_rl["nox_sum"],  2)
            _fuel_rl = round(_em_rl["fuel_sum"], 2)
            _pct = lambda rl, fx: (
                round(((fx - rl) / max(fx, 1e-6)) * 100, 2) if fx else 0.0)
            emissions_writer.write_row({
                "episode": ep, "ep_duration_ms": round(ep_duration_ms, 1),
                "co2_rl_mg":  _co2_rl,
                "nox_rl_mg":  _nox_rl,
                "fuel_rl_mg": _fuel_rl,
                "co2_fx_mg":  round(_co2_fx,  2),
                "nox_fx_mg":  round(_nox_fx,  2),
                "fuel_fx_mg": round(_fuel_fx, 2),
                "co2_saving_pct":  _pct(_co2_rl,  _co2_fx),
                "nox_saving_pct":  _pct(_nox_rl,  _nox_fx),
                "fuel_saving_pct": _pct(_fuel_rl, _fuel_fx),
                "baseline_is_cached": True,
            })

        # ── TensorBoard ────────────────────────────────────────────────────
        if tb_logger is not None:
            tb_logger.log_episode_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss)

        # ── Fixed-Time comparison ─────────────────────────────────────────
        FT_REFRESH_INTERVAL = 50
        if (fixed_controllers and ep % FT_REFRESH_INTERVAL == 0 and ep > 0):
            _ft_seed_label_ip = (f"seed={base_seed} (fixed)"
                                 if fixed_seed else "random seed")
            if fixed_seed and seed_enable:
                _ft_refresh_cmd = sumo_cmd + ["--seed", str(base_seed)]
            else:
                _ft_refresh_cmd = sumo_cmd
            try:
                _log(f"[IPPO Ep {ep}] Refreshing Fixed-Time baseline ({_ft_seed_label_ip})…")
                _sr = _run_fixed_time_episode(
                    _ft_refresh_cmd, max(900, sim_max_steps),
                    BASE_PORT + 701, fixed_controllers)
                cached_fx_result = {
                    k: v for k, v in _sr.items() if not k.startswith("__")}
                for _k in ("__ev_cleared", "__ev_avg_wait", "__cars_passed",
                           "__co2_total", "__nox_total", "__fuel_total"):
                    cached_fx_result[_k] = _sr.get(_k, 0.0)
                cached_fx_result["__ft_baseline_ep"] = ep
            except Exception as e:
                _log(f"[IPPO Ep {ep}] FT refresh failed: {e}")

        fx_reward = {tl: 0.0 for tl in INTERSECTIONS}
        fx_queue  = {tl: 0.0 for tl in INTERSECTIONS}
        fx_wait   = {tl: 0.0 for tl in INTERSECTIONS}
        if fixed_controllers and cached_fx_result:
            if _ui:
                _ui.begin_fixed(ep)
            for tl, data in cached_fx_result.items():
                if tl.startswith("__"):
                    continue
                fx_reward[tl] = (float(np.mean(data["rewards"]))
                                 if data["rewards"] else 0.0)
                fx_queue[tl]  = (float(np.mean(data["queues"]))
                                 if data["queues"] else 0.0)
                fx_wait[tl]   = (float(np.mean(data["waits"]))
                                 if data["waits"] else 0.0)

        # ── Histories ─────────────────────────────────────────────────────
        _rl_ep_stats = _snapshot_ep_stats()
        _rl_ev_avg   = (
            round(_rl_ep_stats["ev_wait_sum"] / _rl_ep_stats["ev_cleared"], 3)
            if _rl_ep_stats["ev_cleared"] > 0 else None)

        if rl_history is not None:
            for tl in INTERSECTIONS:
                rl_history["reward"][tl].append(ep_reward.get(tl, 0.0))
                rl_history["queue"][tl].append(ep_queue.get(tl, 0.0))
                rl_history["wait"][tl].append(ep_wait.get(tl, 0.0))
            rl_history["ev_cleared"].append(_rl_ep_stats["ev_cleared"])
            rl_history["ev_avg_wait"].append(_rl_ev_avg)
            rl_history["cars_passed"].append(_rl_ep_stats["cars_passed"])
            rl_history["co2_total"].append(round(_rl_ep_stats["co2_sum"], 2))
            rl_history["nox_total"].append(round(_rl_ep_stats["nox_sum"], 2))
            rl_history["fuel_total"].append(round(_rl_ep_stats["fuel_sum"], 2))

        if fixed_history is not None:
            for tl in INTERSECTIONS:
                fixed_history["reward"][tl].append(fx_reward.get(tl, 0.0))
                fixed_history["queue"][tl].append(fx_queue.get(tl, 0.0))
                fixed_history["wait"][tl].append(fx_wait.get(tl, 0.0))
            fixed_history["ev_cleared"].append(
                cached_fx_result.get("__ev_cleared", 0)
                if cached_fx_result else 0)
            fixed_history["ev_avg_wait"].append(
                cached_fx_result.get("__ev_avg_wait", None)
                if cached_fx_result else None)
            fixed_history["cars_passed"].append(
                cached_fx_result.get("__cars_passed", 0)
                if cached_fx_result else 0)
            fixed_history["co2_total"].append(
                cached_fx_result.get("__co2_total", 0.0)
                if cached_fx_result else 0.0)
            fixed_history["nox_total"].append(
                cached_fx_result.get("__nox_total", 0.0)
                if cached_fx_result else 0.0)
            fixed_history["fuel_total"].append(
                cached_fx_result.get("__fuel_total", 0.0)
                if cached_fx_result else 0.0)

        # ── Rich UI ───────────────────────────────────────────────────────
        if _ui:
            _ui.record_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                          ep_stats=_rl_ep_stats)
            if fixed_controllers and cached_fx_result:
                _ui.record_fixed(fx_reward, fx_queue, fx_wait,
                                 fx_stats=cached_fx_result)

        # ── Auto-save ─────────────────────────────────────────────────────
        auto_saver.save_episode(
            ep, ep_reward, ep_queue, ep_wait, ep_loss,
            fx_reward, fx_queue, fx_wait)
        if ep % auto_saver.save_every == 0 or ep == n_episodes - 1:
            auto_saver.tag_latest(ep)

        if _sumo_closed_by_user:
            _log("[INFO] SUMO GUI closed — stopping.", style="bold yellow")
            ckpt.save_now(emergency=True)
            break

        _log(f"[IPPO Ep {ep}] Done. "
             f"avg_r={float(np.mean(list(ep_reward.values()))):+.4f}  "
             f"avg_q={float(np.mean(list(ep_queue.values()))):.2f}  "
             f"avg_w={float(np.mean(list(ep_wait.values()))):.2f}",
             style="dim cyan")

    _log("IPPO sequential loop complete.", style="bold green")


# ══════════════════════════════════════════════════════════════════════════════
# Parallel (batch) training loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_parallel_ippo(
    agents:           Dict[str, IPPOAgent],
    sumo_cmd:         list,
    n_episodes:       int,
    sim_max_steps:    int,
    start_ep:         int,
    n_workers:        int,
    device:           torch.device,
    plotter,
    reward_writer,
    queue_writer,
    wait_writer,
    loss_writer,
    ckpt,
    proc_ref:         list,
    tb_logger         = None,
    fixed_controllers = None,
    cached_fx_result: "dict | None" = None,
    rl_history:       "dict | None" = None,
    fixed_history:    "dict | None" = None,
    rich_ui           = None,
    auto_saver        = None,
    emissions_writer  = None,
    shared_critic:    "IPPOCentralCritic | None" = None,
    shared_critic_optimizer: "optim.Adam | None" = None,
    shared_critic_scheduler = None,
    seed_enable:      bool = False,
    base_seed:        int  = 42,
    fixed_seed:       bool = False,
) -> None:
    """
    Batched-sequential parallel mode for IPPO.
    n_workers episodes are collected sequentially (same process, same agents),
    then one shared PPO update is applied.
    On Windows this is always n_workers=1 (forced by run_training_ippo).
    """
    _ui = rich_ui

    def _log(msg, style="dim white"):
        if _ui and RICH_AVAILABLE:
            _ui.log(msg, style=style)
        else:
            print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)

    batch_no = 0
    for batch_start in range(start_ep, n_episodes, n_workers):
        batch_eps    = list(range(batch_start,
                                  min(batch_start + n_workers, n_episodes)))
        batch_no    += 1
        ep_port_base = BASE_PORT + (batch_start % 400) + 1

        _log(f"[IPPO Batch {batch_no}] "
             f"Eps {batch_eps[0]}-{batch_eps[-1]} …")

        batch_rewards = {tl: [] for tl in INTERSECTIONS}
        batch_queues  = {tl: [] for tl in INTERSECTIONS}
        batch_waits   = {tl: [] for tl in INTERSECTIONS}
        batch_peaks   = {tl: [] for tl in INTERSECTIONS}
        batch_p95s    = {tl: [] for tl in INTERSECTIONS}
        batch_dur_ms: list[float] = []
        batch_em_stats: list[dict] = []
        # Accumulates centralised-critic global-state vectors across every
        # episode in the batch, in the same order transitions are pushed to
        # each agent's rollout_buffer, so train_step() receives an array
        # correctly aligned with buf.states[:buf.size].
        batch_global_states: Dict[str, list] = {tid: [] for tid in agents}

        for w_idx, ep in enumerate(batch_eps):
            ckpt.update_episode(ep)
            ep_port      = ep_port_base + w_idx
            _par_start   = time.time() * 1000.0

            for ag in agents.values():
                ag.step_rewards.clear(); ag.step_queues.clear()
                ag.step_losses.clear();  ag.step_waits.clear()
                ag._ev_phi_prev = ag._ev_phi_curr = 0.0
            IPPOAgent.clear_lane_length_cache()
            _reset_ep_stats()

            try:
                if seed_enable:
                    ep_seed_val = base_seed if fixed_seed else base_seed + ep
                    random.seed(ep_seed_val)
                    np.random.seed(ep_seed_val)
                    torch.manual_seed(ep_seed_val)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(ep_seed_val)
                        torch.cuda.manual_seed_all(ep_seed_val)
                    _ep_sumo_cmd = sumo_cmd + ["--seed", str(ep_seed_val)]
                    _log(f"[IPPO Ep {ep}] Deterministic seed = {ep_seed_val}"
                         + (" (fixed)" if fixed_seed else f" (base+ep={base_seed}+{ep})"))
                else:
                    _ep_sumo_cmd = sumo_cmd
                _ep_sumo_cmd = list(_ep_sumo_cmd) + _xml_output_args(os.path.join("results", "ippo_xml"))
                _proc = _start_sumo(_ep_sumo_cmd, port=ep_port)
                proc_ref[0] = _proc
            except Exception as exc:
                _log(f"[ERROR] Ep {ep}: SUMO failed: {exc}", style="bold red")
                batch_dur_ms.append(0.0)
                batch_em_stats.append({})
                for tl in INTERSECTIONS:
                    batch_rewards[tl].append(0.0)
                    batch_queues[tl].append(0.0)
                    batch_waits[tl].append(0.0)
                    batch_peaks[tl].append(0.0)
                    batch_p95s[tl].append(0.0)
                continue

            _setup_all_subscriptions(agents)
            for ag in agents.values():
                ag.last_switch_step = -ag.min_green_steps
                ag.current_action_idx = ag._current_step = 0
                try:
                    ag.state = ag._get_state()
                except Exception:
                    ag.state = np.zeros(ag.state_dim, dtype=np.float32)

            raw:              Dict[str, list] = {tid: [] for tid in agents}
            gs_per_step:      Dict[str, list] = {tid: [] for tid in agents}
            first_ag = next(iter(agents.values()))
            use_centralised  = first_ag.critic_mode == "centralised"

            try:
                for step in range(1, sim_max_steps + 1):
                    if step % 50 == 0:
                        if traci.simulation.getMinExpectedNumber() == 0:
                            break
                    _subscribe_vehicles_bulk()
                    _STEP_CACHE.clear()

                    gs_vec = (build_global_state_vec(agents, first_ag.state_dim)
                              if use_centralised else None)
                    cur_states: Dict[str, np.ndarray] = {}
                    actions:    Dict[str, int]         = {}

                    for tid, ag in agents.items():
                        cur_states[tid] = ag.state.copy()
                        a = ag.choose_action()
                        ag.current_action_idx = a
                        ag.apply_action(a, step)
                        actions[tid] = a

                    traci.simulationStep()
                    _STEP_CACHE.bulk_load_lanes()
                    _STEP_CACHE.bulk_load_detectors()
                    _STEP_CACHE.bulk_load_vehicles()

                    done = 1.0 if step >= sim_max_steps else 0.0
                    for tid, ag in agents.items():
                        ns     = ag._get_state()
                        reward = ag.get_reward()
                        raw[tid].append((cur_states[tid], actions[tid],
                                         float(reward), ns.copy(), float(done)))
                        if gs_vec is not None:
                            gs_per_step[tid].append(gs_vec.copy())
                        ag.state = ns
                        ag.step_rewards.append(reward)
                        ag.step_queues.append(_get_tls_queue_from_detectors(tid))
                        ag.step_waits.append(_get_tls_mean_wait(tid))

            except Exception as e:
                _log(f"[IPPO Ep {ep}] Step error: {e}", style="bold red")

            # Fill rollout buffers (accumulate across episodes in batch)
            for tid, ag in agents.items():
                gs_list = gs_per_step[tid] if use_centralised else None
                _fill_ippo_rollout_buffer(ag, raw[tid], gs_list)
                if use_centralised and gs_list:
                    batch_global_states[tid].extend(gs_list)

            _collect_final_emissions()
            batch_em_stats.append(_snapshot_ep_stats())
            _close_sumo(_proc)
            proc_ref[0] = None
            batch_dur_ms.append(time.time() * 1000.0 - _par_start)

            for tl in INTERSECTIONS:
                ag = agents.get(tl)
                if ag and ag.step_rewards:
                    batch_rewards[tl].append(float(np.mean(ag.step_rewards)))
                    batch_queues[tl].append(float(np.mean(ag.step_queues)))
                    batch_waits[tl].append(float(np.mean(ag.step_waits))
                                           if ag.step_waits else 0.0)
                    batch_peaks[tl].append(float(np.max(ag.step_queues)))
                    batch_p95s[tl].append(float(np.percentile(ag.step_waits, 95))
                                          if ag.step_waits else 0.0)
                else:
                    for lst in (batch_rewards, batch_queues, batch_waits,
                                batch_peaks, batch_p95s):
                        lst[tl].append(0.0)

        # ── Shared PPO update (once per batch) ────────────────────────────
        batch_loss = {tl: 0.0 for tl in INTERSECTIONS}
        first_ag   = next(iter(agents.values()))
        for tid, ag in agents.items():
            if ag.rollout_buffer.size > 0:
                gs_arr = None
                if use_centralised and batch_global_states.get(tid):
                    # IPPOBuffer is a fixed-capacity ring buffer: once
                    # more than `capacity` transitions have been pushed this
                    # batch, buf.states[:buf.size] holds only the most recent
                    # `capacity` entries (oldest ones were overwritten).
                    # Truncate the global-states list the same way so row i
                    # of gs_arr lines up with row i of buf.states[:buf.size].
                    _cap = ag.rollout_buffer.capacity
                    _gs_list = batch_global_states[tid][-_cap:]
                    if len(_gs_list) == ag.rollout_buffer.size:
                        gs_arr = np.stack(_gs_list)
                    else:
                        _log(f"[IPPO-WARN] {tid}: global_states count "
                             f"({len(_gs_list)}) != buffer size "
                             f"({ag.rollout_buffer.size}) — skipping "
                             f"centralised critic update this batch.",
                             style="bold yellow")
                _use_shared = (use_centralised
                               and shared_critic_optimizer is not None)
                if _use_shared:
                    shared_critic_optimizer.zero_grad(set_to_none=True)

                loss = ag.train_step(global_states=gs_arr)

                if _use_shared:
                    shared_critic_optimizer.step()

                batch_loss[tid] = loss
                if not ag.step_losses:
                    ag.step_losses.append(loss)

        if shared_critic_scheduler is not None:
            shared_critic_scheduler.step()

        # ── Record each episode in the batch ──────────────────────────────
        for w_idx, ep in enumerate(batch_eps):
            ep_reward = {tl: batch_rewards[tl][w_idx]
                         if w_idx < len(batch_rewards[tl]) else 0.0
                         for tl in INTERSECTIONS}
            ep_queue  = {tl: batch_queues[tl][w_idx]
                         if w_idx < len(batch_queues[tl])  else 0.0
                         for tl in INTERSECTIONS}
            ep_wait   = {tl: batch_waits[tl][w_idx]
                         if w_idx < len(batch_waits[tl])   else 0.0
                         for tl in INTERSECTIONS}
            ep_loss   = batch_loss
            _ep_dur   = batch_dur_ms[w_idx] if w_idx < len(batch_dur_ms) else 0.0
            _q_peak   = {f"{tl}_peak": batch_peaks[tl][w_idx]
                         if w_idx < len(batch_peaks[tl]) else 0.0
                         for tl in INTERSECTIONS}
            _w_p95    = {f"{tl}_p95": batch_p95s[tl][w_idx]
                         if w_idx < len(batch_p95s[tl]) else 0.0
                         for tl in INTERSECTIONS}

            plotter.record(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                           ep_duration_ms=_ep_dur)
            reward_writer.write_row({"episode": ep, **ep_reward,
                                     "ep_duration_ms": round(_ep_dur, 1)})
            queue_writer.write_row({"episode": ep, **ep_queue,
                                    "ep_duration_ms": round(_ep_dur, 1),
                                    **_q_peak})
            wait_writer.write_row({"episode": ep, **ep_wait,
                                   "ep_duration_ms": round(_ep_dur, 1),
                                   **_w_p95})
            loss_writer.write_row({"episode": ep, **ep_loss,
                                   "ep_duration_ms": round(_ep_dur, 1),
                                   "batch_no": batch_no,
                                   "baseline_is_cached": True,
                                   "parallel_batch_loss": True})

            if emissions_writer is not None:
                _em_rl_p = (batch_em_stats[w_idx]
                            if w_idx < len(batch_em_stats) else {})
                _co2_fx_p  = (cached_fx_result.get("__co2_total",  0.0)
                              if cached_fx_result else 0.0)
                _nox_fx_p  = (cached_fx_result.get("__nox_total",  0.0)
                              if cached_fx_result else 0.0)
                _fuel_fx_p = (cached_fx_result.get("__fuel_total", 0.0)
                              if cached_fx_result else 0.0)
                _co2_rl_p  = round(_em_rl_p.get("co2_sum",  0.0), 2)
                _nox_rl_p  = round(_em_rl_p.get("nox_sum",  0.0), 2)
                _fuel_rl_p = round(_em_rl_p.get("fuel_sum", 0.0), 2)
                _pct_p = lambda rl, fx: (
                    round(((fx - rl) / max(fx, 1e-6)) * 100, 2) if fx else 0.0)
                emissions_writer.write_row({
                    "episode": ep, "ep_duration_ms": round(_ep_dur, 1),
                    "co2_rl_mg":  _co2_rl_p,
                    "nox_rl_mg":  _nox_rl_p,
                    "fuel_rl_mg": _fuel_rl_p,
                    "co2_fx_mg":  round(_co2_fx_p,  2),
                    "nox_fx_mg":  round(_nox_fx_p,  2),
                    "fuel_fx_mg": round(_fuel_fx_p, 2),
                    "co2_saving_pct":  _pct_p(_co2_rl_p,  _co2_fx_p),
                    "nox_saving_pct":  _pct_p(_nox_rl_p,  _nox_fx_p),
                    "fuel_saving_pct": _pct_p(_fuel_rl_p, _fuel_fx_p),
                    "baseline_is_cached": True,
                })

            if tb_logger is not None:
                tb_logger.log_episode_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss)

            fx_reward = {tl: 0.0 for tl in INTERSECTIONS}
            fx_queue  = {tl: 0.0 for tl in INTERSECTIONS}
            fx_wait   = {tl: 0.0 for tl in INTERSECTIONS}
            if fixed_controllers and cached_fx_result:
                for tl, data in cached_fx_result.items():
                    if tl.startswith("__"):
                        continue
                    fx_reward[tl] = (float(np.mean(data["rewards"]))
                                     if data["rewards"] else 0.0)
                    fx_queue[tl]  = (float(np.mean(data["queues"]))
                                     if data["queues"] else 0.0)
                    fx_wait[tl]   = (float(np.mean(data["waits"]))
                                     if data["waits"] else 0.0)

            _rl_ep   = _snapshot_ep_stats()
            _ev_avg  = (round(_rl_ep["ev_wait_sum"] / _rl_ep["ev_cleared"], 3)
                        if _rl_ep["ev_cleared"] > 0 else None)

            if rl_history is not None:
                for tl in INTERSECTIONS:
                    rl_history["reward"][tl].append(ep_reward.get(tl, 0.0))
                    rl_history["queue"][tl].append(ep_queue.get(tl, 0.0))
                    rl_history["wait"][tl].append(ep_wait.get(tl, 0.0))
                rl_history["ev_cleared"].append(_rl_ep["ev_cleared"])
                rl_history["ev_avg_wait"].append(_ev_avg)
                rl_history["cars_passed"].append(_rl_ep["cars_passed"])
                rl_history["co2_total"].append(round(_rl_ep["co2_sum"], 2))
                rl_history["nox_total"].append(round(_rl_ep["nox_sum"], 2))
                rl_history["fuel_total"].append(round(_rl_ep["fuel_sum"], 2))

            if fixed_history is not None:
                for tl in INTERSECTIONS:
                    fixed_history["reward"][tl].append(fx_reward.get(tl, 0.0))
                    fixed_history["queue"][tl].append(fx_queue.get(tl, 0.0))
                    fixed_history["wait"][tl].append(fx_wait.get(tl, 0.0))
                fixed_history["ev_cleared"].append(
                    cached_fx_result.get("__ev_cleared", 0)
                    if cached_fx_result else 0)
                fixed_history["ev_avg_wait"].append(
                    cached_fx_result.get("__ev_avg_wait", None)
                    if cached_fx_result else None)
                fixed_history["cars_passed"].append(
                    cached_fx_result.get("__cars_passed", 0)
                    if cached_fx_result else 0)
                fixed_history["co2_total"].append(
                    cached_fx_result.get("__co2_total",  0.0)
                    if cached_fx_result else 0.0)
                fixed_history["nox_total"].append(
                    cached_fx_result.get("__nox_total",  0.0)
                    if cached_fx_result else 0.0)
                fixed_history["fuel_total"].append(
                    cached_fx_result.get("__fuel_total", 0.0)
                    if cached_fx_result else 0.0)

            if _ui:
                _ui.record_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                              ep_stats=_rl_ep)
                if fixed_controllers and cached_fx_result:
                    _ui.record_fixed(fx_reward, fx_queue, fx_wait,
                                     fx_stats=cached_fx_result)

            auto_saver.save_episode(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                                    fx_reward, fx_queue, fx_wait)
            if ep % auto_saver.save_every == 0 or ep == n_episodes - 1:
                auto_saver.tag_latest(ep)

        _log(f"[IPPO Batch {batch_no}] Done. "
             f"avg_r={float(np.mean([v for vals in batch_rewards.values() for v in vals])):+.4f}",
             style="dim cyan")

    _log("IPPO parallel loop complete.", style="bold green")


# ══════════════════════════════════════════════════════════════════════════════
# Main entry-point: run_training_ippo
# ══════════════════════════════════════════════════════════════════════════════

def run_training_ippo(
    sim_file:       str,
    n_episodes:     int  = 1,
    sim_max_steps:  int  = 3600,
    gui:            str  = "disable",
    save_dir:       str  = "./results_ippo",
    resume_from:    "str | None" = None,
    n_workers:      int  = 1,
    save_every:     int  = 10,
    eval_interval:  int  = 10,
    reward_type:    str  = "backpressure",
    ev_priority_type: str = "default",
    # Vanilla-PPO specific
    critic_mode:    str  = "local",
    adaptive_clip:  bool = True,
    seed_enable:    bool = False,
    base_seed:      int  = 42,
    fixed_seed:     bool = False,
    reset_lr:       bool = False,
    cuda_dynamic:   bool = True,
) -> None:
    """
    Full training entry-point for IPPO.

    Parameters mirror run_training() in training_loop.py.
    ``--aggregator`` is absent (no intent sharing by definition).

    critic_mode : {"local", "centralised"}
        "local"       — independent per-agent critic  (fast, stable)
        "centralised" — shared critic over all agents (better baselines)
    """
    if not TRACI_AVAILABLE:
        raise RuntimeError("traci is not available.")

    # ── Experiment configuration ──────────────────────────────────────────
    REWARD_MAP = {
        "backpressure": BackPressureReward(),
        "waiting_time": WaitingTimeReward(),
        "queue":        QueueReductionReward(),
    }
    EV_MAP = {
        "default": DefaultEVPriority(),
        "none":    NoEVPriority(),
    }
    _reward_func = REWARD_MAP.get(reward_type,      BackPressureReward())
    _ev_module   = EV_MAP.get(ev_priority_type,     DefaultEVPriority())

    # Propagate EV_PRIORITY_ENABLED=False into modules that cache the flag at
    # import time (sumo_env, rewards) when EV priority is disabled.
    if ev_priority_type == "none":
        import sys as _sys
        _ev_flag_modules = [
            __name__.rsplit(".", 1)[0] + ".sumo_env",
            __name__.rsplit(".", 1)[0] + ".rewards",
        ]
        for _mod_name in _ev_flag_modules:
            if _mod_name in _sys.modules:
                _sys.modules[_mod_name].EV_PRIORITY_ENABLED = False
        print("[INFO] Emergency-vehicle reward shaping: DISABLED", flush=True)
    else:
        print("[INFO] Emergency-vehicle reward shaping: ENABLED", flush=True)

    import platform as _plat
    _is_windows  = _plat.system() == "Windows"
    if _is_windows and n_workers > 1:
        print(f"[WARNING] Parallel mode (n_workers={n_workers}) is not "
              f"supported on Windows — forcing n_workers=1.", flush=True)
        n_workers = 1

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")

    sumo_binary = (("sumo-gui.exe" if gui == "enable" else "sumo.exe")
                   if _is_windows
                   else ("sumo-gui" if gui == "enable" else "sumo"))
    sumo_cmd = [
        sumo_binary,
        "-c", sim_file,
        "--step-length",      str(int(STEP_LENGTH)),
        "--no-step-log",
        "--time-to-teleport", "300",
        "--collision.action", "warn",
    ]

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if seed_enable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            torch.backends.cudnn.benchmark        = False
            torch.backends.cudnn.deterministic    = True
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif not cuda_dynamic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32       = False
            torch.backends.cudnn.benchmark        = False
            torch.backends.cudnn.deterministic    = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32       = True
            torch.backends.cudnn.benchmark        = True
            torch.backends.cudnn.deterministic    = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    else:
        device  = torch.device("cpu")
        n_cpu   = os.cpu_count() or 4
        try:
            torch.set_num_threads(max(1, n_cpu - 1))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    parallel_mode = n_workers > 1
    mode_label    = f"PARALLEL x{n_workers}" if parallel_mode else "SEQUENTIAL"

    print(f"\n{'='*70}")
    print(f"  OM Network  |  IPPO (No Intent Sharing — Baseline)")
    print(f"  Device      : {device}")
    print(f"  Mode        : {mode_label}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Steps/ep    : {sim_max_steps}")
    print(f"  Reward      : {_reward_func.name}")
    print(f"  Critic mode : {critic_mode}")
    print(f"  EV Priority : {_ev_module.name}")
    print(f"  γ={GAMMA}  lr={LEARNING_RATE}  λ={GAE_LAMBDA}  "
          f"clip={CLIP_RATIO}  entropy={ENTROPY_COEFF}")
    print(f"{'='*70}\n")

    # ── Seed before agent network initialisation ─────────────────────────
    if seed_enable:
        _seed_msg = f"  Seeding torch/numpy/random with base_seed={base_seed}"
        random.seed(base_seed)
        np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed)
            torch.cuda.manual_seed_all(base_seed)
        print(_seed_msg, flush=True)

    # ── Bootstrap SUMO to read topology ───────────────────────────────────
    print("  Initialising agents (bootstrap SUMO) …")
    _proc = _start_sumo(sumo_cmd, port=BASE_PORT)

    shared_critic: "IPPOCentralCritic | None" = None
    agents: Dict[str, IPPOAgent]              = {}

    try:
        # We need a sample state_dim before building the centralised critic.
        # Construct agents first with mode="local", then upgrade if needed.
        _temp_agents: Dict[str, IPPOAgent] = {}
        for tls_id in traci.trafficlight.getIDList():
            try:
                _temp_agents[tls_id] = IPPOAgent(
                    tls_id, device,
                    critic_mode       = "local",   # temp — upgraded below
                    reward_func       = _reward_func,
                    ev_module         = _ev_module,
                    use_adaptive_clip = adaptive_clip,
                )
            except RuntimeError as e:
                print(f"    [WARN] Skipping {tls_id}: {e}")

        if not _temp_agents:
            raise RuntimeError("No usable TLS agents.")

        if critic_mode == "centralised":
            # Build shared critic whose input = sum of ALL agents' state_dims.
            # Agents have different state_dims (different numbers of lanes/
            # detectors per intersection), so we cannot assume they are equal.
            _total_state_dim = sum(
                ag.state_dim for tls_id, ag in _temp_agents.items()
                if tls_id in INTERSECTIONS
            )
            # Use n_agents=1 so the critic's in_dim = _total_state_dim * 1
            shared_critic = IPPOCentralCritic(
                n_agents  = 1,
                state_dim = _total_state_dim,
            ).to(device)
            # Now rebuild agents with centralised mode
            for tls_id, _ag in _temp_agents.items():
                agents[tls_id] = IPPOAgent(
                    tls_id, device,
                    critic_mode       = "centralised",
                    shared_critic     = shared_critic,
                    reward_func       = _reward_func,
                    ev_module         = _ev_module,
                    use_adaptive_clip = adaptive_clip,
                )
        else:
            agents = _temp_agents

        for tls_id, ag in agents.items():
            print(f"    Agent [{tls_id}]"
                  f"  state_dim={ag.state_dim}"
                  f"  action_dim={ag.action_dim}"
                  f"  critic={ag.critic_mode}"
                  f"  phases={ag.action_phases}")

    except Exception as exc:
        _close_sumo(_proc)
        raise RuntimeError(f"Agent init failed: {exc}")
    _close_sumo(_proc)

    # ── Shared critic optimiser ───────────────────────────────────────────
    shared_critic_optimizer: "optim.Adam | None" = None
    shared_critic_scheduler = None
    if shared_critic is not None:
        shared_critic_optimizer = optim.Adam(
            shared_critic.parameters(), lr=LEARNING_RATE)

    # ── Resize rollout buffers to actual sim_max_steps ────────────────────
    if sim_max_steps != MEMORY_CAPACITY:
        print(f"  [FIX] Resizing rollout buffers: "
              f"{MEMORY_CAPACITY} -> {sim_max_steps}")
        for ag in agents.values():
            ag.rollout_buffer = IPPOBuffer(sim_max_steps, ag.state_dim)

    # ── Resume ────────────────────────────────────────────────────────────
    _resume_tag: "str | None" = None
    start_ep = 0

    if resume_from is not None:
        _ckpt_dir_early = os.path.join(save_dir, "checkpoints")

        if resume_from == "latest":
            _ptr_json = os.path.join(_ckpt_dir_early, "latest.json")
            _ptr_link = os.path.join(_ckpt_dir_early, "latest")

            # 1) Preferred: latest.json pointer.
            if os.path.isfile(_ptr_json):
                try:
                    with open(_ptr_json) as pf:
                        _d = json.load(pf)
                    _resume_tag = _d.get("tag") or f"ep{int(_d['latest_ep']):04d}"
                    print(f"  [IPPO-RESUME] 'latest' resolved via latest.json -> {_resume_tag}",
                          flush=True)
                except Exception as e:
                    print(f"  [IPPO-RESUME] WARNING: latest.json parse failed: {e}",
                          flush=True)

            # 2) Fallback: symlink pointer (mirrors training_loop.py behaviour).
            if _resume_tag is None and os.path.islink(_ptr_link):
                _resume_tag = os.path.basename(os.readlink(_ptr_link))
                print(f"  [IPPO-RESUME] 'latest' resolved via symlink -> {_resume_tag}",
                      flush=True)

            # 3) Fallback: scan checkpoints/ for the highest ep#### directory
            #    that actually contains agent weight files. Covers the case
            #    where latest.json was never written (e.g. interrupted mid
            #    save, or an older run predating the pointer file).
            if _resume_tag is None and os.path.isdir(_ckpt_dir_early):
                _ep_dirs = sorted(
                    (d for d in os.listdir(_ckpt_dir_early)
                     if d.startswith("ep") and
                     os.path.isdir(os.path.join(_ckpt_dir_early, d)) and
                     any(f.startswith("ippo_") and f.endswith(".pt")
                         for f in os.listdir(os.path.join(_ckpt_dir_early, d)))),
                    reverse=True,
                )
                if _ep_dirs:
                    _resume_tag = _ep_dirs[0]
                    print(f"  [IPPO-RESUME] 'latest' resolved via directory scan "
                          f"(no pointer found) -> {_resume_tag}", flush=True)

            # 4) Nothing found anywhere — say so loudly instead of silently
            #    training from scratch.
            if _resume_tag is None:
                print(
                    f"  [IPPO-RESUME] WARNING: --resume_from=latest but no "
                    f"checkpoint pointer or ep#### directory with weights "
                    f"found in {_ckpt_dir_early}. Starting from ep 0.",
                    flush=True,
                )
        else:
            # Explicit tag supplied (e.g. "ep0050") — verify it actually
            # exists before trusting it, otherwise warn instead of silently
            # starting fresh.
            _explicit_dir = os.path.join(_ckpt_dir_early, resume_from)
            if os.path.isdir(_explicit_dir) and any(
                f.startswith("ippo_") and f.endswith(".pt")
                for f in os.listdir(_explicit_dir)
            ):
                _resume_tag = resume_from
                print(f"  [IPPO-RESUME] Explicit tag supplied: {_resume_tag}", flush=True)
            else:
                print(
                    f"  [IPPO-RESUME] WARNING: --resume_from={resume_from!r} "
                    f"but {_explicit_dir} has no ippo_*.pt files. "
                    f"Starting from ep 0.",
                    flush=True,
                )

    if _resume_tag is not None:
        _restored_ep = load_ippo_checkpoint(
            agents, save_dir, episode=_resume_tag,
            shared_critic=shared_critic,
            shared_critic_optimizer=shared_critic_optimizer,
        )
        start_ep = _restored_ep + 1
        for ag in agents.values():
            ag.rollout_buffer.clear()
        print(f"  [IPPO-RESUME] Continuing from ep {start_ep} "
              f"({n_episodes - start_ep} remaining)", flush=True)

    # ── LR schedulers ────────────────────────────────────────────────────
    for ag in agents.values():
        ag.setup_lr_schedulers(n_episodes)
    if shared_critic_optimizer is not None:
        shared_critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            shared_critic_optimizer, T_max=max(n_episodes, 1), eta_min=1e-5)

    if reset_lr and resume_from is not None:
        _init_lr = LEARNING_RATE
        for ag in agents.values():
            for pg in ag.actor_optimizer.param_groups:
                pg['lr'] = _init_lr
            for pg in ag.critic_optimizer.param_groups:
                pg['lr'] = _init_lr
        print(f"  [LR-RESET] All IPPO optimizers reset to initial LR={_init_lr}",
              flush=True)

    print(f"  {len(agents)} agent(s) ready. Starting training ({mode_label}) …\n")

    # ── Rich UI ───────────────────────────────────────────────────────────
    rich_ui = RichUI(
        n_episodes    = n_episodes,
        sim_max_steps = sim_max_steps,
        device        = str(device),
        mode          = mode_label,
        intersections = INTERSECTIONS,
        reward_name   = _reward_func.name,
    )

    # ── Fixed-Time baseline ───────────────────────────────────────────────
    fixed_controllers = {
        tl: FixedTimeController(tl)
        for tl in INTERSECTIONS
        if tl in FIXED_TIME_PLANS
    }

    # Build a seed-aware command for the FT baseline so that, when
    # fixed_seed=True, the Fixed-Time comparison is also reproducible.
    if fixed_seed and seed_enable:
        _ft_sumo_cmd = sumo_cmd + ["--seed", str(base_seed)]
    else:
        _ft_sumo_cmd = sumo_cmd

    _cached_fx: dict = {}
    _fx_cache_path = os.path.join(save_dir, "checkpoints", "fixed_time_baseline.json")
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    if fixed_controllers:
        _fx_steps = max(900, sim_max_steps)
        if os.path.isfile(_fx_cache_path):
            try:
                with open(_fx_cache_path) as fcf:
                    _cached_fx = json.load(fcf)
                _tl_keys = {k: v for k, v in _cached_fx.items()
                            if not k.startswith("__")}
                _fq = {tl: float(np.mean(d["queues"])) if d["queues"] else 0.0
                       for tl, d in _tl_keys.items()}
                _fw = {tl: float(np.mean(d["waits"]))  if d["waits"]  else 0.0
                       for tl, d in _tl_keys.items()}
                print(f"  [FixedTime] Loaded from disk "
                      f"(avg_q={np.mean(list(_fq.values())):.2f}  "
                      f"avg_w={np.mean(list(_fw.values())):.2f})", flush=True)
            except Exception as e:
                print(f"  [FixedTime] Cache read failed ({e}) — re-running.",
                      flush=True)
                _cached_fx = {}

        if not _cached_fx:
            _ft_seed_label = (f"seed={base_seed} (fixed)"
                              if fixed_seed else "random seed")
            print(f"  [FixedTime] Running baseline ({_ft_seed_label}) …",
                  flush=True)
            try:
                _sr = _run_fixed_time_episode(
                    _ft_sumo_cmd, _fx_steps, BASE_PORT + 700, fixed_controllers)
                _cached_fx = {k: v for k, v in _sr.items()
                              if not k.startswith("__")}
                for _k in ("__ev_cleared", "__ev_avg_wait", "__cars_passed",
                           "__co2_total", "__nox_total", "__fuel_total"):
                    _cached_fx[_k] = _sr.get(_k, 0.0)
                with open(_fx_cache_path, "w") as fcf:
                    json.dump(_cached_fx, fcf, indent=2)
                print(f"  [FixedTime] Baseline saved -> {_fx_cache_path}",
                      flush=True)
            except Exception as e:
                print(f"  [FixedTime] Baseline run failed: {e} — continuing.",
                      flush=True)

    # ── TensorBoard ───────────────────────────────────────────────────────
    tb_logger = TBComparisonLogger(save_dir)

    # ── Histories ─────────────────────────────────────────────────────────
    rl_history = {
        "reward":      {tl: [] for tl in INTERSECTIONS},
        "queue":       {tl: [] for tl in INTERSECTIONS},
        "wait":        {tl: [] for tl in INTERSECTIONS},
        "ev_cleared":  [], "ev_avg_wait": [], "cars_passed": [],
        "co2_total":   [], "nox_total":   [], "fuel_total":  [],
    }
    fixed_history = {
        "reward":      {tl: [] for tl in INTERSECTIONS},
        "queue":       {tl: [] for tl in INTERSECTIONS},
        "wait":        {tl: [] for tl in INTERSECTIONS},
        "ev_cleared":  [], "ev_avg_wait": [], "cars_passed": [],
        "co2_total":   [], "nox_total":   [], "fuel_total":  [],
    }

    plotter = TrainingPlotter(intersections=INTERSECTIONS, save_dir=plot_dir)

    # ── CSV writers ───────────────────────────────────────────────────────
    _algo_label = f"ippo_{critic_mode}"
    reward_writer = ResultsWriter(
        INTERSECTIONS,
        os.path.join(save_dir, "episode_rewards"),
        header={"sim_file": sim_file, "episodes": n_episodes,
                "reward": _reward_func.name,
                "algorithm": _algo_label, "mode": mode_label},
    )
    _tl_peak_fields = [f"{tl}_peak" for tl in INTERSECTIONS]
    _tl_p95_fields  = [f"{tl}_p95"  for tl in INTERSECTIONS]
    queue_writer    = ResultsWriter(INTERSECTIONS,
                                    os.path.join(save_dir, "episode_queues"),
                                    extra_fields=_tl_peak_fields)
    wait_writer     = ResultsWriter(INTERSECTIONS,
                                    os.path.join(save_dir, "episode_waits"),
                                    extra_fields=_tl_p95_fields)
    loss_writer     = ResultsWriter(INTERSECTIONS,
                                    os.path.join(save_dir, "episode_losses"),
                                    extra_fields=["batch_no",
                                                  "baseline_is_cached",
                                                  "parallel_batch_loss"])
    emissions_writer = ResultsWriter(
        [],
        os.path.join(save_dir, "episode_emissions"),
        header={"description": "Network-wide emission totals per episode",
                "units": "mg", "algorithm": _algo_label},
        extra_fields=["co2_rl_mg", "nox_rl_mg", "fuel_rl_mg",
                      "co2_fx_mg", "nox_fx_mg", "fuel_fx_mg",
                      "co2_saving_pct", "nox_saving_pct", "fuel_saving_pct",
                      "baseline_is_cached"],
    )

    # ── Checkpoint & auto-saver ───────────────────────────────────────────
    proc_ref = [None]
    ckpt     = CheckpointManager(save_dir)
    ckpt.register(agents, plotter,
                  [reward_writer, queue_writer, wait_writer,
                   loss_writer, emissions_writer],
                  proc_ref)

    auto_saver = EpisodeAutoSaver(
        save_dir              = save_dir,
        agents                = agents,
        plotter               = plotter,
        writers               = [reward_writer, queue_writer, wait_writer,
                                  loss_writer, emissions_writer],
        rl_history            = rl_history,
        fixed_history         = fixed_history,
        onnx_export_interval  = 20,
        excel_export_interval = max(1, save_every),
        save_every            = save_every,
        rich_ui               = rich_ui,
    )

    _STEP_INTERVAL = 25
    rich_ui.start()

    try:
        _common_kwargs = dict(
            agents           = agents,
            sumo_cmd         = sumo_cmd,
            n_episodes       = n_episodes,
            sim_max_steps    = sim_max_steps,
            start_ep         = start_ep,
            device           = device,
            plotter          = plotter,
            reward_writer    = reward_writer,
            queue_writer     = queue_writer,
            wait_writer      = wait_writer,
            loss_writer      = loss_writer,
            ckpt             = ckpt,
            proc_ref         = proc_ref,
            tb_logger        = tb_logger,
            fixed_controllers = fixed_controllers,
            cached_fx_result  = _cached_fx,
            rl_history       = rl_history,
            fixed_history    = fixed_history,
            rich_ui          = rich_ui,
            auto_saver       = auto_saver,
            emissions_writer = emissions_writer,
            shared_critic    = shared_critic,
            shared_critic_optimizer = shared_critic_optimizer,
            shared_critic_scheduler = shared_critic_scheduler,
            seed_enable      = seed_enable,
            base_seed        = base_seed,
            fixed_seed       = fixed_seed,
        )

        if parallel_mode:
            _run_parallel_ippo(
                n_workers  = n_workers,
                **_common_kwargs,
            )
        else:
            _run_sequential_ippo(
                step_interval = _STEP_INTERVAL,
                **_common_kwargs,
            )

    except Exception as exc:
        _is_disc = ("Connection closed by SUMO" in str(exc) or
                    "FatalTraCIError" in type(exc).__name__)
        if _is_disc:
            rich_ui.log("[INFO] SUMO connection closed.", style="bold yellow")
        else:
            print(f"\n[ERROR] Unexpected: {exc}", flush=True)
            import traceback; traceback.print_exc()
        ckpt.save_now(emergency=True)
        if proc_ref[0] is not None:
            _close_sumo(proc_ref[0])
        if not _is_disc:
            raise

    finally:
        rich_ui.set_phase("plot")
        ckpt.save_now()
        for w in [reward_writer, queue_writer, wait_writer,
                  loss_writer, emissions_writer]:
            w.close()
        tb_logger.close()

        if plotter._episodes:
            rich_ui.log("Generating plots and Excel …", style="bold white")
            try:
                plotter.plot_all()
                plotter.export_excel("training_results.xlsx")
            except Exception as e:
                rich_ui.log(f"[WARN] Plot/Excel failed: {e}", style="bold red")

            try:
                plot_comparison(save_dir, rl_history, fixed_history,
                                plotter._episodes,
                                ep_duration_ms=plotter._ep_duration_ms)
            except Exception as e:
                rich_ui.log(f"[WARN] Comparison plots failed: {e}",
                            style="bold red")

            _final_xlsx = export_comparison_excel(
                save_dir, rl_history, fixed_history,
                plotter._episodes,
                ep_duration_ms=plotter._ep_duration_ms,
                verbose=True,
            )
            if _final_xlsx:
                rich_ui.log(
                    f"[Excel] Final: {os.path.basename(_final_xlsx)}",
                    style="bold green")

        rich_ui.set_phase("done")
        rich_ui.stop()
        print(f"\nAll IPPO results saved to: {save_dir}")