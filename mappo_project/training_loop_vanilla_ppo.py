"""
training_loop_vanilla_ppo.py
=============================
Single centralized Vanilla PPO training loop for SmartTraffic-OM.

  run_training_vanilla_ppo(...)       — full training entry-point
  _run_sequential_vanilla_ppo(...)    — inner sequential episode loop

Usage:
  python main.py --sim_file sumo/OM.sumocfg --algorithm vanilla_ppo
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim

from .config import (
    GAMMA, GAE_LAMBDA, LEARNING_RATE, ENTROPY_COEFF, VALUE_LOSS_COEFF,
    CLIP_RATIO, BATCH_SIZE, N_EPOCHS, HIDDEN_DIM, MEMORY_CAPACITY,
    TOTAL_STEPS, INTERSECTIONS, TLS_DETECTOR_MAP,
    _QUEUE_CAPACITY_PER_TLS, EV_PRIORITY_ENABLED, MIN_GREEN_STEPS,
    STEP_LENGTH, BASE_PORT, FIXED_TIME_PLANS,
    RICH_AVAILABLE,
)
from .sumo_env import (
    _start_sumo, _close_sumo,
    _setup_all_subscriptions, _subscribe_vehicles_bulk,
    _STEP_CACHE, _reset_ep_stats, _snapshot_ep_stats, _collect_final_emissions,
    _get_tls_queue_from_detectors, _get_tls_mean_wait,
    _xml_output_args, TRACI_AVAILABLE,
)
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
from .vanilla_ppo import (
    VanillaPPOAgent, VanillaPPOBuffer,
    save_vanilla_ppo_checkpoint, load_vanilla_ppo_checkpoint,
)

try:
    import traci
except ImportError:
    traci = None

__all__ = ["run_training_vanilla_ppo", "_run_sequential_vanilla_ppo"]


# ══════════════════════════════════════════════════════════════════════════════
# Sequential training loop
# ══════════════════════════════════════════════════════════════════════════════

def _run_sequential_vanilla_ppo(
    agent:            VanillaPPOAgent,
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
    seed_enable:      bool = False,
    base_seed:        int  = 42,
    fixed_seed:       bool = False,
) -> None:

    _ui = rich_ui
    tls_ids = agent.tls_ids

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

        # Reset per-episode
        for tls_id in tls_ids:
            agent.step_rewards[tls_id].clear()
            agent.step_queues[tls_id].clear()
            agent.step_waits[tls_id].clear()
            agent.step_losses.clear()
            info = agent._tls_info[tls_id]
            info["ev_phi_prev"] = 0.0
            info["ev_phi_curr"] = 0.0

        VanillaPPOAgent.clear_lane_length_cache()

        ep_port = BASE_PORT + (ep % 500) + 1

        if seed_enable:
            ep_seed_val = base_seed if fixed_seed else base_seed + ep
            random.seed(ep_seed_val)
            np.random.seed(ep_seed_val)
            torch.manual_seed(ep_seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(ep_seed_val)
            _ep_sumo_cmd = sumo_cmd + ["--seed", str(ep_seed_val)]
            _log(f"[VPP Ep {ep}] seed={ep_seed_val}"
                 + (" (fixed)" if fixed_seed else ""))
        else:
            _ep_sumo_cmd = sumo_cmd
            _log(f"[VPP Ep {ep}] Starting SUMO on port {ep_port} ...")

        _ep_sumo_cmd = list(_ep_sumo_cmd) + _xml_output_args(
            os.path.join("results", "vanilla_ppo_xml"))
        try:
            _proc = _start_sumo(_ep_sumo_cmd, port=ep_port)
            proc_ref[0] = _proc
        except Exception as exc:
            _log(f"[ERROR] Ep {ep}: Cannot start SUMO: {exc}",
                 style="bold red")
            import traceback; traceback.print_exc()
            break

        _setup_all_subscriptions({"__vpp__": agent})

        # Reset per-intersection state
        for tls_id in tls_ids:
            info = agent._tls_info[tls_id]
            if is_first_ep and start_ep > 0:
                pass
            else:
                info["last_switch_step"]   = -agent.min_green_steps
                info["current_action_idx"] = 0
            agent._current_step = 0
        is_first_ep = False

        _reset_ep_stats()
        _log(f"[VPP Ep {ep}] Running {sim_max_steps} steps ...")

        raw_transitions = []

        try:
            for step in range(1, sim_max_steps + 1):
                if step % 50 == 0:
                    if traci.simulation.getMinExpectedNumber() == 0:
                        _log(f"[VPP Ep {ep}] Early stop at step {step}")
                        break

                if step % step_interval == 0 and _ui:
                    _ui.update_step(step)

                _subscribe_vehicles_bulk()
                _STEP_CACHE.clear()

                # Build global state
                g_state = agent._build_global_state()

                # Choose joint action
                actions = agent.choose_action(deterministic=False)

                # Apply to SUMO
                agent.apply_actions(actions, step)

                traci.simulationStep()

                _STEP_CACHE.bulk_load_lanes()
                _STEP_CACHE.bulk_load_detectors()
                _STEP_CACHE.bulk_load_vehicles()

                done = 1.0 if step >= sim_max_steps else 0.0
                reward = agent.get_reward()

                raw_transitions.append((g_state, actions, reward, done))

                # Per-TLS metrics
                for tls_id in tls_ids:
                    ns = agent._get_tls_state(tls_id)
                    agent.step_rewards[tls_id].append(reward / len(tls_ids))
                    agent.step_queues[tls_id].append(
                        _get_tls_queue_from_detectors(tls_id))
                    agent.step_waits[tls_id].append(_get_tls_mean_wait(tls_id))

        except Exception as step_exc:
            _is_sumo_disconnect  = (
                "Connection closed by SUMO" in str(step_exc) or
                "FatalTraCIError" in type(step_exc).__name__)
            _sumo_closed_by_user = _is_sumo_disconnect
            if _is_sumo_disconnect:
                _log(f"[VPP Ep {ep}] SUMO closed.", style="bold yellow")
            else:
                _log(f"[VPP Ep {ep}] Step error: {step_exc}",
                     style="bold red")

        # ── Fill buffer & train ───────────────────────────────────────────
        if _ui:
            _ui.set_phase("train")
        _log(f"[VPP Ep {ep}] Training ...")

        _fill_buffer(agent, raw_transitions, sim_max_steps)
        loss = agent.train_step()
        if not agent.step_losses:
            agent.step_losses.append(loss)
        elif agent.step_losses[-1] == 0.0 and loss > 0.0:
            agent.step_losses[-1] = loss

        if _ui:
            _ui.set_phase("checkpoint")
        _log(f"[VPP Ep {ep}] Closing SUMO ...")
        _close_sumo(_proc)
        proc_ref[0] = None

        if _sumo_closed_by_user:
            if not any(agent.step_rewards[t] for t in tls_ids):
                ckpt.save_now(emergency=True)
                break

        # ── Episode metrics ───────────────────────────────────────────────
        ep_reward, ep_queue, ep_wait, ep_loss = {}, {}, {}, {}
        ep_queue_peak, ep_wait_p95 = {}, {}
        for tl in tls_ids:
            r = agent.step_rewards[tl]
            q = agent.step_queues[tl]
            w = agent.step_waits[tl]
            ep_reward[tl]    = float(np.mean(r))  if r else 0.0
            ep_queue[tl]     = float(np.mean(q))  if q else 0.0
            ep_wait[tl]      = float(np.mean(w))  if w else 0.0
            ep_queue_peak[tl] = float(np.max(q))  if q else 0.0
            ep_wait_p95[tl]   = float(np.percentile(w, 95)) if w else 0.0
        ep_loss = {tl: (float(np.mean(agent.step_losses))
                        if agent.step_losses else 0.0)
                   for tl in tls_ids}

        ep_duration_ms = time.time() * 1000.0 - _ep_start_ms

        plotter.record(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                       ep_duration_ms=ep_duration_ms)

        reward_writer.write_row(
            {"episode": ep, **ep_reward,
             "ep_duration_ms": round(ep_duration_ms, 1)})
        _q_peak = {f"{tl}_peak": ep_queue_peak.get(tl, 0.0) for tl in tls_ids}
        _w_p95  = {f"{tl}_p95": ep_wait_p95.get(tl, 0.0)   for tl in tls_ids}
        queue_writer.write_row(
            {"episode": ep, **ep_queue,
             "ep_duration_ms": round(ep_duration_ms, 1), **_q_peak})
        wait_writer.write_row(
            {"episode": ep, **ep_wait,
             "ep_duration_ms": round(ep_duration_ms, 1), **_w_p95})
        loss_writer.write_row(
            {"episode": ep, **ep_loss,
             "ep_duration_ms": round(ep_duration_ms, 1),
             "batch_no": 0, "baseline_is_cached": True,
             "parallel_batch_loss": False})

        if emissions_writer is not None:
            _collect_final_emissions()
            _em = _snapshot_ep_stats()
            _co2_fx = (cached_fx_result.get("__co2_total", 0.0)
                       if cached_fx_result else 0.0)
            _nox_fx = (cached_fx_result.get("__nox_total", 0.0)
                       if cached_fx_result else 0.0)
            _fuel_fx = (cached_fx_result.get("__fuel_total", 0.0)
                        if cached_fx_result else 0.0)
            _pct = lambda rl, fx: (
                round(((fx - rl) / max(fx, 1e-6)) * 100, 2) if fx else 0.0)
            emissions_writer.write_row({
                "episode": ep, "ep_duration_ms": round(ep_duration_ms, 1),
                "co2_rl_mg": round(_em["co2_sum"], 2),
                "nox_rl_mg": round(_em["nox_sum"], 2),
                "fuel_rl_mg": round(_em["fuel_sum"], 2),
                "co2_fx_mg": round(_co2_fx, 2),
                "nox_fx_mg": round(_nox_fx, 2),
                "fuel_fx_mg": round(_fuel_fx, 2),
                "co2_saving_pct":  _pct(_em["co2_sum"],  _co2_fx),
                "nox_saving_pct":  _pct(_em["nox_sum"],  _nox_fx),
                "fuel_saving_pct": _pct(_em["fuel_sum"], _fuel_fx),
                "baseline_is_cached": True,
            })

        if tb_logger is not None:
            tb_logger.log_episode_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss)

        # ── Fixed-Time baseline refresh ───────────────────────────────────
        if (fixed_controllers and ep % 50 == 0 and ep > 0):
            _ft_cmd = (sumo_cmd + ["--seed", str(base_seed)]
                       if fixed_seed and seed_enable else sumo_cmd)
            try:
                _log(f"[VPP Ep {ep}] Refreshing Fixed-Time baseline ...")
                _sr = _run_fixed_time_episode(
                    _ft_cmd, max(900, sim_max_steps),
                    BASE_PORT + 701, fixed_controllers)
                cached_fx_result = {
                    k: v for k, v in _sr.items() if not k.startswith("__")}
                for _k in ("__ev_cleared", "__ev_avg_wait", "__cars_passed",
                           "__co2_total", "__nox_total", "__fuel_total"):
                    cached_fx_result[_k] = _sr.get(_k, 0.0)
                cached_fx_result["__ft_baseline_ep"] = ep
            except Exception as e:
                _log(f"[VPP Ep {ep}] FT refresh failed: {e}")

        fx_reward = {tl: 0.0 for tl in tls_ids}
        fx_queue  = {tl: 0.0 for tl in tls_ids}
        fx_wait   = {tl: 0.0 for tl in tls_ids}
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
        _rl_ep = _snapshot_ep_stats()
        _rl_ev = (round(_rl_ep["ev_wait_sum"] / _rl_ep["ev_cleared"], 3)
                  if _rl_ep["ev_cleared"] > 0 else None)

        if rl_history is not None:
            for tl in tls_ids:
                rl_history["reward"][tl].append(ep_reward.get(tl, 0.0))
                rl_history["queue"][tl].append(ep_queue.get(tl, 0.0))
                rl_history["wait"][tl].append(ep_wait.get(tl, 0.0))
            rl_history["ev_cleared"].append(_rl_ep["ev_cleared"])
            rl_history["ev_avg_wait"].append(_rl_ev)
            rl_history["cars_passed"].append(_rl_ep["cars_passed"])
            rl_history["co2_total"].append(round(_rl_ep["co2_sum"], 2))
            rl_history["nox_total"].append(round(_rl_ep["nox_sum"], 2))
            rl_history["fuel_total"].append(round(_rl_ep["fuel_sum"], 2))

        if fixed_history is not None:
            for tl in tls_ids:
                fixed_history["reward"][tl].append(fx_reward.get(tl, 0.0))
                fixed_history["queue"][tl].append(fx_queue.get(tl, 0.0))
                fixed_history["wait"][tl].append(fx_wait.get(tl, 0.0))
            fixed_history["ev_cleared"].append(
                cached_fx_result.get("__ev_cleared", 0) if cached_fx_result else 0)
            fixed_history["ev_avg_wait"].append(
                cached_fx_result.get("__ev_avg_wait") if cached_fx_result else None)
            fixed_history["cars_passed"].append(
                cached_fx_result.get("__cars_passed", 0) if cached_fx_result else 0)
            fixed_history["co2_total"].append(
                cached_fx_result.get("__co2_total", 0.0) if cached_fx_result else 0.0)
            fixed_history["nox_total"].append(
                cached_fx_result.get("__nox_total", 0.0) if cached_fx_result else 0.0)
            fixed_history["fuel_total"].append(
                cached_fx_result.get("__fuel_total", 0.0) if cached_fx_result else 0.0)

        if _ui:
            _ui.record_rl(ep, ep_reward, ep_queue, ep_wait, ep_loss,
                          ep_stats=_rl_ep)
            if fixed_controllers and cached_fx_result:
                _ui.record_fixed(fx_reward, fx_queue, fx_wait,
                                 fx_stats=cached_fx_result)

        auto_saver.save_episode(
            ep, ep_reward, ep_queue, ep_wait, ep_loss,
            fx_reward, fx_queue, fx_wait)
        if ep % auto_saver.save_every == 0 or ep == n_episodes - 1:
            auto_saver.tag_latest(ep)

        if _sumo_closed_by_user:
            _log("[INFO] SUMO closed -- stopping.", style="bold yellow")
            ckpt.save_now(emergency=True)
            break

        _log(f"[VPP Ep {ep}] Done. "
             f"avg_r={float(np.mean(list(ep_reward.values()))):+.4f}  "
             f"avg_q={float(np.mean(list(ep_queue.values()))):.2f}  "
             f"avg_w={float(np.mean(list(ep_wait.values()))):.2f}",
             style="dim cyan")

    _log("VanillaPPO sequential loop complete.", style="bold green")


def _fill_buffer(
    agent:           VanillaPPOAgent,
    transitions:     list,
    sim_max_steps:   int,
) -> None:
    """Batch-fill the rollout buffer from raw transitions."""
    if not transitions:
        return

    n = len(transitions)
    all_s = np.stack([t[0] for t in transitions])

    # Joint actions: (n, n_intersections)
    all_a = np.stack([np.array([t[1][tls] for tls in agent.tls_ids],
                               dtype=np.int64)
                      for t in transitions])

    all_r = np.array([t[2] for t in transitions], dtype=np.float32)
    all_d = np.array([t[3] for t in transitions], dtype=np.float32)

    # Batch value & log-prob inference
    all_values = agent.get_values_batch(all_s)
    all_lp     = agent.get_joint_log_probs_batch(all_s, all_a)

    # Next-state values
    all_nv = np.zeros(n, dtype=np.float32)
    non_term = np.where(all_d == 0.0)[0]
    if len(non_term) > 0:
        # Next states are the current states shifted by 1
        # (the last transition has no next state, so value=0)
        if len(transitions) > 1:
            next_s = all_s[1:]  # shift
            next_nv = np.zeros(len(transitions), dtype=np.float32)
            next_nv[:-1] = agent.get_values_batch(next_s)
            all_nv = next_nv

    for i in range(n):
        agent.rollout_buffer.push(
            all_s[i], all_a[i], float(all_r[i]),
            float(all_values[i]), float(all_lp[i]),
            float(all_d[i]), float(all_nv[i]),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Full training entry-point
# ══════════════════════════════════════════════════════════════════════════════

def run_training_vanilla_ppo(
    sim_file:       str,
    n_episodes:     int  = 1,
    sim_max_steps:  int  = 3600,
    gui:            str  = "disable",
    save_dir:       str  = "./results_vanilla_ppo",
    resume_from:    "str | None" = None,
    n_workers:      int  = 1,
    save_every:     int  = 10,
    reward_type:    str  = "backpressure",
    ev_priority_type: str = "default",
    seed_enable:    bool = False,
    base_seed:      int  = 42,
    fixed_seed:     bool = False,
    reset_lr:       bool = False,
    cuda_dynamic:   bool = True,
) -> None:

    if not TRACI_AVAILABLE:
        raise RuntimeError("traci is not available.")

    REWARD_MAP = {
        "backpressure": BackPressureReward(),
        "waiting_time": WaitingTimeReward(),
        "queue":        QueueReductionReward(),
    }
    EV_MAP = {"default": DefaultEVPriority(), "none": NoEVPriority()}
    _reward_func = REWARD_MAP.get(reward_type, BackPressureReward())
    _ev_module   = EV_MAP.get(ev_priority_type, DefaultEVPriority())

    if ev_priority_type == "none":
        for _mod_name in [__name__.rsplit(".", 1)[0] + ".sumo_env",
                          __name__.rsplit(".", 1)[0] + ".rewards"]:
            if _mod_name in sys.modules:
                sys.modules[_mod_name].EV_PRIORITY_ENABLED = False
        print("[INFO] EV reward shaping: DISABLED", flush=True)

    import platform as _plat
    _is_windows = _plat.system() == "Windows"
    if _is_windows and n_workers > 1:
        print(f"[WARNING] Parallel not supported on Windows. n_workers=1.",
              flush=True)

    os.makedirs(save_dir, exist_ok=True)
    plot_dir = os.path.join(save_dir, "plots")

    sumo_binary = (("sumo-gui.exe" if gui == "enable" else "sumo.exe")
                   if _is_windows
                   else ("sumo-gui" if gui == "enable" else "sumo"))
    sumo_cmd = [
        sumo_binary, "-c", sim_file,
        "--step-length", str(int(STEP_LENGTH)),
        "--no-step-log", "--time-to-teleport", "300",
        "--collision.action", "warn",
    ]

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if seed_enable:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif not cuda_dynamic:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    else:
        device = torch.device("cpu")
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
            torch.set_num_interop_threads(1)
        except RuntimeError: pass

    print(f"\n{'='*70}")
    print(f"  OM Network  |  Single-Agent Vanilla PPO")
    print(f"  Device      : {device}")
    print(f"  Episodes    : {n_episodes}")
    print(f"  Steps/ep    : {sim_max_steps}")
    print(f"  Reward      : {_reward_func.name}")
    print(f"  EV Priority : {_ev_module.name}")
    print(f"  gamma={GAMMA}  lr={LEARNING_RATE}  lambda={GAE_LAMBDA}  "
          f"clip={CLIP_RATIO}  entropy={ENTROPY_COEFF}")
    print(f"{'='*70}\n")

    if seed_enable:
        random.seed(base_seed); np.random.seed(base_seed)
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(base_seed)

    # ── Bootstrap SUMO ────────────────────────────────────────────────────
    print("  Bootstrapping SUMO ...")
    _proc = _start_sumo(sumo_cmd, port=BASE_PORT)
    try:
        agent = VanillaPPOAgent(
            device, reward_func=_reward_func, ev_module=_ev_module)
    except Exception as exc:
        _close_sumo(_proc)
        raise RuntimeError(f"Agent init failed: {exc}")
    _close_sumo(_proc)

    # ── Resize buffer ─────────────────────────────────────────────────────
    if sim_max_steps != MEMORY_CAPACITY:
        print(f"  [FIX] Resizing buffer: {MEMORY_CAPACITY} -> {sim_max_steps}")
        agent.rollout_buffer = VanillaPPOBuffer(
            sim_max_steps, agent.global_state_dim, agent.n_intersections)

    # ── Resume ────────────────────────────────────────────────────────────
    _resume_tag: "str | None" = None
    start_ep = 0
    if resume_from is not None:
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        if resume_from == "latest":
            ptr = os.path.join(ckpt_dir, "latest.json")
            if os.path.isfile(ptr):
                with open(ptr) as pf:
                    _resume_tag = json.load(pf).get("tag", "emergency")
        else:
            _resume_tag = resume_from

    if _resume_tag is not None:
        _restored = load_vanilla_ppo_checkpoint(agent, save_dir,
                                                episode=_resume_tag)
        start_ep = _restored + 1
        agent.rollout_buffer.clear()
        print(f"  [VPP-RESUME] Continuing from ep {start_ep}", flush=True)

    agent.setup_lr_scheduler(n_episodes)

    if reset_lr and resume_from is not None:
        for pg in agent.optimizer.param_groups:
            pg['lr'] = LEARNING_RATE
        print(f"  [LR-RESET] LR={LEARNING_RATE}", flush=True)

    print(f"  Single agent ready. Starting training ...\n")

    # ── Rich UI ───────────────────────────────────────────────────────────
    rich_ui = RichUI(
        n_episodes=n_episodes, sim_max_steps=sim_max_steps,
        device=str(device), mode="SEQUENTIAL",
        intersections=INTERSECTIONS, reward_name=_reward_func.name,
    )

    fixed_controllers = {
        tl: FixedTimeController(tl)
        for tl in INTERSECTIONS if tl in FIXED_TIME_PLANS
    }

    _ft_cmd = (sumo_cmd + ["--seed", str(base_seed)]
               if fixed_seed and seed_enable else sumo_cmd)
    _cached_fx: dict = {}
    _fx_path = os.path.join(save_dir, "checkpoints", "fixed_time_baseline.json")
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)

    if fixed_controllers:
        if os.path.isfile(_fx_path):
            try:
                with open(_fx_path) as f:
                    _cached_fx = json.load(f)
                _tl = {k: v for k, v in _cached_fx.items()
                       if not k.startswith("__")}
                print(f"  [FixedTime] Loaded from disk.", flush=True)
            except Exception:
                _cached_fx = {}
        if not _cached_fx:
            print(f"  [FixedTime] Running baseline ...", flush=True)
            try:
                _sr = _run_fixed_time_episode(
                    _ft_cmd, max(900, sim_max_steps),
                    BASE_PORT + 700, fixed_controllers)
                _cached_fx = {k: v for k, v in _sr.items()
                              if not k.startswith("__")}
                for _k in ("__ev_cleared", "__ev_avg_wait", "__cars_passed",
                           "__co2_total", "__nox_total", "__fuel_total"):
                    _cached_fx[_k] = _sr.get(_k, 0.0)
                with open(_fx_path, "w") as f:
                    json.dump(_cached_fx, f, indent=2)
            except Exception as e:
                print(f"  [FixedTime] Failed: {e}", flush=True)

    tb_logger = TBComparisonLogger(save_dir)

    tls_ids = agent.tls_ids
    rl_history = {
        "reward": {t: [] for t in tls_ids}, "queue": {t: [] for t in tls_ids},
        "wait": {t: [] for t in tls_ids},
        "ev_cleared": [], "ev_avg_wait": [], "cars_passed": [],
        "co2_total": [], "nox_total": [], "fuel_total": [],
    }
    fixed_history = {
        "reward": {t: [] for t in tls_ids}, "queue": {t: [] for t in tls_ids},
        "wait": {t: [] for t in tls_ids},
        "ev_cleared": [], "ev_avg_wait": [], "cars_passed": [],
        "co2_total": [], "nox_total": [], "fuel_total": [],
    }

    plotter = TrainingPlotter(intersections=INTERSECTIONS, save_dir=plot_dir)

    _algo = "vanilla_ppo"
    reward_writer = ResultsWriter(
        INTERSECTIONS, os.path.join(save_dir, "episode_rewards"),
        header={"sim_file": sim_file, "episodes": n_episodes,
                "reward": _reward_func.name, "algorithm": _algo})
    queue_writer = ResultsWriter(
        INTERSECTIONS, os.path.join(save_dir, "episode_queues"),
        extra_fields=[f"{tl}_peak" for tl in tls_ids])
    wait_writer = ResultsWriter(
        INTERSECTIONS, os.path.join(save_dir, "episode_waits"),
        extra_fields=[f"{tl}_p95" for tl in tls_ids])
    loss_writer = ResultsWriter(
        INTERSECTIONS, os.path.join(save_dir, "episode_losses"),
        extra_fields=["batch_no", "baseline_is_cached", "parallel_batch_loss"])
    emissions_writer = ResultsWriter(
        [], os.path.join(save_dir, "episode_emissions"),
        header={"description": "Emission totals", "units": "mg",
                "algorithm": _algo},
        extra_fields=["co2_rl_mg", "nox_rl_mg", "fuel_rl_mg",
                      "co2_fx_mg", "nox_fx_mg", "fuel_fx_mg",
                      "co2_saving_pct", "nox_saving_pct", "fuel_saving_pct",
                      "baseline_is_cached"])

    proc_ref = [None]
    ckpt = CheckpointManager(save_dir)
    # Wrap agent in a dict for CheckpointManager compatibility
    _agent_dict = {"__vpp__": agent}
    ckpt.register(_agent_dict, plotter,
                  [reward_writer, queue_writer, wait_writer,
                   loss_writer, emissions_writer], proc_ref)

    auto_saver = EpisodeAutoSaver(
        save_dir=save_dir, agents=_agent_dict, plotter=plotter,
        writers=[reward_writer, queue_writer, wait_writer,
                 loss_writer, emissions_writer],
        rl_history=rl_history, fixed_history=fixed_history,
        onnx_export_interval=20, excel_export_interval=max(1, save_every),
        save_every=save_every, rich_ui=rich_ui)

    rich_ui.start()

    try:
        _run_sequential_vanilla_ppo(
            agent=agent, sumo_cmd=sumo_cmd, n_episodes=n_episodes,
            sim_max_steps=sim_max_steps, start_ep=start_ep,
            device=device, step_interval=25, plotter=plotter,
            reward_writer=reward_writer, queue_writer=queue_writer,
            wait_writer=wait_writer, loss_writer=loss_writer,
            ckpt=ckpt, proc_ref=proc_ref, tb_logger=tb_logger,
            fixed_controllers=fixed_controllers, cached_fx_result=_cached_fx,
            rl_history=rl_history, fixed_history=fixed_history,
            rich_ui=rich_ui, auto_saver=auto_saver,
            emissions_writer=emissions_writer,
            seed_enable=seed_enable, base_seed=base_seed,
            fixed_seed=fixed_seed,
        )
    except Exception as exc:
        _is_disc = ("Connection closed by SUMO" in str(exc) or
                    "FatalTraCIError" in type(exc).__name__)
        if _is_disc:
            rich_ui.log("[INFO] SUMO closed.", style="bold yellow")
        else:
            print(f"\n[ERROR] {exc}", flush=True)
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
            rich_ui.log("Generating plots ...", style="bold white")
            try:
                plotter.plot_all()
                plotter.export_excel("training_results.xlsx")
            except Exception as e:
                rich_ui.log(f"[WARN] Plot failed: {e}", style="bold red")
            try:
                plot_comparison(save_dir, rl_history, fixed_history,
                                plotter._episodes,
                                ep_duration_ms=plotter._ep_duration_ms)
            except Exception as e:
                rich_ui.log(f"[WARN] Comparison failed: {e}", style="bold red")
            _xlsx = export_comparison_excel(
                save_dir, rl_history, fixed_history,
                plotter._episodes,
                ep_duration_ms=plotter._ep_duration_ms, verbose=True)
            if _xlsx:
                rich_ui.log(f"[Excel] {os.path.basename(_xlsx)}",
                            style="bold green")
        rich_ui.set_phase("done")
        rich_ui.stop()
        print(f"\nAll VanillaPPO results saved to: {save_dir}")
