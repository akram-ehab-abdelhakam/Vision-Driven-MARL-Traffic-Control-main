"""run_experiments.py — Sequential, reproducible algorithm-comparison runner.

Runs the three algorithm variants back-to-back — IPPO, Flat-GAT, and
HierGAT (full MAPPO+HierGAT) — each with a different ``--save_dir`` so
results never overwrite each other.

Determinism
-----------
This runner enforces the exact same deterministic setup used for the
single-run fixes applied to ``training_loop.py`` / ``training_loop_ippo.py``:

  * ``--seed_enable`` is always passed through (on by default here), which
    makes every training loop set, per episode:
        random.seed(ep_seed) / np.random.seed(ep_seed) / torch.manual_seed(ep_seed)
        cudnn.benchmark=False, cudnn.deterministic=True,
        torch.use_deterministic_algorithms(True), CUBLAS_WORKSPACE_CONFIG=":4096:8"
    and feeds the same ``--seed`` into SUMO's own command line.
  * ``PYTHONHASHSEED`` is fixed in the *subprocess* environment before each
    training loop is spawned. This has to happen before the child
    interpreter boots, so it is set as an env var passed to
    ``subprocess.run(..., env=...)`` rather than inside the training code.
  * Every experiment uses the same ``--base_seed`` so IPPO, Flat-GAT, and
    HierGAT all see identical per-episode SUMO seeds — required for a fair,
    paired comparison.

Reproducibility across machines is still bounded by hardware/driver/SUMO
version differences (see training_loop_ippo.py fix notes) — this script
guarantees reproducibility *given* a fixed environment, not bit-identical
results across different GPUs/CUDA/cuDNN/SUMO versions.

Usage
-----
    python -m mappo_project.run_experiments --sim_file sumo/OM.sumocfg

    # Only run two of the three algorithms:
    python -m mappo_project.run_experiments --sim_file sumo/OM.sumocfg \\
        --only ippo hiergat

    # Disable determinism (fast, non-reproducible) if ever needed:
    python -m mappo_project.run_experiments --sim_file sumo/OM.sumocfg \\
        --no-seed_enable

All other flags (--n_episodes, --n_workers, etc.) are passed through to
every sub-experiment unchanged.  Per-experiment overrides (algorithm +
CLI flags each maps to) are defined in the ``EXPERIMENTS`` list below.

Experiment tuple format
-----------------------
    (save_tag, algorithm, extra_flags)

where:
    algorithm   : "ippo" | "flat_gat" | "mappo_gat"   (--algorithm value)
    extra_flags : dict of extra CLI overrides specific to that algorithm
                  (e.g. {"--ippo_critic": "local"})
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
import time

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DEFINITIONS — the three algorithm variants being compared.
# Columns: (save_tag, algorithm, extra_flags)
# ═══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = [
    # Independent PPO baseline — no intent sharing between agents.
    ("ippo",    "ippo",       {"--ippo_critic": "local"}),
    # MAPPO + single-level GATConv over all agents (no hierarchy).
    ("flatgat", "flat_gat",   {}),
    # Full MAPPO + HierGAT (two-level: local cluster -> global inter-cluster).
    ("hiergat", "mappo_gat",  {"--aggregator": "gat"}),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IPPO / Flat-GAT / HierGAT experiments sequentially, deterministically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    parser.add_argument(
        "--sim_file", required=True,
        help="Path to the SUMO .sumocfg simulation config file",
    )
    # Pass-through flags shared by all experiments
    parser.add_argument("--n_episodes",  type=int,  default=300,
                        help="Training episodes per experiment")
    parser.add_argument("--max_steps",   type=int,  default=3600,
                        help="Simulation steps per episode")
    parser.add_argument("--n_workers",   type=int,  default=1,
                        help="Parallel SUMO workers per experiment")
    parser.add_argument("--save_every",  type=int,  default=10,
                        help="Checkpoint / plot interval (episodes)")
    parser.add_argument("--gui",
                        choices=["enable", "disable"], default="disable",
                        help="Enable SUMO GUI")
    parser.add_argument("--reward",
                        choices=["backpressure", "waiting_time", "queue", "max_pressure"],
                        default="backpressure",
                        help="Reward function shared by every experiment (fair comparison)")
    parser.add_argument("--ev_priority",
                        choices=["default", "none"], default="default",
                        help="EV priority module shared by every experiment")
    parser.add_argument("--adaptive_lr",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="KL-adaptive LR controller, applied identically to every "
                             "algorithm that supports it (fairness parity)")
    parser.add_argument("--base_save_dir", default="./results",
                        help="Root output directory; each experiment gets a sub-folder")
    # Subset selector: run only specific experiments by save_tag
    parser.add_argument("--only", nargs="*", default=None,
                        metavar="TAG",
                        help="If set, run only experiments whose save_tag is listed here "
                             f"(available: {[e[0] for e in EXPERIMENTS]})")

    # ── Reproducibility (on by default; mirrors training_loop.py / training_loop_ippo.py) ──
    parser.add_argument(
        "--seed_enable",
        action=argparse.BooleanOptionalAction, default=True,
        help="Enable deterministic per-episode seeding for every experiment "
             "(cudnn.deterministic=True, cudnn.benchmark=False, fixed PYTHONHASHSEED, "
             "identical SUMO --seed per episode across algorithms). Default: on. "
             "Use --no-seed_enable for the old fast/non-reproducible behaviour.",
    )
    parser.add_argument(
        "--base_seed", type=int, default=42,
        help="Base seed shared by every experiment. Using the same value across "
             "IPPO/Flat-GAT/HierGAT is what makes the comparison paired and fair.",
    )
    parser.add_argument(
        "--fixed_seed",
        action=argparse.BooleanOptionalAction, default=False,
        help="Use the same seed for every episode (fixed_seed=True means "
             "base_seed is used directly without per-episode increment). "
             "Default: disabled (seed = base_seed + ep).",
    )
    parser.add_argument(
        "--hash_seed", type=int, default=None,
        help="PYTHONHASHSEED for every training subprocess. Defaults to --base_seed "
             "when not given, so all runs also agree on hash-order-dependent behaviour.",
    )
    return parser.parse_args()


def _run_experiment(
    save_tag: str,
    algorithm: str,
    extra_flags: dict,
    base_args: argparse.Namespace,
) -> int:
    """Launch one training run as a subprocess. Returns the exit code."""
    save_dir = os.path.join(base_args.base_save_dir, save_tag)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mappo_project.training_loop",
        "--sim_file",    base_args.sim_file,
        "--n_episodes",  str(base_args.n_episodes),
        "--max_steps",   str(base_args.max_steps),
        "--n_workers",   str(base_args.n_workers),
        "--save_every",  str(base_args.save_every),
        "--gui",         base_args.gui,
        "--save_dir",    save_dir,
        "--reward",      base_args.reward,
        "--ev_priority", base_args.ev_priority,
        "--algorithm",   algorithm,
    ]
    if algorithm != "ippo":
        # --adaptive_lr / --no-adaptive_lr — not a CLI flag for ippo: IPPO's
        # KL-adaptive LR controller (added for fairness parity with MAPPO) is
        # wired on unconditionally inside run_training_ippo/agent.py, so there
        # is nothing to pass through here.
        cmd.append("--adaptive_lr" if base_args.adaptive_lr else "--no-adaptive_lr")

    # Reproducibility flags — identical --base_seed across every algorithm
    # is what makes IPPO vs Flat-GAT vs HierGAT a fair, paired comparison.
    cmd.append("--seed_enable" if base_args.seed_enable else "--no-seed_enable")
    cmd += ["--base_seed", str(base_args.base_seed)]
    cmd.append("--fixed_seed" if base_args.fixed_seed else "--no-fixed_seed")

    for flag, value in extra_flags.items():
        cmd += [flag, str(value)]

    # PYTHONHASHSEED must be fixed *before* the child interpreter boots —
    # setting os.environ inside the training script has no effect on its own
    # process. Passing it through subprocess env is the only place this can
    # be enforced reliably for each spawned run.
    env = os.environ.copy()
    if base_args.seed_enable:
        hash_seed = base_args.hash_seed if base_args.hash_seed is not None else base_args.base_seed
        env["PYTHONHASHSEED"] = str(hash_seed)

    print(f"\n{'='*70}", flush=True)
    print(f"  EXPERIMENT: {save_tag}", flush=True)
    print(f"    algorithm     = {algorithm}", flush=True)
    print(f"    reward        = {base_args.reward}", flush=True)
    print(f"    ev_priority   = {base_args.ev_priority}", flush=True)
    print(f"    seed_enable   = {base_args.seed_enable}", flush=True)
    print(f"    base_seed     = {base_args.base_seed}", flush=True)
    print(f"    fixed_seed    = {base_args.fixed_seed}", flush=True)
    if base_args.seed_enable:
        print(f"    PYTHONHASHSEED = {env['PYTHONHASHSEED']}", flush=True)
    if extra_flags:
        print(f"    extra_flags   = {extra_flags}", flush=True)
    print(f"    save_dir      = {save_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = int(elapsed % 60)
    duration_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"

    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"\n  [{save_tag}] finished in {duration_str} — {status}\n", flush=True)
    return result.returncode


def main() -> None:
    args = _parse_args()

    experiments = EXPERIMENTS
    if args.only:
        only_set = set(args.only)
        experiments = [e for e in EXPERIMENTS if e[0] in only_set]
        if not experiments:
            print(f"[run_experiments] No experiments match --only {args.only}. "
                  f"Available tags: {[e[0] for e in EXPERIMENTS]}", flush=True)
            sys.exit(1)

    print(f"\n{'='*70}", flush=True)
    print(f"  run_experiments.py — {len(experiments)} experiment(s) queued", flush=True)
    print(f"  base_save_dir : {args.base_save_dir}", flush=True)
    print(f"  sim_file      : {args.sim_file}", flush=True)
    print(f"  n_episodes    : {args.n_episodes}", flush=True)
    print(f"  seed_enable   : {args.seed_enable}", flush=True)
    print(f"  base_seed     : {args.base_seed}", flush=True)
    if not args.seed_enable:
        print(f"  [WARNING] Determinism disabled (--no-seed_enable) — runs will "
              f"NOT be reproducible and results across algorithms will not be paired.",
              flush=True)
    print(f"{'='*70}", flush=True)

    results: list[tuple[str, int]] = []
    wall_t0 = time.time()

    for save_tag, algorithm, extra_flags in experiments:
        rc = _run_experiment(save_tag, algorithm, extra_flags, args)
        results.append((save_tag, rc))

    # ── Summary ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - wall_t0
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = int(total_elapsed % 60)
    total_str = f"{h}h {m:02d}m {s:02d}s" if h > 0 else f"{m}m {s:02d}s"

    print(f"\n{'='*70}", flush=True)
    print(f"  All experiments complete — total wall time: {total_str}", flush=True)
    print(f"{'='*70}", flush=True)
    n_ok   = sum(1 for _, rc in results if rc == 0)
    n_fail = len(results) - n_ok
    for tag, rc in results:
        status = "✓ OK" if rc == 0 else f"✗ FAILED (rc={rc})"
        print(f"    {tag:<30} {status}", flush=True)
    print(f"\n  {n_ok}/{len(results)} succeeded"
          + (f"  ({n_fail} FAILED)" if n_fail else "") + "\n", flush=True)

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()