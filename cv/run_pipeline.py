"""
run_pipeline.py
Path: (project root — same directory as route_postprocessor.py)

Full offline pipeline:
    cv_stats.json  →  DesRUTGe PPO calibration  →  raw routes.rou.xml
                   →  vType assignment  →  typed_routes.rou.xml

After this script finishes, update your OM.sumocfg:
    <route-files value="<output_path>"/>
Then run your RL system normally — no further changes needed.

Requirements (on system PATH):
    od2trips    (SUMO tools)
    duarouter   (SUMO tools)
    sumo        (SUMO)

Python requirements:
    stable-baselines3, gym, sklearn, torch, numpy

Usage:
python run_pipeline.py --cv_stats desrutge/data/TrafficGeneration/cv_stats.json --om_network desrutge/src/machine_learning/training/om_network  --output generated/om_routes_typed.rou.xml --ppo_steps 300
"""

import argparse
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup — allow running from any working directory as long as the
# desrutge package is present relative to this script's location.
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_DESRUTGE_SRC = os.path.join(_SCRIPT_DIR, "desrutge", "src")
if _DESRUTGE_SRC not in sys.path:
    sys.path.insert(0, _DESRUTGE_SRC)

# ---------------------------------------------------------------------------
# Imports from DesRUTGe and local modules
# ---------------------------------------------------------------------------
try:
    from desrutge.src.machine_learning.dataset.cv_loader import (
        load_cv_type_distribution,
    )
    from desrutge.src.machine_learning.training.Traffic_Generator import (
        Traffic_Generation_Algorithms,
    )
    import desrutge.src.machine_learning.training.SIMULA as _SIMULA_MODULE
except ModuleNotFoundError as exc:
    print(
        f"[ERROR] Cannot import DesRUTGe modules: {exc}\n"
        f"  Make sure desrutge/src is present relative to this script.\n"
        f"  Expected path: {_DESRUTGE_SRC}"
    )
    sys.exit(1)

from route_postprocessor import postprocess_routes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Must match DesRUTGe's internal Norma constant in Traffic_Generator.py
_NORMA = 10000

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SmartTraffic-OM — Offline DesRUTGe Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py \\
      --cv_stats cv_stats.json \\
      --om_network /path/to/om_network \\
      --output generated/typed_routes.rou.xml

  python run_pipeline.py \\
      --cv_stats cv_stats.json \\
      --om_network ./om_network \\
      --output ./generated/routes_final.rou.xml \\
      --ppo_steps 300 \\
      --seed 0
        """,
    )
    p.add_argument(
        "--cv_stats",
        required=True,
        help="Path to cv_stats.json produced by smarttraffic_cv.py",
    )
    p.add_argument(
        "--om_network",
        required=True,
        help=(
            "Path to the OM network directory. Must contain: "
            "osm.net.xml, miniTAZ.xml, detectors.add.xml, sim.sumocfg"
        ),
    )
    p.add_argument(
        "--output",
        default="generated/typed_routes.rou.xml",
        help="Destination path for the final typed routes file "
             "(default: generated/typed_routes.rou.xml)",
    )
    p.add_argument(
        "--ppo_steps",
        type=int,
        default=200,
        help="Total PPO learn() timesteps (default: 200). "
             "DesRUTGe's internal train_model() runs 100 first, "
             "additional steps are run here if ppo_steps > 100.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible vehicle type assignment (default: 42)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_om_network(om_dir: str) -> None:
    required = ["osm.net.xml", "miniTAZ.xml", "detectors.add.xml", "sim.sumocfg"]
    missing  = [f for f in required if not os.path.isfile(os.path.join(om_dir, f))]
    if missing:
        print(
            f"[ERROR] om_network directory is missing required files:\n"
            + "\n".join(f"  {os.path.join(om_dir, f)}" for f in missing)
        )
        sys.exit(1)


def _check_sumo_tools() -> None:
    import shutil
    for tool in ("od2trips", "duarouter", "sumo"):
        if shutil.which(tool) is None:
            print(
                f"[ERROR] SUMO tool '{tool}' not found on PATH.\n"
                f"  Add the SUMO bin directory to your PATH and retry."
            )
            sys.exit(1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n{_SEP}")
    print(f"  SmartTraffic-OM  |  Offline DesRUTGe Pipeline")
    print(f"{_SEP}\n")

    t_start = time.time()

    # ── Validate inputs ────────────────────────────────────────────────────
    if not os.path.isfile(args.cv_stats):
        print(f"[ERROR] cv_stats.json not found: {args.cv_stats}")
        sys.exit(1)

    om_abs = os.path.abspath(args.om_network)
    if not os.path.isdir(om_abs):
        print(f"[ERROR] om_network directory not found: {om_abs}")
        sys.exit(1)

    _validate_om_network(om_abs)
    _check_sumo_tools()

    # ── Step 1: Load CV data ───────────────────────────────────────────────
    print(f"[1/4] Loading CV statistics from: {args.cv_stats}")
    trainloader = _SIMULA_MODULE.build_trainloader(scenario="high")
    type_dist     = load_cv_type_distribution(args.cv_stats)
    mean_flow_vph = sum(trainloader) / len(trainloader)
    ppo_target    = mean_flow_vph / _NORMA

    print(f"      Hourly profile (first 6 slots): {trainloader[:6]} ...")
    print(f"      Mean flow    : {mean_flow_vph:.1f} veh/hr")
    print(f"      PPO target   : {ppo_target:.5f}  "
          f"  (= {mean_flow_vph:.1f} / {_NORMA})")
    print(f"      Type dist    : "
          + "  ".join(f"{k}={round(v*100,1)}%" for k, v in type_dist.items()))

    # ── Step 2: Redirect DesRUTGe SIMULA to OM network ────────────────────
    print(f"\n[2/4] Redirecting DesRUTGe to OM network: {om_abs}")
    _SIMULA_MODULE.OM_NETWORK_DIR = om_abs

    # ── Step 3: Run DesRUTGe PPO calibration + SUMO route generation ──────
    print(f"\n[3/4] Running DesRUTGe PPO calibration "
          f"(target steps: {args.ppo_steps}) ...")

    conf = {"node_id": om_abs}
    algo = Traffic_Generation_Algorithms(conf=conf)

    # train_model() runs 100 steps internally and calls simulacion() which
    # writes om_abs/routes.rou.xml via the full OD → duarouter chain.
    algo.train_model(trainloader, conf)

    # Run additional steps if requested
    extra_steps = args.ppo_steps - 100
    if extra_steps > 0:
        print(f"      Running {extra_steps} additional PPO steps ...")
        algo.model.learn(total_timesteps=extra_steps)

    raw_rou = os.path.join(om_abs, "routes.rou.xml")
    if not os.path.isfile(raw_rou):
        print(
            f"[ERROR] DesRUTGe did not generate a route file.\n"
            f"  Expected: {raw_rou}\n"
            f"  Check that od2trips and duarouter ran successfully above."
        )
        sys.exit(1)

    raw_size = os.path.getsize(raw_rou)
    print(f"      Raw routes   : {raw_rou}  ({raw_size:,} bytes)")

    # ── Step 4: Post-process — assign SUMO vTypes from CV distribution ─────
    print(f"\n[4/4] Assigning vehicle types from CV distribution ...")
    typed_rou = postprocess_routes(
        input_path=raw_rou,
        output_path=args.output,
        type_distribution=type_dist,
        seed=args.seed,
    )

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    typed_size = os.path.getsize(typed_rou)

    print(f"{_SEP}")
    print(f"  PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"{_SEP}")
    print(f"  Input CV stats : {os.path.abspath(args.cv_stats)}")
    print(f"  OM network     : {om_abs}")
    print(f"  Raw routes     : {raw_rou}  ({raw_size:,} bytes)")
    print(f"  Typed routes   : {os.path.abspath(typed_rou)}  ({typed_size:,} bytes)")
    print()
    print(f"  Next step — update your OM.sumocfg:")
    print(f'    <route-files value="{os.path.abspath(typed_rou)}"/>')
    print(f"  Then run your RL system normally.")
    print(f"{_SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()