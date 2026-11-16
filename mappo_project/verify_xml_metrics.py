"""verify_xml_metrics.py

Run this ON YOUR MACHINE (where SUMO + traci are installed) to prove the
CO2/NOx/fuel accounting is correct on a REAL episode, not just fixture data.

It runs a short Fixed-Time baseline episode against OM.sumocfg with XML
output enabled, then prints the TraCI-vs-XML validation report so you can
see the actual numbers side by side.

Usage (from inside the mappo_project/ folder or with it on PYTHONPATH):
    python verify_xml_metrics.py --steps 300 --port 8813

Notes:
  - Uses OM.sumocfg's net/route/additional files as configured; make sure
    you run this from a directory where those relative paths resolve, or
    point --sim-file at your .sumocfg explicitly.
  - Writes tripinfo.xml / emissions.xml / fcd.xml under ./xml_verify_out/
    so you can also inspect them by hand afterwards.
  - fcd.xml is not required for CO2 verification and is skipped by default
    (use --with-fcd to also enable it -- it's the slowest output to write).
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# Make sure the package is importable whether this script sits inside
# mappo_project/ or next to it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(_THIS_DIR) == "mappo_project":
    sys.path.insert(0, os.path.dirname(_THIS_DIR))
    PKG = "mappo_project"
else:
    sys.path.insert(0, _THIS_DIR)
    PKG = "mappo_project"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim-file", default="sumo/OM.sumocfg",
                     help="Path to the .sumocfg file (default: sumo_network/OM.sumocfg)")
    ap.add_argument("--steps", type=int, default=300,
                     help="Max simulation steps to run (default: 300; full episode is 3600)")
    ap.add_argument("--port", type=int, default=8813)
    ap.add_argument("--gui", action="store_true", help="Run sumo-gui instead of headless sumo")
    ap.add_argument("--with-fcd", action="store_true",
                     help="Also enable fcd.xml output (slower, not needed for CO2 checks)")
    ap.add_argument("--out-dir", default="xml_verify_out",
                     help="Directory to write tripinfo/emissions/fcd XML into")
    ap.add_argument("--rel-tol", type=float, default=0.05,
                     help="Relative tolerance for TraCI-vs-XML agreement (default 5%%)")
    args = ap.parse_args()

    import importlib
    config = importlib.import_module(f"{PKG}.config")
    sumo_env = importlib.import_module(f"{PKG}.sumo_env")
    metrics_and_eval = importlib.import_module(f"{PKG}.metrics_and_eval")
    xml_metrics = importlib.import_module(f"{PKG}.xml_metrics")

    # -- force-enable XML output/validation for this run, regardless of
    #    whatever config.py currently has, so this script is self-contained.
    #
    # IMPORTANT: sumo_env.py and metrics_and_eval.py both do
    # `from .config import *`, which COPIES these names into their own
    # module namespace at import time. Setting config.ENABLE_XML_OUTPUT
    # afterwards does NOT retroactively update those copies -- Python's
    # `import *` binds a new name, it is not a live reference back to
    # config's namespace. So we must patch every module that star-imported
    # these constants, or functions like sumo_env._xml_output_args() will
    # keep reading their own stale, unpatched copy (False) and silently
    # never add the --tripinfo-output/--emission-output/--fcd-output flags.
    _xml_flag_overrides = {
        "ENABLE_XML_OUTPUT":      True,
        "ENABLE_XML_VALIDATION":  True,
        "XML_OUTPUT_DIR":         args.out_dir,
        "XML_VALIDATION_REL_TOL": args.rel_tol,
    }
    for _mod in (config, sumo_env, metrics_and_eval):
        for _name, _val in _xml_flag_overrides.items():
            setattr(_mod, _name, _val)

    os.makedirs(args.out_dir, exist_ok=True)

    is_windows = sys.platform.startswith("win")
    if is_windows:
        sumo_binary = "sumo-gui.exe" if args.gui else "sumo.exe"
    else:
        sumo_binary = "sumo-gui" if args.gui else "sumo"

    sumo_cmd = [
        sumo_binary,
        "-c", args.sim_file,
        "--step-length", str(int(config.STEP_LENGTH)),
        "--no-step-log",
        "--time-to-teleport", "300",
        "--collision.action", "warn",
    ]

    # Build simple fixed-time controllers for every configured intersection.
    controllers = {
        tls_id: metrics_and_eval.FixedTimeController(tls_id)
        for tls_id in config.INTERSECTIONS
    }

    print(f"[verify] Running {args.steps} steps against {args.sim_file} "
          f"(XML output -> {args.out_dir}/fixed_time/)", flush=True)

    result = metrics_and_eval._run_fixed_time_episode(
        sumo_cmd=sumo_cmd,
        sim_max_steps=args.steps,
        port=args.port,
        controllers=controllers,
        rich_ui=None,
    )

    print("\n" + "=" * 70)
    print("TraCI-side episode totals")
    print("=" * 70)
    print(f"  cars_passed : {result['__cars_passed']}")
    print(f"  ev_cleared  : {result['__ev_cleared']}")
    print(f"  CO2 (kg)    : {round(result['__co2_total']  / 1_000_000, 3)}")
    print(f"  NOx (g)     : {round(result['__nox_total']  / 1_000,      3)}")
    print(f"  fuel (kg)   : {round(result['__fuel_total'] / 1_000_000, 3)}")

    xml_m = result.get("__xml_metrics")
    val   = result.get("__xml_validation")

    print("\n" + "=" * 70)
    print("XML-side episode totals (from tripinfo.xml)")
    print("=" * 70)
    if xml_m is None:
        print("  [!] No XML metrics were loaded -- check that tripinfo.xml was "
              "written under the out-dir shown above.")
    else:
        print(f"  cars_passed : {xml_m['cars_passed']}")
        print(f"  ev_cleared  : {xml_m['ev_cleared']}")
        print(f"  CO2 (kg)    : {round(xml_m['co2_total']  / 1_000_000, 3)}")
        print(f"  NOx (g)     : {round(xml_m['nox_total']  / 1_000,      3)}")
        print(f"  fuel (kg)   : {round(xml_m['fuel_total'] / 1_000_000, 3)}")
        em = xml_m.get("emissions_xml_totals")
        if em:
            print(f"\n  Cross-check from emissions.xml (integrated rate*step_length):")
            print(f"    CO2 (kg)  : {round(em['CO2']  / 1_000_000, 3)}")
            print(f"    NOx (g)   : {round(em['NOx']  / 1_000,      3)}")
            print(f"    fuel (kg) : {round(em['fuel'] / 1_000_000, 3)}")
            print(f"    step_length used: {em['step_length']}"
                  f"  irregular={em['step_length_irregular']}")

    print("\n" + "=" * 70)
    print("VALIDATION REPORT (tolerance = "
          f"max({config.XML_VALIDATION_ABS_TOL}, {args.rel_tol*100:.0f}% of XML value))")
    print("=" * 70)
    if val is None:
        print("  [!] Validation did not run (xml_metrics missing).")
    else:
        for field, info in val.items():
            if field in ("all_consistent", "warnings"):
                continue
            status = "OK " if info["consistent"] else "FAIL"
            print(f"  [{status}] {field:<12} traci={info['traci']:<14} "
                  f"xml={info['xml']:<14} diff={info['diff']:<10.2f} "
                  f"({info['diff_pct']}%)")
        print()
        print(f"  ALL CONSISTENT: {val['all_consistent']}")
        for w in val.get("warnings", []):
            print(f"  [WARNING] {w}")

    report_path = os.path.join(args.out_dir, "verification_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "traci": {
                "cars_passed": result["__cars_passed"],
                "ev_cleared":  result["__ev_cleared"],
                "co2_total":   result["__co2_total"],
                "nox_total":   result["__nox_total"],
                "fuel_total":  result["__fuel_total"],
            },
            "xml": xml_m,
            "validation": val,
        }, f, indent=2)
    print(f"\n[verify] Full report written to {report_path}")


if __name__ == "__main__":
    main()
