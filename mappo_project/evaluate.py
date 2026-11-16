from mappo_project.config import *
from mappo_project.sumo_env import _start_sumo, _close_sumo
from mappo_project.agent import MAPPOAgent
from mappo_project.metrics_and_eval import _run_evaluation_episode, load_checkpoint
import torch
import os
import json
import datetime
import webbrowser
from pathlib import Path
import argparse
def run_evaluation(sim_file, save_dir, max_steps, gui, checkpoint="latest", deterministic=True, report=False, open_report=False):
    print("Loading agents...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agents
    agents = {}
    import traci
    # Dummy start to get TLS IDs
    proc = _start_sumo(["sumo", "-c", sim_file], port=BASE_PORT)
    for tls_id in traci.trafficlight.getIDList():
        agents[tls_id] = MAPPOAgent(tls_id, device)
    _close_sumo(proc)
    
    # Load latest checkpoint
    loaded_ep = load_checkpoint(agents, save_dir, episode=checkpoint)
    if loaded_ep == 0:
        print("Warning: No checkpoint found. Evaluating with random weights.")
    else:
        print(f"Loaded checkpoint from episode {loaded_ep}")

    sumo_cmd = [
        "sumo-gui" if gui == "enable" else "sumo",
        "-c", sim_file,
        "--step-length", str(int(STEP_LENGTH)),
        "--no-step-log",
        "--time-to-teleport", "300"
        "--collision.action", "warn"
    ]
    
    print("Starting evaluation episode...")
    metrics = _run_evaluation_episode(sumo_cmd, agents, max_steps, BASE_PORT + 1, deterministic=deterministic)
    
    print("\n=== Evaluation Results ===")
    print(f"Average Reward: {metrics['avg_reward']:.4f}")
    print(f"Average Queue:  {metrics['avg_queue']:.4f}")
    print(f"Average Wait:   {metrics['avg_wait']:.4f}")
    print(f"Cars Passed:    {metrics['cars_passed']}")
    if metrics['ev_cleared'] > 0:
        print(f"EVs Cleared:    {metrics['ev_cleared']} (Avg Wait: {metrics['ev_avg_wait']}s)")
    _co2_kg = metrics.get('co2_total', 0) / 1_000_000
    _nox_g  = metrics.get('nox_total', 0) / 1_000
    _fuel_kg = metrics.get('fuel_total', 0) / 1_000_000
    if _co2_kg or _nox_g or _fuel_kg:
        print(f"CO₂ (kg) : {_co2_kg:.3f}")
        print(f"NOx (g)  : {_nox_g:.3f}")
        print(f"Fuel (kg): {_fuel_kg:.3f}")
    print("==========================")

    # After evaluation, optionally generate report and open it
    report_path = None
    if report:
        report_path = generate_report(metrics, save_dir)
    if open_report and report_path:
        webbrowser.open(f'file://{report_path}')
    return metrics

def generate_report(metrics: dict, save_dir: str):
    """Generate an HTML report summarizing evaluation metrics for research consumption.
    The report includes a styled table and timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = Path(save_dir) / "evaluation_report.html"
    html_content = f"""
<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>Evaluation Report</title>
<style>
  body {{font-family: 'Segoe UI', Tahoma, sans-serif; background: #f9f9f9; color:#333; padding:20px;}}
  h1 {{color:#2c3e50;}}
  table {{border-collapse: collapse; width: 60%; margin-top:20px;}}
  th, td {{border:1px solid #ddd; padding:8px; text-align:left;}}
  th {{background:#2c3e50; color:white;}}
  tr:nth-child(even){{background:#f2f2f2;}}
  .footer {{margin-top:20px; font-size:0.9em; color:#777;}}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<p><strong>Generated:</strong> {timestamp}</p>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Average Reward</td><td>{metrics.get('avg_reward', 0):.4f}</td></tr>
<tr><td>Average Queue</td><td>{metrics.get('avg_queue', 0):.4f}</td></tr>
<tr><td>Average Wait</td><td>{metrics.get('avg_wait', 0):.4f}</td></tr>
<tr><td>Cars Passed</td><td>{metrics.get('cars_passed', 0)}</td></tr>
"""
    if metrics.get('ev_cleared', 0) > 0:
        html_content += f"<tr><td>EVs Cleared</td><td>{metrics['ev_cleared']} (Avg Wait: {metrics['ev_avg_wait']}s)</td></tr>"
    html_content += """
</table>
<div class='footer'>Evaluation script version: 1.0</div>
</body>
</html>
"""
    report_path.write_text(html_content, encoding='utf-8')
    print(f"HTML report written to {report_path}")
    return str(report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_file", required=True)
    parser.add_argument("--save_dir", default="./results")
    parser.add_argument("--max_steps", type=int, default=3600)
    parser.add_argument("--gui", choices=["enable", "disable"], default="enable")
    parser.add_argument("--report", action='store_true', help="Generate an HTML report of the evaluation results.")
    parser.add_argument("--open_report", action='store_true', help="Open the generated HTML report in the default web browser after evaluation.")
    parser.add_argument("--checkpoint", default="latest", help="Checkpoint tag (e.g., ep0200 or latest)")
    parser.add_argument("--deterministic", type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to False to enable random exploration (stochastic testing)")
    args = parser.parse_args()
    metrics = run_evaluation(args.sim_file, args.save_dir, args.max_steps, args.gui, args.checkpoint, args.deterministic, report=args.report, open_report=args.open_report)
