# Vision-Driven MARL Traffic Control

Multi-agent reinforcement learning system for adaptive traffic signal control, using MAPPO with hierarchical GAT intent sharing.

## Overview

This project implements a multi-agent reinforcement learning system that optimizes traffic signal control at urban intersections. It combines MAPPO (Multi-Agent Proximal Policy Optimization) with a hierarchical graph-attention (GAT) intent-sharing mechanism.

**Key features:**
- **MAPPO + Hierarchical GAT** — Multi-agent PPO with local-cluster + gateway attention
- **KL-adaptive learning rates** — Automatic scaling based on policy KL divergence
- **Emergency vehicle priority** — Dedicated reward shaping for emergency vehicles
- **SUMO simulation** — Evaluated on a real intersection network

## Repository Structure

```
Vision-Driven-MARL-Traffic-Control/
├── evalaution.py                      # Evaluation script (Webster-Pressure baseline)
├── requirements.txt              # Python dependencies
│
├── mappo_project/                # Core Python package
│   ├── agent.py                  # MAPPOAgent, RolloutBuffer
│   ├── config.py                 # Hyperparameters, network maps
│   ├── networks.py               # Neural network architectures
│   ├── intent_aggregators.py     # GAT / flat GAT / mean intent aggregators
│   ├── training_loop.py          # Main MAPPO+GAT training loop
│   ├── training_loop_ippo.py     # IPPO training loop
│   ├── training_loop_vanilla_ppo.py  # Vanilla PPO training loop
│   ├── ippo.py                   # Independent PPO implementation
│   ├── vanilla_ppo.py            # Vanilla PPO implementation
│   ├── rewards.py                # Reward functions
│   ├── kl_adaptive_lr.py         # KL-adaptive learning rate controller
│   ├── ev_priority.py            # Emergency vehicle priority logic
│   ├── sumo_env.py               # SUMO environment wrapper
│   ├── eval_framework.py         # Evaluation framework
│   ├── evaluate.py               # Evaluation script
│   ├── run_experiments.py        # Full ablation experiment runner
│   ├── metrics_and_eval.py       # Metrics and evaluation utilities
│   ├── visualization.py          # Plotting / visualization
│   ├── xml_metrics.py            # XML metrics parser
│   └── verify_xml_metrics.py     # Cross-check TraCI vs SUMO XML metrics
│
├── sumo_train/         # SUMO network files for training
│   ├── OM.sumocfg                # SUMO simulation config
│   ├── OM.net.xml                # Road network
│   ├── OM.rou.xml                # Vehicle routes
│   ├── OM.add.xml                # Detectors and additional logic
│   └── e2_*.xml                  # Lane-area detector files
│
└── sumo_eval/          # SUMO network files for evaluation
    ├── OM.sumocfg                # SUMO simulation config
    ├── OM.net.xml                # Road network
    ├── OM.rou.xml                # Vehicle routes
    └── e2_*.xml                  # Lane-area detector files
```

## Requirements

- **SUMO** (>= 1.18) — [Installation guide](https://sumo.dlr.de/docs/Installing/index.html)
- **Python** 3.10+
- **PyTorch** (with CUDA recommended for GPU training)
- **PyG** (torch-geometric) for GAT layers

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/mmm005721-hue/Vision-Driven-MARL-Traffic-Control.git
cd Vision-Driven-MARL-Traffic-Control

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure SUMO is installed and on your PATH
sumo --version   # should print SUMO version >= 1.18
```

## Usage

### Train MAPPO + Hierarchical GAT (default)

```bash
python mappo_project/training_loop.py --sim_file sumo_train/OM.sumocfg --n_episodes 300
```

### Train baselines

```bash
# IPPO
python mappo_project/training_loop_ippo.py --sim_file sumo_train/OM.sumocfg

# Vanilla PPO
python mappo_project/training_loop_vanilla_ppo.py --sim_file sumo_train/OM.sumocfg
```

### Evaluate

```bash
# Compare RL vs Webster-Pressure baseline
python evalaution.py --sim_file sumo_eval/OM.sumocfg --mode compare

# RL-only evaluation
python evalaution.py --sim_file sumo_eval/OM.sumocfg --mode rl

# IPPO evaluation
python evalaution.py --sim_file sumo_eval/OM.sumocfg --mode rl --algorithm ippo --ckpt_dir ./checkpoints/ep0240
```

### Key CLI Flags

| Flag | Description |
|---|---|
| `--sim_file` | Path to SUMO `.sumocfg` file |
| `--mode` | `compare` (default), `rl`, `webster` |
| `--algorithm` | `mappo` (default), `ippo` |
| `--gui` | `enable` or `disable` SUMO GUI |
| `--report` | Generate HTML report |
| `--n_episodes` | Number of training episodes |

## Evaluation Modes

| Mode | Description |
|---|---|
| `compare` | Webster-Pressure vs RL (default) |
| `rl` | RL-only evaluation |
| `webster` | Webster-Pressure only (no checkpoint required) |

## Citation

```bibtex
@misc{vision-driven-marl-traffic,
  title     = {{Vision-Driven MARL Traffic Control}: Multi-Agent RL for Adaptive Traffic Signal Control},
  author    = {{Sphinx University --- Faculty of Computers and Artificial Intelligence}},
  year      = {2025},
  url       = {https://github.com/mmm005721-hue/Vision-Driven-MARL-Traffic-Control},
  license   = {MIT}
}
```

## License

This project is licensed under the MIT License.
