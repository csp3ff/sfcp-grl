# Adaptive Orchestration of Service Function Chains in SAGIN-MEC via Graph Reinforcement Learning

[简体中文](README.md) | **English**

This repository contains the implementation of **“Adaptive Orchestration of Service Function Chains in SAGIN-MEC via Graph Reinforcement Learning.”** It uses a graph Reinforement Learning (GRL) with proximal policy optimization (PPO) to place service function chains (SFCs) on space–air–ground integrated network (SAGIN).

## Requirements

- Python>=3.10
- PyTorch, NumPy, NetworkX, pandas, PyYAML, and Matplotlib

Run all commands below from the repository root. The physical network topology is provided at `dataset/topology/network.gpickle`; SFC requests are generated during execution.

## Train

Review or edit `train.yml`, then start training:

```bash
python train.py
```

The main configuration fields are:

| Field | Purpose |
| --- | --- |
| `num_episode` | Number of training episodes |
| `total_time_steps` | Time slots per episode |
| `arrival_rate` | Probability of generating an SFC request in each time slot |
| `net_path` | Path to the physical network topology |
| `model` | Model name; currently `GCN_PPO` |
| `GCN_PPO` | Learning rates, batch sizes, and other PPO parameters |

Training creates `results/GCN_PPO/<timestamp>/` with the following files:

```text
results/GCN_PPO/<timestamp>/
├── config.yml                 # Configuration saved for this run
├── model/
│   ├── actor.pth              # Best actor checkpoint by episode reward
│   └── critic.pth             # Matching critic checkpoint
└── train/
    └── <timestamp>.csv        # Per-episode metrics
```

## Test

Pass the `train` directory from a completed run to `test.py`. Replace `YYYYMMDD_HHMMSS` with the timestamp created by training:

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train
```

The default test uses the arrival rate and time horizon saved in that run's `config.yml`. To use a different arrival rate, pass `--a`:

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train --a 0.3
```

To run the test that reports statistics for 500-slot intervals, use `--t 2`:

```bash
python test.py --p results/GCN_PPO/YYYYMMDD_HHMMSS/train --t 2
```

This mode uses 10,000 time slots. Its arrival rate starts at `0.1` by default (or the value supplied with `--a`) and increases by `0.05` for each 500-slot interval. Test output is written to `results/GCN_PPO/<timestamp>/test/`: the default mode writes `<timestamp>.csv`, while `--t 2` writes `interval_results_<timestamp>.csv`.

## Repository layout

| Path | Contents |
| --- | --- |
| `model/` | GCN actor and critic networks and PPO implementation |
| `controller/` | SFC placement, routing, and resource constraint handling |
| `environment/` | Physical and virtual networks, simulation environment, and evaluation |
| `evolution/` | Training and testing loops and metric recording |
| `utils/` | SFC request generation and component selection |
| `dataset/topology/` | Physical network topology used by the simulation |
| `train.py`, `test.py`, `train.yml` | Entry points and training configuration |
