# Cessna 172 RL Flight Control

**Reinforcement Learning for Longitudinal Flight Control of a Cessna 172 Skyhawk**

A curriculum-trained PPO agent evaluated against a classical PID controller across turbulence, wind shear, step-climb, and degraded control authority scenarios using JSBSim.

## Overview

This repository contains the MATLAB source code, trained agent, and evaluation results for an undergraduate dissertation investigating whether a Proximal Policy Optimisation (PPO) agent can match or exceed the performance of a hand-tuned PID controller for altitude and speed tracking on a Cessna 172 Skyhawk in the JSBSim flight dynamics simulator.

### Key Findings

| Scenario | PID | PPO Agent |
|----------|-----|-----------|
| Moderate turbulence (Sev 1–3) | Comparable | Comparable |
| Severe turbulence (Sev 5) | Higher RMS | Lower RMS, smoother |
| Step climb (CG-varied MC) | Precise overshoot, fast settle | Sluggish, steady-state error |
| Wind shear (adverse MC) | Maintained stall margin | Dropped below stall in some runs |
| Elevator icing (40% authority) | **Divergent — permanent altitude loss** | **Stable — maintained target** |

## Prerequisites

- **MATLAB** R2024a or later
- **Reinforcement Learning Toolbox**
- **Python** 3.12 (for JSBSim bridge)
- **JSBSim** — see [docs/setup.md](docs/setup.md) for installation

## Repository Structure

```
cessna172-rl-control/
├── environment/            # RL environment class (JSBSim wrapper)
├── training/               # Curriculum learning scripts (Phases 1–4 + finetune)
├── models/                 # Trained PPO agent (.mat)
├── evaluation/             # Single-run evaluation scenarios
├── monte_carlo/            # Monte Carlo and parametric sweep scripts
├── visualisation/          # FlightGear 3D demo
├── results/                # Figures (300 DPI) and raw data (.mat)
└── docs/                   # Setup and installation guide
```

### Evaluation Scripts → Dissertation Sections

| Script | Description |
|--------|-------------|
| `evaluation/step_climb.m` | 500ft step-climb transient response |
| `evaluation/turbulence_sweep.m` | Severity 1–5 sweep with 10-run MC |
| `evaluation/severe_turbulence.m` | Severity 5 detailed time-series |
| `evaluation/wind_shear.m` | Microburst wind shear penetration |
| `evaluation/elevator_icing.m` | Elevator icing + downdraft (key finding) |
| `monte_carlo/step_climb_mc.m` | Parametric MC — CG/payload variation |
| `monte_carlo/severe_turbulence_mc.m` | Stochastic MC — 10 seed realisations |
| `monte_carlo/wind_shear_mc.m` | Adverse MC — randomised onset/severity |
| `monte_carlo/icing_authority_sweep.m` | Elevator Authority sweep — 80% down to 40% |

## Quick Start

```matlab
% 1. Run the most important scenario (elevator icing)
run('evaluation/elevator_icing.m')

% 2. Run the parametric authority sweep
run('monte_carlo/icing_authority_sweep.m')

% 3. Run the turbulence severity sweep
run('evaluation/turbulence_sweep.m')
```

## Training

The agent was trained using a 4-phase curriculum:

| Phase | Ailerons | Throttle | Turbulence | Episodes |
|-------|----------|----------|------------|----------|
| 1 | 0% | Fixed 80% | Severity 1 | 1200 |
| 2 | 50% | Fixed 80% | Severity 1–2 | 3500 |
| 3 | 100% | Fixed 80% | Severity 1–3 | 5000 |
| 4 | 100% | 100% (learned) | Severity 1–4 | 5460 |

Followed by conservative fine-tuning with a reduced learning rate.

## Licence

This project was completed as an undergraduate dissertation at Loughborough University. All rights reserved.
