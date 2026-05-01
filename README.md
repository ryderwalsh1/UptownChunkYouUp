# Dual-System Cognitive Control with Lambda-Modulated TD Learning

A cognitive neuroscience research project implementing a biologically-inspired dual-system architecture that models fast habitual processing and slow deliberative memory retrieval with adaptive cognitive control.

## Overview

This repository implements a reinforcement learning agent with two processing systems:

- **Fast System**: A GRU-based policy network for quick, habitual responses
- **Slow System**: Episodic memory retrieval for deliberative, optimal decision-making
- **Meta-Controller**: Learns when to allocate cognitive resources to slow processing
- **Lambda Modulation**: Adaptively adjusts temporal credit assignment based on control demand

The system learns to balance reward maximization, processing efficiency, and cognitive control costs while navigating graph-structured maze environments.

## Repository Organization

```
.
├── train.py                    # Main training script (entry point)
├── README.md                   # This file
├── CLAUDE.md                   # AI assistant guidance for the codebase
│
├── src/                        # Core implementation modules
│   ├── agent.py               # Cognitive agent orchestrating all components
│   ├── fast.py                # Fast GRU-based policy network
│   ├── slow.py                # Slow episodic memory system
│   ├── controller.py          # Meta-controller for system selection
│   ├── conflict_map.py        # Long-term control demand estimates
│   ├── lambda_modulator.py    # Adaptive TD(λ) parameter modulation
│   ├── maze_env.py            # RL environment wrapper
│   ├── corridors.py           # Maze graph generation
│   ├── em.py                  # Episodic memory module
│   └── simple_fast.py         # Simplified fast network variant
│
├── experiments/                # Experimental training scripts
│   ├── run_lambda_experiment.py
│   ├── run_fixed_lambda_baseline.py
│   ├── run_elementary_topology_experiment.py
│   ├── train_and_plot_adaptive_lambda_comparison.py
│   ├── train_collection.py
│   ├── train_hyperparam_sweep.py
│   ├── small_grid_sweep.py
│   └── debug_training.py
│
├── scripts/                    # Standalone plotting & visualization
│   ├── plot_collection.py
│   ├── plot_elementary_topologies.py
│   ├── plot_smallgridsweep.py
│   ├── visualize_hyperparameter_interpolation.py
│   └── inspect_agent.py
│
├── tests/                      # Test scripts
│   ├── test_lambda_experiment.py
│   └── test_remaining_topologies.py
│
├── docs/                       # Documentation & figures
│   ├── implementation.md       # Detailed architecture specification
│   ├── fast_state_entropy_explanation.md
│   ├── agent_architecture.png
│   ├── lambda_modulation_curves.png
│   ├── elementary_topologies.png
│   ├── stage2_training_skip_failure_mode.png
│   └── plot_config.yaml
│
├── outputs/                    # Generated outputs (gitignored)
│   ├── checkpoints/           # Saved model weights
│   ├── results/               # Training results & metrics
│   └── figures/               # Generated plots & visualizations
│
└── legacy/                     # Previous implementation attempts
```

## Quick Start

### Prerequisites

- Python 3.12+
- PyTorch
- NumPy, NetworkX, Matplotlib, tqdm

### Setup

```bash
# Activate virtual environment
source venv/bin/activate

# Run main training
python train.py
```

### Training

The main training script implements a two-stage approach:

**Stage 1** (500 episodes): Pretrain the fast network alone
```bash
python train.py
```

**Stage 2** (10,000 episodes): Add slow memory and meta-controller, train the full system

### Testing Components

Each core module can be tested independently:

```bash
python src/agent.py           # Test full cognitive agent
python src/fast.py           # Test fast network
python src/slow.py           # Test slow memory
python src/controller.py     # Test meta-controller
python src/maze_env.py       # Test maze environment
```

### Running Experiments

```bash
# Lambda modulation experiments
python experiments/run_lambda_experiment.py

# Fixed lambda baseline comparison
python experiments/run_fixed_lambda_baseline.py

# Topology experiments
python experiments/run_elementary_topology_experiment.py

# Hyperparameter sweeps
python experiments/train_hyperparam_sweep.py
```

### Visualization

```bash
# Plot training results
python scripts/plot_collection.py

# Visualize agent behavior
python scripts/inspect_agent.py

# Generate topology plots
python scripts/plot_elementary_topologies.py
```

## Core Architecture

### Components

1. **Fast Network** ([src/fast.py](src/fast.py))
   - GRU-based policy network for habitual processing
   - Outputs action distributions and value estimates
   - Trained with actor-critic TD(λ)

2. **Slow Memory** ([src/slow.py](src/slow.py))
   - Dictionary-based episodic retrieval
   - Pre-populated with optimal shortest-path actions
   - Not trained - pure memory lookup

3. **Meta-Controller** ([src/controller.py](src/controller.py))
   - Decides when to use slow vs. fast processing
   - Learns to balance reward, efficiency, and control cost
   - Trained via reinforcement learning

4. **Conflict Map** ([src/conflict_map.py](src/conflict_map.py))
   - Tracks long-term control demand per state
   - Exponential moving average of fast-slow KL divergence
   - Informs lambda modulation

5. **Lambda Modulator** ([src/lambda_modulator.py](src/lambda_modulator.py))
   - Maps control demand to TD(λ) parameter
   - High demand → low λ (local credit assignment)
   - Low demand → high λ (long-horizon backup, chunking)

6. **Cognitive Agent** ([src/agent.py](src/agent.py))
   - Orchestrates all components
   - Implements sample-based arbitration
   - Coordinates training and decision-making

### Key Design Principles

- **Sample-based arbitration**: Controller samples which system to use, preserving exploration
- **Control cost penalty**: Slow processing incurs cost, forcing selective use
- **Adaptive credit assignment**: Lambda modulation enables implicit chunking
- **Episodic memory as retrieval**: Slow system is not learned, only accessed
- **Separation of signals**: Confidence (entropy) vs. conflict (KL divergence)

## Environment

The agent operates in graph-structured mazes:
- **Maze Graph** ([src/corridors.py](src/corridors.py)): Generates mazes parameterized by corridor density (0=junctions, 1=corridors)
- **Actions**: Direct movement to any adjacent node
- **Goals**: Navigate to target nodes with minimal steps

## Documentation

- **[docs/implementation.md](docs/implementation.md)**: Complete architecture specification
- **[CLAUDE.md](CLAUDE.md)**: Developer guidance for working with this codebase
- **[docs/fast_state_entropy_explanation.md](docs/fast_state_entropy_explanation.md)**: Explanation of confidence signals

## Training Details

- Two-stage training prevents interference between fast network and controller
- Gradient clipping (0.5) for stability
- Conflict map uses small alpha (0.01) for long-term stability
- Lambda modulator uses β=2.0 for superlinear mapping
- Checkpoints saved to `outputs/checkpoints/`
- Training curves and metrics saved to `outputs/results/`

## Output Structure

All generated outputs are organized under `outputs/`:

- `outputs/checkpoints/`: Model weights at various training stages
- `outputs/results/`: Training metrics, performance data, trajectory logs
- `outputs/figures/`: Generated plots and visualizations

These directories are gitignored to keep the repository clean.

## Citation

This research project is part of a cognitive neuroscience thesis exploring dual-system cognitive control and adaptive temporal credit assignment.

## License

Research project - contact for usage permissions.
