"""
Hyperparameter Sweep Training Script

Trains the cognitive agent on a single maze topology (8x8, corridor=1.0, seed=60)
while varying hyperparameters:
- conflict_alpha: [0.01, 0.05, 0.1, 0.2]
- lambda_beta: [1.5, 2.0, 3.0, 5.0]
- w_long: [0.2, 0.5, 0.8, 0.9]

Each parameter is swept independently while keeping others at default values.
Results are saved to results/hyperparam_sweep_<timestamp>/ directory.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count

from maze_env import MazeEnvironment
from agent import CognitiveAgent
from corridors import MazeGraph
from train import Stage2Trainer, plot_stage2_curves, plot_conflict_map_heatmap


# Default hyperparameters
DEFAULT_CONFLICT_ALPHA = 0.05
DEFAULT_LAMBDA_BETA = 2.0
DEFAULT_W_LONG = 0.8

# Hyperparameter sweep values
CONFLICT_ALPHA_VALUES = [0.01, 0.05, 0.1, 0.2]
LAMBDA_BETA_VALUES = [1.5, 2.0, 3.0, 5.0]
W_LONG_VALUES = [0.2, 0.5, 0.8, 0.9]

# Fixed maze topology
MAZE_LENGTH = 8
MAZE_WIDTH = 8
MAZE_CORRIDOR = 1.0
MAZE_SEED = 60

# Fixed training configuration
FIXED_START_NODE = (7, 0)
GOAL_IS_DEADEND = True
NUM_EPISODES_STAGE2 = 10000
LR_FAST = 3e-4
LR_CONTROLLER = 1e-3
GAMMA = 0.99
CONTROL_COST = 0.15


def run_single_training(hyperparam_name, hyperparam_value,
                       conflict_alpha, lambda_beta, w_long, save_dir):
    """
    Run a single training run with specified hyperparameters.

    This is a standalone function to enable multiprocessing parallelization.

    Parameters:
    -----------
    hyperparam_name : str
        Name of the hyperparameter being swept
    hyperparam_value : float
        Value of the hyperparameter being swept
    conflict_alpha : float
        Conflict map learning rate
    lambda_beta : float
        Lambda modulation exponent
    w_long : float
        Weight for long-term conflict (w_short = 1 - w_long)
    save_dir : str
        Directory to save results

    Returns:
    --------
    result_summary : dict
        Summary of training results
    """
    # Prevent PyTorch from spawning multiple threads per process
    torch.set_num_threads(1)

    w_short = 1.0 - w_long

    # Create maze and environment (fresh for this process)
    maze = MazeGraph(length=MAZE_LENGTH, width=MAZE_WIDTH, corridor=MAZE_CORRIDOR, seed=MAZE_SEED)
    env = MazeEnvironment(
        length=MAZE_LENGTH, width=MAZE_WIDTH, corridor=MAZE_CORRIDOR, seed=MAZE_SEED,
        control_cost=CONTROL_COST,
        fixed_start_node=FIXED_START_NODE,
        goal_is_deadend=GOAL_IS_DEADEND
    )

    # Create agent with specified hyperparameters
    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze.get_graph(),
        control_cost=CONTROL_COST,
        conflict_alpha=conflict_alpha,
        lambda_beta=lambda_beta,
        w_long=w_long,
        w_short=w_short
    )

    # Prepare metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'sweep_parameter': hyperparam_name,
        'sweep_value': hyperparam_value,
        'environment': {
            'length': MAZE_LENGTH,
            'width': MAZE_WIDTH,
            'corridor': MAZE_CORRIDOR,
            'seed': MAZE_SEED,
            'num_nodes': env.num_nodes,
            'num_actions': env.num_actions,
            'num_deadends': len(env.deadend_nodes),
            'fixed_start_node': list(FIXED_START_NODE),
            'goal_is_deadend': GOAL_IS_DEADEND,
        },
        'training': {
            'num_episodes_stage2': NUM_EPISODES_STAGE2,
            'log_interval': NUM_EPISODES_STAGE2 // 100,
            'lr_fast': LR_FAST,
            'lr_controller': LR_CONTROLLER,
            'gamma': GAMMA,
        },
        'agent': {
            'control_cost': CONTROL_COST,
            'conflict_alpha': conflict_alpha,
            'lambda_beta': lambda_beta,
            'w_long': w_long,
            'w_short': w_short,
        },
        'architecture': {
            'fast_hidden_dim': agent.fast_network.hidden_dim,
            'fast_embedding_dim': agent.fast_network.embedding_dim,
            'controller_hidden_dim': agent.controller.hidden_dim,
            'controller_embedding_dim': agent.controller.embedding_dim,
        }
    }

    # Save metadata
    os.makedirs(save_dir, exist_ok=True)
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Train Stage 2
    stage2_trainer = Stage2Trainer(env, agent, lr_fast=LR_FAST, lr_controller=LR_CONTROLLER, gamma=GAMMA)
    stage2_metrics = stage2_trainer.train(
        num_episodes=NUM_EPISODES_STAGE2,
        log_interval=NUM_EPISODES_STAGE2 // 100,
        save_interval=None,  # Don't save intermediate checkpoints for sweep
        save_dir=save_dir,
        metadata=metadata
    )

    # Save final agent
    agent.save(os.path.join(save_dir, 'agent_final.pt'))

    # Plot Stage 2 results
    plot_stage2_curves(stage2_metrics, save_path=os.path.join(save_dir, 'stage2_training.png'))

    # Plot conflict map heatmap
    plot_conflict_map_heatmap(agent, maze, save_path=os.path.join(save_dir, 'conflict_map_heatmap.png'))

    # Save metrics to JSON
    # Convert lists to serializable format
    metrics_serializable = {
        'episode_rewards': stage2_metrics['episode_rewards'],
        'episode_lengths': stage2_metrics['episode_lengths'],
        'success_rate': stage2_metrics['success_rate'],
        'p_slow': stage2_metrics['p_slow'],
        'p_fast': stage2_metrics['p_fast'],
        'mean_lambda': stage2_metrics['mean_lambda'],
        'mean_fast_entropy': stage2_metrics['mean_fast_entropy'],
        'mean_kl_divergence': stage2_metrics['mean_kl_divergence'],
        'used_slow_count': stage2_metrics['used_slow_count'],
        'used_fast_count': stage2_metrics['used_fast_count'],
        'mean_delta': stage2_metrics['mean_delta'],
    }

    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    # Return summary for aggregate reporting
    result_summary = {
        'hyperparam_name': hyperparam_name,
        'hyperparam_value': hyperparam_value,
        'final_success_rate': np.mean(stage2_metrics['success_rate'][-100:]),
        'final_reward': np.mean(stage2_metrics['episode_rewards'][-100:]),
        'final_p_slow': np.mean(stage2_metrics['p_slow'][-100:]),
        'final_lambda': np.mean(stage2_metrics['mean_lambda'][-100:])
    }

    return result_summary


def run_conflict_alpha_sweep(base_dir):
    """Run sweep over conflict_alpha values using parallel processing."""
    print("\n" + "=" * 80)
    print("SWEEPING CONFLICT_ALPHA")
    print("=" * 80)
    print(f"Running {len(CONFLICT_ALPHA_VALUES)} configurations in parallel...")

    # Create list of all configurations to run
    configs_to_run = [
        (
            'conflict_alpha',
            alpha,
            alpha,
            DEFAULT_LAMBDA_BETA,
            DEFAULT_W_LONG,
            os.path.join(base_dir, f'conflict_alpha_{alpha}')
        )
        for alpha in CONFLICT_ALPHA_VALUES
    ]

    # Run all configurations in parallel
    with Pool() as pool:
        all_results = pool.starmap(run_single_training, configs_to_run)

    # Collect results
    results = {}
    for result_summary in all_results:
        alpha = result_summary['hyperparam_value']
        results[alpha] = {
            'final_success_rate': result_summary['final_success_rate'],
            'final_reward': result_summary['final_reward'],
            'final_p_slow': result_summary['final_p_slow'],
            'final_lambda': result_summary['final_lambda']
        }
        print(f"\n✓ conflict_alpha={alpha} complete")
        print(f"  Final success rate: {results[alpha]['final_success_rate']:.2%}")
        print(f"  Final reward: {results[alpha]['final_reward']:.2f}")

    return results


def run_lambda_beta_sweep(base_dir):
    """Run sweep over lambda_beta values using parallel processing."""
    print("\n" + "=" * 80)
    print("SWEEPING LAMBDA_BETA")
    print("=" * 80)
    print(f"Running {len(LAMBDA_BETA_VALUES)} configurations in parallel...")

    # Create list of all configurations to run
    configs_to_run = [
        (
            'lambda_beta',
            beta,
            DEFAULT_CONFLICT_ALPHA,
            beta,
            DEFAULT_W_LONG,
            os.path.join(base_dir, f'lambda_beta_{beta}')
        )
        for beta in LAMBDA_BETA_VALUES
    ]

    # Run all configurations in parallel
    with Pool() as pool:
        all_results = pool.starmap(run_single_training, configs_to_run)

    # Collect results
    results = {}
    for result_summary in all_results:
        beta = result_summary['hyperparam_value']
        results[beta] = {
            'final_success_rate': result_summary['final_success_rate'],
            'final_reward': result_summary['final_reward'],
            'final_p_slow': result_summary['final_p_slow'],
            'final_lambda': result_summary['final_lambda']
        }
        print(f"\n✓ lambda_beta={beta} complete")
        print(f"  Final success rate: {results[beta]['final_success_rate']:.2%}")
        print(f"  Final reward: {results[beta]['final_reward']:.2f}")

    return results


def run_w_long_sweep(base_dir):
    """Run sweep over w_long values using parallel processing."""
    print("\n" + "=" * 80)
    print("SWEEPING W_LONG")
    print("=" * 80)
    print(f"Running {len(W_LONG_VALUES)} configurations in parallel...")

    # Create list of all configurations to run
    configs_to_run = [
        (
            'w_long',
            w_long,
            DEFAULT_CONFLICT_ALPHA,
            DEFAULT_LAMBDA_BETA,
            w_long,
            os.path.join(base_dir, f'w_long_{w_long}')
        )
        for w_long in W_LONG_VALUES
    ]

    # Run all configurations in parallel
    with Pool() as pool:
        all_results = pool.starmap(run_single_training, configs_to_run)

    # Collect results
    results = {}
    for result_summary in all_results:
        w_long = result_summary['hyperparam_value']
        results[w_long] = {
            'final_success_rate': result_summary['final_success_rate'],
            'final_reward': result_summary['final_reward'],
            'final_p_slow': result_summary['final_p_slow'],
            'final_lambda': result_summary['final_lambda']
        }
        print(f"\n✓ w_long={w_long} complete")
        print(f"  Final success rate: {results[w_long]['final_success_rate']:.2%}")
        print(f"  Final reward: {results[w_long]['final_reward']:.2f}")

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("HYPERPARAMETER SWEEP TRAINING")
    print("=" * 80)
    print(f"Maze topology: {MAZE_LENGTH}x{MAZE_WIDTH}, corridor={MAZE_CORRIDOR}, seed={MAZE_SEED}")
    print(f"Training episodes: {NUM_EPISODES_STAGE2}")
    print(f"\nSweep ranges:")
    print(f"  conflict_alpha: {CONFLICT_ALPHA_VALUES}")
    print(f"  lambda_beta: {LAMBDA_BETA_VALUES}")
    print(f"  w_long: {W_LONG_VALUES}")
    print(f"\nDefaults:")
    print(f"  conflict_alpha: {DEFAULT_CONFLICT_ALPHA}")
    print(f"  lambda_beta: {DEFAULT_LAMBDA_BETA}")
    print(f"  w_long: {DEFAULT_W_LONG}")
    print("=" * 80)

    # Create timestamped base directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f'results/hyperparam_sweep_{timestamp}'
    os.makedirs(base_dir, exist_ok=True)

    print(f"\nResults will be saved to: {base_dir}")
    print(f"Available CPU cores: {cpu_count()}")
    print()

    # Run all sweeps
    all_results = {}

    # Sweep 1: conflict_alpha
    alpha_results = run_conflict_alpha_sweep(base_dir)
    all_results['conflict_alpha'] = alpha_results

    # Sweep 2: lambda_beta
    beta_results = run_lambda_beta_sweep(base_dir)
    all_results['lambda_beta'] = beta_results

    # Sweep 3: w_long
    w_long_results = run_w_long_sweep(base_dir)
    all_results['w_long'] = w_long_results

    # Save summary
    summary = {
        'timestamp': timestamp,
        'maze_config': {
            'length': MAZE_LENGTH,
            'width': MAZE_WIDTH,
            'corridor': MAZE_CORRIDOR,
            'seed': MAZE_SEED
        },
        'training_config': {
            'num_episodes_stage2': NUM_EPISODES_STAGE2,
            'lr_fast': LR_FAST,
            'lr_controller': LR_CONTROLLER,
            'gamma': GAMMA,
            'control_cost': CONTROL_COST
        },
        'sweep_config': {
            'conflict_alpha_values': CONFLICT_ALPHA_VALUES,
            'lambda_beta_values': LAMBDA_BETA_VALUES,
            'w_long_values': W_LONG_VALUES,
            'defaults': {
                'conflict_alpha': DEFAULT_CONFLICT_ALPHA,
                'lambda_beta': DEFAULT_LAMBDA_BETA,
                'w_long': DEFAULT_W_LONG
            }
        },
        'results': all_results
    }

    summary_path = os.path.join(base_dir, 'sweep_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {base_dir}")
    print(f"Summary saved to: {summary_path}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)

    print("\nCONFLICT_ALPHA SWEEP:")
    print(f"{'Value':<15} {'Success Rate':<15} {'Reward':<15} {'p(slow)':<15} {'Lambda':<15}")
    print("-" * 75)
    for alpha, results in alpha_results.items():
        print(f"{alpha:<15.2f} {results['final_success_rate']:<15.2%} "
              f"{results['final_reward']:<15.2f} {results['final_p_slow']:<15.3f} "
              f"{results['final_lambda']:<15.3f}")

    print("\nLAMBDA_BETA SWEEP:")
    print(f"{'Value':<15} {'Success Rate':<15} {'Reward':<15} {'p(slow)':<15} {'Lambda':<15}")
    print("-" * 75)
    for beta, results in beta_results.items():
        print(f"{beta:<15.2f} {results['final_success_rate']:<15.2%} "
              f"{results['final_reward']:<15.2f} {results['final_p_slow']:<15.3f} "
              f"{results['final_lambda']:<15.3f}")

    print("\nW_LONG SWEEP:")
    print(f"{'Value':<15} {'Success Rate':<15} {'Reward':<15} {'p(slow)':<15} {'Lambda':<15}")
    print("-" * 75)
    for w_long, results in w_long_results.items():
        print(f"{w_long:<15.2f} {results['final_success_rate']:<15.2%} "
              f"{results['final_reward']:<15.2f} {results['final_p_slow']:<15.3f} "
              f"{results['final_lambda']:<15.3f}")

    print("\n✓ All sweeps complete!")
