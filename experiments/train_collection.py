"""
Training Collection Script

Trains agents across multiple corridor parameters (0-1) on 8x8 mazes.
Collects comprehensive statistics including node-type-specific metrics.

For each corridor parameter:
- Trains for 10,000 episodes (Stage 2 only, no pretraining)
- Saves checkpoints every 1,000 episodes
- Collects detailed per-episode statistics:
  1. Junction density
  2. Conflict map values (snapshot every 1,000 episodes)
  3. Fast/slow usage rate by node type
  4. Average entropy/divergence/conflict by system usage
  5. Average lambda by node type
  6. Average conflict map value by node type
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
from train import Stage2Trainer


def classify_node_type(graph, node):
    """
    Classify node by degree.

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph
    node : tuple
        Node position (row, col)

    Returns:
    --------
    node_type : str
        'dead_end' (degree 1), 'corridor' (degree 2), or 'junction' (degree >= 3)
    """
    degree = graph.degree(node)
    if degree == 1:
        return 'dead_end'
    elif degree == 2:
        return 'corridor'
    else:  # degree >= 3
        return 'junction'


def compute_junction_density(graph):
    """
    Compute junction density (fraction of nodes that are junctions).

    Parameters:
    -----------
    graph : nx.Graph
        Maze graph

    Returns:
    --------
    junction_density : float
        Fraction of nodes with degree >= 3
    """
    num_junctions = sum(1 for node in graph.nodes() if graph.degree(node) >= 3)
    return num_junctions / graph.number_of_nodes()


def compute_episode_statistics(trajectory, graph, agent, env):
    """
    Compute comprehensive per-episode statistics.

    Parameters:
    -----------
    trajectory : dict
        Episode trajectory from Stage2Trainer.collect_trajectory()
    graph : nx.Graph
        Maze graph
    agent : CognitiveAgent
        The cognitive agent
    env : MazeEnvironment
        The environment (for idx_to_node mapping)

    Returns:
    --------
    stats : dict
        Comprehensive episode statistics
    """
    step_infos = trajectory['step_info']

    # Get node sequence from state indices
    node_sequence = []
    for step_info in step_infos:
        state_idx = step_info['state_idx']
        node_sequence.append(env.idx_to_node[state_idx])

    # Initialize accumulators
    stats = {
        # By node type
        'junction': {'count': 0, 'used_fast': 0, 'used_slow': 0, 'lambdas': [], 'conflict_values': [],
                     'entropies': [], 'divergences': []},
        'corridor': {'count': 0, 'used_fast': 0, 'used_slow': 0, 'lambdas': [], 'conflict_values': [],
                     'entropies': [], 'divergences': []},
        'dead_end': {'count': 0, 'used_fast': 0, 'used_slow': 0, 'lambdas': [], 'conflict_values': [],
                     'entropies': [], 'divergences': []},

        # By system usage
        'used_fast': {'entropies': [], 'divergences': [], 'conflict_values': []},
        'used_slow': {'entropies': [], 'divergences': [], 'conflict_values': []},
    }

    # Process each step
    for i, (node, step_info) in enumerate(zip(node_sequence, step_infos)):
        node_type = classify_node_type(graph, node)

        # Compute lambda for this step
        lambda_val = agent.compute_lambda(step_info['conflict_value'], step_info['p_slow'])

        # Get conflict map value for this state
        conflict_value = step_info['conflict_value']

        # Accumulate by node type
        stats[node_type]['count'] += 1
        stats[node_type]['lambdas'].append(lambda_val)
        stats[node_type]['conflict_values'].append(conflict_value)
        stats[node_type]['entropies'].append(step_info['fast_entropy'])
        stats[node_type]['divergences'].append(step_info['kl_divergence'])

        if step_info['used_slow']:
            stats[node_type]['used_slow'] += 1
        else:
            stats[node_type]['used_fast'] += 1

        # Accumulate by system usage
        system_key = 'used_slow' if step_info['used_slow'] else 'used_fast'
        stats[system_key]['entropies'].append(step_info['fast_entropy'])
        stats[system_key]['divergences'].append(step_info['kl_divergence'])
        stats[system_key]['conflict_values'].append(conflict_value)

    # Compute averages
    result = {
        'junction_density': compute_junction_density(graph),
    }

    # Node type statistics
    for node_type in ['junction', 'corridor', 'dead_end']:
        node_stats = stats[node_type]
        total = node_stats['count']

        if total > 0:
            result[f'{node_type}_use_fast_rate'] = node_stats['used_fast'] / total
            result[f'{node_type}_use_slow_rate'] = node_stats['used_slow'] / total
            result[f'{node_type}_mean_lambda'] = np.mean(node_stats['lambdas'])
            result[f'{node_type}_mean_conflict'] = np.mean(node_stats['conflict_values'])
            result[f'{node_type}_mean_entropy'] = np.mean(node_stats['entropies'])
            result[f'{node_type}_mean_divergence'] = np.mean(node_stats['divergences'])
        else:
            result[f'{node_type}_use_fast_rate'] = np.nan
            result[f'{node_type}_use_slow_rate'] = np.nan
            result[f'{node_type}_mean_lambda'] = np.nan
            result[f'{node_type}_mean_conflict'] = np.nan
            result[f'{node_type}_mean_entropy'] = np.nan
            result[f'{node_type}_mean_divergence'] = np.nan

    # System usage statistics
    for system in ['used_fast', 'used_slow']:
        sys_stats = stats[system]

        if len(sys_stats['entropies']) > 0:
            result[f'{system}_mean_entropy'] = np.mean(sys_stats['entropies'])
            result[f'{system}_mean_divergence'] = np.mean(sys_stats['divergences'])
            result[f'{system}_mean_conflict'] = np.mean(sys_stats['conflict_values'])
        else:
            result[f'{system}_mean_entropy'] = np.nan
            result[f'{system}_mean_divergence'] = np.nan
            result[f'{system}_mean_conflict'] = np.nan

    return result


class ExtendedStage2Trainer(Stage2Trainer):
    """
    Extended Stage2Trainer with comprehensive statistics collection.
    """

    def __init__(self, env, agent, graph, lr_fast=3e-4, lr_controller=1e-3, gamma=0.99, lambda_init=0.95):
        """
        Initialize extended trainer.

        Parameters:
        -----------
        env : MazeEnvironment
            Training environment
        agent : CognitiveAgent
            The cognitive agent
        graph : nx.Graph
            Maze graph for node classification
        lr_fast : float
            Learning rate for fast network
        lr_controller : float
            Learning rate for controller
        gamma : float
            Discount factor
        lambda_init : float
            Initial lambda
        """
        super().__init__(env, agent, lr_fast, lr_controller, gamma, lambda_init)
        self.graph = graph

    def train(self, num_episodes=10000, log_interval=100, temperature_schedule=None,
              save_interval=1000, save_dir='results/collection'):
        """
        Train with extended statistics collection.

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        log_interval : int
            Logging interval
        temperature_schedule : callable, optional
            Function mapping episode -> temperature
        save_interval : int
            Save checkpoints and conflict maps every save_interval episodes
        save_dir : str
            Directory to save results

        Returns:
        --------
        metrics : dict
            Extended training metrics
        """
        # Standard Stage 2 metrics
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'p_slow': [],
            'p_fast': [],
            'mean_lambda': [],
            'lambda_values': [],
            'mean_fast_entropy': [],
            'mean_kl_divergence': [],
            'used_slow_count': [],
            'used_fast_count': [],
            'mean_delta': [],
            'losses': [],

            # Extended statistics (per-episode trajectories)
            'junction_density': [],

            # Node type statistics
            'junction_use_fast_rate': [],
            'junction_use_slow_rate': [],
            'junction_mean_lambda': [],
            'junction_mean_conflict': [],
            'junction_mean_entropy': [],
            'junction_mean_divergence': [],

            'corridor_use_fast_rate': [],
            'corridor_use_slow_rate': [],
            'corridor_mean_lambda': [],
            'corridor_mean_conflict': [],
            'corridor_mean_entropy': [],
            'corridor_mean_divergence': [],

            'dead_end_use_fast_rate': [],
            'dead_end_use_slow_rate': [],
            'dead_end_mean_lambda': [],
            'dead_end_mean_conflict': [],
            'dead_end_mean_entropy': [],
            'dead_end_mean_divergence': [],

            # System usage statistics
            'used_fast_mean_entropy': [],
            'used_fast_mean_divergence': [],
            'used_fast_mean_conflict': [],

            'used_slow_mean_entropy': [],
            'used_slow_mean_divergence': [],
            'used_slow_mean_conflict': [],
        }

        print("=" * 70)
        print("STAGE 2: TRAINING WITH EXTENDED STATISTICS COLLECTION")
        print("=" * 70)

        # Create save directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'conflict_maps'), exist_ok=True)

        for episode in tqdm(range(num_episodes), desc="Stage 2 Extended"):
            # Get temperature for this episode
            if temperature_schedule is not None:
                temperature = temperature_schedule(episode)
            else:
                temperature = 1.0

            # Collect trajectory
            trajectory, episode_reward, episode_length, success = self.collect_trajectory(
                temperature=temperature
            )

            # Train on trajectory
            loss_dict = self.train_step(trajectory)

            # Compute standard metrics
            step_infos = trajectory['step_info']
            mean_p_slow = np.mean([info['p_slow'] for info in step_infos])
            mean_p_fast = 1.0 - mean_p_slow

            lambdas = []
            for info in step_infos:
                lam = self.agent.compute_lambda(info['conflict_value'], info['p_slow'])
                lambdas.append(lam)
            mean_lambda = np.mean(lambdas)

            mean_fast_entropy = np.mean([info['fast_entropy'] for info in step_infos])
            mean_kl_divergence = np.mean([info['kl_divergence'] for info in step_infos])

            used_slow_count = sum([1 for info in step_infos if info['used_slow']])
            used_fast_count = len(step_infos) - used_slow_count

            mean_delta = np.mean([info['delta'].item() for info in step_infos])

            # Log standard metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_lengths'].append(episode_length)
            metrics['success_rate'].append(1.0 if success else 0.0)
            metrics['p_slow'].append(mean_p_slow)
            metrics['p_fast'].append(mean_p_fast)
            metrics['mean_lambda'].append(mean_lambda)
            metrics['lambda_values'].append(lambdas)
            metrics['mean_fast_entropy'].append(mean_fast_entropy)
            metrics['mean_kl_divergence'].append(mean_kl_divergence)
            metrics['used_slow_count'].append(used_slow_count)
            metrics['used_fast_count'].append(used_fast_count)
            metrics['mean_delta'].append(mean_delta)
            metrics['losses'].append(loss_dict)

            # Compute extended statistics
            extended_stats = compute_episode_statistics(trajectory, self.graph, self.agent, self.env)

            # Log extended metrics
            for key, value in extended_stats.items():
                if key in metrics:
                    metrics[key].append(value)

            # Periodic logging
            if (episode + 1) % log_interval == 0:
                recent_rewards = metrics['episode_rewards'][-log_interval:]
                recent_success = metrics['success_rate'][-log_interval:]
                recent_lengths = metrics['episode_lengths'][-log_interval:]
                recent_p_slow = metrics['p_slow'][-log_interval:]
                recent_lambda = metrics['mean_lambda'][-log_interval:]

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Mean reward: {np.mean(recent_rewards):.2f}")
                print(f"  Mean length: {np.mean(recent_lengths):.1f}")
                print(f"  Success rate: {np.mean(recent_success):.2%}")
                print(f"  Mean p(slow): {np.mean(recent_p_slow):.3f}")
                print(f"  Mean lambda: {np.mean(recent_lambda):.3f}")

            # Periodic checkpoint and conflict map saving
            if save_interval is not None and (episode + 1) % save_interval == 0:
                # Save agent checkpoint
                checkpoint_path = os.path.join(save_dir, 'checkpoints', f'agent_{episode + 1}.pt')
                self.agent.save(checkpoint_path)

                # Save conflict map snapshot
                conflict_map_path = os.path.join(save_dir, 'conflict_maps', f'conflict_map_{episode + 1}.npz')
                np.savez(
                    conflict_map_path,
                    conflict_values=self.agent.conflict_map.conflict_values,
                    episode=episode + 1
                )

                print(f"\n  Checkpoint saved: {checkpoint_path}")
                print(f"  Conflict map saved: {conflict_map_path}")

        print("\n" + "=" * 70)
        print("STAGE 2 EXTENDED COMPLETE")
        print("=" * 70)

        return metrics


def train_corridor_parameter(corridor, seed=60, num_episodes=10000, save_interval=1000,
                             base_dir='results/collection'):
    """
    Train agent for a single corridor parameter.

    This is a standalone function to enable multiprocessing parallelization.

    Parameters:
    -----------
    corridor : float
        Corridor parameter (0-1)
    seed : int
        Random seed
    num_episodes : int
        Number of training episodes
    save_interval : int
        Save checkpoints every save_interval episodes
    base_dir : str
        Base directory for saving results

    Returns:
    --------
    metrics : dict
        Training metrics
    """
    # Prevent PyTorch from spawning multiple threads per process
    torch.set_num_threads(1)
    # Environment configuration
    fixed_start_node = (7, 0)
    goal_is_deadend = True
    length = 8
    width = 8

    lr_fast = 3e-4
    lr_controller = 1e-3
    gamma = 0.99

    control_cost = 0.15
    conflict_alpha = 0.05
    lambda_beta = 2.0
    w_long = 0.8
    w_short = 1 - w_long

    # Create save directory
    save_dir = os.path.join(base_dir, f'corridor_{corridor:.1f}')
    os.makedirs(save_dir, exist_ok=True)

    # Create environment and agent
    maze = MazeGraph(length=length, width=width, corridor=corridor, seed=seed)
    env = MazeEnvironment(
        length=length, width=width, corridor=corridor, seed=seed,
        control_cost=control_cost,
        fixed_start_node=fixed_start_node,
        goal_is_deadend=goal_is_deadend
    )

    agent = CognitiveAgent(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        maze_graph=maze.get_graph(),
        control_cost=control_cost,
        conflict_alpha=conflict_alpha,
        lambda_beta=lambda_beta,
        w_long=w_long,
        w_short=w_short
    )

    # Create metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'environment': {
            'length': length,
            'width': width,
            'corridor': corridor,
            'seed': seed,
            'num_nodes': env.num_nodes,
            'num_actions': env.num_actions,
            'num_deadends': len(env.deadend_nodes),
            'fixed_start_node': list(fixed_start_node) if fixed_start_node else None,
            'goal_is_deadend': goal_is_deadend,
        },
        'training': {
            'num_episodes': num_episodes,
            'save_interval': save_interval,
            'lr_fast': lr_fast,
            'lr_controller': lr_controller,
            'gamma': gamma,
        },
        'agent': {
            'control_cost': control_cost,
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
    metadata_path = os.path.join(save_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Training corridor={corridor:.1f}")
    print(f"{'=' * 70}")
    print(f"Environment: {env.num_nodes} nodes, {env.num_actions} actions")
    print(f"Save directory: {save_dir}")

    # Train
    trainer = ExtendedStage2Trainer(
        env, agent, maze.get_graph(),
        lr_fast=lr_fast, lr_controller=lr_controller, gamma=gamma
    )

    metrics = trainer.train(
        num_episodes=num_episodes,
        log_interval=num_episodes // 100,
        save_interval=save_interval,
        save_dir=save_dir
    )

    # Save final agent
    agent.save(os.path.join(save_dir, 'checkpoints', 'agent_final.pt'))

    # Save metrics (convert lambda_values to regular lists for JSON serialization)
    metrics_serializable = {}
    for key, value in metrics.items():
        if key == 'lambda_values':
            # Convert nested lists of lambda values
            metrics_serializable[key] = [list(lam_list) for lam_list in value]
        elif key == 'losses':
            # Convert loss dictionaries
            metrics_serializable[key] = [
                {k: float(v['loss']) if isinstance(v, dict) and 'loss' in v else float(v)
                 for k, v in loss_dict.items()}
                for loss_dict in value
            ]
        else:
            # Convert numpy types to Python types
            if isinstance(value, list):
                metrics_serializable[key] = [float(x) if not np.isnan(x) else None for x in value]
            else:
                metrics_serializable[key] = value

    metrics_path = os.path.join(save_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Training complete for corridor={corridor:.1f}")

    return metrics


if __name__ == "__main__":
    print("=" * 70)
    print("TRAINING COLLECTION: 8x8 Mazes, Corridor Parameters 0-1")
    print("=" * 70)

    # Configuration
    corridor_params = np.arange(0, 1.1, 0.1)  # 0.0, 0.1, ..., 1.0
    seed = 60
    num_episodes = 10000
    save_interval = 1000
    base_dir = 'results/collection'

    print(f"\nConfiguration:")
    print(f"  Corridor parameters: {list(corridor_params)}")
    print(f"  Seed: {seed}")
    print(f"  Episodes per agent: {num_episodes}")
    print(f"  Save interval: {save_interval}")
    print(f"  Base directory: {base_dir}")
    print(f"  Available CPU cores: {cpu_count()}")
    print(f"\nTotal agents to train: {len(corridor_params)}")

    # Create list of all configurations to run in parallel
    configs_to_run = [
        (corridor, seed, num_episodes, save_interval, base_dir)
        for corridor in corridor_params
    ]

    print(f"\nRunning {len(configs_to_run)} corridor configurations in parallel...")
    print()

    # Train all corridor parameters in parallel
    with Pool() as pool:
        all_metrics_list = pool.starmap(train_corridor_parameter, configs_to_run)

    # Organize results by corridor parameter
    all_metrics = {}
    for i, corridor in enumerate(corridor_params):
        all_metrics[f'corridor_{corridor:.1f}'] = all_metrics_list[i]

    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {base_dir}")
    print("\nDirectory structure:")
    print(f"  {base_dir}/")
    print(f"    corridor_0.0/")
    print(f"      metadata.json")
    print(f"      metrics.json")
    print(f"      checkpoints/")
    print(f"      conflict_maps/")
    print(f"    ...")
    print(f"    corridor_1.0/")
    print("\n✓ Collection complete!")
