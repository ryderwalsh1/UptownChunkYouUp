"""
Adaptive Lambda Comparison: Training and Plotting

Trains the full CognitiveAgent with dynamic lambda modulation on the same
mazes as the fixed lambda baseline, then generates comparison plots.

Usage:
------
# Train and plot
python train_and_plot_adaptive_lambda_comparison.py

# Train only
python train_and_plot_adaptive_lambda_comparison.py --train-only

# Plot only (from existing results)
python train_and_plot_adaptive_lambda_comparison.py --plot-only \
    --results-dir adaptive_lambda_results_20260413_123456 \
    --baseline-dir fixed_lambda_baseline_results_20260413_052244
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import networkx as nx
import torch.nn.functional as F

from maze_env import MazeEnvironment
from agent import CognitiveAgent
from fast import FastNetworkTrainer
from controller import MetaControllerTrainer
from lambda_experiment.evaluation_metrics import (
    compute_junction_decision_accuracy,
    compute_junction_action_entropy,
    compute_junction_policy_margin,
    compute_wrong_turn_rate
)

# Publication-quality plotting defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


class AdaptiveLambdaTrainer:
    """Trainer for full cognitive agent with dynamic lambda modulation."""

    def __init__(self, env, agent, lr_fast=3e-4, lr_controller=1e-3, gamma=0.99):
        """
        Initialize trainer (no Stage 1 pretraining).

        Parameters:
        -----------
        env : MazeEnvironment
            Training environment
        agent : CognitiveAgent
            The cognitive agent
        lr_fast : float
            Learning rate for fast network
        lr_controller : float
            Learning rate for controller
        gamma : float
            Discount factor
        """
        self.env = env
        self.agent = agent
        self.gamma = gamma

        # Trainers for each component
        self.fast_trainer = FastNetworkTrainer(
            agent.fast_network,
            lr=lr_fast,
            gamma=gamma,
            lambda_=0.95,  # Will be modulated dynamically
            entropy_coef=0.01,
            value_coef=0.5,
            teacher_coef=10.0  # Match baseline (very important!)
        )
        self.controller_trainer = MetaControllerTrainer(
            agent.controller,
            lr=lr_controller,
            gamma=gamma
        )

    def collect_trajectory(self, max_steps=100, temperature=1.0):
        """
        Collect trajectory using full agent (fast + slow + controller).

        Returns:
        --------
        trajectory : dict
            Full trajectory data
        episode_info : dict
            Episode-level metrics
        """
        state = self.env.reset()
        self.agent.reset()

        # Trajectory storage
        trajectory = {
            # For fast network
            'fast_states': [],
            'fast_goals': [],
            'fast_actions': [],
            'fast_log_probs': [],
            'fast_values': [],
            'fast_rewards': [],
            'fast_dones': [],
            'fast_lambdas': [],

            # For controller
            'control_states': [],
            'control_fast_entropies': [],
            'control_kl_divergences': [],
            'control_conflict_values': [],
            'control_actions': [],
            'control_log_probs': [],
            'control_rewards': [],
            'control_dones': [],

            # For analysis
            'step_info': [],
            'node_sequence': [],
            'action_probs': [],
        }

        episode_reward = 0.0
        success = False
        graph = self.env.graph

        # Get shortest path for junction metrics
        start_node = state['current_pos']
        goal_node = state['goal_pos']
        if nx.has_path(graph, start_node, goal_node):
            shortest_path = nx.shortest_path(graph, start_node, goal_node)
        else:
            shortest_path = []

        for step in range(max_steps):
            # Get state encoding
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Full agent step
            step_info = self.agent.step(state_encoding, goal_encoding, temperature=temperature)

            # Compute action probabilities for junction metrics
            action_logits = step_info['fast_logits'] if not step_info['used_slow'] else step_info['slow_logits']
            action_probs = F.softmax(action_logits, dim=-1).squeeze().detach().cpu().numpy()

            # Take environment step
            next_state, reward, done, info = self.env.step(
                step_info['action'],
                used_slow=step_info['used_slow']
            )

            # Compute lambda for this step
            lambda_val = self.agent.compute_lambda(
                step_info['conflict_value'],
                step_info['p_slow']
            )

            # Update conflict map
            self.agent.update_conflict_map(step_info['state_idx'], step_info['kl_divergence'])

            # Store trajectory data for fast network
            trajectory['fast_states'].append(state_encoding.squeeze(0))
            trajectory['fast_goals'].append(goal_encoding.squeeze(0))
            trajectory['fast_actions'].append(step_info['action'])
            trajectory['fast_log_probs'].append(step_info['action_log_prob'])
            trajectory['fast_values'].append(step_info['fast_value'].squeeze())
            trajectory['fast_rewards'].append(reward)
            trajectory['fast_dones'].append(done)
            trajectory['fast_lambdas'].append(lambda_val)

            # Controller trajectory
            trajectory['control_states'].append(state_encoding.squeeze(0))
            trajectory['control_fast_entropies'].append(torch.tensor(step_info['fast_entropy']))
            trajectory['control_kl_divergences'].append(torch.tensor(step_info['kl_divergence']))
            trajectory['control_conflict_values'].append(torch.tensor(step_info['conflict_value']))
            trajectory['control_actions'].append(step_info['control_action'])
            trajectory['control_log_probs'].append(step_info['control_log_prob'])
            trajectory['control_rewards'].append(reward)
            trajectory['control_dones'].append(done)

            # Analysis data
            trajectory['step_info'].append(step_info)
            trajectory['node_sequence'].append(state['current_pos'])
            trajectory['action_probs'].append(action_probs)

            episode_reward += reward
            state = next_state

            if done:
                success = info['reached_goal']
                break

        # Store final state for bootstrapping
        final_state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32)
        final_goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32)
        trajectory['next_state'] = final_state_encoding
        trajectory['next_goal'] = final_goal_encoding

        # Compute episode metrics
        episode_length = len(trajectory['step_info'])
        optimal_path_length = len(shortest_path) - 1 if len(shortest_path) > 1 else 1

        # Junction metrics
        junction_metrics = {}
        if len(shortest_path) > 0:
            junction_acc = compute_junction_decision_accuracy(
                {'node_sequence': trajectory['node_sequence']}, graph, shortest_path
            )
            junction_metrics.update(junction_acc)

            junction_entropy = compute_junction_action_entropy(
                {'node_sequence': trajectory['node_sequence']}, graph, trajectory['action_probs']
            )
            junction_metrics.update(junction_entropy)

            junction_margin = compute_junction_policy_margin(
                {'node_sequence': trajectory['node_sequence']}, graph, trajectory['action_probs']
            )
            junction_metrics.update(junction_margin)

            wrong_turn = compute_wrong_turn_rate(
                {'node_sequence': trajectory['node_sequence']}, graph, shortest_path
            )
            junction_metrics['wrong_turn_rate'] = wrong_turn

        episode_info = {
            'episode_rewards': episode_reward,  # Use plural to match metrics dict
            'episode_lengths': episode_length,  # Use plural to match metrics dict
            'success_rate': 1.0 if success else 0.0,  # Use success_rate to match metrics dict
            'optimal_path_length': optimal_path_length,
            'optimality_ratio': episode_length / optimal_path_length if optimal_path_length > 0 else -1,
            'junction_accuracy': junction_metrics.get('junction_accuracy', np.nan),
            'mean_junction_entropy': junction_metrics.get('mean_junction_entropy', np.nan),
            'mean_junction_margin': junction_metrics.get('mean_junction_margin', np.nan),
            'wrong_turn_rate': junction_metrics.get('wrong_turn_rate', np.nan),
            'mean_p_slow': np.mean([info['p_slow'] for info in trajectory['step_info']]),
            'mean_lambda': np.mean(trajectory['fast_lambdas']),
            'mean_fast_entropy': np.mean([info['fast_entropy'] for info in trajectory['step_info']]),
            'mean_kl_divergence': np.mean([info['kl_divergence'] for info in trajectory['step_info']]),
            'used_slow_count': sum([1 for info in trajectory['step_info'] if info['used_slow']]),
        }

        return trajectory, episode_info

    def train_step(self, trajectory):
        """
        Train on collected trajectory.

        Parameters:
        -----------
        trajectory : dict
            Trajectory data

        Returns:
        --------
        loss_dict : dict
            Loss values for each component
        """
        loss_dict = {}

        # Train fast network with per-step lambda modulation
        fast_traj = {
            'states': trajectory['fast_states'],
            'goals': trajectory['fast_goals'],
            'actions': trajectory['fast_actions'],
            'log_probs': trajectory['fast_log_probs'],
            'values': trajectory['fast_values'],
            'rewards': trajectory['fast_rewards'],
            'dones': trajectory['fast_dones'],
            'lambdas': trajectory['fast_lambdas'],  # Dynamic lambda values
            'next_state': trajectory['next_state'],
            'next_goal': trajectory['next_goal']
        }
        fast_loss = self.fast_trainer.train_step(fast_traj)
        loss_dict['fast'] = fast_loss

        # Train controller
        control_traj = {
            'states': trajectory['control_states'],
            'fast_entropies': trajectory['control_fast_entropies'],
            'kl_divergences': trajectory['control_kl_divergences'],
            'conflict_values': trajectory['control_conflict_values'],
            'control_actions': trajectory['control_actions'],
            'control_log_probs': trajectory['control_log_probs'],
            'rewards': trajectory['control_rewards'],
            'dones': trajectory['control_dones']
        }
        controller_loss = self.controller_trainer.train_step(control_traj)
        loss_dict['controller'] = controller_loss

        return loss_dict

    def train(self, num_episodes=10000, log_interval=100, temperature=1.0):
        """
        Train full agent.

        Parameters:
        -----------
        num_episodes : int
            Number of training episodes
        log_interval : int
            Logging interval
        temperature : float
            Temperature for control policy sampling

        Returns:
        --------
        metrics : dict
            Training metrics
        """
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'optimality_ratio': [],
            'junction_accuracy': [],
            'mean_junction_entropy': [],
            'mean_junction_margin': [],
            'wrong_turn_rate': [],
            'mean_p_slow': [],
            'mean_lambda': [],
            'mean_fast_entropy': [],
            'mean_kl_divergence': [],
            'used_slow_count': [],
        }

        for episode in tqdm(range(num_episodes), desc="Training adaptive lambda agent"):
            # Collect trajectory
            trajectory, episode_info = self.collect_trajectory(temperature=temperature)

            # Train on trajectory
            loss_dict = self.train_step(trajectory)

            # Log metrics
            for key, value in episode_info.items():
                if key in metrics:
                    metrics[key].append(value)

            # Periodic logging
            if (episode + 1) % log_interval == 0:
                recent_rewards = metrics['episode_rewards'][-log_interval:]
                recent_success = metrics['success_rate'][-log_interval:]
                recent_p_slow = metrics['mean_p_slow'][-log_interval:]
                recent_lambda = metrics['mean_lambda'][-log_interval:]

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Mean reward: {np.mean(recent_rewards):.2f}")
                print(f"  Success rate: {np.mean(recent_success):.2%}")
                print(f"  Mean p(slow): {np.mean(recent_p_slow):.3f}")
                print(f"  Mean lambda: {np.mean(recent_lambda):.3f}")

        return metrics


def train_adaptive_lambda_comparison(corridor_values=[0.0, 0.5, 1.0], seed=60,
                                     num_episodes=10000, maze_size=8,
                                     output_dir=None):
    """
    Train adaptive lambda agent on specified mazes.

    Parameters:
    -----------
    corridor_values : list of float
        Corridor parameters for maze generation
    seed : int
        Random seed
    num_episodes : int
        Number of training episodes
    maze_size : int
        Size of maze
    output_dir : str, optional
        Output directory for results

    Returns:
    --------
    results_dir : Path
        Path to results directory
    """
    if output_dir is None:
        output_dir = f'adaptive_lambda_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    results_dir = Path(output_dir)
    results_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 80)
    print("Adaptive Lambda Training")
    print("=" * 80)
    print(f"Corridor values: {corridor_values}")
    print(f"Seed: {seed}")
    print(f"Episodes: {num_episodes}")
    print(f"Maze size: {maze_size}×{maze_size}")
    print(f"Output directory: {results_dir}")
    print()

    for corridor in corridor_values:
        print(f"\n{'=' * 80}")
        print(f"Training corridor={corridor}")
        print('=' * 80)

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create environment
        env = MazeEnvironment(
            length=maze_size,
            width=maze_size,
            corridor=corridor,
            seed=seed,
            control_cost=0.2,  # Control cost for using slow processing
            fixed_start_node=(7, 0),
            goal_is_deadend=True
        )

        # Create cognitive agent with dynamic lambda modulation
        agent = CognitiveAgent(
            num_nodes=env.num_nodes,
            num_actions=env.num_actions,
            maze_graph=env.graph,
            embedding_dim=64,
            hidden_dim=128,
            conflict_alpha=0.05,
            lambda_beta=2.0,
            w_long=0.8,
            w_short=0.2,
            control_cost=0.2,
            fixed_lambda=None  # Enable dynamic lambda modulation
        )

        # Create trainer
        trainer = AdaptiveLambdaTrainer(
            env=env,
            agent=agent,
            lr_fast=3e-4,
            lr_controller=1e-3,
            gamma=0.99
        )

        # Train
        metrics = trainer.train(
            num_episodes=num_episodes,
            log_interval=100,
            temperature=1.0
        )

        # Save results
        corridor_dir = results_dir / f'corridor_{corridor:.1f}'
        corridor_dir.mkdir(exist_ok=True)

        metrics_file = corridor_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            metrics_serializable = {}
            for key, value in metrics.items():
                if isinstance(value, list):
                    metrics_serializable[key] = [float(v) if not np.isnan(v) else None for v in value]
                else:
                    metrics_serializable[key] = value
            json.dump(metrics_serializable, f, indent=2)

        print(f"\nMetrics saved to: {metrics_file}")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)

    return results_dir


class ComparisonPlotter:
    """Load and plot comparison between fixed and adaptive lambda."""

    def __init__(self, baseline_dir, adaptive_dir, output_dir='comparison_plots',
                 smooth_window=200):
        """
        Initialize plotter.

        Parameters:
        -----------
        baseline_dir : str or Path
            Directory with fixed lambda baseline results
        adaptive_dir : str or Path
            Directory with adaptive lambda results
        output_dir : str or Path
            Directory to save plots
        smooth_window : int
            Window size for smoothing curves
        """
        self.baseline_dir = Path(baseline_dir)
        self.adaptive_dir = Path(adaptive_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.smooth_window = smooth_window

        # Load data
        print(f"Loading baseline from: {self.baseline_dir}")
        self.baseline_df = pd.read_csv(self.baseline_dir / 'results.csv')

        print(f"Loading adaptive from: {self.adaptive_dir}")
        self.adaptive_metrics = {}
        for corridor in [0.0, 0.5, 1.0]:
            metrics_file = self.adaptive_dir / f'corridor_{corridor:.1f}' / 'metrics.json'
            with open(metrics_file, 'r') as f:
                self.adaptive_metrics[corridor] = json.load(f)

        print(f"Output directory: {self.output_dir}")

    def _get_adaptive_metric(self, corridor, metric_name):
        """
        Get adaptive metric with backward compatibility.

        Tries multiple possible key names to handle old and new formats.

        Parameters:
        -----------
        corridor : float
            Corridor value
        metric_name : str
            Metric name to retrieve (e.g., 'episode_lengths', 'success_rate')

        Returns:
        --------
        list or None
            Metric values if found, None otherwise
        """
        # Define alternative key mappings for backward compatibility
        key_alternatives = {
            'episode_lengths': ['episode_lengths', 'episode_length'],
            'success_rate': ['success_rate', 'success'],
            'episode_rewards': ['episode_rewards', 'episode_reward'],
        }

        # Get list of possible keys for this metric
        possible_keys = key_alternatives.get(metric_name, [metric_name])

        # Try each possible key
        for key in possible_keys:
            if key in self.adaptive_metrics[corridor]:
                data = self.adaptive_metrics[corridor][key]
                # If it's 'success' (boolean), convert to rate
                if key == 'success' and isinstance(data, list):
                    return [1.0 if x else 0.0 for x in data]
                return data

        return None

    def _smooth(self, data, window=None):
        """Apply uniform smoothing to 1D data."""
        if window is None:
            window = self.smooth_window

        if len(data) < window:
            return np.array(data)

        data_array = np.array(data, dtype=float)
        # Replace None with NaN
        data_array[data_array == None] = np.nan

        # Forward fill NaNs
        mask = np.isnan(data_array)
        if np.any(mask):
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) == 0:
                return data_array
            for i in range(len(data_array)):
                if mask[i]:
                    prev_valid = valid_indices[valid_indices < i]
                    if len(prev_valid) > 0:
                        data_array[i] = data_array[prev_valid[-1]]

        # Convolve with uniform kernel
        kernel = np.ones(window) / window
        smoothed = np.convolve(data_array, kernel, mode='valid')

        return smoothed

    def plot_reward_comparison(self):
        """Plot episode reward comparison across mazes."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        lambda_values = np.arange(0, 1.1, 0.1)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambda_values)))

        for idx, corridor in enumerate([0.0, 0.5, 1.0]):
            ax = axes[idx]

            # Plot fixed lambda curves
            for i, lam in enumerate(lambda_values):
                subset = self.baseline_df[
                    (self.baseline_df['corridor'] == corridor) &
                    (self.baseline_df['lambda'] == lam)
                ].sort_values('episode')

                if len(subset) > 0:
                    rewards = subset['episode_reward'].values
                    smoothed = self._smooth(rewards)
                    episodes = np.arange(len(smoothed))
                    ax.plot(episodes, smoothed, color=colors[i], alpha=0.6,
                           linewidth=1.5, label=f'λ={lam:.1f}')

            # Plot adaptive lambda curve
            adaptive_rewards = self._get_adaptive_metric(corridor, 'episode_rewards')
            if adaptive_rewards is not None:
                smoothed = self._smooth(adaptive_rewards)
                episodes = np.arange(len(smoothed))
                ax.plot(episodes, smoothed, color='black', linewidth=2.5,
                       label='Adaptive λ', marker='o', markevery=len(episodes)//10,
                       markersize=4)

            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Reward')
            ax.set_title(f'Corridor = {corridor}')
            ax.grid(alpha=0.3)
            if idx == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'reward_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'reward_comparison.pdf', bbox_inches='tight')
        print(f"Saved: reward_comparison.png")
        plt.close()

    def plot_performance_metrics(self):
        """Plot performance metrics grid (episode_length, optimality_ratio, success_rate)."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        lambda_values = np.arange(0, 1.1, 0.1)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambda_values)))

        # Baseline uses singular keys, adaptive uses plural keys
        baseline_metrics = ['episode_length', 'optimality_ratio', 'success']
        adaptive_metrics = ['episode_lengths', 'optimality_ratio', 'success_rate']
        metric_labels = ['Episode Length', 'Optimality Ratio', 'Success Rate']

        for row_idx, (baseline_metric, adaptive_metric, label) in enumerate(
                zip(baseline_metrics, adaptive_metrics, metric_labels)):
            for col_idx, corridor in enumerate([0.0, 0.5, 1.0]):
                ax = axes[row_idx, col_idx]

                # Plot fixed lambda curves
                for i, lam in enumerate(lambda_values):
                    subset = self.baseline_df[
                        (self.baseline_df['corridor'] == corridor) &
                        (self.baseline_df['lambda'] == lam)
                    ].sort_values('episode')

                    if len(subset) > 0:
                        values = subset[baseline_metric].values
                        smoothed = self._smooth(values)
                        episodes = np.arange(len(smoothed))
                        ax.plot(episodes, smoothed, color=colors[i], alpha=0.6,
                               linewidth=1.5)

                # Plot adaptive lambda curve
                adaptive_values = self._get_adaptive_metric(corridor, adaptive_metric)
                if adaptive_values is not None:
                    smoothed = self._smooth(adaptive_values)
                    episodes = np.arange(len(smoothed))
                    ax.plot(episodes, smoothed, color='black', linewidth=2.5,
                           marker='o', markevery=len(episodes)//10, markersize=4)

                ax.set_xlabel('Episode')
                ax.set_ylabel(label)
                if row_idx == 0:
                    ax.set_title(f'Corridor = {corridor}')
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'performance_metrics.pdf', bbox_inches='tight')
        print(f"Saved: performance_metrics.png")
        plt.close()

    def plot_junction_metrics(self):
        """Plot junction metrics grid."""
        fig, axes = plt.subplots(4, 3, figsize=(18, 18))
        lambda_values = np.arange(0, 1.1, 0.1)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambda_values)))

        metrics = ['junction_accuracy', 'mean_junction_entropy',
                  'mean_junction_margin', 'wrong_turn_rate']
        metric_labels = ['Junction Accuracy', 'Junction Entropy',
                        'Junction Margin', 'Wrong Turn Rate']

        for row_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            for col_idx, corridor in enumerate([0.0, 0.5, 1.0]):
                ax = axes[row_idx, col_idx]

                # Plot fixed lambda curves
                for i, lam in enumerate(lambda_values):
                    subset = self.baseline_df[
                        (self.baseline_df['corridor'] == corridor) &
                        (self.baseline_df['lambda'] == lam)
                    ].sort_values('episode')

                    if len(subset) > 0 and metric in subset.columns:
                        values = subset[metric].values
                        smoothed = self._smooth(values)
                        episodes = np.arange(len(smoothed))
                        ax.plot(episodes, smoothed, color=colors[i], alpha=0.6,
                               linewidth=1.5)

                # Plot adaptive lambda curve
                adaptive_values = self._get_adaptive_metric(corridor, metric)
                if adaptive_values is not None:
                    smoothed = self._smooth(adaptive_values)
                    episodes = np.arange(len(smoothed))
                    ax.plot(episodes, smoothed, color='black', linewidth=2.5,
                           marker='o', markevery=len(episodes)//10, markersize=4)

                ax.set_xlabel('Episode')
                ax.set_ylabel(label)
                if row_idx == 0:
                    ax.set_title(f'Corridor = {corridor}')
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'junction_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'junction_metrics.pdf', bbox_inches='tight')
        print(f"Saved: junction_metrics.png")
        plt.close()

    def plot_policy_metrics(self):
        """Plot policy metrics grid."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        lambda_values = np.arange(0, 1.1, 0.1)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(lambda_values)))

        # For baseline: consultation_rate, mean_policy_entropy, correction_rate
        # For adaptive: mean_p_slow, mean_fast_entropy, (no correction)
        baseline_metrics = ['consultation_rate', 'mean_policy_entropy', 'correction_rate']
        adaptive_metrics_keys = ['mean_p_slow', 'mean_fast_entropy', 'correction_rate']
        metric_labels = ['Memory Usage (Consultation/p_slow)', 'Policy Entropy', 'Correction Rate']

        for row_idx, (baseline_metric, adaptive_metric, label) in enumerate(
                zip(baseline_metrics, adaptive_metrics_keys, metric_labels)):
            for col_idx, corridor in enumerate([0.0, 0.5, 1.0]):
                ax = axes[row_idx, col_idx]

                # Plot fixed lambda curves
                for i, lam in enumerate(lambda_values):
                    subset = self.baseline_df[
                        (self.baseline_df['corridor'] == corridor) &
                        (self.baseline_df['lambda'] == lam)
                    ].sort_values('episode')

                    if len(subset) > 0 and baseline_metric in subset.columns:
                        values = subset[baseline_metric].values
                        smoothed = self._smooth(values)
                        episodes = np.arange(len(smoothed))
                        ax.plot(episodes, smoothed, color=colors[i], alpha=0.6,
                               linewidth=1.5)

                # Plot adaptive lambda curve (use appropriate metric)
                adaptive_values = self._get_adaptive_metric(corridor, adaptive_metric)
                if adaptive_values is not None:
                    smoothed = self._smooth(adaptive_values)
                    episodes = np.arange(len(smoothed))
                    ax.plot(episodes, smoothed, color='black', linewidth=2.5,
                           marker='o', markevery=len(episodes)//10, markersize=4)

                ax.set_xlabel('Episode')
                ax.set_ylabel(label)
                if row_idx == 0:
                    ax.set_title(f'Corridor = {corridor}')
                ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'policy_metrics.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'policy_metrics.pdf', bbox_inches='tight')
        print(f"Saved: policy_metrics.png")
        plt.close()

    def plot_all(self):
        """Generate all comparison plots."""
        print("\n" + "=" * 80)
        print("Generating comparison plots")
        print("=" * 80)

        self.plot_reward_comparison()
        self.plot_performance_metrics()
        self.plot_junction_metrics()
        self.plot_policy_metrics()

        print("\n" + "=" * 80)
        print("Plotting complete!")
        print(f"Plots saved to: {self.output_dir}")
        print("=" * 80)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train adaptive lambda agent and plot comparison with fixed lambda baseline'
    )
    parser.add_argument('--train-only', action='store_true',
                       help='Only run training (skip plotting)')
    parser.add_argument('--plot-only', action='store_true',
                       help='Only generate plots (skip training)')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Adaptive lambda results directory (for --plot-only)')
    parser.add_argument('--baseline-dir', type=str, default=None,
                       help='Fixed lambda baseline directory (for --plot-only)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for training results')

    args = parser.parse_args()

    if args.plot_only:
        # Plot only mode
        if args.results_dir is None or args.baseline_dir is None:
            print("Error: --results-dir and --baseline-dir required for --plot-only")
            return

        plotter = ComparisonPlotter(
            baseline_dir=args.baseline_dir,
            adaptive_dir=args.results_dir,
            output_dir='comparison_plots'
        )
        plotter.plot_all()

    elif args.train_only:
        # Train only mode
        results_dir = train_adaptive_lambda_comparison(
            corridor_values=[0.0, 0.5, 1.0],
            seed=60,
            num_episodes=10000,
            maze_size=8,
            output_dir=args.output_dir
        )
        print(f"\nTraining complete. To plot, run:")
        print(f"python train_and_plot_adaptive_lambda_comparison.py --plot-only \\")
        print(f"  --results-dir {results_dir} \\")
        print(f"  --baseline-dir <fixed_lambda_baseline_dir>")

    else:
        # Train and plot
        results_dir = train_adaptive_lambda_comparison(
            corridor_values=[0.0, 0.5, 1.0],
            seed=60,
            num_episodes=10000,
            maze_size=8,
            output_dir=args.output_dir
        )

        if args.baseline_dir is None:
            print("\nError: --baseline-dir required for plotting")
            print("Please provide the fixed lambda baseline directory:")
            print(f"python train_and_plot_adaptive_lambda_comparison.py --plot-only \\")
            print(f"  --results-dir {results_dir} \\")
            print(f"  --baseline-dir <fixed_lambda_baseline_dir>")
        else:
            plotter = ComparisonPlotter(
                baseline_dir=args.baseline_dir,
                adaptive_dir=results_dir,
                output_dir='comparison_plots'
            )
            plotter.plot_all()


if __name__ == "__main__":
    main()
