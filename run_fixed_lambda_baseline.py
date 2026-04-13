"""
Fixed Lambda Baseline Experiment

Trains the primitive entropy-gated agent (FastNetwork + SlowMemory)
with fixed TD(λ) values to establish a baseline for comparison with
the dynamic lambda modulation in the full CognitiveAgent.

This script trains on 8x8 mazes with corridor={0.0, 0.5, 1.0} and
sweeps lambda from 0.0 to 1.0 in steps of 0.1.
"""

import torch
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from maze_env import MazeEnvironment
from fast import FastNetwork, FastNetworkTrainer
from slow import SlowMemory
from lambda_experiment.evaluation_metrics import (
    compute_auc, compute_episodes_to_threshold, compute_steps_to_threshold,
    compute_timeout_rate, compute_junction_decision_accuracy,
    compute_junction_action_entropy, compute_junction_policy_margin,
    compute_wrong_turn_rate
)
import networkx as nx
import torch.nn.functional as F


def run_single_configuration(corridor, lambda_val, seed, config):
    """
    Run training for a single (corridor, λ, seed) configuration.

    This is a standalone function to enable multiprocessing parallelization.

    Parameters:
    -----------
    corridor : float
        Corridor parameter for maze generation (0.0=junctions, 1.0=corridors)
    lambda_val : float
        TD(λ) parameter value (fixed for this run)
    seed : int
        Random seed
    config : dict
        Experiment configuration containing hyperparameters

    Returns:
    --------
    run_results : list of dict
        Episode-level results
    """
    # Prevent PyTorch from spawning multiple threads per process
    torch.set_num_threads(1)

    # Extract config parameters
    num_episodes = config['num_episodes']
    lr = config.get('lr', 3e-4)
    gamma = config.get('gamma', 0.99)
    entropy_coef = config.get('entropy_coef', 0.01)
    teacher_coef = config.get('teacher_coef', 10.0)
    value_coef = config.get('value_coef', 0.5)
    tau = config.get('tau', 1.0)
    consultation_temperature = config.get('consultation_temperature', 2.0)
    hard_teacher_force = config.get('hard_teacher_force', False)
    memory_consultation_cost = config.get('memory_consultation_cost', 0.0)
    memory_correction_cost = config.get('memory_correction_cost', 0.0)
    maze_size = config.get('maze_size', 8)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create environment
    env = MazeEnvironment(
        length=maze_size,
        width=maze_size,
        corridor=corridor,
        seed=seed,
        control_cost=0.0,  # No control cost for this experiment
        fixed_start_node=(0, 0),
        goal_is_deadend=True  # Goals at dead-ends
    )

    # Create network and trainer with FIXED lambda
    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        embedding_dim=64,
        hidden_dim=128,
        prospection_head=False  # No prospection head
    )

    trainer = FastNetworkTrainer(
        network=network,
        lr=lr,
        gamma=gamma,
        lambda_=lambda_val,  # FIXED lambda for this run
        entropy_coef=entropy_coef,
        teacher_coef=teacher_coef,
        value_coef=value_coef
    )

    # Create slow memory and initialize with optimal paths
    slow_memory = SlowMemory(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions
    )
    slow_memory.initialize_memory(env.maze)

    # Training loop
    run_results = []

    for episode in tqdm(range(num_episodes),
                       desc=f"corridor={corridor} λ={lambda_val:.1f} seed={seed}",
                       leave=False):
        state = env.reset()
        network.reset_hidden(batch_size=1)

        # Collect trajectory
        trajectory = {
            'states': [],
            'goals': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'log_probs': [],
            'values': [],
            'hiddens': [],
            'next_state': None,
            'next_goal': None,
            'node_sequence': [],  # For junction metrics
            'action_probs': [],   # For junction metrics
            'used_slow': [],      # For teacher forcing
            'policy_entropies': [],  # For tracking consultation behavior
            'consulted_memory': [],  # Whether memory was consulted
            'policy_corrected': [],  # Whether policy was corrected
        }

        episode_reward = 0.0
        episode_length = 0
        success = False
        max_steps = env.max_steps

        # Get shortest path for junction metrics
        start_node = state['current_pos']
        goal_node = state['goal_pos']
        graph = env.graph
        if nx.has_path(graph, start_node, goal_node):
            shortest_path = nx.shortest_path(graph, start_node, goal_node)
        else:
            shortest_path = []

        for step in range(max_steps):
            # Get state encoding
            state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
            goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

            # Forward pass
            action_logits, _, value, hidden = network(
                state_encoding, goal_encoding, network.hidden
            )

            # Compute policy entropy
            policy_entropy = network.compute_entropy(action_logits).item()

            # Compute action probabilities for junction metrics
            action_probs = F.softmax(action_logits, dim=-1).squeeze().detach().cpu().numpy()

            # Entropy-gated memory consultation
            # Probability of consulting memory: sigmoid((entropy - tau) / temperature)
            consultation_logit = (policy_entropy - tau) / consultation_temperature
            consultation_prob = torch.sigmoid(torch.tensor(consultation_logit)).item()
            consult_memory = np.random.random() < consultation_prob

            # Get optimal action from memory for potential use
            memory_action = slow_memory.query(state_encoding, goal_encoding)
            memory_action_idx = memory_action.argmax().item()

            # Decide on action and whether to use teacher forcing
            teacher_force_this_step = False
            policy_corrected = False
            memory_cost = 0.0  # Cost for memory intervention

            if consult_memory:
                # Memory consulted: use memory action and apply teacher forcing
                action_to_take = memory_action_idx
                teacher_force_this_step = True
                consulted = True
                memory_cost = memory_consultation_cost
            else:
                # Memory not consulted: sample from policy
                sampled_action, log_prob = network.sample_action(action_logits)
                sampled_action_idx = sampled_action.item()
                consulted = False

                if hard_teacher_force:
                    # Check if sampled action matches memory
                    if sampled_action_idx != memory_action_idx:
                        # Mismatch: correct to memory action and apply teacher forcing
                        action_to_take = memory_action_idx
                        teacher_force_this_step = True
                        policy_corrected = True
                        memory_cost = memory_correction_cost
                    else:
                        # Match: use sampled action, no teacher forcing
                        action_to_take = sampled_action_idx
                        teacher_force_this_step = False
                else:
                    # hard_teacher_force=False: always use sampled action
                    action_to_take = sampled_action_idx
                    teacher_force_this_step = False

            # Get log probability for the action we're storing in trajectory
            action_tensor = torch.tensor([action_to_take], dtype=torch.long)
            log_prob = network.get_log_prob(action_logits, action_tensor)

            # Store trajectory data
            trajectory['states'].append(torch.tensor(state['current_encoding']))
            trajectory['goals'].append(torch.tensor(state['goal_encoding']))
            trajectory['actions'].append(action_to_take)
            trajectory['log_probs'].append(log_prob)
            trajectory['values'].append(value.squeeze())
            trajectory['hiddens'].append(hidden)
            trajectory['node_sequence'].append(state['current_pos'])
            trajectory['action_probs'].append(action_probs)
            trajectory['used_slow'].append(teacher_force_this_step)
            trajectory['policy_entropies'].append(policy_entropy)
            trajectory['consulted_memory'].append(consulted)
            trajectory['policy_corrected'].append(policy_corrected)

            # Take step
            next_state, reward, done, info = env.step(action_to_take, used_slow=False)

            # Apply memory intervention cost to reward
            reward_with_cost = reward - memory_cost

            trajectory['rewards'].append(reward_with_cost)
            trajectory['dones'].append(done)

            episode_reward += reward_with_cost
            episode_length += 1

            if done:
                success = info['reached_goal']
                trajectory['next_state'] = torch.tensor(next_state['current_encoding'])
                trajectory['next_goal'] = torch.tensor(next_state['goal_encoding'])
                break

            state = next_state

        # If didn't finish, store final state
        if trajectory['next_state'] is None:
            trajectory['next_state'] = torch.tensor(state['current_encoding'])
            trajectory['next_goal'] = torch.tensor(state['goal_encoding'])

        # Train on trajectory
        loss_dict = trainer.train_step(trajectory)

        # Compute junction decision metrics
        junction_metrics = {}
        if len(shortest_path) > 0:
            junction_acc = compute_junction_decision_accuracy(
                trajectory, graph, shortest_path
            )
            junction_metrics.update(junction_acc)

            junction_entropy = compute_junction_action_entropy(
                trajectory, graph, trajectory['action_probs']
            )
            junction_metrics.update(junction_entropy)

            junction_margin = compute_junction_policy_margin(
                trajectory, graph, trajectory['action_probs']
            )
            junction_metrics.update(junction_margin)

            wrong_turn = compute_wrong_turn_rate(trajectory, graph, shortest_path)
            junction_metrics['wrong_turn_rate'] = wrong_turn

        # Compute consultation and correction statistics
        consultation_rate = np.mean(trajectory['consulted_memory']) if len(trajectory['consulted_memory']) > 0 else 0.0
        correction_rate = np.mean(trajectory['policy_corrected']) if len(trajectory['policy_corrected']) > 0 else 0.0
        mean_policy_entropy = np.mean(trajectory['policy_entropies']) if len(trajectory['policy_entropies']) > 0 else 0.0

        # Compute optimal path length
        optimal_path_length = len(shortest_path) - 1 if len(shortest_path) > 1 else 1

        # Store episode results
        episode_result = {
            # Configuration
            'topology': f'{corridor} corridor',  # Format as string for compatibility with evaluation metrics
            'corridor': corridor,
            'lambda': lambda_val,
            'seed': seed,
            'episode': episode,
            # Episode metrics
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            'optimal_path_length': optimal_path_length,
            'optimality_ratio': episode_length / optimal_path_length if optimal_path_length > 0 else -1,
            # Loss components
            'loss': loss_dict['loss'],
            'policy_loss': loss_dict['policy_loss'],
            'value_loss': loss_dict['value_loss'],
            'entropy_loss': loss_dict['entropy_loss'],
            'mean_entropy': loss_dict['mean_entropy'],
            'mean_value': loss_dict['mean_value'],
            # Policy metrics
            'consultation_rate': consultation_rate,
            'correction_rate': correction_rate,
            'mean_policy_entropy': mean_policy_entropy,
        }

        # Add junction decision metrics (if computed)
        if junction_metrics:
            episode_result.update({
                'num_junctions_visited': junction_metrics.get('num_junctions_visited', 0),
                'num_correct_junction_choices': junction_metrics.get('num_correct_junction_choices', 0),
                'junction_accuracy': junction_metrics.get('junction_accuracy', np.nan),
                'mean_junction_entropy': junction_metrics.get('mean_junction_entropy', np.nan),
                'mean_corridor_entropy': junction_metrics.get('mean_corridor_entropy', np.nan),
                'mean_junction_margin': junction_metrics.get('mean_junction_margin', np.nan),
                'mean_corridor_margin': junction_metrics.get('mean_corridor_margin', np.nan),
                'wrong_turn_rate': junction_metrics.get('wrong_turn_rate', np.nan),
            })

        run_results.append(episode_result)

    return run_results


class FixedLambdaBaselineExperiment:
    """Main experiment runner for fixed lambda baseline."""

    def __init__(self, config):
        """
        Initialize experiment.

        Parameters:
        -----------
        config : dict
            Experiment configuration with keys:
            - lambda_values: list of λ values to test
            - corridor_values: list of corridor parameters
            - seeds: list of random seeds
            - num_episodes: number of training episodes per run
            - maze_size: size of maze (default: 8)
            - lr: learning rate
            - gamma: discount factor
            - entropy_coef: entropy regularization coefficient
            - value_coef: value loss coefficient
            - tau: threshold entropy for memory consultation
            - consultation_temperature: temperature for sigmoid gating
            - hard_teacher_force: whether to correct wrong policy samples
            - memory_consultation_cost: reward penalty for consulting memory
            - memory_correction_cost: reward penalty for being corrected
            - output_dir: directory for saving results
        """
        self.config = config
        self.lambda_values = config['lambda_values']
        self.corridor_values = config['corridor_values']
        self.seeds = config['seeds']
        self.num_episodes = config['num_episodes']
        self.output_dir = config.get('output_dir', 'fixed_lambda_baseline_results')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Results storage
        self.results = []

    def run(self):
        """
        Run full experiment sweep using parallel processing.
        """
        print("=" * 80)
        print("Fixed Lambda Baseline Experiment")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Corridor values: {self.corridor_values}")
        print(f"  Lambda values: {self.lambda_values}")
        print(f"  Seeds: {self.seeds}")
        print(f"  Episodes per run: {self.num_episodes}")
        print(f"  Total runs: {len(self.corridor_values) * len(self.lambda_values) * len(self.seeds)}")
        print(f"  Available CPU cores: {cpu_count()}")
        print(f"  Output directory: {self.output_dir}")
        print()

        # Create list of all configurations to run in parallel
        configs_to_run = [
            (corridor, lambda_val, seed, self.config)
            for corridor in self.corridor_values
            for lambda_val in self.lambda_values
            for seed in self.seeds
        ]

        total_configs = len(configs_to_run)
        print(f"Running {total_configs} configurations in parallel...")
        print()

        # Run all configurations in parallel
        with Pool() as pool:
            all_results = pool.starmap(run_single_configuration, configs_to_run)

        # Flatten results (each config returns a list of episode results)
        for run_results in all_results:
            self.results.extend(run_results)

        # Save all results
        self.save_results()

        print("\n" + "=" * 80)
        print("Experiment complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)

        # Compute aggregate metrics
        print("\nComputing aggregate metrics...")
        self.compute_aggregate_metrics()
        print("Aggregate metrics saved!")

    def compute_aggregate_metrics(self):
        """
        Compute aggregate metrics across episodes (AUC, thresholds, etc.).
        """
        if len(self.results) == 0:
            print("  No results to process!")
            return

        df = pd.DataFrame(self.results)

        # Compute AUC for success rate
        print("  - Computing AUC...")
        auc_results = compute_auc(df, metric='success', x_axis='episode')
        auc_file = os.path.join(self.output_dir, 'auc_results.csv')
        auc_results.to_csv(auc_file, index=False)

        # Compute episodes to threshold
        print("  - Computing episodes to threshold...")
        episodes_threshold = compute_episodes_to_threshold(df)
        episodes_file = os.path.join(self.output_dir, 'episodes_to_threshold.csv')
        episodes_threshold.to_csv(episodes_file, index=False)

        # Compute steps to threshold
        print("  - Computing steps to threshold...")
        steps_threshold = compute_steps_to_threshold(df)
        steps_file = os.path.join(self.output_dir, 'steps_to_threshold.csv')
        steps_threshold.to_csv(steps_file, index=False)

        # Compute timeout rate
        print("  - Computing timeout rate...")
        timeout_results = compute_timeout_rate(df)
        timeout_file = os.path.join(self.output_dir, 'timeout_rate.csv')
        timeout_results.to_csv(timeout_file, index=False)

        # Compute summary statistics per condition
        print("  - Computing summary statistics...")
        summary = df.groupby(['corridor', 'lambda', 'seed']).agg({
            'success': 'mean',
            'episode_reward': 'mean',
            'episode_length': 'mean',
            'optimality_ratio': 'mean',
            'junction_accuracy': 'mean',
            'wrong_turn_rate': 'mean',
            'consultation_rate': 'mean',
            'correction_rate': 'mean',
            'mean_policy_entropy': 'mean',
        }).reset_index()

        summary_file = os.path.join(self.output_dir, 'summary_statistics.csv')
        summary.to_csv(summary_file, index=False)

    def save_results(self):
        """
        Save results to CSV file.
        """
        df = pd.DataFrame(self.results)
        output_file = os.path.join(self.output_dir, 'results.csv')
        df.to_csv(output_file, index=False)

        # Also save config
        config_file = os.path.join(self.output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)


def compute_metrics_from_directory(results_dir):
    """
    Compute aggregate metrics from a saved results directory.

    Parameters:
    -----------
    results_dir : str
        Path to results directory containing results.csv

    Example:
    --------
    >>> python run_fixed_lambda_baseline.py --compute-metrics fixed_lambda_baseline_results_20260413_052244
    """
    results_csv = os.path.join(results_dir, 'results.csv')

    if not os.path.exists(results_csv):
        print(f"Error: results.csv not found in {results_dir}")
        return

    print("=" * 80)
    print("Computing aggregate metrics from saved results")
    print("=" * 80)
    print(f"Loading results from: {results_csv}")

    df = pd.read_csv(results_csv)
    print(f"Loaded {len(df)} episodes")

    # Backward compatibility: add 'topology' column if it doesn't exist
    if 'topology' not in df.columns:
        if 'corridor' in df.columns:
            print("  Note: Adding 'topology' column for backward compatibility")
            df['topology'] = df['corridor'].apply(lambda x: f'{x} corridor')
        else:
            print("Error: Neither 'topology' nor 'corridor' column found in results")
            return

    # Create a minimal experiment object to use its compute_aggregate_metrics method
    dummy_config = {
        'lambda_values': [],
        'corridor_values': [],
        'seeds': [],
        'num_episodes': 0,
        'output_dir': results_dir
    }

    experiment = FixedLambdaBaselineExperiment(dummy_config)
    experiment.results = df.to_dict('records')

    print("\nComputing aggregate metrics...")
    experiment.compute_aggregate_metrics()

    print("\n" + "=" * 80)
    print("Metrics computation complete!")
    print(f"Results saved to: {results_dir}")
    print("=" * 80)


def main():
    """
    Main entry point for fixed lambda baseline experiment.

    Usage:
    ------
    # Run full experiment
    python run_fixed_lambda_baseline.py

    # Compute metrics from existing results
    python run_fixed_lambda_baseline.py --compute-metrics <results_directory>
    """
    import sys

    # Check if running in metrics-only mode
    if len(sys.argv) > 1 and sys.argv[1] == '--compute-metrics':
        if len(sys.argv) < 3:
            print("Usage: python run_fixed_lambda_baseline.py --compute-metrics <results_directory>")
            sys.exit(1)

        results_dir = sys.argv[2]
        compute_metrics_from_directory(results_dir)
        return

    # Experiment configuration
    config = {
        'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'corridor_values': [0.0, 0.5, 1.0],
        'seeds': [60],
        'num_episodes': 10000,
        'maze_size': 8,
        'lr': 3e-4,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'teacher_coef': 10.0,
        'value_coef': 0.5,
        'tau': 0.6,  # Threshold entropy for memory consultation
        'consultation_temperature': 0.5,  # Temperature for sigmoid gating
        'hard_teacher_force': False,  # Whether to correct wrong policy samples
        'memory_consultation_cost': 0.0,  # Reward penalty for consulting memory
        'memory_correction_cost': 0.0,  # Reward penalty for being corrected
        'output_dir': f'fixed_lambda_baseline_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Create and run experiment
    experiment = FixedLambdaBaselineExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
