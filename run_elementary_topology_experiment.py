"""
Elementary Topology Lambda Experiment

Tests how optimal TD(λ) depends on elementary maze topologies with
controlled structural properties (pure corridors, single branches,
intersections, corridor chains, and trees).

This complements run_lambda_experiment.py which uses procedurally
generated mazes. Elementary topologies provide interpretable anchor
cases for understanding topology-λ relationships.
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
from lambda_experiment.topology_generators import ALL_ELEMENTARY_TOPOLOGIES, generate_topology
from lambda_experiment.topology_metrics import compute_all_metrics
from lambda_experiment.credit_diagnostics import compute_all_credit_diagnostics
from lambda_experiment.evaluation_metrics import (
    compute_auc, compute_episodes_to_threshold, compute_steps_to_threshold,
    compute_timeout_rate, compute_junction_decision_accuracy,
    compute_junction_action_entropy, compute_junction_policy_margin,
    compute_wrong_turn_rate
)
import networkx as nx
import torch.nn.functional as F


def run_single_configuration(topology_name, lambda_val, seed, config, training_scheme=None):
    """
    Run training for a single (topology, λ, seed) configuration.

    This is a standalone function to enable multiprocessing parallelization.

    Parameters:
    -----------
    topology_name : str
        Name of topology from ALL_ELEMENTARY_TOPOLOGIES
    lambda_val : float
        TD(λ) parameter value
    seed : int
        Random seed
    config : dict
        Experiment configuration containing hyperparameters
    training_scheme : str, optional
        Training scheme to use for branch topologies:
        - 'branch_alternating': Alternate between two branch goals (like dead_end_goal)
        - 'branch_switch': Train on one branch, then switch to the other
        - None: Use default curated pairs

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
    hard_teacher_force = config.get('hard_teacher_force', True)
    memory_consultation_cost = config.get('memory_consultation_cost', 0.01)
    memory_correction_cost = config.get('memory_correction_cost', 0.01)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate topology
    topology_config = ALL_ELEMENTARY_TOPOLOGIES[topology_name]
    maze = generate_topology(topology_config, seed=seed)
    graph = maze.get_graph()

    # Compute topology metrics (path-independent)
    topo_metrics = compute_all_metrics(graph, maze=maze)

    # Create environment
    env = MazeEnvironment(
        length=maze.length,
        width=maze.width,
        corridor=maze.corridor,
        seed=seed,
        control_cost=0.0,  # No control cost for this experiment
        fixed_start_node=None,  # Will be set by curated pairs
        goal_is_deadend=False  # Will be set by curated pairs
    )
    env.maze = maze  # Use the generated maze
    env.graph = graph
    env.nodes_list = list(graph.nodes())
    env.node_to_idx = {node: idx for idx, node in enumerate(env.nodes_list)}
    env.idx_to_node = {idx: node for node, idx in env.node_to_idx.items()}
    env.num_nodes = len(env.nodes_list)
    env.deadend_nodes = [node for node in env.nodes_list if graph.degree(node) == 1]

    # Get curated start-goal pairs from the maze
    if hasattr(maze, 'start_goal_pairs') and maze.start_goal_pairs is not None:
        curated_pairs = maze.start_goal_pairs
    else:
        # Fallback: if no curated pairs, use all dead-end pairs
        curated_pairs = None

    # Override curated pairs for branch topology training schemes
    is_branch_topology = 'branch_pre' in topology_name
    if is_branch_topology and training_scheme is not None:
        # Extract branch parameters from topology name
        # e.g., 'branch_pre5_post5' -> pre=5, post=5
        parts = topology_name.split('_')
        pre_length = int(parts[1].replace('pre', ''))
        post_length = int(parts[2].replace('post', ''))

        # Start node is always the beginning of the corridor
        start_node = (0, 0)

        # Two branch ends
        branch1_goal = (post_length, pre_length - 1)   # Right branch
        branch2_goal = (-post_length, pre_length - 1)  # Left branch

        if training_scheme == 'branch_alternating':
            # Alternate between the two branch goals each episode
            curated_pairs = [(start_node, branch1_goal), (start_node, branch2_goal)]
        elif training_scheme == 'branch_switch':
            # We'll handle this in the training loop (single pair, but we'll switch midway)
            curated_pairs = [(start_node, branch1_goal)]  # Start with branch 1
            switch_episode = num_episodes // 2  # Store for later use
        else:
            # Keep default curated pairs
            pass

    # Create network and trainer
    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        embedding_dim=64,
        hidden_dim=128,
        prospection_head=False  # No prospection head for this experiment
    )

    trainer = FastNetworkTrainer(
        network=network,
        lr=lr,
        gamma=gamma,
        lambda_=lambda_val,
        entropy_coef=entropy_coef,
        teacher_coef=teacher_coef,
        value_coef=value_coef
    )

    # Create slow memory and initialize with optimal paths
    slow_memory = SlowMemory(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions
    )
    slow_memory.initialize_memory(maze)

    # Training loop
    run_results = []

    for episode in tqdm(range(num_episodes),
                       desc=f"{topology_name} λ={lambda_val:.1f} seed={seed}",
                       leave=False):
        # Handle branch_switch scheme: switch goal midway through training
        if is_branch_topology and training_scheme == 'branch_switch' and episode == switch_episode:
            # Switch to branch 2
            curated_pairs = [(start_node, branch2_goal)]

        # Reset environment with curated start-goal pairs
        if curated_pairs is not None:
            # Cycle through curated pairs
            pair_idx = episode % len(curated_pairs)
            start_pos, goal_pos = curated_pairs[pair_idx]
            state = env.reset(start_pos=start_pos, goal_pos=goal_pos)
        else:
            # Random sampling fallback
            state = env.reset()

        network.reset_hidden(batch_size=1)

        # Compute path-dependent topology metrics for this start-goal pair
        path_metrics = compute_all_metrics(
            graph,
            start=state['current_pos'],
            goal=state['goal_pos']
        )

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
            'node_sequence': [],  # For credit diagnostics
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
            state_idx = state_encoding.argmax().item()
            goal_idx = goal_encoding.argmax().item()
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
            # (for policy gradient, we need log prob of action_to_take)
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

        # Compute credit diagnostics (only on successful episodes for cleaner signal)
        credit_diag = {}
        if success and 'advantages' in loss_dict and 'td_errors' in loss_dict:
            advantages = loss_dict['advantages']
            td_errors = loss_dict['td_errors']
            node_sequence = trajectory['node_sequence']

            credit_diag = compute_all_credit_diagnostics(
                advantages=advantages,
                node_sequence=node_sequence,
                graph=graph,
                shortest_path=shortest_path,
                action_probs=trajectory['action_probs'],
                td_errors=td_errors
            )

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
        consulted_indices = [i for i, c in enumerate(trajectory['consulted_memory']) if c]
        mean_consultation_entropy = np.mean([trajectory['policy_entropies'][i] for i in consulted_indices]) if consulted_indices else 0.0

        # Store episode results
        episode_result = {
            # Configuration
            'topology': topology_name,
            'lambda': lambda_val,
            'seed': seed,
            'episode': episode,
            'training_scheme': training_scheme if training_scheme else 'default',
            # Episode metrics
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'success': success,
            'optimal_path_length': path_metrics['shortest_path_length'],
            'optimality_ratio': episode_length / path_metrics['shortest_path_length'] if path_metrics['shortest_path_length'] > 0 else -1,
            # Loss components
            'loss': loss_dict['loss'],
            'policy_loss': loss_dict['policy_loss'],
            'value_loss': loss_dict['value_loss'],
            'entropy_loss': loss_dict['entropy_loss'],
            'mean_entropy': loss_dict['mean_entropy'],
            'mean_value': loss_dict['mean_value'],
            # Memory consultation metrics
            'consultation_rate': consultation_rate,
            'correction_rate': correction_rate,
            'mean_policy_entropy': mean_policy_entropy,
            'mean_consultation_entropy': mean_consultation_entropy,
            # Topology metrics (global)
            'topo_num_dead_ends': topo_metrics['num_dead_ends'],
            'topo_num_corridors': topo_metrics['num_corridors'],
            'topo_num_junctions': topo_metrics['num_junctions'],
            'topo_frac_corridors': topo_metrics['frac_corridors'],
            'topo_frac_junctions': topo_metrics['frac_junctions'],
            'topo_mean_corridor_length': topo_metrics['mean_corridor_length'],
            'topo_junction_density': topo_metrics['junction_density'],
            'topo_mean_global_corr_dec_ratio': topo_metrics['mean_global_corr_dec_ratio'],
            'topo_median_global_corr_dec_ratio': topo_metrics['median_global_corr_dec_ratio'],
            'topo_std_global_corr_dec_ratio': topo_metrics['std_global_corr_dec_ratio'],
            'topo_spatial_homogeneity': topo_metrics.get('spatial_homogeneity', 1.0),
            'topo_spatial_heterogeneity': topo_metrics.get('spatial_heterogeneity', 0.0),
            # Path-specific metrics
            'path_num_corridor_nodes': path_metrics['num_corridor_nodes_on_path'],
            'path_num_junction_nodes': path_metrics['num_junction_nodes_on_path'],
            'path_corr_dec_ratio': path_metrics['corr_dec_ratio'],
        }

        # Add credit diagnostics (if computed)
        if credit_diag:
            episode_result.update({
                'effective_credit_distance': credit_diag.get('effective_credit_distance', np.nan),
                'C_corridor': credit_diag.get('C_corridor', np.nan),
                'C_junction': credit_diag.get('C_junction', np.nan),
                'C_dead_end': credit_diag.get('C_dead_end', np.nan),
                'junction_corridor_ratio': credit_diag.get('junction_corridor_ratio', np.nan),
                'dead_corridor_ratio': credit_diag.get('dead_corridor_ratio', np.nan),
                'mean_decision_localization': credit_diag.get('mean_decision_localization', np.nan),
                'num_junctions_analyzed': credit_diag.get('num_junctions_analyzed', 0),
                'mean_junction_action_gap': credit_diag.get('mean_junction_action_gap', np.nan),
                'mean_upstream_local_ratio': credit_diag.get('mean_upstream_local_ratio', np.nan),
            })

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


class ElementaryTopologyExperiment:
    """Main experiment runner for testing λ across elementary topologies."""

    def __init__(self, config):
        """
        Initialize experiment.

        Parameters:
        -----------
        config : dict
            Experiment configuration with keys:
            - lambda_values: list of λ values to test
            - seeds: list of random seeds
            - num_episodes: number of training episodes per run
            - topologies: list of topology names to test
            - lr: learning rate
            - gamma: discount factor
            - entropy_coef: entropy regularization coefficient
            - value_coef: value loss coefficient
            - tau: threshold entropy for memory consultation (default: 1.0)
            - consultation_temperature: temperature for sigmoid gating (default: 2.0)
            - hard_teacher_force: whether to correct wrong policy samples (default: True)
            - output_dir: directory for saving results
        """
        self.config = config
        self.lambda_values = config['lambda_values']
        self.seeds = config['seeds']
        self.num_episodes = config['num_episodes']
        self.topologies = config['topologies']
        self.lr = config.get('lr', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.teacher_coef = config.get('teacher_coef', 10.0)
        self.value_coef = config.get('value_coef', 0.5)
        self.tau = config.get('tau', 1.0)
        self.consultation_temperature = config.get('consultation_temperature', 2.0)
        self.hard_teacher_force = config.get('hard_teacher_force', True)
        self.memory_consultation_cost = config.get('memory_consultation_cost', 0.01)
        self.memory_correction_cost = config.get('memory_correction_cost', 0.01)
        self.output_dir = config.get('output_dir', 'elementary_topology_results')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Results storage (dict of lists, keyed by training_scheme)
        self.results_by_scheme = {}

    def run(self):
        """
        Run full experiment sweep over topologies, λ values, and seeds using parallel processing.
        """
        print("=" * 80)
        print("Elementary Topology Lambda Experiment")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Topologies: {len(self.topologies)} elementary topologies")
        print(f"  Lambda values: {self.lambda_values}")
        print(f"  Seeds: {self.seeds}")
        print(f"  Episodes per run: {self.num_episodes}")
        print(f"  Available CPU cores: {cpu_count()}")
        print(f"  Output directory: {self.output_dir}")
        print()

        # Create list of all configurations to run in parallel
        configs_to_run = []
        for topology_name in self.topologies:
            # Check if this is a branch topology
            is_branch_topology = 'branch_pre' in topology_name

            if is_branch_topology:
                # For branch topologies, run both training schemes
                for training_scheme in ['branch_alternating', 'branch_switch']:
                    for lambda_val in self.lambda_values:
                        for seed in self.seeds:
                            configs_to_run.append((topology_name, lambda_val, seed, self.config, training_scheme))
            else:
                # For non-branch topologies, use default scheme
                for lambda_val in self.lambda_values:
                    for seed in self.seeds:
                        configs_to_run.append((topology_name, lambda_val, seed, self.config, None))

        total_configs = len(configs_to_run)
        print(f"Total runs: {total_configs}")
        print(f"Running {total_configs} configurations in parallel...")
        print()

        # Run all configurations in parallel
        with Pool() as pool:
            all_results = pool.starmap(run_single_configuration, configs_to_run)

        # Organize results by training scheme
        for i, run_results in enumerate(all_results):
            # Get the training scheme from the config
            _, _, _, _, training_scheme = configs_to_run[i]
            scheme_key = training_scheme if training_scheme else 'default'

            if scheme_key not in self.results_by_scheme:
                self.results_by_scheme[scheme_key] = []

            self.results_by_scheme[scheme_key].extend(run_results)

        # Save results for each scheme separately
        self.save_results()

        print("\n" + "=" * 80)
        print("Experiment complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)

        # Compute aggregate metrics for each scheme
        print("\nComputing aggregate metrics...")
        self.compute_aggregate_metrics()
        print("Aggregate metrics saved!")

    def compute_aggregate_metrics(self, results_csv_path=None, output_dir=None):
        """
        Compute aggregate metrics across episodes (AUC, thresholds, etc.).

        Parameters:
        -----------
        results_csv_path : str, optional
            Path to saved results CSV. If provided, loads from file.
            If None, uses self.results_by_scheme from current experiment run.
        output_dir : str, optional
            Directory to save metrics. Required if results_csv_path is provided.
        """
        # Load data either from file or from current results
        if results_csv_path is not None:
            if output_dir is None:
                output_dir = os.path.dirname(results_csv_path)
            print(f"  Loading results from: {results_csv_path}")
            df = pd.read_csv(results_csv_path)
            print(f"  Loaded {len(df)} episodes")
        else:
            # Process each training scheme separately
            if len(self.results_by_scheme) == 0:
                print("  No results to process!")
                return

            for scheme_key, results in self.results_by_scheme.items():
                if len(results) == 0:
                    continue

                print(f"\n  Processing training scheme: {scheme_key}")
                df = pd.DataFrame(results)
                scheme_output_dir = os.path.join(self.output_dir, scheme_key)

                self._compute_metrics_for_dataframe(df, scheme_output_dir)
            return

        # If loading from file, compute for that dataframe
        self._compute_metrics_for_dataframe(df, output_dir)

    def _compute_metrics_for_dataframe(self, df, output_dir):
        """
        Helper to compute metrics for a single dataframe.

        Parameters:
        -----------
        df : pd.DataFrame
            Episode results dataframe
        output_dir : str
            Directory to save metrics
        """
        # Determine groupby columns for aggregate metrics
        base_group_cols = ['topology', 'lambda', 'seed']
        group_cols = base_group_cols

        # Compute AUC for success rate
        print("    - Computing AUC...")
        auc_results = compute_auc(df, metric='success', x_axis='episode')
        auc_file = os.path.join(output_dir, 'auc_results.csv')
        auc_results.to_csv(auc_file, index=False)

        # Compute episodes to threshold
        print("    - Computing episodes to threshold...")
        episodes_threshold = compute_episodes_to_threshold(df)
        episodes_file = os.path.join(output_dir, 'episodes_to_threshold.csv')
        episodes_threshold.to_csv(episodes_file, index=False)

        # Compute steps to threshold
        print("    - Computing steps to threshold...")
        steps_threshold = compute_steps_to_threshold(df)
        steps_file = os.path.join(output_dir, 'steps_to_threshold.csv')
        steps_threshold.to_csv(steps_file, index=False)

        # Compute timeout rate
        print("    - Computing timeout rate...")
        timeout_results = compute_timeout_rate(df)
        timeout_file = os.path.join(output_dir, 'timeout_rate.csv')
        timeout_results.to_csv(timeout_file, index=False)

        # Compute summary statistics per condition
        print("    - Computing summary statistics...")

        # Build aggregation dict only for columns that exist
        agg_dict = {}
        possible_metrics = {
            'success': 'mean',
            'episode_reward': 'mean',
            'episode_length': 'mean',
            'optimality_ratio': 'mean',
            'junction_accuracy': 'mean',
            'junction_corridor_ratio': 'mean',
            'effective_credit_distance': 'mean',
            'mean_decision_localization': 'mean',
            'wrong_turn_rate': 'mean',
            'consultation_rate': 'mean',
            'correction_rate': 'mean',
            'mean_policy_entropy': 'mean',
            'mean_consultation_entropy': 'mean',
        }

        # Only aggregate columns that exist in the dataframe
        for col, agg_func in possible_metrics.items():
            if col in df.columns:
                agg_dict[col] = agg_func

        if len(agg_dict) > 0:
            summary = df.groupby(group_cols).agg(agg_dict).reset_index()
            summary_file = os.path.join(output_dir, 'summary_statistics.csv')
            summary.to_csv(summary_file, index=False)
        else:
            print("    Warning: No metrics available for summary statistics")

    def save_results(self):
        """
        Save results to CSV files, organized by training scheme.
        """
        for scheme_key, results in self.results_by_scheme.items():
            if len(results) == 0:
                continue

            # Create subdirectory for this training scheme
            scheme_output_dir = os.path.join(self.output_dir, scheme_key)
            os.makedirs(scheme_output_dir, exist_ok=True)

            # Save results CSV
            df = pd.DataFrame(results)
            output_file = os.path.join(scheme_output_dir, 'results.csv')
            df.to_csv(output_file, index=False)

            # Save config
            config_file = os.path.join(scheme_output_dir, 'config.json')
            with open(config_file, 'w') as f:
                # Add training scheme to config
                config_with_scheme = self.config.copy()
                config_with_scheme['training_scheme'] = scheme_key
                json.dump(config_with_scheme, f, indent=2)

            print(f"  Saved {len(results)} episodes to {scheme_output_dir}/")


def main():
    """
    Main entry point for elementary topology lambda experiment.

    Command-line arguments:
    -----------------------
    --topology : str
        Name of specific topology to train (e.g., 'corridor_L10', 'branch_pre5_post5').
        If not provided, runs all topologies.
    --output_dir : str
        Base output directory. Results will be saved to:
        {output_dir}/{topology_name}/
        Default: 'elementary_topology_results'
    --lambdas : str
        Comma-separated lambda values (e.g., '0.0,0.5,1.0').
        Default: '0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0'
    --seeds : str
        Comma-separated random seeds (e.g., '60,61,62').
        Default: '60,61,62,63,64,65'
    --episodes : int
        Number of training episodes per run.
        Default: 3000
    --list : flag
        List all available topologies and exit.

    Examples:
    ---------
    # Train single topology
    python run_elementary_topology_experiment.py --topology corridor_L10

    # Train with custom output directory
    python run_elementary_topology_experiment.py --topology corridor_L10 --output_dir my_results

    # Train with specific lambda values
    python run_elementary_topology_experiment.py --topology branch_pre5_post5 --lambdas 0.0,0.5,1.0

    # List all available topologies
    python run_elementary_topology_experiment.py --list
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Run TD(λ) experiment on elementary maze topologies',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--topology', type=str, default=None,
                        help='Specific topology to train (e.g., corridor_L10). If not provided, runs all topologies.')
    parser.add_argument('--output_dir', type=str, default='elementary_topology_results',
                        help='Base output directory. Results saved to {output_dir}/{topology_name}/')
    parser.add_argument('--lambdas', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated lambda values')
    parser.add_argument('--seeds', type=str, default='60',
                        help='Comma-separated random seeds')
    parser.add_argument('--episodes', type=int, default=500,
                        help='Number of training episodes per run')
    parser.add_argument('--list', action='store_true',
                        help='List all available topologies and exit')

    args = parser.parse_args()

    # List topologies if requested
    if args.list:
        print("Available elementary topologies:")
        print("=" * 80)

        # Group by family
        families = {
            'corridor': 'Family A: Pure Corridors',
            'single_branch': 'Family B: Single Branch',
            'intersection': 'Family C: Intersections',
            'corridor_chain': 'Family D: Corridor Chains',
            'tree': 'Family E: Trees'
        }

        for family_key, family_name in families.items():
            print(f"\n{family_name}:")
            for topo_name, topo_config in ALL_ELEMENTARY_TOPOLOGIES.items():
                if topo_config['type'] == family_key:
                    print(f"  - {topo_name}")

        print(f"\nTotal: {len(ALL_ELEMENTARY_TOPOLOGIES)} topologies")
        return

    # Parse lambda values
    lambda_values = [float(x.strip()) for x in args.lambdas.split(',')]

    # Parse seeds
    seeds = [int(x.strip()) for x in args.seeds.split(',')]

    # Determine which topologies to run
    if args.topology:
        # Single topology
        if args.topology not in ALL_ELEMENTARY_TOPOLOGIES:
            print(f"Error: Topology '{args.topology}' not found.")
            print(f"Use --list to see available topologies.")
            return

        topologies = [args.topology]
        print(f"Running experiment for topology: {args.topology}")
    else:
        # All topologies
        topologies = list(ALL_ELEMENTARY_TOPOLOGIES.keys())
        print(f"Running experiment for all {len(topologies)} topologies")

    # Create base output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run experiment for each topology
    for topology_name in topologies:
        print("\n" + "=" * 80)
        print(f"Topology: {topology_name}")
        print("=" * 80)

        # Create topology-specific output directory
        topology_output_dir = os.path.join(args.output_dir, topology_name)
        os.makedirs(topology_output_dir, exist_ok=True)

        # Experiment configuration
        config = {
            'lambda_values': lambda_values,
            'seeds': seeds,
            'num_episodes': args.episodes,
            'topologies': [topology_name],  # Single topology per run
            'lr': 3e-4,
            'gamma': 0.99,
            'entropy_coef': 0.01,
            'teacher_coef': 10.0,
            'value_coef': 0.5,
            'tau': 0.6,  # Threshold entropy for memory consultation
            'consultation_temperature': 0.5,  # Temperature for sigmoid gating
            'hard_teacher_force': False,  # Whether to correct wrong policy samples
            'memory_consultation_cost': 0.0,  # Reward penalty for consulting memory
            'memory_correction_cost': 0.0,  # Reward penalty for being corrected by memory
            'output_dir': topology_output_dir,
        }

        # Create and run experiment
        experiment = ElementaryTopologyExperiment(config)
        experiment.run()

        print(f"\nResults saved to: {topology_output_dir}")


if __name__ == "__main__":
    main()
