"""
Lambda Experiment: Topology-Dependent Optimal TD(λ)

Investigates how the optimal trace parameter λ depends on maze topology
for a goal-conditioned recurrent agent trained with actor-critic TD(λ).

Central hypothesis: The best λ increases with temporal separation between
consequential decisions, and decreases with branching ambiguity.
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
from lambda_experiment.topology_generators import ALL_TOPOLOGIES, generate_topology
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


def run_single_configuration(topology_name, lambda_val, seed, config):
    """
    Run training for a single (topology, λ, seed) configuration.

    This is a standalone function to enable multiprocessing parallelization.

    Parameters:
    -----------
    topology_name : str
        Name of topology from ALL_TOPOLOGIES
    lambda_val : float
        TD(λ) parameter value
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
    hard_teacher_force = config.get('hard_teacher_force', True)
    memory_consultation_cost = config.get('memory_consultation_cost', 0.01)
    memory_correction_cost = config.get('memory_correction_cost', 0.01)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate topology
    topology_config = ALL_TOPOLOGIES[topology_name]
    maze = generate_topology(topology_config, seed=seed)
    graph = maze.get_graph()

    # Compute topology metrics (path-independent)
    topo_metrics = compute_all_metrics(graph)

    # Create environment
    env = MazeEnvironment(
        length=maze.length,
        width=maze.width,
        corridor=maze.corridor,
        seed=seed,
        control_cost=0.0,  # No control cost for this experiment
        fixed_start_node=(0,0),  # Random start
        goal_is_deadend=True  # Goals at dead-ends
    )
    env.maze = maze  # Use the generated maze
    env.graph = graph
    env.nodes_list = list(graph.nodes())
    env.node_to_idx = {node: idx for idx, node in enumerate(env.nodes_list)}
    env.idx_to_node = {idx: node for node, idx in env.node_to_idx.items()}
    env.num_nodes = len(env.nodes_list)
    env.deadend_nodes = [node for node in env.nodes_list if graph.degree(node) == 1]

    # Create network and trainer
    network = FastNetwork(
        num_nodes=env.num_nodes,
        num_actions=env.num_actions,
        embedding_dim=64,
        hidden_dim=128,
        prospection_head=False  # No prospection head for this experiment
    )

    # Determine learning rate (adaptive or fixed)
    if lr == 'adaptive':
        # Use 2D interpolation based on lambda and junction density
        from run_lambda_experiment import LambdaExperiment
        junction_density = topo_metrics['junction_density']
        learning_rate = LambdaExperiment.get_optimal_learning_rate_2d(lambda_val, junction_density)
    else:
        learning_rate = lr

    trainer = FastNetworkTrainer(
        network=network,
        lr=learning_rate,
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
        # Reset environment
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


class LambdaExperiment:
    """Main experiment runner for testing λ across topologies."""

    # Optimal hyperparameters from small grid sweep (see plot_smallgridsweep.py)
    # Optimized for: 0.7 × (AUC × memory_independence) + 0.3 × stability
    OPTIMAL_HYPERPARAMETERS = {
        0.0: {'lr': 6e-4, 'teacher_coef': 10.0, 'tau': 0.4},
        0.2: {'lr': 6e-4, 'teacher_coef': 6.5, 'tau': 0.6},
        0.4: {'lr': 6e-4, 'teacher_coef': 10.0, 'tau': 0.6},
        0.6: {'lr': 6e-4, 'teacher_coef': 3.0, 'tau': 0.6},
        0.8: {'lr': 6e-4, 'teacher_coef': 6.5, 'tau': 0.6},
        1.0: {'lr': 6e-4, 'teacher_coef': 3.0, 'tau': 0.4},
    }

    # 2D optimal hyperparameters: {junction_density: {lambda: {lr}}}
    # Junction density = fraction of nodes with degree >= 3
    # Only learning rate varies with topology; teacher_coef and tau are fixed config parameters
    OPTIMAL_HYPERPARAMETERS_2D = {
        0.3: {  # Higher junction density (more branching)
            0.0: {'lr': 3e-4},
            0.2: {'lr': 3e-4},
            0.4: {'lr': 6e-4},
            0.6: {'lr': 6e-4},
            0.8: {'lr': 6e-4},
            1.0: {'lr': 6e-4},
        },
        0.1: {  # Lower junction density (fewer branching points)
            0.0: {'lr': 1.4e-4},
            0.2: {'lr': 1.4e-4},
            0.4: {'lr': 1.4e-4},
            0.6: {'lr': 1.4e-4},
            0.8: {'lr': 3e-4},
            1.0: {'lr': 6e-4},
        }
    }

    @staticmethod
    def get_optimal_hyperparameters_for_lambda(lambda_val):
        """
        Get optimal hyperparameters for a given lambda value.

        Uses linear interpolation between empirically optimal points.

        Parameters:
        -----------
        lambda_val : float
            Lambda value in [0, 1]

        Returns:
        --------
        dict with keys: lr, teacher_coef, tau
        """
        optimal_points = LambdaExperiment.OPTIMAL_HYPERPARAMETERS

        # If exact match, return it
        if lambda_val in optimal_points:
            return optimal_points[lambda_val].copy()

        # Find surrounding points for interpolation
        lambda_keys = sorted(optimal_points.keys())

        # Clamp to valid range
        if lambda_val < lambda_keys[0]:
            return optimal_points[lambda_keys[0]].copy()
        if lambda_val > lambda_keys[-1]:
            return optimal_points[lambda_keys[-1]].copy()

        # Find bracketing points
        upper_idx = next(i for i, k in enumerate(lambda_keys) if k > lambda_val)
        lower_key = lambda_keys[upper_idx - 1]
        upper_key = lambda_keys[upper_idx]

        # Linear interpolation weight
        alpha = (lambda_val - lower_key) / (upper_key - lower_key)

        # Interpolate each parameter
        lower_params = optimal_points[lower_key]
        upper_params = optimal_points[upper_key]

        return {
            'lr': lower_params['lr'] * (1 - alpha) + upper_params['lr'] * alpha,
            'teacher_coef': lower_params['teacher_coef'] * (1 - alpha) + upper_params['teacher_coef'] * alpha,
            'tau': lower_params['tau'] * (1 - alpha) + upper_params['tau'] * alpha,
        }

    @staticmethod
    def get_optimal_learning_rate_2d(lambda_val, junction_density):
        """
        Get optimal learning rate for a given lambda and junction density using 2D bilinear interpolation.

        Junction density is the fraction of nodes with degree >= 3 in the maze.

        Parameters:
        -----------
        lambda_val : float
            Lambda value in [0, 1]
        junction_density : float
            Junction density (typically in [0.1, 0.3])

        Returns:
        --------
        float: interpolated learning rate
        """
        optimal_2d = LambdaExperiment.OPTIMAL_HYPERPARAMETERS_2D

        # Get available junction densities and lambda values
        jd_keys = sorted(optimal_2d.keys())
        lambda_keys = sorted(optimal_2d[jd_keys[0]].keys())  # Assume all jd have same lambda keys

        # Clamp junction density to measured range
        jd_clamped = max(min(junction_density, jd_keys[-1]), jd_keys[0])

        # Clamp lambda to [0, 1]
        lambda_clamped = max(min(lambda_val, lambda_keys[-1]), lambda_keys[0])

        # Check for exact match
        if jd_clamped in jd_keys and lambda_clamped in lambda_keys:
            return optimal_2d[jd_clamped][lambda_clamped]['lr']

        # Find bracketing junction densities
        if jd_clamped <= jd_keys[0]:
            jd1, jd2 = jd_keys[0], jd_keys[0]
            jd_alpha = 0.0
        elif jd_clamped >= jd_keys[-1]:
            jd1, jd2 = jd_keys[-1], jd_keys[-1]
            jd_alpha = 0.0
        else:
            jd2_idx = next(i for i, k in enumerate(jd_keys) if k > jd_clamped)
            jd1 = jd_keys[jd2_idx - 1]
            jd2 = jd_keys[jd2_idx]
            jd_alpha = (jd_clamped - jd1) / (jd2 - jd1)

        # Find bracketing lambda values
        if lambda_clamped <= lambda_keys[0]:
            lam1, lam2 = lambda_keys[0], lambda_keys[0]
            lam_alpha = 0.0
        elif lambda_clamped >= lambda_keys[-1]:
            lam1, lam2 = lambda_keys[-1], lambda_keys[-1]
            lam_alpha = 0.0
        else:
            lam2_idx = next(i for i, k in enumerate(lambda_keys) if k > lambda_clamped)
            lam1 = lambda_keys[lam2_idx - 1]
            lam2 = lambda_keys[lam2_idx]
            lam_alpha = (lambda_clamped - lam1) / (lam2 - lam1)

        # Get four corner points (only lr is stored)
        p11 = optimal_2d[jd1][lam1]  # (jd1, lam1)
        p12 = optimal_2d[jd1][lam2]  # (jd1, lam2)
        p21 = optimal_2d[jd2][lam1]  # (jd2, lam1)
        p22 = optimal_2d[jd2][lam2]  # (jd2, lam2)

        # Helper function for linear interpolation
        def lerp_param(v1, v2, alpha):
            return v1 * (1 - alpha) + v2 * alpha

        # Interpolate along lambda axis for each junction density
        q1_lr = lerp_param(p11['lr'], p12['lr'], lam_alpha)
        q2_lr = lerp_param(p21['lr'], p22['lr'], lam_alpha)

        # Interpolate along junction density axis and return learning rate
        return lerp_param(q1_lr, q2_lr, jd_alpha)

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
        self.output_dir = config.get('output_dir', 'lambda_experiment_results')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Results storage
        self.results = []

    def run(self):
        """
        Run full experiment sweep over topologies, λ values, and seeds using parallel processing.
        """
        print("=" * 80)
        print("Lambda Experiment: Topology-Dependent Optimal TD(λ)")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Topologies: {self.topologies}")
        print(f"  Lambda values: {self.lambda_values}")
        print(f"  Seeds: {self.seeds}")
        print(f"  Episodes per run: {self.num_episodes}")
        print(f"  Total runs: {len(self.topologies) * len(self.lambda_values) * len(self.seeds)}")
        print(f"  Available CPU cores: {cpu_count()}")
        print(f"  Output directory: {self.output_dir}")
        print()

        # Create list of all configurations to run in parallel
        configs_to_run = [
            (topology_name, lambda_val, seed, self.config)
            for topology_name in self.topologies
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

    def compute_aggregate_metrics(self, results_csv_path=None):
        """
        Compute aggregate metrics across episodes (AUC, thresholds, etc.).

        Parameters:
        -----------
        results_csv_path : str, optional
            Path to saved results CSV. If provided, loads from file.
            If None, uses self.results from current experiment run.
        """
        # Load data either from file or from current results
        if results_csv_path is not None:
            print(f"  Loading results from: {results_csv_path}")
            df = pd.read_csv(results_csv_path)
            print(f"  Loaded {len(df)} episodes")
            # Use directory of CSV file for output
            output_dir = os.path.dirname(results_csv_path)
        else:
            if len(self.results) == 0:
                print("  No results to process!")
                return
            df = pd.DataFrame(self.results)
            output_dir = self.output_dir

        # Compute AUC for success rate
        print("  - Computing AUC...")
        auc_results = compute_auc(df, metric='success', x_axis='episode')
        auc_file = os.path.join(output_dir, 'auc_results.csv')
        auc_results.to_csv(auc_file, index=False)

        # Compute episodes to threshold
        print("  - Computing episodes to threshold...")
        episodes_threshold = compute_episodes_to_threshold(df)
        episodes_file = os.path.join(output_dir, 'episodes_to_threshold.csv')
        episodes_threshold.to_csv(episodes_file, index=False)

        # Compute steps to threshold
        print("  - Computing steps to threshold...")
        steps_threshold = compute_steps_to_threshold(df)
        steps_file = os.path.join(output_dir, 'steps_to_threshold.csv')
        steps_threshold.to_csv(steps_file, index=False)

        # Compute timeout rate
        print("  - Computing timeout rate...")
        timeout_results = compute_timeout_rate(df)
        timeout_file = os.path.join(output_dir, 'timeout_rate.csv')
        timeout_results.to_csv(timeout_file, index=False)

        # Compute summary statistics per condition
        print("  - Computing summary statistics...")
        summary = df.groupby(['topology', 'lambda', 'seed']).agg({
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
        }).reset_index()

        summary_file = os.path.join(output_dir, 'summary_statistics.csv')
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


def compute_metrics_from_csv(csv_path):
    """
    Standalone function to compute aggregate metrics from a saved CSV file.

    This can be called from another script or interactively.

    Parameters:
    -----------
    csv_path : str
        Path to results.csv file

    Example:
    --------
    >>> from lambda_experiment import compute_metrics_from_csv
    >>> compute_metrics_from_csv('lambda_experiment_results_20260331_211814/results.csv')
    """
    # Create a minimal experiment object just to use its compute_aggregate_metrics method
    dummy_config = {
        'lambda_values': [],
        'seeds': [],
        'num_episodes': 0,
        'topologies': [],
        'output_dir': '.'
    }
    exp = LambdaExperiment(dummy_config)
    exp.compute_aggregate_metrics(results_csv_path=csv_path)


def main():
    """
    Main entry point for lambda experiment.
    """
    import sys

    # Check if running in analysis mode (computing metrics from existing CSV)
    if len(sys.argv) > 1 and sys.argv[1] == '--compute-metrics':
        if len(sys.argv) < 3:
            print("Usage: python lambda_experiment.py --compute-metrics <path_to_results.csv>")
            sys.exit(1)
        csv_path = sys.argv[2]
        print("=" * 80)
        print("Computing aggregate metrics from saved results")
        print("=" * 80)
        compute_metrics_from_csv(csv_path)
        return

    # Experiment configuration
    config = {
        'lambda_values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'seeds': [60,61,62,63,64,65],
        'num_episodes': 3000,
        'topologies': [
            # Procedural topologies (Family D)
            '0.0 corridor',
            '0.1 corridor',
            '0.2 corridor',
            '0.3 corridor',
            '0.4 corridor',
            '0.5 corridor',
            '0.6 corridor',
            '0.7 corridor',
            '0.8 corridor',
            '0.9 corridor',
            '1.0 corridor',
        ],
        'lr': 'adaptive',  # Use 2D interpolation based on lambda and junction density
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'teacher_coef': 10.0, 
        'value_coef': 0.5,
        'tau': 0.6,  # Fixed value - threshold entropy for memory consultation
        'consultation_temperature': 0.5,  # Temperature for sigmoid gating
        'hard_teacher_force': False,  # Whether to correct wrong policy samples
        'memory_consultation_cost': 0.0,  # Reward penalty for consulting memory
        'memory_correction_cost': 0.0,  # Reward penalty for being corrected by memory
        'output_dir': f'lambda_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Create and run experiment
    experiment = LambdaExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
