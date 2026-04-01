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

from maze_env import MazeEnvironment
from fast import FastNetwork, FastNetworkTrainer
from lambda_experiment.topology_generators import ALL_TOPOLOGIES, generate_topology
from lambda_experiment.topology_metrics import compute_all_metrics


class LambdaExperiment:
    """Main experiment runner for testing λ across topologies."""

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
        self.value_coef = config.get('value_coef', 0.5)
        self.output_dir = config.get('output_dir', 'lambda_experiment_results')

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Results storage
        self.results = []

    def run_single_configuration(self, topology_name, lambda_val, seed):
        """
        Run training for a single (topology, λ, seed) configuration.

        Parameters:
        -----------
        topology_name : str
            Name of topology from ALL_TOPOLOGIES
        lambda_val : float
            TD(λ) parameter value
        seed : int
            Random seed

        Returns:
        --------
        run_results : list of dict
            Episode-level results
        """
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
            fixed_start_node=None,  # Random start
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

        trainer = FastNetworkTrainer(
            network=network,
            lr=self.lr,
            gamma=self.gamma,
            lambda_=lambda_val,
            entropy_coef=self.entropy_coef,
            value_coef=self.value_coef
        )

        # Training loop
        run_results = []

        for episode in tqdm(range(self.num_episodes),
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
            }

            episode_reward = 0.0
            episode_length = 0
            success = False
            max_steps = env.max_steps

            for step in range(max_steps):
                # Get state encoding
                state_encoding = torch.tensor(state['current_encoding'], dtype=torch.float32).unsqueeze(0)
                goal_encoding = torch.tensor(state['goal_encoding'], dtype=torch.float32).unsqueeze(0)

                # Forward pass
                action_logits, _, value, hidden = network(
                    state_encoding, goal_encoding, network.hidden
                )

                # Sample action
                action, log_prob = network.sample_action(action_logits)

                # Store trajectory data
                trajectory['states'].append(torch.tensor(state['current_encoding']))
                trajectory['goals'].append(torch.tensor(state['goal_encoding']))
                trajectory['actions'].append(action.item())
                trajectory['log_probs'].append(log_prob)
                trajectory['values'].append(value.squeeze())
                trajectory['hiddens'].append(hidden)

                # Take step
                next_state, reward, done, info = env.step(action.item(), used_slow=False)

                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)

                episode_reward += reward
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

            run_results.append(episode_result)

        return run_results

    def run(self):
        """
        Run full experiment sweep over topologies, λ values, and seeds.
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
        print(f"  Output directory: {self.output_dir}")
        print()

        # Nested loops over all configurations
        total_configs = len(self.topologies) * len(self.lambda_values) * len(self.seeds)
        config_idx = 0

        for topology_name in self.topologies:
            for lambda_val in self.lambda_values:
                for seed in self.seeds:
                    config_idx += 1
                    print(f"\n[{config_idx}/{total_configs}] Running: {topology_name}, λ={lambda_val:.2f}, seed={seed}")

                    # Run single configuration
                    run_results = self.run_single_configuration(topology_name, lambda_val, seed)

                    # Accumulate results
                    self.results.extend(run_results)

                    # Save intermediate results after each configuration
                    self.save_results()

        print("\n" + "=" * 80)
        print("Experiment complete!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 80)

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


def main():
    """
    Main entry point for lambda experiment.
    """
    # Experiment configuration
    config = {
        'lambda_values': [0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0],  # Coarse sweep
        'seeds': [42, 43, 44],  # 3 seeds
        'num_episodes': 300,  # Short training for initial tests
        'topologies': [
            # Elementary topologies (Family A: corridors)
            'corridor_short',
            'corridor_medium',
            'corridor_long',
            # Procedural topologies (Family D)
            'proc_junction_heavy',
            'proc_mixed',
            'proc_corridor_heavy',
        ],
        'lr': 3e-4,
        'gamma': 0.99,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'output_dir': f'lambda_experiment_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
    }

    # Create and run experiment
    experiment = LambdaExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
