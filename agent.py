"""
Cognitive Agent

Orchestrates fast, slow, controller, conflict map, and lambda modulator.
Implements the full cognitive control architecture with sample-based arbitration.
"""

import torch
import numpy as np
from fast import FastNetwork
from slow import SlowMemory
from controller import MetaController
from conflict_map import ConflictMap, compute_kl_divergence
from lambda_modulator import LambdaModulator


class CognitiveAgent:
    def __init__(self, num_nodes, num_actions, maze_graph=None,
                 embedding_dim=64, hidden_dim=128,
                 conflict_alpha=0.01, lambda_beta=2.0,
                 w_long=0.8, w_short=0.2,
                 control_cost=0.01, fixed_lambda=None):
        """
        Initialize cognitive agent.

        Parameters:
        -----------
        num_nodes : int
            Number of states in environment
        num_actions : int
            Number of actions
        maze_graph : networkx.Graph, optional
            Maze graph for initializing slow network memory
        embedding_dim : int
            Embedding dimension for networks
        hidden_dim : int
            Hidden dimension for networks
        conflict_alpha : float
            EMA update rate for conflict map
        lambda_beta : float
            Power-law exponent for lambda modulation
        w_long : float
            Weight for long-term signal in lambda
        w_short : float
            Weight for short-term signal in lambda
        control_cost : float
            Cost penalty for using slow processing
        fixed_lambda : float, optional
            If provided, lambda modulator always returns this value (disables modulation)
        """
        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.control_cost = control_cost

        # Create networks
        self.fast_network = FastNetwork(num_nodes, num_actions, embedding_dim, hidden_dim)
        self.slow_memory = SlowMemory(num_nodes, num_actions)
        self.controller = MetaController(num_nodes, embedding_dim, hidden_dim, control_cost=control_cost)

        # Initialize slow memory if maze graph provided
        if maze_graph is not None:
            self.slow_memory.initialize_memory(maze_graph)

        # Create conflict map
        self.conflict_map = ConflictMap(num_nodes, alpha=conflict_alpha)

        # Create lambda modulator
        self.lambda_modulator = LambdaModulator(beta=lambda_beta, w_long=w_long, w_short=w_short, fixed_lambda=fixed_lambda)

        # Hidden state for fast network
        self.fast_hidden = None

        # Statistics
        self.stats = {
            'fast_selected': 0,
            'slow_selected': 0,
            'total_steps': 0,
            'total_control_cost': 0.0
        }

    def reset(self):
        """Reset agent for new episode."""
        self.fast_hidden = None
        self.fast_network.reset_hidden(batch_size=1)

    def step(self, state_encoding, goal_encoding, temperature=1.0, train_mode=True):
        """
        Perform single step of cognitive processing.

        Implements the per-step computation from Section 13 of the spec:
        1. Encode state/history with fast GRU
        2. Retrieve episodic content
        3. Compute slow policy logits
        4. Compute controller features
        5. Compute controller meta-values and control policy
        6. Sample control action
        7. Use selected branch to sample environment action

        Parameters:
        -----------
        state_encoding : torch.Tensor
            Current state encoding [1, num_nodes]
        goal_encoding : torch.Tensor
            Goal state encoding [1, num_nodes]
        temperature : float
            Temperature for control policy sampling
        train_mode : bool
            Whether in training mode (affects sampling)

        Returns:
        --------
        step_info : dict
            Dictionary containing all step information
        """
        with torch.no_grad() if not train_mode else torch.enable_grad():
            # 1. Fast network forward pass
            fast_logits, prospection_logits, fast_value, self.fast_hidden = self.fast_network(
                state_encoding, goal_encoding, self.fast_hidden
            )

            # 2. Slow memory query (pure retrieval, no network)
            slow_logits = self.slow_memory.query(state_encoding, goal_encoding)

            # 3. Compute fast entropy
            fast_entropy = self.fast_network.compute_entropy(fast_logits)

            # 4. Compute KL divergence between fast and slow
            kl_divergence = compute_kl_divergence(fast_logits, slow_logits)

            # 5. Get conflict map value
            state_idx = state_encoding.argmax().item()
            conflict_value = torch.tensor(self.conflict_map.get(state_idx), dtype=torch.float32)

            # 6. Controller forward pass (outputs delta = Q_slow - Q_fast)
            delta = self.controller(
                state_encoding,
                fast_entropy,
                kl_divergence,
                conflict_value.unsqueeze(0)
            )

            # 7. Sample control action
            control_action, control_log_prob, control_probs = self.controller.sample_control_action(
                delta, temperature
            )

            # 8. Get p_slow for lambda modulation
            p_slow = self.controller.get_slow_probability(delta)

            # 9. Select policy branch based on control action
            used_slow = (control_action.item() == self.controller.USE_SLOW)

            if used_slow:
                action_logits = slow_logits
                self.stats['slow_selected'] += 1
            else:
                action_logits = fast_logits
                self.stats['fast_selected'] += 1

            # Value always comes from fast network (slow memory doesn't estimate values)
            value = fast_value

            # 10. Sample environment action from selected policy
            action, action_log_prob = self.fast_network.sample_action(action_logits)

            # Update statistics
            self.stats['total_steps'] += 1
            if used_slow:
                self.stats['total_control_cost'] += self.control_cost

        # Return comprehensive step information
        return {
            'action': action.item(),
            'action_log_prob': action_log_prob,
            'value': value,
            'control_action': control_action.item(),
            'control_log_prob': control_log_prob,
            'used_slow': used_slow,
            'p_slow': p_slow.item(),
            'fast_logits': fast_logits,
            'prospection_logits': prospection_logits,
            'slow_logits': slow_logits,
            'fast_value': fast_value,
            'fast_entropy': fast_entropy.item(),
            'kl_divergence': kl_divergence.item(),
            'conflict_value': conflict_value.item(),
            'state_idx': state_idx,
            'delta': delta,  # Advantage (Q_slow - Q_fast)
            'control_probs': control_probs
        }

    def update_conflict_map(self, state_idx, kl_divergence):
        """
        Update conflict map after step.

        Parameters:
        -----------
        state_idx : int
            State index
        kl_divergence : float
            KL divergence between fast and slow
        """
        self.conflict_map.update(state_idx, kl_divergence)

    def compute_lambda(self, conflict_value, p_slow):
        """
        Compute lambda for current step.

        Parameters:
        -----------
        conflict_value : float
            Conflict map value
        p_slow : float
            Probability of using slow processing

        Returns:
        --------
        lambda_val : float
            Lambda value for eligibility traces
        """
        return self.lambda_modulator.compute_lambda(conflict_value, p_slow)

    def get_statistics(self):
        """Get agent statistics."""
        stats = self.stats.copy()
        if stats['total_steps'] > 0:
            stats['p_slow_empirical'] = stats['slow_selected'] / stats['total_steps']
            stats['mean_control_cost'] = stats['total_control_cost'] / stats['total_steps']
        else:
            stats['p_slow_empirical'] = 0.0
            stats['mean_control_cost'] = 0.0

        # Add conflict map statistics
        stats['conflict_map'] = self.conflict_map.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset statistics."""
        self.stats = {
            'fast_selected': 0,
            'slow_selected': 0,
            'total_steps': 0,
            'total_control_cost': 0.0
        }

    def save(self, filepath):
        """
        Save agent to file.

        Parameters:
        -----------
        filepath : str
            Path to save file
        """
        torch.save({
            'fast_network': self.fast_network.state_dict(),
            # Note: slow_memory is not saved (it's re-initialized from maze graph)
            'controller': self.controller.state_dict(),
            'conflict_map': {
                'conflict_values': self.conflict_map.conflict_values,
                'update_counts': self.conflict_map.update_counts,
                'alpha': self.conflict_map.alpha
            },
            'lambda_modulator': {
                'beta': self.lambda_modulator.beta,
                'w_long': self.lambda_modulator.w_long,
                'w_short': self.lambda_modulator.w_short,
                'fixed_lambda': self.lambda_modulator.fixed_lambda
            },
            'stats': self.stats
        }, filepath)

    def load(self, filepath):
        """
        Load agent from file.

        Parameters:
        -----------
        filepath : str
            Path to load file
        """
        checkpoint = torch.load(filepath, weights_only=False)

        self.fast_network.load_state_dict(checkpoint['fast_network'])
        # Note: slow_memory is not loaded (it's re-initialized from maze graph)
        self.controller.load_state_dict(checkpoint['controller'])

        self.conflict_map.conflict_values = checkpoint['conflict_map']['conflict_values']
        self.conflict_map.update_counts = checkpoint['conflict_map']['update_counts']
        self.conflict_map.alpha = checkpoint['conflict_map']['alpha']

        self.lambda_modulator.beta = checkpoint['lambda_modulator']['beta']
        self.lambda_modulator.w_long = checkpoint['lambda_modulator']['w_long']
        self.lambda_modulator.w_short = checkpoint['lambda_modulator']['w_short']
        self.lambda_modulator.fixed_lambda = checkpoint['lambda_modulator'].get('fixed_lambda', None)

        self.stats = checkpoint['stats']


if __name__ == "__main__":
    print("Testing CognitiveAgent...")

    from corridors import MazeGraph

    # Create maze
    print("\nCreating maze...")
    maze = MazeGraph(length=4, width=4, corridor=0.5, seed=42)
    graph = maze.get_graph()
    num_nodes = graph.number_of_nodes()
    num_actions = 5  # Direction-based actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL

    print(f"Maze has {num_nodes} nodes")

    # Create agent
    print("\nCreating agent...")
    agent = CognitiveAgent(
        num_nodes=num_nodes,
        num_actions=num_actions,
        maze_graph=maze,  # Pass MazeGraph object, not raw graph
        control_cost=0.01
    )

    print(f"Agent created with:")
    print(f"  Fast network: {sum(p.numel() for p in agent.fast_network.parameters())} params")
    print(f"  Slow memory: episodic retrieval (no trainable params)")
    print(f"  Controller: {sum(p.numel() for p in agent.controller.parameters())} params")

    # Test single step
    print("\nTesting single step...")
    agent.reset()

    state_encoding = torch.zeros(1, num_nodes)
    state_encoding[0, 0] = 1.0

    goal_encoding = torch.zeros(1, num_nodes)
    goal_encoding[0, num_nodes-1] = 1.0

    step_info = agent.step(state_encoding, goal_encoding, train_mode=False)

    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}
    print(f"Step completed:")
    print(f"  Action: {action_names[step_info['action']]} ({step_info['action']})")
    print(f"  Used slow: {step_info['used_slow']}")
    print(f"  p_slow: {step_info['p_slow']:.3f}")
    print(f"  Fast entropy: {step_info['fast_entropy']:.3f}")
    print(f"  KL divergence: {step_info['kl_divergence']:.3f}")
    print(f"  Conflict value: {step_info['conflict_value']:.3f}")

    # Compute lambda
    lambda_val = agent.compute_lambda(step_info['conflict_value'], step_info['p_slow'])
    print(f"  Lambda: {lambda_val:.3f}")

    # Update conflict map
    agent.update_conflict_map(step_info['state_idx'], step_info['kl_divergence'])

    # Test multiple steps
    print("\nTesting episode with 10 steps...")
    agent.reset()

    for i in range(10):
        state_idx = np.random.randint(num_nodes)
        goal_idx = np.random.randint(num_nodes)

        state_encoding = torch.zeros(1, num_nodes)
        state_encoding[0, state_idx] = 1.0

        goal_encoding = torch.zeros(1, num_nodes)
        goal_encoding[0, goal_idx] = 1.0

        step_info = agent.step(state_encoding, goal_encoding, train_mode=False)
        agent.update_conflict_map(step_info['state_idx'], step_info['kl_divergence'])

        action_name = action_names[step_info['action']]
        print(f"  Step {i+1}: action={action_name} ({step_info['action']}), used_slow={step_info['used_slow']}, "
              f"kl={step_info['kl_divergence']:.3f}")

    # Get statistics
    print("\nAgent statistics:")
    stats = agent.get_statistics()
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Fast selected: {stats['fast_selected']}")
    print(f"  Slow selected: {stats['slow_selected']}")
    print(f"  p(slow) empirical: {stats['p_slow_empirical']:.3f}")
    print(f"  Mean control cost: {stats['mean_control_cost']:.4f}")

    print("\nConflict map statistics:")
    for key, value in stats['conflict_map'].items():
        print(f"  {key}: {value}")

    # Test save/load
    print("\nTesting save/load...")
    agent.save('/tmp/cognitive_agent_test.pt')
    print("  Saved agent")

    agent2 = CognitiveAgent(num_nodes=num_nodes, num_actions=num_actions, maze_graph=maze)
    agent2.load('/tmp/cognitive_agent_test.pt')
    print("  Loaded agent")

    # Verify statistics match
    stats2 = agent2.get_statistics()
    print(f"  Stats match: {stats['total_steps'] == stats2['total_steps']}")

    print("\n✓ CognitiveAgent tests passed!")
