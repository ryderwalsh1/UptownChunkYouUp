"""
Slow Policy Network

Episodic-memory-informed policy using PsyNeuLink.
Represents deliberate, flexible, memory-guided processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import psyneulink as pnl
import networkx as nx
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='psyneulink')


class SlowNetwork(nn.Module):
    def __init__(self, num_nodes, num_actions, embedding_dim=64, hidden_dim=128,
                 memory_softmax_gain=15.0, memory_field_weights=None):
        """
        Initialize slow policy network with episodic memory.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions
        embedding_dim : int
            Dimension of state and goal embeddings
        hidden_dim : int
            Dimension of hidden layers
        memory_softmax_gain : float
            Softmax gain for episodic memory retrieval
        memory_field_weights : list, optional
            Weights for memory fields [source, target, answer]
        """
        super(SlowNetwork, self).__init__()

        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # State and goal embeddings
        self.state_embedding = nn.Linear(num_nodes, embedding_dim)
        self.goal_embedding = nn.Linear(num_nodes, embedding_dim)

        # Memory retrieval embedding (process retrieved content)
        # Retrieved memory is a one-hot over nodes (next step suggestion)
        self.memory_embedding = nn.Linear(num_nodes, embedding_dim)

        # Combine embeddings with retrieved memory
        # Input: state_embed + goal_embed + memory_embed
        self.fusion_layer = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head (outputs action logits)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Value head (outputs state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Episodic memory (PsyNeuLink EMComposition)
        self.episodic_memory = None
        self.memory_softmax_gain = memory_softmax_gain
        self.memory_field_weights = memory_field_weights or [1.0, 1.0, None]

    def initialize_memory(self, maze_graph):
        """
        Initialize episodic memory with shortest path information from maze.

        Parameters:
        -----------
        maze_graph : networkx.Graph
            The maze graph
        """
        # Build shortest path memories
        nodes_list = list(maze_graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes_list)}

        memories = []
        for source in nodes_list:
            for target in nodes_list:
                if source != target and nx.has_path(maze_graph, source, target):
                    path = nx.shortest_path(maze_graph, source, target)
                    if len(path) > 1:
                        next_step = path[1]

                        # Create one-hot encodings
                        source_encoding = [0] * self.num_nodes
                        source_encoding[node_to_idx[source]] = 1

                        target_encoding = [0] * self.num_nodes
                        target_encoding[node_to_idx[target]] = 1

                        next_step_encoding = [0] * self.num_nodes
                        next_step_encoding[node_to_idx[next_step]] = 1

                        memories.append({
                            'Source': source_encoding,
                            'Target': target_encoding,
                            'Answer': next_step_encoding
                        })

        if len(memories) == 0:
            raise ValueError("No memories created from maze graph")

        memory_capacity = len(memories)
        memory_array = np.array([[m['Source'], m['Target'], m['Answer']] for m in memories])

        # Create EMComposition
        self.episodic_memory = pnl.EMComposition(
            memory_template=memory_array,
            memory_capacity=memory_capacity,
            field_names=['Source', 'Target', 'Answer'],
            field_weights=self.memory_field_weights,
            storage_prob=0.0,  # Don't store new memories during queries
            memory_decay_rate=0.0,
            softmax_gain=self.memory_softmax_gain,
            normalize_memories=True,
            enable_learning=False,
            name='Slow_Memory'
        )

        print(f"Initialized episodic memory with {len(memories)} navigation memories")

    def retrieve_memory(self, state_encoding, goal_encoding):
        """
        Retrieve episodic memory given current state and goal.

        Parameters:
        -----------
        state_encoding : np.ndarray
            One-hot encoding of current state [num_nodes]
        goal_encoding : np.ndarray
            One-hot encoding of goal state [num_nodes]

        Returns:
        --------
        retrieved : np.ndarray
            Retrieved next-step suggestion [num_nodes]
        """
        if self.episodic_memory is None:
            raise ValueError("Episodic memory not initialized. Call initialize_memory() first.")

        # Query the episodic memory
        query_inputs = {
            self.episodic_memory.query_input_nodes[0]: state_encoding.tolist(),
            self.episodic_memory.query_input_nodes[1]: goal_encoding.tolist()
        }

        results = self.episodic_memory.run(inputs=query_inputs)

        # Extract the retrieved answer (next step suggestion)
        retrieved = np.array(results[2], dtype=np.float32)  # Answer field

        return retrieved

    def forward(self, state_encoding, goal_encoding, retrieved_memory=None):
        """
        Forward pass through slow network.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        goal_encoding : torch.Tensor
            One-hot encoding of goal state [batch_size, num_nodes]
        retrieved_memory : torch.Tensor, optional
            Retrieved memory content [batch_size, num_nodes]
            If None, will retrieve from episodic memory

        Returns:
        --------
        action_logits : torch.Tensor
            Logits over actions [batch_size, num_actions]
        value : torch.Tensor
            State value estimate [batch_size, 1]
        retrieved_memory : torch.Tensor
            Retrieved memory used [batch_size, num_nodes]
        """
        batch_size = state_encoding.shape[0]

        # Retrieve from episodic memory if not provided
        if retrieved_memory is None:
            retrieved_list = []
            for i in range(batch_size):
                state_np = state_encoding[i].detach().cpu().numpy()
                goal_np = goal_encoding[i].detach().cpu().numpy()
                retrieved = self.retrieve_memory(state_np, goal_np)
                retrieved_list.append(retrieved)

            retrieved_memory = torch.tensor(np.array(retrieved_list), dtype=torch.float32)

        # Embed inputs
        state_embed = F.relu(self.state_embedding(state_encoding))
        goal_embed = F.relu(self.goal_embedding(goal_encoding))
        memory_embed = F.relu(self.memory_embedding(retrieved_memory))

        # Fuse embeddings
        combined = torch.cat([state_embed, goal_embed, memory_embed], dim=-1)
        fused = self.fusion_layer(combined)

        # Generate action logits and value
        action_logits = self.policy_head(fused)
        value = self.value_head(fused)

        return action_logits, value, retrieved_memory

    def get_action_distribution(self, action_logits):
        """Convert action logits to probability distribution."""
        return F.softmax(action_logits, dim=-1)

    def compute_entropy(self, action_logits):
        """Compute entropy of action distribution."""
        probs = self.get_action_distribution(action_logits)
        log_probs = F.log_softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def sample_action(self, action_logits):
        """Sample action from policy distribution."""
        probs = self.get_action_distribution(action_logits)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, action_logits, action):
        """Get log probability of specific action."""
        log_probs = F.log_softmax(action_logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


class SlowNetworkTrainer:
    """Trainer for slow network."""

    def __init__(self, network, lr=3e-4, gamma=0.99, lambda_=0.95, entropy_coef=0.01, value_coef=0.5):
        """
        Initialize trainer.

        Parameters:
        -----------
        network : SlowNetwork
            The slow network to train
        lr : float
            Learning rate
        gamma : float
            Discount factor
        lambda_ : float
            TD(λ) trace decay parameter
        entropy_coef : float
            Coefficient for entropy bonus
        value_coef : float
            Coefficient for value loss
        """
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

    def compute_gae(self, rewards, values, dones, next_value, gamma=None, lambda_=None):
        """Compute Generalized Advantage Estimation (GAE)."""
        if gamma is None:
            gamma = self.gamma
        if lambda_ is None:
            lambda_ = self.lambda_

        advantages = []
        gae = 0
        T = len(rewards)

        # Stack values if they're 0-d tensors, otherwise cat
        if values[0].dim() == 0:
            values = torch.stack(values)  # [T]
        else:
            values = torch.cat(values)  # [T]
        values_list = values.tolist()

        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values_list[t + 1]

            if dones[t]:
                delta = rewards[t] - values_list[t]
                gae = delta
            else:
                delta = rewards[t] + gamma * next_val - values_list[t]
                gae = delta + gamma * lambda_ * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values

        return advantages, returns

    def train_step(self, trajectory):
        """
        Perform single training step on trajectory.

        Parameters:
        -----------
        trajectory : dict
            Dictionary containing trajectory data

        Returns:
        --------
        loss_dict : dict
            Dictionary with loss components
        """
        states = torch.stack(trajectory['states'])
        goals = torch.stack(trajectory['goals'])
        memories = torch.stack(trajectory['memories'])  # Pre-retrieved memories
        actions = torch.tensor(trajectory['actions'], dtype=torch.long)
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        old_values = trajectory['values']

        # Compute next value
        with torch.no_grad():
            next_state = trajectory['next_state'].unsqueeze(0)
            next_goal = trajectory['next_goal'].unsqueeze(0)
            next_memory = trajectory['next_memory'].unsqueeze(0)
            _, next_value, _ = self.network(next_state, next_goal, next_memory)
            next_value = next_value.squeeze()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, old_values, dones, next_value)

        # Normalize advantages (only if we have more than 1 sample)
        if len(advantages) > 1:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std
            else:
                advantages = advantages - advantages.mean()
        # If only 1 sample, no normalization needed

        # Recompute outputs with current network
        action_logits, values, _ = self.network(states, goals, memories)
        values = values.squeeze()

        # Compute losses
        log_probs = self.network.get_log_prob(action_logits, actions)
        entropies = self.network.compute_entropy(action_logits)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns)
        entropy_loss = -entropies.mean()

        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_entropy': entropies.mean().item(),
            'mean_value': values.mean().item()
        }


if __name__ == "__main__":
    print("Testing Slow Network...")

    # Create a simple test graph
    from corridors import MazeGraph

    print("\nCreating maze...")
    maze = MazeGraph(length=4, width=4, corridor=0.5, seed=42)
    graph = maze.get_graph()
    num_nodes = graph.number_of_nodes()
    num_actions = num_nodes  # Actions are node indices

    print(f"Maze has {num_nodes} nodes")

    # Create slow network
    network = SlowNetwork(num_nodes, num_actions)
    print(f"\nCreated SlowNetwork with {sum(p.numel() for p in network.parameters())} parameters")

    # Initialize memory
    print("\nInitializing episodic memory...")
    network.initialize_memory(graph)

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4

    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # First node

    goal_encoding = torch.zeros(batch_size, num_nodes)
    goal_encoding[:, num_nodes-1] = 1.0  # Last node

    action_logits, value, retrieved = network(state_encoding, goal_encoding)
    print(f"  Action logits shape: {action_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Retrieved memory shape: {retrieved.shape}")
    print(f"  Retrieved memory (first sample): {retrieved[0].argmax().item()}")

    # Compute entropy
    entropy = network.compute_entropy(action_logits)
    print(f"  Mean entropy: {entropy.mean().item():.3f}")

    # Sample action
    action, log_prob = network.sample_action(action_logits)
    print(f"  Sampled actions: {action}")
    print(f"  Log probs: {log_prob}")

    print("\n✓ Slow network tests passed!")
