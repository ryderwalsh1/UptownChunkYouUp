"""
Meta-Controller Network

Control-value network that decides whether to use fast or slow processing.
Estimates the value of allocating additional computational resources.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MetaController(nn.Module):
    def __init__(self, num_nodes, embedding_dim=64, hidden_dim=128, control_cost=0.01):
        """
        Initialize meta-controller network with advantage-based formulation.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (for state embedding)
        embedding_dim : int
            Dimension of state embedding
        hidden_dim : int
            Dimension of hidden layers
        control_cost : float
            Cost penalty for using slow processing (used as decision threshold)
        """
        super(MetaController, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.control_cost = control_cost
        self.num_control_actions = 2  # Still 2 actions: use_fast, use_slow

        # Control action mapping
        self.USE_FAST = 0
        self.USE_SLOW = 1

        # State embedding
        self.state_embedding = nn.Linear(num_nodes, embedding_dim)

        # Input features:
        # - state embedding: embedding_dim
        # - fast entropy: 1
        # - KL divergence: 1
        # - conflict map value: 1
        input_dim = embedding_dim + 3

        # Advantage network
        # Outputs delta = Q_slow - Q_fast (advantage of using slow processing)
        self.meta_value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single advantage value
        )

    def forward(self, state_encoding, fast_entropy, kl_divergence, conflict_map_value):
        """
        Forward pass through meta-controller.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        fast_entropy : torch.Tensor
            Entropy of fast policy [batch_size]
        kl_divergence : torch.Tensor
            KL divergence between fast and slow [batch_size]
        conflict_map_value : torch.Tensor
            Long-term conflict value from conflict map [batch_size]

        Returns:
        --------
        delta : torch.Tensor
            Advantage of using slow processing (Q_slow - Q_fast) [batch_size, 1]
        """
        batch_size = state_encoding.shape[0]

        # Embed state
        state_embed = F.relu(self.state_embedding(state_encoding))

        # Ensure all inputs are proper shape
        if fast_entropy.dim() == 0:
            fast_entropy = fast_entropy.unsqueeze(0)
        if kl_divergence.dim() == 0:
            kl_divergence = kl_divergence.unsqueeze(0)
        if conflict_map_value.dim() == 0:
            conflict_map_value = conflict_map_value.unsqueeze(0)

        # Reshape to [batch_size, 1] if needed
        if fast_entropy.dim() == 1:
            fast_entropy = fast_entropy.unsqueeze(1)
        if kl_divergence.dim() == 1:
            kl_divergence = kl_divergence.unsqueeze(1)
        if conflict_map_value.dim() == 1:
            conflict_map_value = conflict_map_value.unsqueeze(1)

        # Concatenate features
        features = torch.cat([
            state_embed,
            fast_entropy,
            kl_divergence,
            conflict_map_value
        ], dim=-1)

        # Compute advantage (delta = Q_slow - Q_fast)
        delta = self.meta_value_net(features)

        return delta

    def get_control_policy(self, delta, temperature=1.0):
        """
        Convert advantage delta to control policy via sigmoid.

        Uses: p(slow) = sigmoid((delta - control_cost) / temperature)

        Parameters:
        -----------
        delta : torch.Tensor
            Advantage of using slow (Q_slow - Q_fast) [batch_size, 1]
        temperature : float
            Temperature for sigmoid (higher = more exploration)

        Returns:
        --------
        probs : torch.Tensor
            Probability distribution over control actions [batch_size, 2]
            probs[:, 0] = p(fast), probs[:, 1] = p(slow)
        """
        # Apply control cost threshold and temperature
        logit = (delta - self.control_cost) / temperature
        p_slow = torch.sigmoid(logit)  # [batch_size, 1]
        p_fast = 1 - p_slow

        # Return as [batch_size, 2] for compatibility
        probs = torch.cat([p_fast, p_slow], dim=-1)
        return probs

    def sample_control_action(self, delta, temperature=1.0):
        """
        Sample control action from policy.

        Parameters:
        -----------
        delta : torch.Tensor
            Advantage of using slow (Q_slow - Q_fast) [batch_size, 1]
        temperature : float
            Temperature for sigmoid

        Returns:
        --------
        action : torch.Tensor
            Sampled control action [batch_size]
        log_prob : torch.Tensor
            Log probability of sampled action [batch_size]
        probs : torch.Tensor
            Full probability distribution [batch_size, 2]
        """
        probs = self.get_control_policy(delta, temperature)

        # Check for NaN or Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"WARNING: NaN or Inf detected in control probabilities!")
            print(f"  delta: {delta}")
            print(f"  probs: {probs}")
            # Replace with uniform distribution
            probs = torch.ones_like(probs) / probs.shape[-1]

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, probs

    def get_slow_probability(self, delta, temperature=1.0):
        """
        Get probability of using slow processing.

        This is used for lambda modulation.

        Parameters:
        -----------
        delta : torch.Tensor
            Advantage of using slow (Q_slow - Q_fast) [batch_size, 1]
        temperature : float
            Temperature for sigmoid

        Returns:
        --------
        p_slow : torch.Tensor
            Probability of use_slow action [batch_size]
        """
        probs = self.get_control_policy(delta, temperature)
        return probs[:, self.USE_SLOW]

    def compute_entropy(self, delta, temperature=1.0):
        """
        Compute entropy of control policy.

        Parameters:
        -----------
        delta : torch.Tensor
            Advantage of using slow (Q_slow - Q_fast) [batch_size, 1]
        temperature : float
            Temperature for sigmoid

        Returns:
        --------
        entropy : torch.Tensor
            Entropy of control policy [batch_size]
        """
        probs = self.get_control_policy(delta, temperature)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class MetaControllerTrainer:
    """Trainer for meta-controller using reinforcement learning."""

    def __init__(self, controller, lr=1e-3, gamma=0.99, entropy_coef=0.01):
        """
        Initialize trainer.

        Parameters:
        -----------
        controller : MetaController
            The controller to train
        lr : float
            Learning rate
        gamma : float
            Discount factor for meta-level returns
        entropy_coef : float
            Coefficient for entropy bonus
        """
        self.controller = controller
        self.optimizer = torch.optim.Adam(controller.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef

    def compute_returns(self, rewards, gamma=None):
        """
        Compute discounted returns.

        Parameters:
        -----------
        rewards : list of float
            Rewards for each step
        gamma : float, optional
            Discount factor

        Returns:
        --------
        returns : torch.Tensor
            Discounted returns [T]
        """
        if gamma is None:
            gamma = self.gamma

        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        return returns

    def train_step(self, trajectory):
        """
        Perform single training step on trajectory.

        The controller is trained with RL from the returns induced by its
        control choices. It learns where slow processing improves outcomes
        and where it's not worth the cost.

        Parameters:
        -----------
        trajectory : dict
            Dictionary containing:
            - states: list of state encodings
            - fast_entropies: list of fast entropies
            - kl_divergences: list of KL divergences
            - conflict_values: list of conflict map values
            - control_actions: list of control actions taken
            - control_log_probs: list of log probs for control actions
            - rewards: list of rewards (including control costs)
            - dones: list of done flags

        Returns:
        --------
        loss_dict : dict
            Dictionary with loss components
        """
        states = torch.stack(trajectory['states'])
        fast_entropies = torch.stack(trajectory['fast_entropies'])
        kl_divergences = torch.stack(trajectory['kl_divergences'])
        conflict_values = torch.stack(trajectory['conflict_values'])
        control_actions = torch.tensor(trajectory['control_actions'], dtype=torch.long)
        old_log_probs = torch.stack(trajectory['control_log_probs'])
        rewards = trajectory['rewards']

        # Compute returns
        returns = self.compute_returns(rewards)

        # Normalize returns (only if we have more than 1 sample)
        if len(returns) > 1:
            returns_std = returns.std()
            if returns_std > 1e-8:
                returns = (returns - returns.mean()) / returns_std
            else:
                returns = returns - returns.mean()
        # If only 1 sample, no normalization needed

        # Recompute delta (advantage) with current controller
        delta = self.controller(states, fast_entropies, kl_divergences, conflict_values)

        # Get log probs and entropy
        probs = self.controller.get_control_policy(delta)
        log_probs = torch.log(probs.gather(1, control_actions.unsqueeze(1)).squeeze(1) + 1e-10)
        entropies = self.controller.compute_entropy(delta)

        # Policy loss (REINFORCE)
        policy_loss = -(log_probs * returns.detach()).mean()

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropies.mean()

        # Total loss
        loss = policy_loss + self.entropy_coef * entropy_loss

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected in controller! Skipping update.")
            print(f"  policy_loss: {policy_loss.item() if not torch.isnan(policy_loss) else 'NaN'}")
            print(f"  entropy_loss: {entropy_loss.item() if not torch.isnan(entropy_loss) else 'NaN'}")
            return {
                'loss': float('nan'),
                'policy_loss': policy_loss.item() if not torch.isnan(policy_loss) else float('nan'),
                'entropy_loss': entropy_loss.item() if not torch.isnan(entropy_loss) else float('nan'),
                'mean_entropy': 0.0,
                'mean_return': 0.0,
                'p_slow_mean': 0.5
            }

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 0.5)
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'mean_entropy': entropies.mean().item(),
            'mean_return': returns.mean().item(),
            'p_slow_mean': probs[:, self.controller.USE_SLOW].mean().item()
        }


if __name__ == "__main__":
    print("Testing MetaController...")

    num_nodes = 64
    batch_size = 8
    control_cost = 0.3

    # Create controller
    controller = MetaController(num_nodes, control_cost=control_cost)
    print(f"Created MetaController with {sum(p.numel() for p in controller.parameters())} parameters")
    print(f"Control cost: {control_cost}")

    # Create random inputs
    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # One-hot

    fast_entropy = torch.rand(batch_size)
    kl_divergence = torch.rand(batch_size) * 2
    conflict_map_value = torch.rand(batch_size)

    # Forward pass
    print("\nForward pass:")
    delta = controller(state_encoding, fast_entropy, kl_divergence, conflict_map_value)
    print(f"  Delta shape: {delta.shape}")
    print(f"  Delta (Q_slow - Q_fast):\n{delta.squeeze()}")

    # Get control policy
    probs = controller.get_control_policy(delta)
    print(f"\n  Control policy (probabilities):")
    print(f"    use_fast: {probs[:, controller.USE_FAST]}")
    print(f"    use_slow: {probs[:, controller.USE_SLOW]}")
    print(f"    Mean p(slow): {probs[:, controller.USE_SLOW].mean().item():.3f}")

    # Sample control action
    action, log_prob, probs = controller.sample_control_action(delta)
    print(f"\n  Sampled control actions: {action}")
    print(f"  Log probs: {log_prob}")

    # Get slow probability for lambda modulation
    p_slow = controller.get_slow_probability(delta)
    print(f"\n  p_slow (for lambda modulation): {p_slow}")

    # Compute entropy
    entropy = controller.compute_entropy(delta)
    print(f"  Control policy entropy: {entropy.mean().item():.3f}")

    # Test trainer
    print("\nTesting MetaControllerTrainer...")
    trainer = MetaControllerTrainer(controller, lr=1e-3)

    # Create dummy trajectory
    T = 10
    trajectory = {
        'states': [torch.zeros(num_nodes) for _ in range(T)],
        'fast_entropies': [torch.tensor(np.random.rand()) for _ in range(T)],
        'kl_divergences': [torch.tensor(np.random.rand() * 2) for _ in range(T)],
        'conflict_values': [torch.tensor(np.random.rand()) for _ in range(T)],
        'control_actions': [np.random.randint(2) for _ in range(T)],
        'control_log_probs': [torch.tensor(np.random.randn()) for _ in range(T)],
        'rewards': [np.random.randn() for _ in range(T)],
        'dones': [False] * (T-1) + [True]
    }

    # Make states one-hot
    for i, state in enumerate(trajectory['states']):
        state[i % num_nodes] = 1.0

    loss_dict = trainer.train_step(trajectory)
    print(f"Training step completed:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ MetaController tests passed!")
