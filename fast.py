"""
Fast Policy Network

GRU-based policy network representing habitual/intuitive processing.
Produces action logits and value estimates from compact state representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FastNetwork(nn.Module):
    def __init__(self, num_nodes, num_actions, embedding_dim=64, hidden_dim=128, prospection_head=True):
        """
        Initialize fast policy network.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions
        embedding_dim : int
            Dimension of state and goal embeddings
        hidden_dim : int
            Dimension of GRU hidden state
        prospection_head : bool
            Whether to include the prospection head (default: True)
        """
        super(FastNetwork, self).__init__()

        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.has_prospection_head = prospection_head

        # State and goal embeddings
        self.state_embedding = nn.Linear(num_nodes, embedding_dim)
        self.goal_embedding = nn.Linear(num_nodes, embedding_dim)

        # GRU for sequential processing
        # Input: concatenated state and goal embeddings
        self.gru = nn.GRU(
            input_size=embedding_dim * 2,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Action head (outputs direction-space action logits: 5 actions)
        # This is the causal policy that actually selects environment actions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL
        )

        # Prospection head (outputs node-space predictions: num_nodes)
        # This is an auxiliary predictive head trained on future-state targets
        if self.has_prospection_head:
            self.prospection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_nodes)
            )
        else:
            self.prospection_head = None

        # Value head (outputs state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # GRU hidden state
        self.hidden = None

    def reset_hidden(self, batch_size=1):
        """Reset GRU hidden state."""
        self.hidden = torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, state_encoding, goal_encoding, hidden=None):
        """
        Forward pass through fast network.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        goal_encoding : torch.Tensor
            One-hot encoding of goal state [batch_size, num_nodes]
        hidden : torch.Tensor, optional
            Previous GRU hidden state [1, batch_size, hidden_dim]

        Returns:
        --------
        action_logits : torch.Tensor
            Logits over direction actions [batch_size, 5]
        prospection_logits : torch.Tensor
            Logits over node predictions [batch_size, num_nodes]
        value : torch.Tensor
            State value estimate [batch_size, 1]
        hidden : torch.Tensor
            Updated GRU hidden state [1, batch_size, hidden_dim]
        """
        batch_size = state_encoding.shape[0]

        # Embed state and goal
        state_embed = F.relu(self.state_embedding(state_encoding))
        goal_embed = F.relu(self.goal_embedding(goal_encoding))

        # Concatenate embeddings
        combined = torch.cat([state_embed, goal_embed], dim=-1)  # [batch_size, embedding_dim * 2]

        # Add sequence dimension for GRU
        combined = combined.unsqueeze(1)  # [batch_size, 1, embedding_dim * 2]

        # Pass through GRU
        if hidden is None:
            hidden = torch.zeros(1, batch_size, self.hidden_dim)

        gru_out, hidden = self.gru(combined, hidden)
        gru_out = gru_out.squeeze(1)  # [batch_size, hidden_dim]

        # Generate outputs from heads
        action_logits = self.action_head(gru_out)  # [batch_size, 5]
        prospection_logits = self.prospection_head(gru_out) if self.has_prospection_head else None  # [batch_size, num_nodes] or None
        value = self.value_head(gru_out)  # [batch_size, 1]

        return action_logits, prospection_logits, value, hidden

    def get_action_distribution(self, action_logits):
        """
        Convert action logits to probability distribution.

        Parameters:
        -----------
        action_logits : torch.Tensor
            Raw action logits [batch_size, num_actions]

        Returns:
        --------
        probs : torch.Tensor
            Action probabilities [batch_size, num_actions]
        """
        return F.softmax(action_logits, dim=-1)

    def compute_entropy(self, action_logits):
        """
        Compute entropy of action distribution.

        High entropy = low confidence / high uncertainty
        Low entropy = high confidence / low uncertainty

        Parameters:
        -----------
        action_logits : torch.Tensor
            Raw action logits [batch_size, num_actions]

        Returns:
        --------
        entropy : torch.Tensor
            Entropy of action distribution [batch_size]
        """
        probs = self.get_action_distribution(action_logits)
        log_probs = F.log_softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def sample_action(self, action_logits):
        """
        Sample action from policy distribution.

        Parameters:
        -----------
        action_logits : torch.Tensor
            Raw action logits [batch_size, num_actions]

        Returns:
        --------
        action : torch.Tensor
            Sampled action indices [batch_size]
        log_prob : torch.Tensor
            Log probability of sampled action [batch_size]
        """
        probs = self.get_action_distribution(action_logits)

        # Check for NaN or Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(f"WARNING: NaN or Inf detected in action probabilities!")
            print(f"  action_logits: min={action_logits.min().item():.4f}, max={action_logits.max().item():.4f}")
            print(f"  probs: min={probs.min().item():.4f}, max={probs.max().item():.4f}")
            print(f"  Contains NaN: {torch.isnan(action_logits).any()}")
            print(f"  Contains Inf: {torch.isinf(action_logits).any()}")
            # Replace NaN/Inf with uniform distribution
            probs = torch.ones_like(probs) / probs.shape[-1]

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(self, action_logits, action):
        """
        Get log probability of specific action.

        Parameters:
        -----------
        action_logits : torch.Tensor
            Raw action logits [batch_size, num_actions]
        action : torch.Tensor
            Action indices [batch_size]

        Returns:
        --------
        log_prob : torch.Tensor
            Log probability of action [batch_size]
        """
        log_probs = F.log_softmax(action_logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(1)).squeeze(1)


class FastNetworkTrainer:
    """Trainer for fast network using actor-critic with TD(λ)."""

    def __init__(self, network, lr=3e-4, gamma=0.99, lambda_=0.95, entropy_coef=0.01, value_coef=0.5, teacher_coef=1.0, prospection_coef=0.5):
        """
        Initialize trainer.

        Parameters:
        -----------
        network : FastNetwork
            The fast network to train
        lr : float
            Learning rate
        gamma : float
            Discount factor
        lambda_ : float
            TD(λ) trace decay parameter (will be modulated later)
        entropy_coef : float
            Coefficient for entropy bonus
        value_coef : float
            Coefficient for value loss
        teacher_coef : float
            Coefficient for teacher forcing loss when slow memory is used
        prospection_coef : float
            Coefficient for prospection auxiliary loss
        """
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.teacher_coef = teacher_coef
        self.prospection_coef = prospection_coef

    def compute_prospection_targets(self, trajectory, lambdas=None):
        """
        Compute λ-weighted future-state targets for prospection head.

        For each timestep t, builds soft target over future visited nodes:
        target_t[j] = Σ_{k≥1} [Π_{m=0}^{k-1} λ_{t+m}] × 𝟙[s_{t+k} = j]

        Parameters:
        -----------
        trajectory : dict
            Trajectory with 'states' (list of state encodings)
        lambdas : list of float, optional
            Lambda values for each timestep. If None, use self.lambda_

        Returns:
        --------
        targets : torch.Tensor
            Prospection targets [T, num_nodes] (normalized probability distributions)
        """
        states = trajectory['states']
        T = len(states)
        num_nodes = states[0].shape[0]

        # If lambdas not provided, use constant lambda
        if lambdas is None:
            lambdas = [self.lambda_] * T

        targets = []

        for t in range(T):
            target = torch.zeros(num_nodes)
            lambda_product = 1.0

            # Sum over future timesteps
            for k in range(1, T - t):
                # Accumulate lambda product
                lambda_product *= lambdas[t + k - 1]

                # Get state at t+k
                future_state = states[t + k]  # one-hot encoding
                state_idx = future_state.argmax().item()

                # Add contribution
                target[state_idx] += lambda_product

            # Normalize to probability distribution
            if target.sum() > 0:
                target = target / target.sum()
            else:
                # If no future states (end of trajectory), use uniform
                target = torch.ones(num_nodes) / num_nodes

            targets.append(target)

        return torch.stack(targets)

    def compute_gae(self, rewards, values, dones, next_value, gamma=None, lambda_=None):
        """
        Compute Generalized Advantage Estimation (GAE).

        Parameters:
        -----------
        rewards : list of float
            Rewards for each step
        values : list of torch.Tensor
            Value estimates for each step
        dones : list of bool
            Done flags for each step
        next_value : torch.Tensor
            Value estimate for final next state
        gamma : float, optional
            Discount factor (uses self.gamma if None)
        lambda_ : float, optional
            GAE lambda parameter (uses self.lambda_ if None)

        Returns:
        --------
        advantages : torch.Tensor
            Advantage estimates [T]
        returns : torch.Tensor
            Return estimates [T]
        """
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

        # Work backwards from end of trajectory
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values_list[t + 1]

            # TD error
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
            Dictionary containing:
            - states: list of state encodings
            - goals: list of goal encodings
            - actions: list of actions taken
            - rewards: list of rewards received
            - dones: list of done flags
            - log_probs: list of action log probabilities
            - values: list of value estimates
            - next_state: final next state encoding
            - next_goal: final next goal encoding
            - hiddens: list of hidden states
            - used_slow: (optional) list of bool flags indicating if slow was used

        Returns:
        --------
        loss_dict : dict
            Dictionary with loss components
        """
        states = torch.stack(trajectory['states'])
        goals = torch.stack(trajectory['goals'])
        actions = torch.tensor(trajectory['actions'], dtype=torch.long)
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        old_log_probs = torch.stack(trajectory['log_probs'])
        old_values = trajectory['values']

        # Get slow usage flags if available (for teacher forcing)
        used_slow = trajectory.get('used_slow', None)

        # Compute next value
        with torch.no_grad():
            next_state = trajectory['next_state'].unsqueeze(0)
            next_goal = trajectory['next_goal'].unsqueeze(0)
            _, _, next_value, _ = self.network(next_state, next_goal)
            next_value = next_value.squeeze()

        # Compute advantages and returns using GAE
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
        action_logits_list = []
        prospection_logits_list = []
        values_list = []
        entropies_list = []

        hidden = None
        for t in range(len(states)):
            state = states[t:t+1]
            goal = goals[t:t+1]

            action_logits, prospection_logits, value, hidden = self.network(state, goal, hidden)
            action_logits_list.append(action_logits)
            prospection_logits_list.append(prospection_logits)
            values_list.append(value.squeeze())
            entropies_list.append(self.network.compute_entropy(action_logits))

        action_logits = torch.cat(action_logits_list, dim=0)
        prospection_logits = torch.cat(prospection_logits_list, dim=0)
        values = torch.stack(values_list)
        entropies = torch.stack(entropies_list)

        # Compute new log probabilities
        log_probs = self.network.get_log_prob(action_logits, actions)

        # Policy loss (negative because we want to maximize expected return)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (encourage exploration)
        entropy_loss = -entropies.mean()

        # Teacher forcing loss: when slow memory is used, supervise fast network to match slow action
        teacher_loss = torch.tensor(0.0)
        if used_slow is not None and any(used_slow):
            # Create mask for steps where slow was used
            slow_mask = torch.tensor(used_slow, dtype=torch.bool)
            if slow_mask.any():
                # Cross-entropy loss between fast logits and actual actions (from slow memory)
                teacher_loss = F.cross_entropy(
                    action_logits[slow_mask],
                    actions[slow_mask]
                )

        # Prospection loss: auxiliary supervised loss on future-state predictions
        prospection_loss = torch.tensor(0.0)
        if self.network.has_prospection_head and 'lambdas' in trajectory:
            # Compute prospection targets using lambda-weighted future states
            prospection_targets = self.compute_prospection_targets(trajectory, trajectory['lambdas'])
            # Cross-entropy loss between prospection logits and soft targets
            prospection_probs = F.log_softmax(prospection_logits, dim=-1)
            prospection_loss = -(prospection_targets * prospection_probs).sum(dim=-1).mean()

        # Total loss
        loss = (policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss +
                self.teacher_coef * teacher_loss + self.prospection_coef * prospection_loss)

        # Check for NaN loss
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected! Skipping update.")
            print(f"  policy_loss: {policy_loss.item()}")
            print(f"  value_loss: {value_loss.item()}")
            print(f"  entropy_loss: {entropy_loss.item()}")
            print(f"  teacher_loss: {teacher_loss.item()}")
            return {
                'loss': float('nan'),
                'policy_loss': policy_loss.item() if not torch.isnan(policy_loss) else float('nan'),
                'value_loss': value_loss.item() if not torch.isnan(value_loss) else float('nan'),
                'entropy_loss': entropy_loss.item() if not torch.isnan(entropy_loss) else float('nan'),
                'teacher_loss': teacher_loss.item() if not torch.isnan(teacher_loss) else float('nan'),
                'prospection_loss': prospection_loss.item() if not torch.isnan(prospection_loss) else float('nan'),
                'mean_entropy': 0.0,
                'mean_value': 0.0
            }

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
            'teacher_loss': teacher_loss.item(),
            'prospection_loss': prospection_loss.item(),
            'mean_entropy': entropies.mean().item(),
            'mean_value': values.mean().item()
        }


if __name__ == "__main__":
    # Test the fast network with dual heads
    print("Testing Fast Network with dual heads...")

    num_nodes = 64
    num_actions = 5  # Direction-based actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=IDENTIFY_GOAL
    batch_size = 8

    # Create network
    network = FastNetwork(num_nodes, num_actions)
    print(f"Created FastNetwork with {sum(p.numel() for p in network.parameters())} parameters")

    # Create random inputs
    state_encoding = torch.zeros(batch_size, num_nodes)
    state_encoding[:, 0] = 1.0  # One-hot

    goal_encoding = torch.zeros(batch_size, num_nodes)
    goal_encoding[:, num_nodes-1] = 1.0  # One-hot

    # Forward pass
    action_logits, prospection_logits, value, hidden = network(state_encoding, goal_encoding)
    print(f"\nForward pass:")
    print(f"  Action logits shape: {action_logits.shape} (direction-space)")
    print(f"  Prospection logits shape: {prospection_logits.shape} (node-space)")
    print(f"  Value shape: {value.shape}")
    print(f"  Hidden shape: {hidden.shape}")

    # Compute entropy
    entropy = network.compute_entropy(action_logits)
    print(f"  Action entropy: {entropy.mean().item():.3f}")

    # Sample action
    action, log_prob = network.sample_action(action_logits)
    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT', 4: 'IDENTIFY_GOAL'}
    print(f"  Sampled actions: {[action_names[a.item()] for a in action]}")
    print(f"  Log probs: {log_prob}")

    # Test trainer
    print("\nTesting FastNetworkTrainer...")
    trainer = FastNetworkTrainer(network, lr=1e-3)

    # Create dummy trajectory
    T = 10
    trajectory = {
        'states': [torch.zeros(num_nodes) for _ in range(T)],
        'goals': [torch.zeros(num_nodes) for _ in range(T)],
        'actions': [np.random.randint(5) for _ in range(T)],  # Direction actions
        'rewards': [np.random.randn() for _ in range(T)],
        'dones': [False] * (T-1) + [True],
        'log_probs': [torch.tensor(0.0) for _ in range(T)],
        'values': [torch.tensor(0.0) for _ in range(T)],
        'next_state': torch.zeros(num_nodes),
        'next_goal': torch.zeros(num_nodes),
        'hiddens': [None] * T,
        'lambdas': [0.95] * T  # Add lambdas for prospection training
    }

    # Make states one-hot
    for i, state in enumerate(trajectory['states']):
        state[i % num_nodes] = 1.0
    for i, goal in enumerate(trajectory['goals']):
        goal[(i+1) % num_nodes] = 1.0

    trajectory['next_state'][0] = 1.0
    trajectory['next_goal'][1] = 1.0

    loss_dict = trainer.train_step(trajectory)
    print(f"Training step completed:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print("\n✓ Fast network tests passed!")
