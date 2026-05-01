"""
Simplified Fast Network (Feedforward instead of Recurrent)

MLP-based policy network that mirrors FastNetwork architecture exactly,
except using a feedforward MLP instead of GRU for feature extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SimpleFastNetwork(nn.Module):
    """
    Feedforward policy network (MLP instead of GRU).

    Mirrors FastNetwork exactly except:
    - Uses MLP instead of GRU for sequential processing
    - No hidden state to maintain between steps
    """

    def __init__(self, num_nodes, num_actions, embedding_dim=64, hidden_dim=128, prospection_head=True):
        """
        Initialize simple fast policy network.

        Parameters:
        -----------
        num_nodes : int
            Number of nodes in the maze (state space size)
        num_actions : int
            Number of possible actions
        embedding_dim : int
            Dimension of state and goal embeddings
        hidden_dim : int
            Dimension of MLP hidden layers
        prospection_head : bool
            Whether to include the prospection head (default: True)
        """
        super(SimpleFastNetwork, self).__init__()

        self.num_nodes = num_nodes
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.has_prospection_head = prospection_head

        # State and goal embeddings
        self.state_embedding = nn.Linear(num_nodes, embedding_dim)
        self.goal_embedding = nn.Linear(num_nodes, embedding_dim)

        # MLP for sequential processing (replaces GRU)
        # Input: concatenated state and goal embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
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

    def reset_hidden(self, batch_size=1):
        """Dummy method for compatibility with GRU-based interface."""
        pass

    def forward(self, state_encoding, goal_encoding, hidden=None):
        """
        Forward pass through simple fast network.

        Parameters:
        -----------
        state_encoding : torch.Tensor
            One-hot encoding of current state [batch_size, num_nodes]
        goal_encoding : torch.Tensor
            One-hot encoding of goal state [batch_size, num_nodes]
        hidden : torch.Tensor, optional
            Unused (for compatibility with GRU interface)

        Returns:
        --------
        action_logits : torch.Tensor
            Logits over direction actions [batch_size, 5]
        prospection_logits : torch.Tensor or None
            Logits over node predictions [batch_size, num_nodes] (if prospection_head=True)
        value : torch.Tensor
            State value estimate [batch_size, 1]
        hidden : None
            Always None (no hidden state for feedforward network)
        """
        batch_size = state_encoding.shape[0]

        # Embed state and goal
        state_embed = F.relu(self.state_embedding(state_encoding))
        goal_embed = F.relu(self.goal_embedding(goal_encoding))

        # Concatenate embeddings
        combined = torch.cat([state_embed, goal_embed], dim=-1)  # [batch_size, embedding_dim * 2]

        # Pass through MLP (replaces GRU processing)
        mlp_out = self.mlp(combined)  # [batch_size, hidden_dim]

        # Generate outputs from heads
        action_logits = self.action_head(mlp_out)  # [batch_size, 5]
        prospection_logits = self.prospection_head(mlp_out) if self.has_prospection_head else None  # [batch_size, num_nodes] or None
        value = self.value_head(mlp_out)  # [batch_size, 1]

        return action_logits, prospection_logits, value, None

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


if __name__ == "__main__":
    print("Testing SimpleFastNetwork (feedforward)...")

    num_nodes = 25
    num_actions = 5
    batch_size = 4

    # Test without prospection head
    print("\n--- Testing without prospection head ---")
    network = SimpleFastNetwork(num_nodes, num_actions, prospection_head=False)

    # Create dummy inputs
    state = torch.randn(batch_size, num_nodes)
    goal = torch.randn(batch_size, num_nodes)

    # Forward pass
    action_logits, prospection_logits, value, hidden = network(state, goal)

    print(f"Action logits shape: {action_logits.shape}")  # Should be [4, 5]
    print(f"Prospection logits: {prospection_logits}")  # Should be None
    print(f"Value shape: {value.shape}")  # Should be [4, 1]
    print(f"Hidden: {hidden}")  # Should be None

    # Sample action
    action, log_prob = network.sample_action(action_logits)
    print(f"Sampled actions: {action}")
    print(f"Log probs: {log_prob}")

    # Compute entropy
    entropy = network.compute_entropy(action_logits)
    print(f"Entropy: {entropy}")

    # Test with prospection head
    print("\n--- Testing with prospection head ---")
    network_with_prosp = SimpleFastNetwork(num_nodes, num_actions, prospection_head=True)

    action_logits, prospection_logits, value, hidden = network_with_prosp(state, goal)

    print(f"Action logits shape: {action_logits.shape}")  # Should be [4, 5]
    print(f"Prospection logits shape: {prospection_logits.shape}")  # Should be [4, 25]
    print(f"Value shape: {value.shape}")  # Should be [4, 1]
    print(f"Hidden: {hidden}")  # Should be None

    print("\n✓ SimpleFastNetwork tests passed!")
